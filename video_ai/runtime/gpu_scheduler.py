"""
GPU Scheduler - Intelligent GPU/TPU Dispatch with Quantization

This module handles:
- Job queuing and priority scheduling
- GPU resource allocation
- Dynamic quantization (FP16/BF16/INT8/NVFP8)
- Model pruning for inference speed
- Progressive preview generation
- Multi-model orchestration
"""

import os
import gc
import time
import logging
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels"""
    REALTIME = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 10


class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    LOADING = "loading"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QuantizationMode(Enum):
    """Quantization precision modes"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    NVFP8 = "nvfp8"
    DYNAMIC = "dynamic"


@dataclass
class ScheduledJob:
    """Represents a scheduled generation job"""
    id: str
    priority: JobPriority
    status: JobStatus
    model_name: str
    prompt: str
    config: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    estimated_time: Optional[float] = None
    vram_required_gb: float = 0.0
    error: Optional[str] = None
    result: Optional[Any] = None
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        """Compare for priority queue (higher priority = lower value for min-heap)"""
        return self.priority.value > other.priority.value


@dataclass
class ResourceAllocation:
    """GPU resource allocation for a job"""
    job_id: str
    device_id: int
    vram_allocated_gb: float
    quantization_mode: QuantizationMode
    chunk_size: int
    use_cpu_offload: bool


class GPUScheduler:
    """
    Intelligent GPU/TPU dispatch scheduler.
    
    Features:
    - Priority-based job queue
    - Dynamic resource allocation
    - Automatic quantization selection
    - Model switching based on resources
    - Progressive preview (~30 sec target)
    - Preemption for high-priority jobs
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize GPU scheduler.
        
        Args:
            config_dir: Path to configuration directory
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        
        self.config_dir = Path(config_dir)
        self.hardware_config = self._load_hardware_config()
        self.scheduler_config = self.hardware_config.get('scheduler', {})
        
        # Job queues
        self._job_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._active_jobs: Dict[str, ScheduledJob] = {}
        self._completed_jobs: Dict[str, ScheduledJob] = {}
        
        # Resource tracking
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._total_vram_gb: float = 10.0  # RTX 3080
        self._available_vram_gb: float = 10.0
        
        # Threading
        self._lock = threading.RLock()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Callbacks
        self._progress_callbacks: Dict[str, Callable] = {}
        
        # Session manager (lazy loaded)
        self._cuda_session = None
        
        logger.info("GPUScheduler initialized")
    
    def _load_hardware_config(self) -> Dict[str, Any]:
        """Load hardware configuration"""
        config_path = self.config_dir / "hardware.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    @property
    def cuda_session(self):
        """Lazy-load CUDA session"""
        if self._cuda_session is None:
            from .cuda_session import CUDASession
            self._cuda_session = CUDASession(self.config_dir)
            self._cuda_session.initialize()
            
            # Update VRAM info
            mem_info = self._cuda_session.get_memory_info()
            self._total_vram_gb = mem_info['total']
            self._available_vram_gb = mem_info['free']
        
        return self._cuda_session
    
    def start(self):
        """Start the scheduler background thread"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="GPUScheduler"
        )
        self._scheduler_thread.start()
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        logger.info("Scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                # Check for available resources
                self._update_resource_status()
                
                # Process queue
                self._process_queue()
                
                # Small delay to prevent busy-waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(1.0)
    
    def _update_resource_status(self):
        """Update available GPU resources"""
        try:
            mem_info = self.cuda_session.get_memory_info()
            self._available_vram_gb = mem_info['free']
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
    
    def _process_queue(self):
        """Process jobs from the queue"""
        with self._lock:
            # Check max concurrent jobs
            max_concurrent = self.scheduler_config.get('max_concurrent_jobs', 2)
            if len(self._active_jobs) >= max_concurrent:
                return
            
            # Get next job
            try:
                job: ScheduledJob = self._job_queue.get_nowait()
            except queue.Empty:
                return
            
            # Check resource availability
            if job.vram_required_gb > self._available_vram_gb:
                # Check for preemption
                if job.priority.value >= JobPriority.HIGH.value:
                    if self._try_preempt(job.vram_required_gb):
                        pass  # Resources freed, continue
                    else:
                        # Requeue job
                        self._job_queue.put(job)
                        return
                else:
                    # Requeue job
                    self._job_queue.put(job)
                    return
            
            # Allocate resources and start job
            allocation = self._allocate_resources(job)
            if allocation:
                self._start_job(job, allocation)
    
    def _try_preempt(self, required_vram: float) -> bool:
        """Try to preempt lower-priority jobs to free resources"""
        # Find jobs to preempt (lowest priority first)
        candidates = sorted(
            self._active_jobs.values(),
            key=lambda j: j.priority.value
        )
        
        freed_vram = 0.0
        jobs_to_pause = []
        
        for job in candidates:
            if job.priority.value >= JobPriority.HIGH.value:
                break  # Don't preempt high priority jobs
            
            allocation = self._allocations.get(job.id)
            if allocation:
                freed_vram += allocation.vram_allocated_gb
                jobs_to_pause.append(job)
                
                if freed_vram >= required_vram:
                    break
        
        if freed_vram >= required_vram:
            for job in jobs_to_pause:
                self._pause_job(job.id)
            return True
        
        return False
    
    def _allocate_resources(self, job: ScheduledJob) -> Optional[ResourceAllocation]:
        """
        Allocate GPU resources for a job.
        
        Determines optimal quantization based on available VRAM.
        """
        # Determine quantization mode based on available VRAM
        quant_mode = self._select_quantization(job.vram_required_gb)
        
        # Calculate adjusted VRAM requirement
        vram_multiplier = {
            QuantizationMode.FP32: 1.0,
            QuantizationMode.FP16: 0.5,
            QuantizationMode.BF16: 0.5,
            QuantizationMode.INT8: 0.25,
            QuantizationMode.NVFP8: 0.25,
            QuantizationMode.DYNAMIC: 0.5
        }
        
        adjusted_vram = job.vram_required_gb * vram_multiplier.get(quant_mode, 1.0)
        
        # Determine chunk size based on available memory
        chunk_size = self._calculate_chunk_size(adjusted_vram)
        
        # Determine if CPU offload is needed
        use_cpu_offload = adjusted_vram > self._available_vram_gb * 0.9
        
        allocation = ResourceAllocation(
            job_id=job.id,
            device_id=0,
            vram_allocated_gb=adjusted_vram,
            quantization_mode=quant_mode,
            chunk_size=chunk_size,
            use_cpu_offload=use_cpu_offload
        )
        
        self._allocations[job.id] = allocation
        self._available_vram_gb -= adjusted_vram
        
        logger.info(f"Allocated {adjusted_vram:.1f}GB VRAM for job {job.id} "
                   f"(quantization={quant_mode.value}, chunk_size={chunk_size})")
        
        return allocation
    
    def _select_quantization(self, base_vram_required: float) -> QuantizationMode:
        """Select optimal quantization mode based on resources"""
        quant_config = self.hardware_config.get('quantization', {})
        
        if not quant_config.get('enabled', True):
            return QuantizationMode.FP32
        
        # Check dynamic quantization
        if quant_config.get('dynamic_quantization', True):
            # Select based on available VRAM
            if self._available_vram_gb >= base_vram_required:
                return QuantizationMode.FP32
            elif self._available_vram_gb >= base_vram_required * 0.5:
                return QuantizationMode.FP16
            elif self._available_vram_gb >= base_vram_required * 0.25:
                return QuantizationMode.INT8
            else:
                return QuantizationMode.INT8
        
        # Use default precision
        default = quant_config.get('default_precision', 'fp16')
        return QuantizationMode(default)
    
    def _calculate_chunk_size(self, vram_allocated: float) -> int:
        """Calculate optimal chunk size for processing"""
        # Base chunk sizes from config
        res_scaling = self.hardware_config.get('resolution_scaling', {})
        
        if vram_allocated >= 8:
            return res_scaling.get('ultra_vram', {}).get('chunk_size', 8)
        elif vram_allocated >= 6:
            return res_scaling.get('high_vram', {}).get('chunk_size', 16)
        elif vram_allocated >= 4:
            return res_scaling.get('medium_vram', {}).get('chunk_size', 24)
        else:
            return res_scaling.get('low_vram', {}).get('chunk_size', 32)
    
    def _start_job(self, job: ScheduledJob, allocation: ResourceAllocation):
        """Start executing a job"""
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self._active_jobs[job.id] = job
        
        # Start job in background thread
        thread = threading.Thread(
            target=self._execute_job,
            args=(job, allocation),
            daemon=True,
            name=f"Job-{job.id}"
        )
        thread.start()
    
    def _execute_job(self, job: ScheduledJob, allocation: ResourceAllocation):
        """Execute a job (runs in background thread)"""
        try:
            logger.info(f"Executing job {job.id}: {job.model_name}")
            
            # TODO: Actual model execution
            # This is where the inference engine would be called
            
            # Simulate progress for now
            for i in range(10):
                if job.status == JobStatus.CANCELLED:
                    break
                if job.status == JobStatus.PAUSED:
                    # Wait for resume
                    while job.status == JobStatus.PAUSED:
                        time.sleep(0.1)
                
                job.progress = (i + 1) / 10
                
                # Report progress
                if job.id in self._progress_callbacks:
                    self._progress_callbacks[job.id](job.progress, f"Step {i+1}/10")
                
                time.sleep(0.5)  # Simulate work
            
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            
            logger.info(f"Job {job.id} completed in {job.completed_at - job.started_at:.1f}s")
            
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
        
        finally:
            # Release resources
            self._release_resources(job.id)
            
            # Move to completed
            with self._lock:
                if job.id in self._active_jobs:
                    del self._active_jobs[job.id]
                self._completed_jobs[job.id] = job
            
            # Call completion callback
            if job.callback:
                job.callback(job)
    
    def _release_resources(self, job_id: str):
        """Release resources allocated to a job"""
        with self._lock:
            if job_id in self._allocations:
                allocation = self._allocations[job_id]
                self._available_vram_gb += allocation.vram_allocated_gb
                del self._allocations[job_id]
                logger.info(f"Released {allocation.vram_allocated_gb:.1f}GB VRAM from job {job_id}")
    
    def _pause_job(self, job_id: str):
        """Pause a running job"""
        with self._lock:
            if job_id in self._active_jobs:
                job = self._active_jobs[job_id]
                job.status = JobStatus.PAUSED
                logger.info(f"Paused job {job_id}")
    
    def submit(
        self,
        job_id: str,
        model_name: str,
        prompt: str,
        config: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        vram_required_gb: float = 6.0,
        callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> ScheduledJob:
        """
        Submit a new job to the scheduler.
        
        Args:
            job_id: Unique job identifier
            model_name: Name of model to use
            prompt: Generation prompt
            config: Generation configuration
            priority: Job priority
            vram_required_gb: Estimated VRAM requirement
            callback: Completion callback
            progress_callback: Progress callback
            
        Returns:
            ScheduledJob object
        """
        job = ScheduledJob(
            id=job_id,
            priority=priority,
            status=JobStatus.QUEUED,
            model_name=model_name,
            prompt=prompt,
            config=config,
            vram_required_gb=vram_required_gb,
            callback=callback
        )
        
        if progress_callback:
            self._progress_callbacks[job_id] = progress_callback
        
        self._job_queue.put(job)
        logger.info(f"Submitted job {job_id} with priority {priority.name}")
        
        return job
    
    def submit_preview(
        self,
        job_id: str,
        model_name: str,
        prompt: str,
        config: Dict[str, Any],
        target_time_seconds: float = 30.0
    ) -> ScheduledJob:
        """
        Submit a fast preview job (~30 sec target).
        
        Uses optimized settings for rapid feedback.
        """
        preview_config = self.hardware_config.get('progressive_preview', {})
        preview_settings = preview_config.get('preview_settings', {})
        
        # Apply preview optimizations
        config = config.copy()
        config['resolution_scale'] = preview_settings.get('resolution_scale', 0.25)
        config['steps_scale'] = preview_settings.get('steps_scale', 0.3)
        config['skip_refinement'] = preview_settings.get('skip_refinement', True)
        config['use_fast_sampler'] = preview_settings.get('use_fast_sampler', True)
        
        # Use fast model for preview
        fast_model = 'accvideo'
        
        return self.submit(
            job_id=job_id,
            model_name=fast_model,
            prompt=prompt,
            config=config,
            priority=JobPriority.HIGH,
            vram_required_gb=4.0
        )
    
    def cancel(self, job_id: str) -> bool:
        """Cancel a job"""
        with self._lock:
            if job_id in self._active_jobs:
                job = self._active_jobs[job_id]
                job.status = JobStatus.CANCELLED
                logger.info(f"Cancelled job {job_id}")
                return True
        return False
    
    def get_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status"""
        with self._lock:
            if job_id in self._active_jobs:
                return self._active_jobs[job_id].status
            if job_id in self._completed_jobs:
                return self._completed_jobs[job_id].status
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status"""
        with self._lock:
            return {
                'queued_jobs': self._job_queue.qsize(),
                'active_jobs': len(self._active_jobs),
                'completed_jobs': len(self._completed_jobs),
                'total_vram_gb': self._total_vram_gb,
                'available_vram_gb': self._available_vram_gb,
                'active_allocations': [
                    {
                        'job_id': a.job_id,
                        'vram_gb': a.vram_allocated_gb,
                        'quantization': a.quantization_mode.value
                    }
                    for a in self._allocations.values()
                ]
            }


class ModelPruner:
    """
    Model pruning utilities for inference optimization.
    
    Implements structured and unstructured pruning
    for faster inference without significant quality loss.
    """
    
    @staticmethod
    def prune_attention_heads(model, keep_ratio: float = 0.9):
        """Prune attention heads based on importance"""
        # TODO: Implement attention head pruning
        logger.info(f"Attention head pruning with keep_ratio={keep_ratio}")
        return model
    
    @staticmethod
    def prune_channels(model, keep_ratio: float = 0.9):
        """Prune convolutional channels"""
        # TODO: Implement channel pruning
        logger.info(f"Channel pruning with keep_ratio={keep_ratio}")
        return model
    
    @staticmethod
    def apply_magnitude_pruning(model, sparsity: float = 0.3):
        """Apply magnitude-based unstructured pruning"""
        try:
            import torch.nn.utils.prune as prune
            
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
            
            logger.info(f"Applied magnitude pruning with sparsity={sparsity}")
            return model
            
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
            return model


class DynamicQuantizer:
    """
    Dynamic quantization utilities.
    
    Implements runtime quantization for
    optimized inference based on hardware capabilities.
    """
    
    def __init__(self, device_info: Dict[str, Any] = None):
        self.device_info = device_info or {}
    
    def quantize(
        self,
        model,
        mode: QuantizationMode = QuantizationMode.FP16,
        calibration_data: Optional[List] = None
    ):
        """
        Quantize model to specified precision.
        
        Args:
            model: PyTorch model
            mode: Target quantization mode
            calibration_data: Data for INT8 calibration
            
        Returns:
            Quantized model
        """
        try:
            import torch
            
            if mode == QuantizationMode.FP16:
                return model.half()
            
            elif mode == QuantizationMode.BF16:
                return model.to(torch.bfloat16)
            
            elif mode == QuantizationMode.INT8:
                if calibration_data is None:
                    # Dynamic quantization
                    return torch.quantization.quantize_dynamic(
                        model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                else:
                    # Static quantization with calibration
                    logger.info("INT8 static quantization requires calibration")
                    return model
            
            elif mode == QuantizationMode.NVFP8:
                # FP8 requires specific NVIDIA hardware (H100+)
                logger.warning("NVFP8 requires H100 or newer GPU")
                return model.half()
            
            else:
                return model
                
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model
    
    def auto_select_precision(self, model_size_gb: float, available_vram_gb: float) -> QuantizationMode:
        """Automatically select best precision based on resources"""
        if available_vram_gb >= model_size_gb * 2:
            return QuantizationMode.FP32
        elif available_vram_gb >= model_size_gb:
            return QuantizationMode.FP16
        elif available_vram_gb >= model_size_gb * 0.5:
            return QuantizationMode.INT8
        else:
            return QuantizationMode.INT8
