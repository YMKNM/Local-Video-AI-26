"""
Generation Planner - The AI Agent Brain

This module is the central orchestrator that:
- Coordinates all generation components
- Makes intelligent decisions about resources
- Plans and executes video generation pipelines
- Manages the complete generation workflow
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import yaml

from .prompt_engine import PromptEngine, ExpandedPrompt
from .resource_monitor import ResourceMonitor, ResourceStatus
from .retry_logic import RetryManager, RetryResult, FallbackChain

logger = logging.getLogger(__name__)


@dataclass
class GenerationJob:
    """Represents a video generation job"""
    id: str
    prompt: str
    expanded_prompt: Optional[ExpandedPrompt] = None
    width: int = 854
    height: int = 480
    num_frames: int = 144  # 6 seconds at 24fps
    fps: int = 24
    duration_seconds: float = 6.0
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    negative_prompt: str = ""
    model_name: str = "ltx-video-2b"
    output_path: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        if self.expanded_prompt:
            result['expanded_prompt'] = self.expanded_prompt.to_dict()
        return result
    
    @property
    def duration(self) -> Optional[float]:
        """Get generation duration"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class GenerationResult:
    """Result of a video generation"""
    success: bool
    job: GenerationJob
    output_path: Optional[str]
    frame_paths: List[str]
    metadata: Dict[str, Any]
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class GenerationPlanner:
    """
    The AI Agent Brain - Central orchestrator for video generation.
    
    This is the director, not the renderer. It:
    - Validates and expands prompts
    - Decides optimal generation parameters
    - Monitors resources and adjusts dynamically
    - Coordinates model loading and inference
    - Handles failures with intelligent retry
    - Assembles final video output
    """
    
    def __init__(self, config_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        """
        Initialize the generation planner.
        
        Args:
            config_dir: Path to configuration directory
            output_dir: Path to output directory
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "outputs"
        
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-components
        self.prompt_engine = PromptEngine(config_dir)
        self.resource_monitor = ResourceMonitor(config_dir)
        self.retry_manager = RetryManager(config_dir=config_dir)
        self.fallback_chain = FallbackChain()
        
        # Load configurations
        self.defaults = self._load_defaults()
        self.model_config = self._load_model_config()
        
        # Job tracking
        self.current_job: Optional[GenerationJob] = None
        self.job_history: List[GenerationJob] = []
        
        # Runtime components (lazy loaded)
        self._inference_engine = None
        self._video_assembler = None
        
        logger.info("GenerationPlanner initialized")
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default settings"""
        defaults_path = self.config_dir / "defaults.yaml"
        if defaults_path.exists():
            with open(defaults_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        model_path = self.config_dir / "models.yaml"
        if model_path.exists():
            with open(model_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _generate_job_id(self) -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import random
        suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
        return f"job_{timestamp}_{suffix}"
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, str]:
        """
        Validate user prompt.
        
        Args:
            prompt: User's prompt string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.prompt_engine.validate_prompt(prompt)
    
    def plan_generation(
        self,
        prompt: str,
        duration_seconds: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        quality_preset: Optional[str] = None
    ) -> GenerationJob:
        """
        Plan a video generation job.
        
        This method:
        1. Validates the prompt
        2. Expands the prompt with details
        3. Checks resource availability
        4. Adjusts parameters for optimal quality
        5. Creates a generation job
        
        Args:
            prompt: User's prompt
            duration_seconds: Video duration
            width: Frame width
            height: Frame height
            fps: Frames per second
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
            seed: Random seed
            model_name: Model to use
            quality_preset: Quality preset (fast, balanced, quality)
            
        Returns:
            Configured GenerationJob
        """
        # Validate prompt
        is_valid, error = self.validate_prompt(prompt)
        if not is_valid:
            raise ValueError(f"Invalid prompt: {error}")
        
        # Apply quality preset if specified
        gen_config = self.defaults.get('generation', {})
        presets = self.defaults.get('presets', {})
        
        if quality_preset and quality_preset in presets:
            preset = presets[quality_preset]
            width = width or preset.get('width')
            height = height or preset.get('height')
            num_inference_steps = num_inference_steps or preset.get('num_inference_steps')
            guidance_scale = guidance_scale or preset.get('guidance_scale')
        
        # Apply defaults (Wan2.1-compatible)
        duration_seconds = duration_seconds or gen_config.get('duration_seconds', 2.0)
        width = width or gen_config.get('width', 832)
        height = height or gen_config.get('height', 480)
        fps = fps or gen_config.get('fps', 16)
        num_inference_steps = num_inference_steps or gen_config.get('num_inference_steps', 30)
        guidance_scale = guidance_scale or gen_config.get('guidance_scale', 5.0)
        
        # Get active model — prefer explicit model_name, then registry default
        if not model_name:
            try:
                from ..runtime.diffusers_pipeline import DEFAULT_MODEL
                model_name = DEFAULT_MODEL
            except ImportError:
                model_name = 'wan2.1-t2v-1.3b'
        
        # Calculate number of frames — prefer explicit num_frames from caller
        if num_frames is None:
            num_frames = int(duration_seconds * fps)
        else:
            # Derive duration from explicit frame count
            duration_seconds = num_frames / fps
        
        # Check resources and adjust if needed
        resource_status = self.resource_monitor.get_resource_status()
        
        # Adjust for available resources
        suggested = self.resource_monitor.suggest_reduced_settings(
            width, height, num_frames
        )
        
        if suggested['reduced']:
            logger.warning(
                f"Reducing settings for available VRAM: "
                f"{width}x{height}@{num_frames} -> "
                f"{suggested['width']}x{suggested['height']}@{suggested['frames']}"
            )
            width = suggested['width']
            height = suggested['height']
            num_frames = suggested['frames']
            duration_seconds = num_frames / fps
        
        # Expand the prompt (model-family-aware for negative prompts & tags)
        model_family = ""
        try:
            from ..runtime.model_registry import get_model
            spec = get_model(model_name)
            if spec:
                model_family = spec.family or ""
        except Exception:
            pass

        expanded = self.prompt_engine.expand(
            prompt,
            fps=fps,
            duration=duration_seconds,
            resolution=(width, height),
            seed=seed,
            model_family=model_family,
        )
        
        # Use seed from expanded prompt if not specified
        seed = seed or expanded.seed
        
        # Create job
        job = GenerationJob(
            id=self._generate_job_id(),
            prompt=prompt,
            expanded_prompt=expanded,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            duration_seconds=duration_seconds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            negative_prompt=negative_prompt if negative_prompt else expanded.negative,
            model_name=model_name,
            metadata={
                'resource_status': {
                    'gpu_vram_total': resource_status.gpu.vram_total_gb,
                    'gpu_vram_free': resource_status.gpu.vram_free_gb,
                    'gpu_backend': resource_status.gpu.backend,
                    'settings_reduced': suggested['reduced']
                },
                'quality_preset': quality_preset,
                'warnings': resource_status.warnings
            }
        )
        
        logger.info(f"Planned job {job.id}: {width}x{height} @ {fps}fps, {num_frames} frames")
        
        return job
    
    def execute_job(self, job: GenerationJob) -> GenerationResult:
        """
        Execute a planned generation job.
        
        Args:
            job: The generation job to execute
            
        Returns:
            GenerationResult with output details
        """
        self.current_job = job
        job.status = "running"
        job.started_at = time.time()
        
        logger.info(f"Executing job {job.id}")
        
        frame_paths = []
        warnings = []
        
        try:
            # Create output directory for this job
            job_output_dir = self.output_dir / job.id
            job_output_dir.mkdir(parents=True, exist_ok=True)
            frames_dir = job_output_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            # Get inference engine and ensure correct model is loaded
            inference_engine = self._get_inference_engine()
            
            # Load / switch to the model specified in the job
            current = inference_engine.get_current_model_id()
            if current != job.model_name:
                logger.info(f"Switching model to {job.model_name} (was {current})")
                inference_engine.set_model(job.model_name)
            elif not inference_engine._diffusers_mode and inference_engine._text_encoder_session is None:
                # Engine exists but nothing loaded yet
                inference_engine.load_models(model_id=job.model_name)
            
            # Execute generation with retry logic
            result = self.retry_manager.execute_with_retry(
                inference_engine.generate_frames,
                {
                    'prompt': job.expanded_prompt.expanded,
                    'negative_prompt': job.negative_prompt,
                    'width': job.width,
                    'height': job.height,
                    'num_frames': job.num_frames,
                    'num_inference_steps': job.num_inference_steps,
                    'guidance_scale': job.guidance_scale,
                    'seed': job.seed,
                    'output_dir': str(frames_dir)
                },
                description=f"frame generation for job {job.id}"
            )
            
            if not result.success:
                raise RuntimeError(f"Frame generation failed: {result.final_error}")
            
            frame_paths = result.result or []
            
            if result.parameters_adjusted:
                warnings.append("Parameters were adjusted due to resource constraints")
            
            # Assemble video
            video_assembler = self._get_video_assembler()
            output_path = job_output_dir / f"{job.id}.mp4"
            
            video_assembler.assemble(
                frame_paths=frame_paths,
                output_path=str(output_path),
                fps=job.fps
            )
            
            job.output_path = str(output_path)
            job.status = "completed"
            job.completed_at = time.time()
            
            # Save metadata
            self._save_job_metadata(job, job_output_dir)
            
            logger.info(f"Job {job.id} completed successfully in {job.duration:.2f}s")
            
            return GenerationResult(
                success=True,
                job=job,
                output_path=str(output_path),
                frame_paths=frame_paths,
                metadata=job.to_dict(),
                warnings=warnings
            )
            
        except Exception as e:
            job.status = "failed"
            job.completed_at = time.time()
            job.error_message = str(e)
            
            logger.error(f"Job {job.id} failed: {e}")
            
            return GenerationResult(
                success=False,
                job=job,
                output_path=None,
                frame_paths=frame_paths,
                metadata=job.to_dict(),
                error=str(e),
                warnings=warnings
            )
        
        finally:
            self.job_history.append(job)
            self.current_job = None
    
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> GenerationResult:
        """
        High-level generation API - plan and execute in one call.
        
        Args:
            prompt: User's prompt
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult
        """
        job = self.plan_generation(prompt, **kwargs)
        return self.execute_job(job)
    
    def _get_inference_engine(self):
        """Get or create inference engine (lazy loading)"""
        if self._inference_engine is None:
            from ..runtime import InferenceEngine
            self._inference_engine = InferenceEngine(self.config_dir)
        return self._inference_engine
    
    def _get_video_assembler(self):
        """Get or create video assembler (lazy loading)"""
        if self._video_assembler is None:
            from ..video import VideoAssembler
            self._video_assembler = VideoAssembler(self.config_dir)
        return self._video_assembler
    
    def _save_job_metadata(self, job: GenerationJob, output_dir: Path):
        """Save job metadata to JSON file"""
        metadata_path = output_dir / "metadata.json"
        
        metadata = {
            'job': job.to_dict(),
            'generation_time': job.duration,
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'gpu': self.resource_monitor.get_gpu_info().__dict__,
                'system': self.resource_monitor.get_system_info().__dict__
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job by ID"""
        if self.current_job and self.current_job.id == job_id:
            return self.current_job.to_dict()
        
        for job in self.job_history:
            if job.id == job_id:
                return job.to_dict()
        
        return None
    
    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all jobs, optionally filtered by status"""
        jobs = self.job_history.copy()
        if self.current_job:
            jobs.append(self.current_job)
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return [j.to_dict() for j in jobs]
    
    def estimate_generation_time(
        self,
        width: int,
        height: int,
        num_frames: int,
        num_steps: int
    ) -> float:
        """
        Estimate generation time in seconds.
        
        This is a rough estimate based on typical performance.
        
        Args:
            width: Frame width
            height: Frame height
            num_frames: Number of frames
            num_steps: Diffusion steps
            
        Returns:
            Estimated time in seconds
        """
        # Base time per step per frame (rough estimate for DirectML)
        base_time_per_step = 0.5  # seconds
        
        # Scale by resolution
        resolution_factor = (width * height) / (512 * 512)
        
        # Calculate estimate
        estimated = base_time_per_step * num_steps * num_frames * resolution_factor
        
        # Add overhead for model loading, VAE decoding, etc.
        overhead = 10.0  # seconds
        
        return estimated + overhead
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get current system capabilities"""
        status = self.resource_monitor.get_resource_status()
        
        return {
            'gpu': {
                'name': status.gpu.name,
                'vram_total_gb': status.gpu.vram_total_gb,
                'vram_free_gb': status.gpu.vram_free_gb,
                'backend': status.gpu.backend,
                'available': status.gpu.is_available
            },
            'recommended_settings': {
                'resolution': status.recommended_resolution,
                'max_frames': status.recommended_max_frames
            },
            'supported_models': list(
                self.model_config.get('video_diffusion', {}).keys()
            ),
            'warnings': status.warnings
        }


def main():
    """Test the generation planner"""
    logging.basicConfig(level=logging.INFO)
    
    planner = GenerationPlanner()
    
    # Show capabilities
    print("\n=== System Capabilities ===")
    capabilities = planner.get_capabilities()
    print(json.dumps(capabilities, indent=2))
    
    # Plan a job
    print("\n=== Planning Generation Job ===")
    job = planner.plan_generation(
        prompt="A cinematic drone shot over the ocean at sunset",
        duration_seconds=6,
        quality_preset="balanced"
    )
    
    print(f"Job ID: {job.id}")
    print(f"Resolution: {job.width}x{job.height}")
    print(f"Frames: {job.num_frames}")
    print(f"FPS: {job.fps}")
    print(f"Steps: {job.num_inference_steps}")
    print(f"Seed: {job.seed}")
    print(f"\nExpanded Prompt:\n{job.expanded_prompt.expanded}")
    print(f"\nNegative Prompt:\n{job.negative_prompt}")
    
    # Estimate time
    estimated_time = planner.estimate_generation_time(
        job.width, job.height, job.num_frames, job.num_inference_steps
    )
    print(f"\nEstimated generation time: {estimated_time:.1f} seconds")


if __name__ == "__main__":
    main()
