"""
Video AI - Python API

High-level API for programmatic video generation.

Usage:
    from video_ai.api import VideoGenerator
    
    generator = VideoGenerator()
    video = generator.generate("A sunset over the ocean")
    print(video.output_path)
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

from video_ai.agent import GenerationPlanner, PromptEngine, ResourceMonitor
from video_ai.agent.planner import GenerationJob, GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class VideoGenerationConfig:
    """Configuration for video generation"""
    # Video settings
    duration_seconds: float = 6.0
    width: int = 854
    height: int = 480
    fps: int = 24
    
    # Quality settings
    quality_preset: str = "balanced"  # fast, balanced, quality
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    
    # Reproducibility
    seed: Optional[int] = None
    
    # Output
    output_dir: Optional[str] = None
    save_frames: bool = False
    save_metadata: bool = True
    
    # Model
    model_name: Optional[str] = None


class VideoGenerator:
    """
    High-level video generation API.
    
    Example:
        generator = VideoGenerator()
        
        # Simple generation
        result = generator.generate("A sunset over the ocean")
        
        # With options
        result = generator.generate(
            prompt="A cat playing",
            duration=10,
            quality="high",
            seed=42
        )
    """
    
    def __init__(
        self,
        config_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        config: Optional[VideoGenerationConfig] = None
    ):
        """
        Initialize the video generator.
        
        Args:
            config_dir: Path to configuration directory
            output_dir: Default output directory
            config: Default generation configuration
        """
        self.config_dir = Path(config_dir) if config_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.default_config = config or VideoGenerationConfig()
        
        # Lazy-load planner
        self._planner: Optional[GenerationPlanner] = None
        self._prompt_engine: Optional[PromptEngine] = None
        self._resource_monitor: Optional[ResourceMonitor] = None
        
        # Callbacks
        self._progress_callback: Optional[Callable[[float, str], None]] = None
        self._completion_callback: Optional[Callable[[GenerationResult], None]] = None
    
    @property
    def planner(self) -> GenerationPlanner:
        """Get or create generation planner"""
        if self._planner is None:
            self._planner = GenerationPlanner(
                config_dir=self.config_dir,
                output_dir=self.output_dir
            )
        return self._planner
    
    @property
    def prompt_engine(self) -> PromptEngine:
        """Get or create prompt engine"""
        if self._prompt_engine is None:
            self._prompt_engine = PromptEngine(self.config_dir)
        return self._prompt_engine
    
    @property
    def resource_monitor(self) -> ResourceMonitor:
        """Get or create resource monitor"""
        if self._resource_monitor is None:
            self._resource_monitor = ResourceMonitor(self.config_dir)
        return self._resource_monitor
    
    def on_progress(self, callback: Callable[[float, str], None]):
        """
        Set progress callback.
        
        Args:
            callback: Function(progress, message)
        """
        self._progress_callback = callback
        return self
    
    def on_complete(self, callback: Callable[[GenerationResult], None]):
        """
        Set completion callback.
        
        Args:
            callback: Function(result)
        """
        self._completion_callback = callback
        return self
    
    def generate(
        self,
        prompt: str,
        duration: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        quality: Optional[str] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate a video from a text prompt.
        
        Args:
            prompt: Text description of the video
            duration: Video duration in seconds
            width: Frame width
            height: Frame height
            fps: Frames per second
            quality: Quality preset (fast, balanced, quality)
            seed: Random seed for reproducibility
            steps: Number of inference steps
            **kwargs: Additional options
            
        Returns:
            GenerationResult with output path and metadata
        """
        # Merge with defaults
        config = VideoGenerationConfig(
            duration_seconds=duration or self.default_config.duration_seconds,
            width=width or self.default_config.width,
            height=height or self.default_config.height,
            fps=fps or self.default_config.fps,
            quality_preset=quality or self.default_config.quality_preset,
            num_inference_steps=steps or self.default_config.num_inference_steps,
            seed=seed or self.default_config.seed,
        )
        
        # Plan generation
        job = self.planner.plan_generation(
            prompt=prompt,
            duration_seconds=config.duration_seconds,
            width=config.width,
            height=config.height,
            fps=config.fps,
            quality_preset=config.quality_preset,
            num_inference_steps=config.num_inference_steps,
            seed=config.seed,
            **kwargs
        )
        
        # Execute
        result = self.planner.execute_job(job)
        
        # Call completion callback
        if self._completion_callback:
            self._completion_callback(result)
        
        return result
    
    def plan(
        self,
        prompt: str,
        **kwargs
    ) -> GenerationJob:
        """
        Plan a generation without executing it.
        
        Args:
            prompt: Text description
            **kwargs: Generation options
            
        Returns:
            GenerationJob that can be executed later
        """
        return self.planner.plan_generation(prompt, **kwargs)
    
    def execute(self, job: GenerationJob) -> GenerationResult:
        """
        Execute a planned job.
        
        Args:
            job: Pre-planned generation job
            
        Returns:
            GenerationResult
        """
        result = self.planner.execute_job(job)
        
        if self._completion_callback:
            self._completion_callback(result)
        
        return result
    
    def expand_prompt(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Expand a prompt without generating.
        
        Args:
            prompt: User's prompt
            **kwargs: Expansion options
            
        Returns:
            Dictionary with expanded prompt details
        """
        expanded = self.prompt_engine.expand(prompt, **kwargs)
        return expanded.to_dict()
    
    def check_resources(self) -> Dict[str, Any]:
        """
        Check current resource status.
        
        Returns:
            Dictionary with GPU, RAM, and recommendation info
        """
        status = self.resource_monitor.get_resource_status()
        
        return {
            'gpu': {
                'name': status.gpu.name,
                'vram_total_gb': status.gpu.vram_total_gb,
                'vram_free_gb': status.gpu.vram_free_gb,
                'backend': status.gpu.backend,
                'available': status.gpu.is_available
            },
            'system': {
                'ram_total_gb': status.system.ram_total_gb,
                'ram_available_gb': status.system.ram_available_gb,
                'cpu_count': status.system.cpu_count
            },
            'recommendations': {
                'resolution': status.recommended_resolution,
                'max_frames': status.recommended_max_frames,
                'high_quality_possible': status.can_run_high_quality
            },
            'warnings': status.warnings
        }
    
    def can_generate(
        self,
        width: int,
        height: int,
        num_frames: int
    ) -> tuple[bool, str]:
        """
        Check if generation is possible with given parameters.
        
        Args:
            width: Frame width
            height: Frame height
            num_frames: Number of frames
            
        Returns:
            Tuple of (can_generate, message)
        """
        return self.resource_monitor.can_generate(width, height, num_frames)
    
    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List generation jobs.
        
        Args:
            status: Filter by status (pending, running, completed, failed)
            
        Returns:
            List of job dictionaries
        """
        return self.planner.list_jobs(status)
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job dictionary or None
        """
        return self.planner.get_job_status(job_id)


# Convenience functions
def generate(prompt: str, **kwargs) -> GenerationResult:
    """
    Quick generation function.
    
    Args:
        prompt: Text prompt
        **kwargs: Generation options
        
    Returns:
        GenerationResult
    """
    generator = VideoGenerator()
    return generator.generate(prompt, **kwargs)


def expand_prompt(prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Expand a prompt.
    
    Args:
        prompt: Text prompt
        **kwargs: Expansion options
        
    Returns:
        Expanded prompt details
    """
    generator = VideoGenerator()
    return generator.expand_prompt(prompt, **kwargs)


def check_system() -> Dict[str, Any]:
    """
    Check system resources and capabilities.
    
    Returns:
        System status dictionary
    """
    generator = VideoGenerator()
    return generator.check_resources()


__all__ = [
    'VideoGenerator',
    'VideoGenerationConfig',
    'generate',
    'expand_prompt',
    'check_system',
]
