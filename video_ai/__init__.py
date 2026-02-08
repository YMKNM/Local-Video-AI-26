"""
Video AI Main Package
Local AI video generation system for AMD GPUs on Windows
"""

from .agent import GenerationPlanner, PromptEngine, ResourceMonitor, RetryManager

__version__ = "0.1.0"
__author__ = "Video AI Project"

# UI launcher function
def launch_ui(**kwargs):
    """
    Launch the web-based user interface.
    
    Args:
        port: Server port (default: 7860)
        share: Create public link (default: False)
        output_dir: Output directory for videos
        debug: Enable debug logging
    """
    from .ui import launch_ui as _launch
    _launch(**kwargs)

# Main high-level API
class VideoAI:
    """
    High-level API for AI video generation.
    
    Usage:
        from video_ai import VideoAI
        
        ai = VideoAI()
        result = ai.generate("A sunset over the ocean")
        print(result.output_path)
    """
    
    def __init__(self, config_dir=None, output_dir=None):
        """Initialize VideoAI with optional custom directories"""
        self._planner = None
        self._config_dir = config_dir
        self._output_dir = output_dir
    
    @property
    def planner(self):
        """Lazy-load the generation planner"""
        if self._planner is None:
            self._planner = GenerationPlanner(
                config_dir=self._config_dir,
                output_dir=self._output_dir
            )
        return self._planner
    
    def generate(self, prompt: str, **kwargs):
        """
        Generate a video from a text prompt.
        
        Args:
            prompt: Text description of the video
            **kwargs: Additional generation parameters
                - duration_seconds: Video duration (default: 6)
                - width: Frame width (default: 854)
                - height: Frame height (default: 480)
                - fps: Frames per second (default: 24)
                - seed: Random seed for reproducibility
                - quality_preset: 'fast', 'balanced', or 'quality'
        
        Returns:
            GenerationResult with output path and metadata
        """
        return self.planner.generate(prompt, **kwargs)
    
    def plan(self, prompt: str, **kwargs):
        """
        Plan a generation job without executing it.
        
        Returns:
            GenerationJob that can be executed later
        """
        return self.planner.plan_generation(prompt, **kwargs)
    
    def execute(self, job):
        """Execute a planned generation job"""
        return self.planner.execute_job(job)
    
    def get_capabilities(self):
        """Get system capabilities and recommendations"""
        return self.planner.get_capabilities()
    
    def check_resources(self):
        """Check current resource status"""
        return self.planner.resource_monitor.get_resource_status()


# Convenience function
def generate(prompt: str, **kwargs):
    """
    Quick generation function.
    
    Usage:
        from video_ai import generate
        result = generate("A cat playing with a ball")
    """
    ai = VideoAI()
    return ai.generate(prompt, **kwargs)


__all__ = [
    'VideoAI',
    'generate',
    'launch_ui',
    'GenerationPlanner',
    'PromptEngine',
    'ResourceMonitor',
    'RetryManager',
]

# Also expose the advanced API (server module only exports app/run_server;
# VideoGenerator etc. live in the root-level api.py entry point)
try:
    from .api import app as _api_app, run_server
    __all__.extend(['run_server'])
except ImportError:
    pass  # API module not available (fastapi not installed)

# Expose UI module
try:
    from .ui import WebUI, UILogHandler
    __all__.extend(['WebUI', 'UILogHandler'])
except ImportError:
    pass  # UI module not available
