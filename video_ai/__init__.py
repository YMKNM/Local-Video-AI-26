"""
Video AI Main Package
Local AI video generation system for AMD GPUs on Windows
"""

__version__ = "0.1.0"
__author__ = "Video AI Project"


# ── Lazy imports ---------------------------------------------------------
# Heavy modules (torch, gradio, fastapi, diffusers) are NOT imported at
# package level.  They are loaded on first use to keep startup fast and
# avoid import-chain crashes.

def _import_agent():
    from .agent import GenerationPlanner, PromptEngine, ResourceMonitor, RetryManager
    return GenerationPlanner, PromptEngine, ResourceMonitor, RetryManager


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
            from .agent import GenerationPlanner
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
]


def __getattr__(name):
    """Lazy attribute access for heavy sub-module symbols."""
    # Agent classes
    _agent_names = {
        'GenerationPlanner', 'PromptEngine', 'ResourceMonitor', 'RetryManager',
    }
    if name in _agent_names:
        from . import agent as _agent
        return getattr(_agent, name)

    # API server
    if name == 'run_server':
        try:
            from .api import run_server
            return run_server
        except ImportError:
            raise AttributeError(name)

    # UI classes
    _ui_names = {'WebUI', 'UILogHandler'}
    if name in _ui_names:
        try:
            from . import ui as _ui
            return getattr(_ui, name)
        except ImportError:
            raise AttributeError(name)

    raise AttributeError(f"module 'video_ai' has no attribute {name!r}")
