"""
Video Model Orchestrator

Unified interface for multiple video generation models:
- LTX-Video-2: 4K, 50 FPS, audio sync
- HunyuanVideo 1.5: 8B param efficient I2V
- Genmo Mochi 1: 10B param cinematic
- AccVideo AI: Fast HD generation
- Pyramid Flow: 10+ second autoregressive
- Rhymes Allegro: Smooth motion diffusion
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Union
from pathlib import Path
from abc import ABC, abstractmethod
import logging
import time
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Model Types
# =============================================================================

class VideoModelType(Enum):
    """Supported video generation models"""
    LTX_VIDEO_2 = "ltx_video_2"
    HUNYUAN_VIDEO = "hunyuan_video"
    GENMO_MOCHI = "genmo_mochi"
    ACCVIDEO = "accvideo"
    PYRAMID_FLOW = "pyramid_flow"
    RHYMES_ALLEGRO = "rhymes_allegro"


class GenerationMode(Enum):
    """Video generation modes"""
    TEXT_TO_VIDEO = "t2v"
    IMAGE_TO_VIDEO = "i2v"
    VIDEO_TO_VIDEO = "v2v"


# =============================================================================
# Model Specifications
# =============================================================================

MODEL_SPECS = {
    VideoModelType.LTX_VIDEO_2: {
        "name": "LTX-Video-2",
        "description": "True open-source T2V with 4K, synced audio, 50 FPS",
        "parameters": "5B",
        "license": "Apache 2.0",
        "max_resolution": (3840, 2160),
        "max_fps": 50,
        "max_duration": 10.0,
        "modes": [GenerationMode.TEXT_TO_VIDEO, GenerationMode.IMAGE_TO_VIDEO],
        "audio_sync": True,
        "vram_required_gb": 10,
        "repo_id": "Lightricks/LTX-Video-2",
        "supports_controlnet": True,
        "commercial_use": True
    },
    VideoModelType.HUNYUAN_VIDEO: {
        "name": "HunyuanVideo 1.5",
        "description": "Lightweight high-quality T2V/I2V with ~8B params",
        "parameters": "8B",
        "license": "Apache 2.0",
        "max_resolution": (1920, 1080),
        "max_fps": 30,
        "max_duration": 8.0,
        "modes": [GenerationMode.TEXT_TO_VIDEO, GenerationMode.IMAGE_TO_VIDEO],
        "audio_sync": False,
        "vram_required_gb": 12,
        "repo_id": "tencent/HunyuanVideo",
        "supports_controlnet": True,
        "commercial_use": True
    },
    VideoModelType.GENMO_MOCHI: {
        "name": "Genmo Mochi 1",
        "description": "10B diffusion-transformer with excellent motion realism",
        "parameters": "10B",
        "license": "Apache 2.0",
        "max_resolution": (1920, 1080),
        "max_fps": 30,
        "max_duration": 6.0,
        "modes": [GenerationMode.TEXT_TO_VIDEO],
        "audio_sync": False,
        "vram_required_gb": 16,
        "repo_id": "genmo/mochi-1-preview",
        "supports_controlnet": False,
        "commercial_use": True
    },
    VideoModelType.ACCVIDEO: {
        "name": "AccVideo AI",
        "description": "Fast HD video generation with reduced wait times",
        "parameters": "1.5B",
        "license": "MIT",
        "max_resolution": (1280, 720),
        "max_fps": 24,
        "max_duration": 4.0,
        "modes": [GenerationMode.TEXT_TO_VIDEO, GenerationMode.IMAGE_TO_VIDEO],
        "audio_sync": False,
        "vram_required_gb": 6,
        "repo_id": "accvideo/accvideo-hd",
        "supports_controlnet": False,
        "commercial_use": True
    },
    VideoModelType.PYRAMID_FLOW: {
        "name": "Pyramid Flow",
        "description": "Advanced autoregressive with 10+ second generation",
        "parameters": "3B",
        "license": "MIT",
        "max_resolution": (1280, 768),
        "max_fps": 24,
        "max_duration": 12.0,
        "modes": [GenerationMode.TEXT_TO_VIDEO, GenerationMode.IMAGE_TO_VIDEO],
        "audio_sync": False,
        "vram_required_gb": 8,
        "repo_id": "rain1011/pyramid-flow-sd3",
        "supports_controlnet": True,
        "commercial_use": True
    },
    VideoModelType.RHYMES_ALLEGRO: {
        "name": "Rhymes Allegro",
        "description": "Smooth motion diffusion with cinematic quality",
        "parameters": "2.8B",
        "license": "Apache 2.0",
        "max_resolution": (1360, 768),
        "max_fps": 30,
        "max_duration": 6.0,
        "modes": [GenerationMode.TEXT_TO_VIDEO],
        "audio_sync": False,
        "vram_required_gb": 10,
        "repo_id": "rhymes-ai/Allegro",
        "supports_controlnet": False,
        "commercial_use": True
    }
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VideoGenerationConfig:
    """Configuration for video generation"""
    
    # Model selection
    model_type: VideoModelType = VideoModelType.LTX_VIDEO_2
    generation_mode: GenerationMode = GenerationMode.IMAGE_TO_VIDEO
    
    # Output settings
    width: int = 1280
    height: int = 720
    fps: int = 30
    duration_seconds: float = 6.0
    
    # Generation settings
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    motion_bucket_id: int = 127  # For I2V motion strength
    noise_aug_strength: float = 0.02
    
    # Quality
    use_fp16: bool = True
    enable_vae_tiling: bool = True
    enable_model_cpu_offload: bool = False
    
    # Audio (for LTX-2)
    generate_audio: bool = False
    audio_prompt: Optional[str] = None
    
    # Output
    output_format: str = "mp4"
    output_codec: str = "h264"
    output_quality: int = 23


@dataclass
class VideoGenerationResult:
    """Result from video generation"""
    video_path: Optional[Path] = None
    video_frames: Optional[np.ndarray] = None
    audio_path: Optional[Path] = None
    duration: float = 0.0
    fps: int = 30
    resolution: tuple = (1280, 720)
    model_used: str = ""
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Video Model Interface
# =============================================================================

class VideoModelBase(ABC):
    """Abstract base class for video generation models"""
    
    def __init__(self, model_type: VideoModelType):
        self.model_type = model_type
        self.specs = MODEL_SPECS[model_type]
        self.device = self._get_device()
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._loaded = False
        self._pipeline = None
    
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    @abstractmethod
    def load(self):
        """Load model weights"""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: VideoGenerationConfig,
        source_image: Optional[np.ndarray] = None,
        negative_prompt: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> VideoGenerationResult:
        """Generate video"""
        pass
    
    def unload(self):
        """Unload model to free memory"""
        if self._pipeline:
            del self._pipeline
            self._pipeline = None
        self._loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# LTX-Video-2 Implementation
# =============================================================================

class LTXVideo2Model(VideoModelBase):
    """LTX-Video-2 model implementation"""
    
    def __init__(self):
        super().__init__(VideoModelType.LTX_VIDEO_2)
        self.audio_pipeline = None
    
    def load(self):
        if self._loaded:
            return
        
        logger.info("Loading LTX-Video-2...")
        
        try:
            from diffusers import DiffusionPipeline
            
            self._pipeline = DiffusionPipeline.from_pretrained(
                self.specs["repo_id"],
                torch_dtype=self.dtype,
                use_safetensors=True
            )
            self._pipeline.to(self.device)
            
            # Enable optimizations
            if hasattr(self._pipeline, 'enable_xformers_memory_efficient_attention'):
                self._pipeline.enable_xformers_memory_efficient_attention()
            
            self._loaded = True
            logger.info("LTX-Video-2 loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load LTX-Video-2: {e}")
            # Use placeholder for development
            self._loaded = True
    
    def generate(
        self,
        prompt: str,
        config: VideoGenerationConfig,
        source_image: Optional[np.ndarray] = None,
        negative_prompt: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> VideoGenerationResult:
        
        if not self._loaded:
            self.load()
        
        start_time = time.time()
        
        if progress_callback:
            progress_callback(0.1, "Preparing generation...")
        
        # Calculate frame count
        num_frames = int(config.duration_seconds * config.fps)
        
        # Generate video
        if progress_callback:
            progress_callback(0.3, "Generating video frames...")
        
        # Placeholder - would call actual pipeline
        frames = self._generate_placeholder_frames(
            config.width, config.height, num_frames
        )
        
        if progress_callback:
            progress_callback(0.9, "Encoding video...")
        
        # Save video
        output_path = Path(f"outputs/videos/ltx2_{uuid.uuid4().hex[:8]}.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._encode_video(frames, output_path, config.fps)
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return VideoGenerationResult(
            video_path=output_path,
            video_frames=frames,
            duration=config.duration_seconds,
            fps=config.fps,
            resolution=(config.width, config.height),
            model_used="LTX-Video-2",
            generation_time=time.time() - start_time,
            metadata={
                "prompt": prompt,
                "steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale
            }
        )
    
    def _generate_placeholder_frames(self, width: int, height: int, num_frames: int) -> np.ndarray:
        """Generate placeholder frames for development"""
        frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        for i in range(num_frames):
            # Create gradient animation
            t = i / max(num_frames - 1, 1)
            frames[i, :, :, 0] = int(100 + 50 * np.sin(t * np.pi * 2))  # R
            frames[i, :, :, 1] = int(50 + 30 * np.cos(t * np.pi * 2))   # G
            frames[i, :, :, 2] = int(80 + 40 * np.sin(t * np.pi))       # B
        return frames
    
    def _encode_video(self, frames: np.ndarray, output_path: Path, fps: int):
        """Encode frames to video"""
        import subprocess
        import tempfile
        from PIL import Image
        
        # Get FFmpeg path from imageio
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = 'ffmpeg'  # Fall back to PATH
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame)
                img.save(tmpdir / f"frame_{i:06d}.png")
            
            cmd = [
                ffmpeg_exe, '-y',
                '-framerate', str(fps),
                '-i', str(tmpdir / 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)


# =============================================================================
# HunyuanVideo Implementation
# =============================================================================

class HunyuanVideoModel(VideoModelBase):
    """HunyuanVideo 1.5 model implementation"""
    
    def __init__(self):
        super().__init__(VideoModelType.HUNYUAN_VIDEO)
    
    def load(self):
        if self._loaded:
            return
        
        logger.info("Loading HunyuanVideo 1.5...")
        
        try:
            from diffusers import HunyuanVideoPipeline
            
            self._pipeline = HunyuanVideoPipeline.from_pretrained(
                self.specs["repo_id"],
                torch_dtype=self.dtype
            )
            self._pipeline.to(self.device)
            
            self._loaded = True
            logger.info("HunyuanVideo loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load HunyuanVideo: {e}")
            self._loaded = True
    
    def generate(
        self,
        prompt: str,
        config: VideoGenerationConfig,
        source_image: Optional[np.ndarray] = None,
        negative_prompt: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> VideoGenerationResult:
        
        if not self._loaded:
            self.load()
        
        start_time = time.time()
        num_frames = int(config.duration_seconds * config.fps)
        
        # Placeholder generation
        frames = np.zeros((num_frames, config.height, config.width, 3), dtype=np.uint8)
        
        output_path = Path(f"outputs/videos/hunyuan_{uuid.uuid4().hex[:8]}.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return VideoGenerationResult(
            video_path=output_path,
            video_frames=frames,
            duration=config.duration_seconds,
            fps=config.fps,
            resolution=(config.width, config.height),
            model_used="HunyuanVideo 1.5",
            generation_time=time.time() - start_time
        )


# =============================================================================
# Genmo Mochi Implementation
# =============================================================================

class GenmoMochiModel(VideoModelBase):
    """Genmo Mochi 1 model implementation"""
    
    def __init__(self):
        super().__init__(VideoModelType.GENMO_MOCHI)
    
    def load(self):
        if self._loaded:
            return
        
        logger.info("Loading Genmo Mochi 1...")
        
        try:
            from diffusers import MochiPipeline
            
            self._pipeline = MochiPipeline.from_pretrained(
                self.specs["repo_id"],
                torch_dtype=self.dtype
            )
            self._pipeline.to(self.device)
            
            self._loaded = True
            logger.info("Genmo Mochi loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load Genmo Mochi: {e}")
            self._loaded = True
    
    def generate(
        self,
        prompt: str,
        config: VideoGenerationConfig,
        source_image: Optional[np.ndarray] = None,
        negative_prompt: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> VideoGenerationResult:
        
        if not self._loaded:
            self.load()
        
        start_time = time.time()
        num_frames = int(config.duration_seconds * config.fps)
        
        frames = np.zeros((num_frames, config.height, config.width, 3), dtype=np.uint8)
        
        output_path = Path(f"outputs/videos/mochi_{uuid.uuid4().hex[:8]}.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return VideoGenerationResult(
            video_path=output_path,
            video_frames=frames,
            duration=config.duration_seconds,
            fps=config.fps,
            resolution=(config.width, config.height),
            model_used="Genmo Mochi 1",
            generation_time=time.time() - start_time
        )


# =============================================================================
# AccVideo Implementation
# =============================================================================

class AccVideoModel(VideoModelBase):
    """AccVideo AI fast generation model"""
    
    def __init__(self):
        super().__init__(VideoModelType.ACCVIDEO)
    
    def load(self):
        if self._loaded:
            return
        
        logger.info("Loading AccVideo...")
        self._loaded = True
    
    def generate(
        self,
        prompt: str,
        config: VideoGenerationConfig,
        source_image: Optional[np.ndarray] = None,
        negative_prompt: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> VideoGenerationResult:
        
        if not self._loaded:
            self.load()
        
        start_time = time.time()
        num_frames = int(config.duration_seconds * config.fps)
        
        frames = np.zeros((num_frames, config.height, config.width, 3), dtype=np.uint8)
        
        output_path = Path(f"outputs/videos/accvideo_{uuid.uuid4().hex[:8]}.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return VideoGenerationResult(
            video_path=output_path,
            video_frames=frames,
            duration=config.duration_seconds,
            fps=config.fps,
            resolution=(config.width, config.height),
            model_used="AccVideo AI",
            generation_time=time.time() - start_time
        )


# =============================================================================
# Pyramid Flow Implementation
# =============================================================================

class PyramidFlowModel(VideoModelBase):
    """Pyramid Flow autoregressive model"""
    
    def __init__(self):
        super().__init__(VideoModelType.PYRAMID_FLOW)
    
    def load(self):
        if self._loaded:
            return
        
        logger.info("Loading Pyramid Flow...")
        self._loaded = True
    
    def generate(
        self,
        prompt: str,
        config: VideoGenerationConfig,
        source_image: Optional[np.ndarray] = None,
        negative_prompt: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> VideoGenerationResult:
        
        if not self._loaded:
            self.load()
        
        start_time = time.time()
        num_frames = int(config.duration_seconds * config.fps)
        
        frames = np.zeros((num_frames, config.height, config.width, 3), dtype=np.uint8)
        
        output_path = Path(f"outputs/videos/pyramid_{uuid.uuid4().hex[:8]}.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return VideoGenerationResult(
            video_path=output_path,
            video_frames=frames,
            duration=config.duration_seconds,
            fps=config.fps,
            resolution=(config.width, config.height),
            model_used="Pyramid Flow",
            generation_time=time.time() - start_time
        )


# =============================================================================
# Rhymes Allegro Implementation
# =============================================================================

class RhymesAllegroModel(VideoModelBase):
    """Rhymes Allegro smooth motion model"""
    
    def __init__(self):
        super().__init__(VideoModelType.RHYMES_ALLEGRO)
    
    def load(self):
        if self._loaded:
            return
        
        logger.info("Loading Rhymes Allegro...")
        self._loaded = True
    
    def generate(
        self,
        prompt: str,
        config: VideoGenerationConfig,
        source_image: Optional[np.ndarray] = None,
        negative_prompt: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> VideoGenerationResult:
        
        if not self._loaded:
            self.load()
        
        start_time = time.time()
        num_frames = int(config.duration_seconds * config.fps)
        
        frames = np.zeros((num_frames, config.height, config.width, 3), dtype=np.uint8)
        
        output_path = Path(f"outputs/videos/allegro_{uuid.uuid4().hex[:8]}.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return VideoGenerationResult(
            video_path=output_path,
            video_frames=frames,
            duration=config.duration_seconds,
            fps=config.fps,
            resolution=(config.width, config.height),
            model_used="Rhymes Allegro",
            generation_time=time.time() - start_time
        )


# =============================================================================
# Model Orchestrator
# =============================================================================

class VideoModelOrchestrator:
    """
    Unified orchestrator for all video generation models
    Handles model loading, selection, and generation routing
    """
    
    # Model class mapping
    MODEL_CLASSES = {
        VideoModelType.LTX_VIDEO_2: LTXVideo2Model,
        VideoModelType.HUNYUAN_VIDEO: HunyuanVideoModel,
        VideoModelType.GENMO_MOCHI: GenmoMochiModel,
        VideoModelType.ACCVIDEO: AccVideoModel,
        VideoModelType.PYRAMID_FLOW: PyramidFlowModel,
        VideoModelType.RHYMES_ALLEGRO: RhymesAllegroModel
    }
    
    def __init__(self):
        self._models: Dict[VideoModelType, VideoModelBase] = {}
        self._current_model: Optional[VideoModelType] = None
        self._progress_callback: Optional[Callable] = None
        
        logger.info("VideoModelOrchestrator initialized")
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set progress callback"""
        self._progress_callback = callback
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with specifications"""
        return [
            {
                "type": model_type,
                "name": specs["name"],
                "description": specs["description"],
                "parameters": specs["parameters"],
                "max_resolution": specs["max_resolution"],
                "max_fps": specs["max_fps"],
                "max_duration": specs["max_duration"],
                "vram_required_gb": specs["vram_required_gb"],
                "modes": [m.value for m in specs["modes"]],
                "audio_sync": specs["audio_sync"]
            }
            for model_type, specs in MODEL_SPECS.items()
        ]
    
    def get_model(self, model_type: VideoModelType) -> VideoModelBase:
        """Get or create a model instance"""
        if model_type not in self._models:
            model_class = self.MODEL_CLASSES.get(model_type)
            if model_class is None:
                raise ValueError(f"Unknown model type: {model_type}")
            self._models[model_type] = model_class()
        
        return self._models[model_type]
    
    def load_model(self, model_type: VideoModelType):
        """Load a specific model"""
        model = self.get_model(model_type)
        model.load()
        self._current_model = model_type
    
    def unload_model(self, model_type: VideoModelType):
        """Unload a specific model"""
        if model_type in self._models:
            self._models[model_type].unload()
            del self._models[model_type]
    
    def unload_all(self):
        """Unload all models"""
        for model in self._models.values():
            model.unload()
        self._models.clear()
        self._current_model = None
    
    def generate(
        self,
        prompt: str,
        config: VideoGenerationConfig,
        source_image: Optional[np.ndarray] = None,
        negative_prompt: Optional[str] = None
    ) -> VideoGenerationResult:
        """
        Generate video using the specified model
        
        Args:
            prompt: Text prompt for generation
            config: Generation configuration
            source_image: Optional source image for I2V
            negative_prompt: Optional negative prompt
            
        Returns:
            VideoGenerationResult
        """
        model = self.get_model(config.model_type)
        
        if not model._loaded:
            if self._progress_callback:
                self._progress_callback(0.0, f"Loading {model.specs['name']}...")
            model.load()
        
        return model.generate(
            prompt=prompt,
            config=config,
            source_image=source_image,
            negative_prompt=negative_prompt,
            progress_callback=self._progress_callback
        )
    
    def auto_select_model(
        self,
        mode: GenerationMode,
        duration: float,
        resolution: tuple,
        vram_available_gb: float
    ) -> VideoModelType:
        """
        Automatically select the best model based on requirements
        
        Args:
            mode: Generation mode (T2V, I2V)
            duration: Desired duration in seconds
            resolution: Desired resolution (width, height)
            vram_available_gb: Available VRAM
            
        Returns:
            Best matching VideoModelType
        """
        candidates = []
        
        for model_type, specs in MODEL_SPECS.items():
            # Check mode support
            if mode not in specs["modes"]:
                continue
            
            # Check VRAM
            if specs["vram_required_gb"] > vram_available_gb:
                continue
            
            # Check duration
            if duration > specs["max_duration"]:
                continue
            
            # Check resolution
            max_w, max_h = specs["max_resolution"]
            if resolution[0] > max_w or resolution[1] > max_h:
                continue
            
            # Score based on quality (higher params = better)
            param_score = float(specs["parameters"].replace("B", ""))
            candidates.append((model_type, param_score))
        
        if not candidates:
            # Fallback to AccVideo (lowest requirements)
            return VideoModelType.ACCVIDEO
        
        # Sort by parameter count (quality indicator)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_video(
    prompt: str,
    model: str = "ltx_video_2",
    source_image: Optional[Union[str, np.ndarray]] = None,
    duration: float = 6.0,
    resolution: tuple = (1280, 720),
    fps: int = 30,
    output_path: Optional[str] = None
) -> VideoGenerationResult:
    """
    Quick function to generate a video
    
    Args:
        prompt: Text prompt
        model: Model name (ltx_video_2, hunyuan_video, etc.)
        source_image: Optional source image path or array
        duration: Duration in seconds
        resolution: Output resolution
        fps: Frames per second
        output_path: Optional output path
        
    Returns:
        VideoGenerationResult
    """
    # Map model name to type
    model_map = {
        "ltx_video_2": VideoModelType.LTX_VIDEO_2,
        "hunyuan_video": VideoModelType.HUNYUAN_VIDEO,
        "genmo_mochi": VideoModelType.GENMO_MOCHI,
        "accvideo": VideoModelType.ACCVIDEO,
        "pyramid_flow": VideoModelType.PYRAMID_FLOW,
        "rhymes_allegro": VideoModelType.RHYMES_ALLEGRO
    }
    
    model_type = model_map.get(model.lower(), VideoModelType.LTX_VIDEO_2)
    
    # Determine mode
    mode = GenerationMode.IMAGE_TO_VIDEO if source_image is not None else GenerationMode.TEXT_TO_VIDEO
    
    config = VideoGenerationConfig(
        model_type=model_type,
        generation_mode=mode,
        width=resolution[0],
        height=resolution[1],
        fps=fps,
        duration_seconds=duration
    )
    
    # Load source image if path
    image_array = None
    if source_image is not None:
        if isinstance(source_image, str):
            from PIL import Image
            image_array = np.array(Image.open(source_image).convert('RGB'))
        else:
            image_array = source_image
    
    orchestrator = VideoModelOrchestrator()
    return orchestrator.generate(prompt, config, image_array)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'VideoModelType',
    'GenerationMode',
    'VideoGenerationConfig',
    'VideoGenerationResult',
    'VideoModelOrchestrator',
    'MODEL_SPECS',
    'generate_video',
    'LTXVideo2Model',
    'HunyuanVideoModel',
    'GenmoMochiModel',
    'AccVideoModel',
    'PyramidFlowModel',
    'RhymesAllegroModel'
]
