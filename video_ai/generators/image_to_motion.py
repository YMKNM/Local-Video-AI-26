"""
Image-to-Motion Animation Engine

Animates still images with facial motion, lip sync, and micro-expressions
while preserving identity, geometry, and lighting consistency.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from pathlib import Path
import logging
import time
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# Motion Types and Enums
# =============================================================================

class MotionIntensity(Enum):
    """Motion intensity levels - micro > macro for realism"""
    SUBTLE = "subtle"      # Barely perceptible
    NATURAL = "natural"    # Natural human motion
    EXPRESSIVE = "expressive"  # Enhanced but realistic
    DRAMATIC = "dramatic"  # Cinematic exaggeration


class ExpressionType(Enum):
    """Facial expression categories"""
    NEUTRAL = "neutral"
    INTENSE_FOCUS = "intense_focus"
    CONTROLLED_ANGER = "controlled_anger"
    COLD_STARE = "cold_stare"
    SUBTLE_THREAT = "subtle_threat"
    CONFIDENT_SMIRK = "confident_smirk"
    DETERMINED = "determined"
    RUTHLESS = "ruthless"
    PREDATORY = "predatory"


class LipSyncMode(Enum):
    """Lip sync input modes"""
    TEXT_TO_SPEECH = "tts"
    AUDIO_FILE = "audio"
    SILENT_EXPRESSION = "silent"
    PHONEME_SEQUENCE = "phonemes"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MotionConfig:
    """Configuration for image-to-motion animation"""
    
    # Model selection
    motion_model: str = "live_portrait"  # live_portrait, wav2lip, sadtalker
    
    # Motion parameters
    motion_intensity: MotionIntensity = MotionIntensity.NATURAL
    head_motion_scale: float = 0.3  # Micro-movement scale
    eye_motion_scale: float = 0.5
    lip_motion_scale: float = 1.0
    expression_scale: float = 0.7
    
    # Preservation settings
    preserve_identity: bool = True
    identity_preservation_strength: float = 0.95
    preserve_lighting: bool = True
    preserve_texture: bool = True
    
    # Quality settings
    output_fps: int = 30
    output_resolution: Tuple[int, int] = (1920, 1080)
    duration_seconds: float = 6.0
    
    # Motion constraints
    max_head_rotation: float = 15.0  # degrees
    max_head_translation: float = 0.05  # relative to face size
    enable_blink: bool = True
    blink_frequency: float = 0.2  # blinks per second
    
    # Temporal smoothing
    temporal_smoothing: float = 0.8
    motion_blend_frames: int = 3
    
    # Output
    output_format: str = "mp4"
    output_codec: str = "h264"
    output_quality: int = 23  # CRF value


@dataclass
class AnimationResult:
    """Result from motion animation"""
    video_path: Optional[Path] = None
    video_frames: Optional[np.ndarray] = None
    audio_path: Optional[Path] = None
    duration: float = 0.0
    fps: int = 30
    resolution: Tuple[int, int] = (1920, 1080)
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Motion Model Interface
# =============================================================================

class MotionModelBase(ABC):
    """Abstract base class for motion generation models"""
    
    @abstractmethod
    def load(self):
        """Load model weights"""
        pass
    
    @abstractmethod
    def generate_motion(
        self,
        source_image: np.ndarray,
        driving_signal: Any,
        config: MotionConfig
    ) -> np.ndarray:
        """Generate motion frames"""
        pass
    
    @abstractmethod
    def unload(self):
        """Unload model to free memory"""
        pass


# =============================================================================
# Live Portrait Motion Model
# =============================================================================

class LivePortraitModel(MotionModelBase):
    """
    Live Portrait-based motion generation
    Optimized for facial animation with identity preservation
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or self._get_device()
        self.model = None
        self.face_analyzer = None
        self.motion_extractor = None
        self.warping_module = None
        self._loaded = False
    
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def load(self):
        """Load Live Portrait models"""
        if self._loaded:
            return
        
        logger.info("Loading Live Portrait motion model...")
        
        try:
            # Placeholder for actual model loading
            # In production, would load:
            # - Appearance feature extractor
            # - Motion extractor
            # - Warping/spade generator
            
            self._loaded = True
            logger.info("Live Portrait model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Live Portrait: {e}")
            raise
    
    def generate_motion(
        self,
        source_image: np.ndarray,
        driving_signal: Any,
        config: MotionConfig
    ) -> np.ndarray:
        """
        Generate motion frames from source image
        
        Args:
            source_image: Source face image (RGB, HWC)
            driving_signal: Motion driving signal (keypoints, audio, etc.)
            config: Motion configuration
            
        Returns:
            Video frames as numpy array (THWC)
        """
        if not self._loaded:
            self.load()
        
        # Calculate frame count
        num_frames = int(config.duration_seconds * config.output_fps)
        h, w = config.output_resolution[1], config.output_resolution[0]
        
        # Placeholder for motion generation
        # In production:
        # 1. Extract appearance features from source
        # 2. Extract/generate motion keypoints from driving signal
        # 3. Apply motion with warping network
        # 4. Preserve identity through feature blending
        
        frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
        
        # Copy source as placeholder
        from PIL import Image
        source_pil = Image.fromarray(source_image).resize((w, h))
        source_resized = np.array(source_pil)
        
        for i in range(num_frames):
            frames[i] = source_resized
        
        return frames
    
    def unload(self):
        """Unload models"""
        self.model = None
        self.face_analyzer = None
        self.motion_extractor = None
        self.warping_module = None
        self._loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Live Portrait model unloaded")


# =============================================================================
# Lip Sync Engine
# =============================================================================

class LipSyncEngine:
    """
    Phoneme-accurate lip synchronization engine
    Supports TTS, audio files, and silent expressions
    """
    
    # Phoneme to viseme mapping (simplified)
    PHONEME_VISEMES = {
        # Bilabials (lips together)
        'P': 'bilabial', 'B': 'bilabial', 'M': 'bilabial',
        # Labiodentals (teeth on lip)
        'F': 'labiodental', 'V': 'labiodental',
        # Dentals/Alveolars
        'TH': 'dental', 'T': 'alveolar', 'D': 'alveolar',
        'S': 'alveolar', 'Z': 'alveolar', 'N': 'alveolar',
        # Open mouth
        'AA': 'open', 'AE': 'open', 'AH': 'open',
        # Rounded
        'OW': 'rounded', 'UW': 'rounded', 'W': 'rounded',
        # Wide
        'IY': 'wide', 'EY': 'wide', 'EH': 'wide',
        # Rest/neutral
        'SIL': 'neutral', ' ': 'neutral'
    }
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tts_model = None
        self.audio_analyzer = None
        self._loaded = False
    
    def load(self):
        """Load lip sync models"""
        if self._loaded:
            return
        
        logger.info("Loading lip sync engine...")
        
        # Would load:
        # - Text-to-speech model
        # - Audio-to-phoneme model
        # - Viseme generation model
        
        self._loaded = True
        logger.info("Lip sync engine loaded")
    
    def text_to_phonemes(self, text: str) -> List[Dict[str, Any]]:
        """
        Convert text to phoneme sequence with timing
        
        Returns:
            List of dicts with phoneme, start_time, duration
        """
        # Placeholder - would use phonemizer or TTS model
        words = text.split()
        phonemes = []
        current_time = 0.0
        
        for word in words:
            # Approximate phoneme timing
            word_duration = len(word) * 0.08
            phonemes.append({
                'phoneme': word[0].upper() if word else 'SIL',
                'viseme': self.PHONEME_VISEMES.get(word[0].upper(), 'neutral'),
                'start_time': current_time,
                'duration': word_duration
            })
            current_time += word_duration + 0.05  # Word gap
        
        return phonemes
    
    def audio_to_phonemes(self, audio_path: Path) -> List[Dict[str, Any]]:
        """
        Extract phonemes from audio file
        
        Returns:
            List of phoneme dicts with timing
        """
        # Would use forced alignment or audio-to-phoneme model
        # Placeholder implementation
        return []
    
    def generate_lip_keypoints(
        self,
        phonemes: List[Dict[str, Any]],
        fps: int = 30,
        duration: float = 6.0
    ) -> np.ndarray:
        """
        Generate lip keypoint sequence from phonemes
        
        Returns:
            Keypoints array (num_frames, num_keypoints, 2)
        """
        num_frames = int(fps * duration)
        num_lip_keypoints = 20  # Standard lip keypoint count
        
        keypoints = np.zeros((num_frames, num_lip_keypoints, 2))
        
        # Generate keypoints based on phoneme timing
        # Would interpolate between viseme positions
        
        return keypoints
    
    def add_jaw_tension(
        self,
        keypoints: np.ndarray,
        tension_level: float = 0.5
    ) -> np.ndarray:
        """Add jaw tension for power/intensity"""
        # Reduce vertical mouth opening slightly
        # Add clenched jaw appearance
        tensioned = keypoints.copy()
        
        # Placeholder - would modify jaw keypoints
        
        return tensioned
    
    def add_breath_pauses(
        self,
        keypoints: np.ndarray,
        pause_positions: List[float],
        fps: int = 30
    ) -> np.ndarray:
        """Insert breath pauses for realism"""
        # Add subtle mouth opening for breath
        # at specified positions (as fractions of duration)
        
        return keypoints
    
    def unload(self):
        """Unload models"""
        self.tts_model = None
        self.audio_analyzer = None
        self._loaded = False


# =============================================================================
# Expression Engine
# =============================================================================

class ExpressionEngine:
    """
    Facial expression generation engine
    Specialized for intense, powerful expressions
    """
    
    # Expression parameter templates
    EXPRESSION_TEMPLATES = {
        ExpressionType.NEUTRAL: {
            'brow_raise': 0.0, 'brow_furrow': 0.0,
            'eye_squint': 0.0, 'eye_wide': 0.0,
            'nose_flare': 0.0, 'lip_corner_raise': 0.0,
            'jaw_clench': 0.0, 'neck_tension': 0.0
        },
        ExpressionType.INTENSE_FOCUS: {
            'brow_raise': 0.1, 'brow_furrow': 0.4,
            'eye_squint': 0.3, 'eye_wide': 0.0,
            'nose_flare': 0.1, 'lip_corner_raise': -0.1,
            'jaw_clench': 0.3, 'neck_tension': 0.2
        },
        ExpressionType.CONTROLLED_ANGER: {
            'brow_raise': 0.0, 'brow_furrow': 0.7,
            'eye_squint': 0.4, 'eye_wide': 0.0,
            'nose_flare': 0.4, 'lip_corner_raise': -0.3,
            'jaw_clench': 0.8, 'neck_tension': 0.5
        },
        ExpressionType.COLD_STARE: {
            'brow_raise': 0.0, 'brow_furrow': 0.2,
            'eye_squint': 0.1, 'eye_wide': 0.2,
            'nose_flare': 0.0, 'lip_corner_raise': 0.0,
            'jaw_clench': 0.4, 'neck_tension': 0.3
        },
        ExpressionType.SUBTLE_THREAT: {
            'brow_raise': 0.2, 'brow_furrow': 0.3,
            'eye_squint': 0.2, 'eye_wide': 0.1,
            'nose_flare': 0.2, 'lip_corner_raise': 0.1,
            'jaw_clench': 0.5, 'neck_tension': 0.4
        },
        ExpressionType.CONFIDENT_SMIRK: {
            'brow_raise': 0.3, 'brow_furrow': 0.0,
            'eye_squint': 0.2, 'eye_wide': 0.0,
            'nose_flare': 0.0, 'lip_corner_raise': 0.4,
            'jaw_clench': 0.2, 'neck_tension': 0.1
        },
        ExpressionType.DETERMINED: {
            'brow_raise': 0.0, 'brow_furrow': 0.5,
            'eye_squint': 0.3, 'eye_wide': 0.0,
            'nose_flare': 0.2, 'lip_corner_raise': -0.2,
            'jaw_clench': 0.6, 'neck_tension': 0.4
        },
        ExpressionType.RUTHLESS: {
            'brow_raise': 0.0, 'brow_furrow': 0.8,
            'eye_squint': 0.5, 'eye_wide': 0.0,
            'nose_flare': 0.5, 'lip_corner_raise': -0.4,
            'jaw_clench': 0.9, 'neck_tension': 0.7
        },
        ExpressionType.PREDATORY: {
            'brow_raise': 0.1, 'brow_furrow': 0.6,
            'eye_squint': 0.4, 'eye_wide': 0.0,
            'nose_flare': 0.3, 'lip_corner_raise': 0.2,
            'jaw_clench': 0.7, 'neck_tension': 0.5
        }
    }
    
    def __init__(self):
        self.current_expression = ExpressionType.NEUTRAL
    
    def get_expression_params(
        self,
        expression: ExpressionType,
        intensity: float = 1.0
    ) -> Dict[str, float]:
        """
        Get expression parameters scaled by intensity
        
        Args:
            expression: Target expression type
            intensity: Scale factor (0.0 - 1.0)
            
        Returns:
            Dict of expression parameters
        """
        template = self.EXPRESSION_TEMPLATES.get(
            expression,
            self.EXPRESSION_TEMPLATES[ExpressionType.NEUTRAL]
        )
        
        return {k: v * intensity for k, v in template.items()}
    
    def interpolate_expressions(
        self,
        start: ExpressionType,
        end: ExpressionType,
        num_frames: int,
        easing: str = "ease_in_out"
    ) -> List[Dict[str, float]]:
        """
        Generate smooth expression transition
        
        Args:
            start: Starting expression
            end: Ending expression
            num_frames: Number of frames for transition
            easing: Easing function type
            
        Returns:
            List of expression parameters per frame
        """
        start_params = self.get_expression_params(start)
        end_params = self.get_expression_params(end)
        
        frames = []
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)
            
            # Apply easing
            if easing == "ease_in_out":
                t = self._ease_in_out(t)
            elif easing == "ease_in":
                t = t * t
            elif easing == "ease_out":
                t = 1 - (1 - t) ** 2
            
            # Interpolate parameters
            frame_params = {}
            for key in start_params:
                frame_params[key] = start_params[key] + t * (end_params[key] - start_params[key])
            
            frames.append(frame_params)
        
        return frames
    
    def _ease_in_out(self, t: float) -> float:
        """Smooth ease in/out function"""
        return t * t * (3 - 2 * t)
    
    def generate_micro_expressions(
        self,
        base_expression: ExpressionType,
        duration: float,
        fps: int = 30,
        variation: float = 0.1
    ) -> List[Dict[str, float]]:
        """
        Add subtle micro-expression variations for realism
        
        Args:
            base_expression: Base expression to vary around
            duration: Duration in seconds
            fps: Frames per second
            variation: Amount of random variation
            
        Returns:
            List of expression parameters with micro-variations
        """
        num_frames = int(duration * fps)
        base_params = self.get_expression_params(base_expression)
        
        frames = []
        for i in range(num_frames):
            frame_params = {}
            for key, value in base_params.items():
                # Add subtle noise
                noise = np.random.normal(0, variation * abs(value) if value != 0 else variation * 0.1)
                frame_params[key] = np.clip(value + noise, -1.0, 1.0)
            frames.append(frame_params)
        
        # Smooth the variations
        frames = self._smooth_expression_sequence(frames, window=3)
        
        return frames
    
    def _smooth_expression_sequence(
        self,
        frames: List[Dict[str, float]],
        window: int = 3
    ) -> List[Dict[str, float]]:
        """Apply temporal smoothing to expression sequence"""
        if len(frames) < window:
            return frames
        
        smoothed = []
        for i in range(len(frames)):
            start = max(0, i - window // 2)
            end = min(len(frames), i + window // 2 + 1)
            
            frame_params = {}
            for key in frames[0].keys():
                values = [frames[j][key] for j in range(start, end)]
                frame_params[key] = sum(values) / len(values)
            
            smoothed.append(frame_params)
        
        return smoothed


# =============================================================================
# Main Animation Engine
# =============================================================================

class ImageToMotionEngine:
    """
    Main orchestrator for image-to-motion animation
    Combines motion models, lip sync, and expression engines
    """
    
    def __init__(self, config: Optional[MotionConfig] = None):
        self.config = config or MotionConfig()
        
        # Sub-engines
        self.motion_model = LivePortraitModel()
        self.lip_sync = LipSyncEngine()
        self.expression_engine = ExpressionEngine()
        
        # Progress callback
        self._progress_callback: Optional[Callable] = None
        
        logger.info("ImageToMotionEngine initialized")
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates"""
        self._progress_callback = callback
    
    def _report_progress(self, progress: float, message: str):
        """Report progress to callback"""
        if self._progress_callback:
            self._progress_callback(progress, message)
        logger.debug(f"Progress {progress:.1%}: {message}")
    
    def load_models(self):
        """Load all required models"""
        self._report_progress(0.0, "Loading motion models...")
        self.motion_model.load()
        
        self._report_progress(0.5, "Loading lip sync engine...")
        self.lip_sync.load()
        
        self._report_progress(1.0, "All models loaded")
    
    def animate(
        self,
        source_image: Union[np.ndarray, Path, str],
        lip_sync_mode: LipSyncMode = LipSyncMode.SILENT_EXPRESSION,
        lip_sync_input: Optional[Union[str, Path]] = None,
        expression: ExpressionType = ExpressionType.INTENSE_FOCUS,
        expression_intensity: float = 0.8,
        output_path: Optional[Path] = None
    ) -> AnimationResult:
        """
        Animate a source image with facial motion
        
        Args:
            source_image: Source image (array, path, or base64)
            lip_sync_mode: Type of lip sync to apply
            lip_sync_input: Text or audio path for lip sync
            expression: Target expression type
            expression_intensity: Expression intensity (0-1)
            output_path: Path for output video
            
        Returns:
            AnimationResult with video path and metadata
        """
        start_time = time.time()
        
        # Load image
        self._report_progress(0.05, "Loading source image...")
        source_array = self._load_image(source_image)
        
        # Generate expression sequence
        self._report_progress(0.1, "Generating expression sequence...")
        expression_params = self.expression_engine.generate_micro_expressions(
            base_expression=expression,
            duration=self.config.duration_seconds,
            fps=self.config.output_fps,
            variation=0.1 * expression_intensity
        )
        
        # Generate lip sync if needed
        lip_keypoints = None
        audio_path = None
        
        if lip_sync_mode == LipSyncMode.TEXT_TO_SPEECH and lip_sync_input:
            self._report_progress(0.2, "Generating speech and lip sync...")
            phonemes = self.lip_sync.text_to_phonemes(str(lip_sync_input))
            lip_keypoints = self.lip_sync.generate_lip_keypoints(
                phonemes, self.config.output_fps, self.config.duration_seconds
            )
            # Add jaw tension for intensity
            lip_keypoints = self.lip_sync.add_jaw_tension(
                lip_keypoints, tension_level=expression_intensity * 0.5
            )
            
        elif lip_sync_mode == LipSyncMode.AUDIO_FILE and lip_sync_input:
            self._report_progress(0.2, "Analyzing audio for lip sync...")
            audio_path = Path(lip_sync_input)
            phonemes = self.lip_sync.audio_to_phonemes(audio_path)
            lip_keypoints = self.lip_sync.generate_lip_keypoints(
                phonemes, self.config.output_fps, self.config.duration_seconds
            )
        
        # Combine driving signals
        self._report_progress(0.3, "Preparing motion signals...")
        driving_signal = {
            'expression_params': expression_params,
            'lip_keypoints': lip_keypoints,
            'head_motion_scale': self.config.head_motion_scale,
            'eye_motion_scale': self.config.eye_motion_scale
        }
        
        # Generate motion frames
        self._report_progress(0.4, "Generating motion frames...")
        frames = self.motion_model.generate_motion(
            source_image=source_array,
            driving_signal=driving_signal,
            config=self.config
        )
        
        self._report_progress(0.8, "Motion generation complete")
        
        # Encode video
        self._report_progress(0.85, "Encoding video...")
        if output_path is None:
            output_path = Path(f"outputs/animation_{uuid.uuid4().hex[:8]}.{self.config.output_format}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._encode_video(frames, output_path, audio_path)
        
        self._report_progress(1.0, "Animation complete")
        
        return AnimationResult(
            video_path=output_path,
            video_frames=frames,
            audio_path=audio_path,
            duration=self.config.duration_seconds,
            fps=self.config.output_fps,
            resolution=self.config.output_resolution,
            generation_time=time.time() - start_time,
            metadata={
                'expression': expression.value,
                'intensity': expression_intensity,
                'lip_sync_mode': lip_sync_mode.value,
                'motion_model': self.config.motion_model
            }
        )
    
    def _load_image(self, source: Union[np.ndarray, Path, str]) -> np.ndarray:
        """Load image from various sources"""
        if isinstance(source, np.ndarray):
            return source
        
        from PIL import Image
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                img = Image.open(path).convert('RGB')
                return np.array(img)
        
        raise ValueError(f"Could not load image from {source}")
    
    def _encode_video(
        self,
        frames: np.ndarray,
        output_path: Path,
        audio_path: Optional[Path] = None
    ):
        """Encode frames to video file"""
        import subprocess
        import tempfile
        
        # Write frames to temp images and use ffmpeg
        # Placeholder - would use proper video encoding
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Write frames
            from PIL import Image
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame)
                img.save(tmpdir / f"frame_{i:06d}.png")
            
            # FFmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(self.config.output_fps),
                '-i', str(tmpdir / 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-crf', str(self.config.output_quality),
                '-pix_fmt', 'yuv420p'
            ]
            
            if audio_path:
                cmd.extend(['-i', str(audio_path), '-c:a', 'aac', '-shortest'])
            
            cmd.append(str(output_path))
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"Video encoded to {output_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg encoding failed: {e.stderr}")
                raise
    
    def unload_models(self):
        """Unload all models"""
        self.motion_model.unload()
        self.lip_sync.unload()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("All motion models unloaded")


# =============================================================================
# Quick Animation Functions
# =============================================================================

def animate_image(
    image_path: Union[str, Path],
    text: Optional[str] = None,
    audio_path: Optional[str] = None,
    expression: str = "intense_focus",
    intensity: float = 0.8,
    duration: float = 6.0,
    output_path: Optional[str] = None
) -> AnimationResult:
    """
    Quick function to animate an image
    
    Args:
        image_path: Path to source image
        text: Text for TTS lip sync (optional)
        audio_path: Path to audio for lip sync (optional)
        expression: Expression type name
        intensity: Expression intensity (0-1)
        duration: Output duration in seconds
        output_path: Output video path
        
    Returns:
        AnimationResult
    """
    config = MotionConfig(duration_seconds=duration)
    engine = ImageToMotionEngine(config)
    engine.load_models()
    
    # Determine lip sync mode
    if text:
        lip_sync_mode = LipSyncMode.TEXT_TO_SPEECH
        lip_sync_input = text
    elif audio_path:
        lip_sync_mode = LipSyncMode.AUDIO_FILE
        lip_sync_input = audio_path
    else:
        lip_sync_mode = LipSyncMode.SILENT_EXPRESSION
        lip_sync_input = None
    
    # Map expression string to enum
    expression_map = {e.value: e for e in ExpressionType}
    expression_type = expression_map.get(expression, ExpressionType.INTENSE_FOCUS)
    
    result = engine.animate(
        source_image=Path(image_path),
        lip_sync_mode=lip_sync_mode,
        lip_sync_input=lip_sync_input,
        expression=expression_type,
        expression_intensity=intensity,
        output_path=Path(output_path) if output_path else None
    )
    
    engine.unload_models()
    return result
