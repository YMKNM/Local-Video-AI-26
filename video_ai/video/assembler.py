"""
Video Assembler - High-level video assembly from frames

This module handles:
- Frame sequence to video conversion
- Multi-format output support
- Quality presets
- Progress tracking
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Callable
import yaml

from .frame_writer import FrameWriter
from .ffmpeg_wrapper import FFmpegWrapper, EncodingSettings, VideoCodec, PixelFormat, AudioCodec

logger = logging.getLogger(__name__)


class VideoAssembler:
    """
    High-level video assembly from frame sequences.
    
    Provides a simple interface for converting generated frames
    into final video files with various quality options.
    """
    
    # Quality presets with encoding settings
    # All presets now use H.264 Baseline/Main/High profile + yuv420p
    # for universal playback on every device and platform.
    QUALITY_PRESETS = {
        'draft': {
            'codec': VideoCodec.H264,
            'crf': 28,
            'preset': 'ultrafast',
            'pixel_format': PixelFormat.YUV420P,
            'profile': 'baseline',
            'level': '3.1',
            'description': 'Quick preview, lower quality'
        },
        'standard': {
            'codec': VideoCodec.H264,
            'crf': 23,
            'preset': 'medium',
            'pixel_format': PixelFormat.YUV420P,
            'profile': 'high',
            'level': '4.0',
            'description': 'Good balance of quality and speed'
        },
        'high': {
            'codec': VideoCodec.H264,
            'crf': 18,
            'preset': 'slow',
            'pixel_format': PixelFormat.YUV420P,
            'profile': 'high',
            'level': '4.1',
            'description': 'High quality, slower encoding'
        },
        'master': {
            'codec': VideoCodec.H264,
            'crf': 15,
            'preset': 'slower',
            'pixel_format': PixelFormat.YUV420P,
            'profile': 'high',
            'level': '4.2',
            'description': 'Maximum quality H.264 (universal playback)'
        },
        'web': {
            'codec': VideoCodec.H264,
            'crf': 22,
            'preset': 'medium',
            'pixel_format': PixelFormat.YUV420P,
            'profile': 'main',
            'level': '4.0',
            'description': 'Optimized for web streaming'
        },
        'compatible': {
            'codec': VideoCodec.H264,
            'crf': 20,
            'preset': 'medium',
            'pixel_format': PixelFormat.YUV420P,
            'profile': 'baseline',
            'level': '3.1',
            'description': 'Maximum compatibility (all devices & social media)'
        }
    }
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the video assembler.
        
        Args:
            config_dir: Path to configuration directory
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        
        self.config_dir = Path(config_dir)
        self.output_config = self._load_output_config()
        
        # Initialize FFmpeg wrapper
        self.ffmpeg = FFmpegWrapper()
        
        # Progress tracking
        self._progress_callback: Optional[Callable[[float, str], None]] = None
    
    def _load_output_config(self) -> Dict[str, Any]:
        """Load output configuration"""
        defaults_path = self.config_dir / "defaults.yaml"
        if defaults_path.exists():
            with open(defaults_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                return data.get('output', {})
        return {}
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """
        Set progress callback.
        
        Args:
            callback: Function(progress_fraction, message)
        """
        self._progress_callback = callback
    
    def _report_progress(self, progress: float, message: str):
        """Report progress"""
        if self._progress_callback:
            self._progress_callback(progress, message)
        logger.info(f"[{progress*100:.0f}%] {message}")
    
    def get_encoding_settings(
        self,
        quality_preset: str = 'standard',
        **overrides
    ) -> EncodingSettings:
        """
        Get encoding settings for a quality preset.
        
        Args:
            quality_preset: Preset name (draft, standard, high, master, web)
            **overrides: Override specific settings
            
        Returns:
            EncodingSettings object
        """
        if quality_preset not in self.QUALITY_PRESETS:
            logger.warning(f"Unknown preset '{quality_preset}', using 'standard'")
            quality_preset = 'standard'
        
        preset_config = self.QUALITY_PRESETS[quality_preset].copy()
        preset_config.pop('description', None)
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(EncodingSettings, key):
                preset_config[key] = value
        
        # Create settings object
        settings = EncodingSettings(
            codec=preset_config.get('codec', VideoCodec.H264),
            crf=preset_config.get('crf', 23),
            preset=preset_config.get('preset', 'medium'),
            pixel_format=preset_config.get('pixel_format', PixelFormat.YUV420P),
            profile=preset_config.get('profile'),
            level=preset_config.get('level'),
            bitrate=preset_config.get('bitrate'),
            audio_codec=AudioCodec.NONE  # No audio by default for generated videos
        )
        
        return settings
    
    def assemble(
        self,
        frame_paths: Union[List[str], List[Path]],
        output_path: Union[str, Path],
        fps: float = 24.0,
        quality_preset: str = 'standard',
        **kwargs
    ) -> bool:
        """
        Assemble frames into a video file.
        
        Args:
            frame_paths: List of frame file paths
            output_path: Output video path
            fps: Frames per second
            quality_preset: Quality preset name
            **kwargs: Additional encoding options
            
        Returns:
            True if successful
        """
        if not frame_paths:
            logger.error("No frames provided")
            return False
        
        frame_paths = [Path(p) for p in frame_paths]
        output_path = Path(output_path)
        
        # Verify frames exist
        missing = [p for p in frame_paths if not p.exists()]
        if missing:
            logger.error(f"Missing frames: {missing[:5]}...")
            return False
        
        self._report_progress(0.1, f"Preparing to assemble {len(frame_paths)} frames")
        
        # Determine frame pattern
        # Check if frames follow a pattern
        first_frame = frame_paths[0]
        frame_dir = first_frame.parent
        
        # Try to detect pattern from filenames
        import re
        pattern_match = re.search(r'(\d+)', first_frame.stem)
        
        if pattern_match:
            # Use pattern-based input
            prefix = first_frame.stem[:pattern_match.start()]
            suffix = first_frame.stem[pattern_match.end():]
            digit_count = len(pattern_match.group())
            
            frame_pattern = str(frame_dir / f"{prefix}%0{digit_count}d{suffix}{first_frame.suffix}")
        else:
            # Create concat file for arbitrary frame names
            frame_pattern = self._create_concat_list(frame_paths)
        
        self._report_progress(0.2, "Frame pattern detected")
        
        # Get encoding settings
        settings = self.get_encoding_settings(quality_preset, **kwargs)
        
        self._report_progress(0.3, f"Encoding with {quality_preset} preset")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Encode video
        success, message = self.ffmpeg.encode_from_frames(
            frame_pattern=frame_pattern,
            output_path=str(output_path),
            fps=fps,
            settings=settings
        )
        
        if success:
            self._report_progress(1.0, f"Video saved: {output_path}")
            
            # Get video info
            info = self.ffmpeg.get_video_info(str(output_path))
            if info:
                duration = float(info.get('format', {}).get('duration', 0))
                size_mb = int(info.get('format', {}).get('size', 0)) / (1024 * 1024)
                logger.info(f"Video: {duration:.2f}s, {size_mb:.2f} MB")
        else:
            logger.error(f"Encoding failed: {message}")
        
        return success
    
    def _create_concat_list(self, frame_paths: List[Path]) -> str:
        """
        Create a concat list file for FFmpeg.
        
        When frame names don't follow a numeric pattern, we save them
        as individual images and use the frame pattern approach instead.
        
        Args:
            frame_paths: List of frame paths
            
        Returns:
            Frame pattern string for FFmpeg -i input
        """
        import tempfile
        import shutil
        
        # Copy frames into a temp dir with sequential names
        tmpdir = tempfile.mkdtemp(prefix='videoai_frames_')
        for i, src in enumerate(frame_paths):
            dst = Path(tmpdir) / f"frame_{i:06d}{src.suffix}"
            shutil.copy2(str(src), str(dst))
        
        suffix = frame_paths[0].suffix
        return str(Path(tmpdir) / f"frame_%06d{suffix}")
    
    def assemble_from_directory(
        self,
        frame_dir: Union[str, Path],
        output_path: Union[str, Path],
        pattern: str = "*.png",
        fps: float = 24.0,
        quality_preset: str = 'standard',
        **kwargs
    ) -> bool:
        """
        Assemble video from frames in a directory.
        
        Args:
            frame_dir: Directory containing frames
            output_path: Output video path
            pattern: Glob pattern for frames
            fps: Frames per second
            quality_preset: Quality preset
            **kwargs: Additional encoding options
            
        Returns:
            True if successful
        """
        frame_dir = Path(frame_dir)
        
        if not frame_dir.exists():
            logger.error(f"Frame directory not found: {frame_dir}")
            return False
        
        # Find frames
        frame_paths = sorted(frame_dir.glob(pattern))
        
        if not frame_paths:
            logger.error(f"No frames found matching '{pattern}' in {frame_dir}")
            return False
        
        logger.info(f"Found {len(frame_paths)} frames")
        
        return self.assemble(
            frame_paths=frame_paths,
            output_path=output_path,
            fps=fps,
            quality_preset=quality_preset,
            **kwargs
        )
    
    def assemble_with_audio(
        self,
        frame_paths: Union[List[str], List[Path]],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        fps: float = 24.0,
        quality_preset: str = 'standard',
        **kwargs
    ) -> bool:
        """
        Assemble video with audio track.
        
        Args:
            frame_paths: List of frame paths
            audio_path: Path to audio file
            output_path: Output video path
            fps: Frames per second
            quality_preset: Quality preset
            **kwargs: Additional options
            
        Returns:
            True if successful
        """
        import tempfile
        
        # First, create video without audio
        temp_video = tempfile.mktemp(suffix='.mp4')
        
        try:
            success = self.assemble(
                frame_paths=frame_paths,
                output_path=temp_video,
                fps=fps,
                quality_preset=quality_preset,
                **kwargs
            )
            
            if not success:
                return False
            
            # Add audio
            success, message = self.ffmpeg.add_audio(
                video_path=temp_video,
                audio_path=str(audio_path),
                output_path=str(output_path)
            )
            
            return success
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_video):
                os.unlink(temp_video)
    
    def create_preview(
        self,
        frame_paths: Union[List[str], List[Path]],
        output_path: Union[str, Path],
        fps: float = 24.0,
        max_frames: int = 30  # Limit for quick preview
    ) -> bool:
        """
        Create a quick preview video from frames.
        
        Args:
            frame_paths: List of frame paths
            output_path: Output video path
            fps: Frames per second
            max_frames: Maximum frames to include
            
        Returns:
            True if successful
        """
        # Sample frames evenly if we have too many
        if len(frame_paths) > max_frames:
            step = len(frame_paths) // max_frames
            frame_paths = frame_paths[::step][:max_frames]
        
        return self.assemble(
            frame_paths=frame_paths,
            output_path=output_path,
            fps=fps,
            quality_preset='draft'
        )
    
    def get_available_presets(self) -> Dict[str, str]:
        """Get available quality presets with descriptions"""
        return {
            name: config['description']
            for name, config in self.QUALITY_PRESETS.items()
        }


class VideoEditor:
    """
    Basic video editing operations.
    
    Supports trimming, concatenation, and format conversion.
    This is a foundation for future timeline-based editing.
    """
    
    def __init__(self):
        self.ffmpeg = FFmpegWrapper()
    
    def trim(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: Optional[float] = None,
        duration: Optional[float] = None
    ) -> bool:
        """
        Trim a video to a specific time range.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            start_time: Start time in seconds
            end_time: End time in seconds (optional)
            duration: Duration in seconds (optional, alternative to end_time)
            
        Returns:
            True if successful
        """
        if end_time is not None and duration is None:
            duration = end_time - start_time
        
        success, message = self.ffmpeg.encode(
            input_path=input_path,
            output_path=output_path,
            start_time=start_time,
            duration=duration
        )
        
        return success
    
    def concatenate(
        self,
        video_paths: List[str],
        output_path: str,
        reencode: bool = False
    ) -> bool:
        """
        Concatenate multiple videos.
        
        Args:
            video_paths: List of video paths
            output_path: Output path
            reencode: Whether to re-encode
            
        Returns:
            True if successful
        """
        success, message = self.ffmpeg.concatenate_videos(
            video_paths=video_paths,
            output_path=output_path,
            reencode=reencode
        )
        
        return success
    
    def convert_format(
        self,
        input_path: str,
        output_path: str,
        quality_preset: str = 'standard'
    ) -> bool:
        """
        Convert video to different format.
        
        Args:
            input_path: Input video path
            output_path: Output video path (format determined by extension)
            quality_preset: Quality preset
            
        Returns:
            True if successful
        """
        assembler = VideoAssembler()
        settings = assembler.get_encoding_settings(quality_preset)
        
        success, message = self.ffmpeg.encode(
            input_path=input_path,
            output_path=output_path,
            settings=settings
        )
        
        return success


def main():
    """Test video assembler"""
    logging.basicConfig(level=logging.INFO)
    
    assembler = VideoAssembler()
    
    print("\n=== Video Assembler ===")
    print(f"FFmpeg available: {assembler.ffmpeg.is_available()}")
    
    # Show presets
    print("\nAvailable presets:")
    for name, desc in assembler.get_available_presets().items():
        print(f"  {name}: {desc}")
    
    # Show sample settings
    settings = assembler.get_encoding_settings('high')
    print(f"\nHigh quality settings:")
    print(f"  Codec: {settings.codec.value}")
    print(f"  CRF: {settings.crf}")
    print(f"  Preset: {settings.preset}")
    print(f"  Pixel format: {settings.pixel_format.value}")


if __name__ == "__main__":
    main()
