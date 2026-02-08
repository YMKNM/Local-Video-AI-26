"""
FFmpeg Wrapper - Python interface to FFmpeg for video encoding

This module handles:
- FFmpeg command construction
- Video encoding with various codecs
- Audio handling
- Progress tracking
"""

import os
import re
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VideoCodec(Enum):
    """Supported video codecs"""
    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"
    AV1 = "libaom-av1"
    PRORES = "prores_ks"


class AudioCodec(Enum):
    """Supported audio codecs"""
    AAC = "aac"
    MP3 = "libmp3lame"
    OPUS = "libopus"
    FLAC = "flac"
    COPY = "copy"
    NONE = None


class PixelFormat(Enum):
    """Common pixel formats"""
    YUV420P = "yuv420p"  # Most compatible
    YUV444P = "yuv444p"  # Higher quality
    RGB24 = "rgb24"
    RGBA = "rgba"


@dataclass
class EncodingSettings:
    """Video encoding settings"""
    codec: VideoCodec = VideoCodec.H264
    crf: int = 23  # Constant Rate Factor (lower = better quality)
    preset: str = "medium"  # Encoding speed preset
    bitrate: Optional[str] = None  # e.g., "8M" for 8 Mbps
    maxrate: Optional[str] = None
    bufsize: Optional[str] = None
    pixel_format: PixelFormat = PixelFormat.YUV420P
    profile: Optional[str] = None  # e.g., "high" for H.264
    level: Optional[str] = None  # e.g., "4.1" for H.264
    tune: Optional[str] = None  # e.g., "film", "animation"
    
    # Audio settings
    audio_codec: AudioCodec = AudioCodec.AAC
    audio_bitrate: str = "192k"
    audio_sample_rate: int = 48000
    
    # Additional options
    extra_options: List[str] = None


class FFmpegWrapper:
    """
    Python wrapper for FFmpeg operations.
    
    Provides a clean interface for video encoding, format conversion,
    and other FFmpeg operations.
    """
    
    # Preset quality settings
    QUALITY_PRESETS = {
        'low': {'crf': 28, 'preset': 'faster', 'bitrate': '2M'},
        'medium': {'crf': 23, 'preset': 'medium', 'bitrate': '5M'},
        'high': {'crf': 18, 'preset': 'slow', 'bitrate': '10M'},
        'ultra': {'crf': 15, 'preset': 'slower', 'bitrate': '20M'},
        'lossless': {'crf': 0, 'preset': 'veryslow', 'bitrate': None},
    }
    
    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        Initialize FFmpeg wrapper.
        
        Args:
            ffmpeg_path: Optional path to FFmpeg executable
        """
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
        self.ffprobe_path = self._find_ffprobe()
        
        if not self.ffmpeg_path:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg and add it to PATH, "
                "or provide the path explicitly."
            )
        
        self._version = self._get_version()
        logger.info(f"FFmpeg found: {self.ffmpeg_path} (version: {self._version})")
    
    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable"""
        # Try imageio_ffmpeg first (bundled, always works)
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.isfile(exe):
                return exe
        except ImportError:
            pass
        
        # Check common locations on Windows
        possible_paths = [
            "ffmpeg",
            "ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        ]
        
        for path in possible_paths:
            if shutil.which(path):
                return shutil.which(path)
        
        return None
    
    def _find_ffprobe(self) -> Optional[str]:
        """Find FFprobe executable"""
        if self.ffmpeg_path:
            # Only replace the executable name, not directory components
            dirpath = os.path.dirname(self.ffmpeg_path)
            basename = os.path.basename(self.ffmpeg_path).replace('ffmpeg', 'ffprobe')
            ffprobe_path = os.path.join(dirpath, basename) if dirpath else basename
            if os.path.isfile(ffprobe_path):
                return ffprobe_path
            if shutil.which(ffprobe_path):
                return ffprobe_path
        
        return shutil.which('ffprobe') or shutil.which('ffprobe.exe')
    
    def _get_version(self) -> str:
        """Get FFmpeg version"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Extract version from first line
            first_line = result.stdout.split('\n')[0]
            match = re.search(r'ffmpeg version (\S+)', first_line)
            return match.group(1) if match else "unknown"
        except Exception:
            return "unknown"
    
    def is_available(self) -> bool:
        """Check if FFmpeg is available"""
        return self.ffmpeg_path is not None
    
    def get_supported_encoders(self) -> List[str]:
        """Get list of supported video encoders"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-encoders'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            encoders = []
            for line in result.stdout.split('\n'):
                if line.strip().startswith('V'):
                    parts = line.split()
                    if len(parts) >= 2:
                        encoders.append(parts[1])
            
            return encoders
        except Exception:
            return []
    
    def build_command(
        self,
        input_path: str,
        output_path: str,
        settings: Optional[EncodingSettings] = None,
        fps: Optional[float] = None,
        duration: Optional[float] = None,
        start_time: Optional[float] = None,
        overwrite: bool = True
    ) -> List[str]:
        """
        Build FFmpeg command from settings.
        
        Args:
            input_path: Input file or pattern
            output_path: Output file path
            settings: Encoding settings
            fps: Frame rate
            duration: Duration in seconds
            start_time: Start time in seconds
            overwrite: Whether to overwrite existing output
            
        Returns:
            List of command arguments
        """
        if settings is None:
            settings = EncodingSettings()
        
        cmd = [self.ffmpeg_path]
        
        # Overwrite flag
        if overwrite:
            cmd.append('-y')
        
        # Start time
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])
        
        # Frame rate for input (important for image sequences)
        if fps is not None:
            cmd.extend(['-framerate', str(fps)])
        
        # Input
        cmd.extend(['-i', input_path])
        
        # Duration
        if duration is not None:
            cmd.extend(['-t', str(duration)])
        
        # Video codec settings
        cmd.extend(['-c:v', settings.codec.value])
        
        # Quality settings
        if settings.bitrate:
            cmd.extend(['-b:v', settings.bitrate])
            if settings.maxrate:
                cmd.extend(['-maxrate', settings.maxrate])
            if settings.bufsize:
                cmd.extend(['-bufsize', settings.bufsize])
        else:
            cmd.extend(['-crf', str(settings.crf)])
        
        # Preset
        if settings.preset:
            cmd.extend(['-preset', settings.preset])
        
        # Pixel format
        cmd.extend(['-pix_fmt', settings.pixel_format.value])
        
        # Profile and level (for H.264/H.265)
        if settings.profile:
            cmd.extend(['-profile:v', settings.profile])
        if settings.level:
            cmd.extend(['-level', settings.level])
        
        # Tune
        if settings.tune:
            cmd.extend(['-tune', settings.tune])
        
        # Audio settings
        if settings.audio_codec == AudioCodec.NONE:
            cmd.append('-an')  # No audio
        elif settings.audio_codec != AudioCodec.COPY:
            cmd.extend(['-c:a', settings.audio_codec.value])
            cmd.extend(['-b:a', settings.audio_bitrate])
            cmd.extend(['-ar', str(settings.audio_sample_rate)])
        
        # ── Universal compatibility flags ───────────────────────
        codec_name = settings.codec.value if settings.codec else ''
        output_ext = os.path.splitext(output_path)[1].lower()

        if codec_name in ('libx264', 'libx265'):
            # Ensure dimensions are even (required by most H.264/H.265 decoders)
            cmd.extend(['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])

            # BT.709 colour metadata — needed for correct colours on
            # phones, browsers, QuickTime, and social-media re-encoders
            cmd.extend([
                '-colorspace', 'bt709',
                '-color_primaries', 'bt709',
                '-color_trc', 'bt709',
            ])

        if output_ext in ('.mp4', '.m4v', '.mov'):
            # Move the MOOV atom to the front so the file can start
            # playing before it has finished downloading / copying
            cmd.extend(['-movflags', '+faststart'])

        # Extra options
        if settings.extra_options:
            cmd.extend(settings.extra_options)
        
        # Output
        cmd.append(output_path)
        
        return cmd
    
    def encode_from_frames(
        self,
        frame_pattern: str,
        output_path: str,
        fps: float = 24.0,
        settings: Optional[EncodingSettings] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Tuple[bool, str]:
        """
        Encode video from image sequence.
        
        Args:
            frame_pattern: Path pattern for frames (e.g., "frames/%05d.png")
            output_path: Output video file path
            fps: Frame rate
            settings: Encoding settings
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (success, message)
        """
        if settings is None:
            settings = EncodingSettings(
                audio_codec=AudioCodec.NONE,
                profile='high',
                level='4.0',
            )
        if settings.audio_codec != AudioCodec.NONE:
            settings.audio_codec = AudioCodec.NONE  # No audio for frame sequence
        
        cmd = self.build_command(
            input_path=frame_pattern,
            output_path=output_path,
            settings=settings,
            fps=fps
        )
        
        logger.info(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            # Run FFmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Video encoded successfully: {output_path}")
                return True, f"Video saved to {output_path}"
            else:
                logger.error(f"FFmpeg error: {stderr}")
                return False, stderr
                
        except Exception as e:
            logger.error(f"Failed to run FFmpeg: {e}")
            return False, str(e)
    
    def encode(
        self,
        input_path: str,
        output_path: str,
        settings: Optional[EncodingSettings] = None,
        **kwargs
    ) -> Tuple[bool, str]:
        """
        Encode/transcode a video file.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            settings: Encoding settings
            **kwargs: Additional options for build_command
            
        Returns:
            Tuple of (success, message)
        """
        cmd = self.build_command(
            input_path=input_path,
            output_path=output_path,
            settings=settings,
            **kwargs
        )
        
        logger.info(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return True, f"Video saved to {output_path}"
            else:
                return False, result.stderr
                
        except Exception as e:
            return False, str(e)
    
    def get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Get video information using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        if not self.ffprobe_path:
            logger.warning("FFprobe not available")
            return None
        
        try:
            cmd = [
                self.ffprobe_path,
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                return json.loads(result.stdout)
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
        
        return None
    
    def extract_frames(
        self,
        video_path: str,
        output_pattern: str,
        fps: Optional[float] = None,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
        quality: int = 2  # 1-31, lower is better
    ) -> Tuple[bool, str]:
        """
        Extract frames from a video.
        
        Args:
            video_path: Input video path
            output_pattern: Output pattern (e.g., "frames/%05d.png")
            fps: Output frame rate (None = same as input)
            start_time: Start time in seconds
            duration: Duration in seconds
            quality: JPEG quality (for jpg output)
            
        Returns:
            Tuple of (success, message)
        """
        cmd = [self.ffmpeg_path, '-y']
        
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])
        
        cmd.extend(['-i', video_path])
        
        if duration is not None:
            cmd.extend(['-t', str(duration)])
        
        if fps is not None:
            cmd.extend(['-vf', f'fps={fps}'])
        
        # Quality for JPEG
        if output_pattern.endswith('.jpg') or output_pattern.endswith('.jpeg'):
            cmd.extend(['-qscale:v', str(quality)])
        
        cmd.append(output_pattern)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True, f"Frames extracted to {output_pattern}"
            else:
                return False, result.stderr
                
        except Exception as e:
            return False, str(e)
    
    def concatenate_videos(
        self,
        video_paths: List[str],
        output_path: str,
        reencode: bool = False
    ) -> Tuple[bool, str]:
        """
        Concatenate multiple videos.
        
        Args:
            video_paths: List of input video paths
            output_path: Output video path
            reencode: Whether to re-encode (slower but more compatible)
            
        Returns:
            Tuple of (success, message)
        """
        import tempfile
        
        # Create concat file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")
            concat_file = f.name
        
        try:
            cmd = [self.ffmpeg_path, '-y', '-f', 'concat', '-safe', '0', '-i', concat_file]
            
            if reencode:
                cmd.extend(['-c:v', 'libx264', '-c:a', 'aac'])
            else:
                cmd.extend(['-c', 'copy'])
            
            cmd.append(output_path)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True, f"Videos concatenated to {output_path}"
            else:
                return False, result.stderr
                
        finally:
            os.unlink(concat_file)
    
    def add_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        audio_settings: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Add audio track to a video.
        
        Args:
            video_path: Input video path
            audio_path: Audio file path
            output_path: Output video path
            audio_settings: Optional audio encoding settings
            
        Returns:
            Tuple of (success, message)
        """
        cmd = [
            self.ffmpeg_path, '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',  # Match shorter of video/audio
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True, f"Audio added: {output_path}"
            else:
                return False, result.stderr
                
        except Exception as e:
            return False, str(e)


def main():
    """Test FFmpeg wrapper"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        ffmpeg = FFmpegWrapper()
        
        print(f"\n=== FFmpeg Wrapper ===")
        print(f"FFmpeg path: {ffmpeg.ffmpeg_path}")
        print(f"FFprobe path: {ffmpeg.ffprobe_path}")
        print(f"Version: {ffmpeg._version}")
        
        # Show some encoders
        encoders = ffmpeg.get_supported_encoders()
        print(f"\nAvailable encoders: {len(encoders)}")
        print(f"Sample: {encoders[:10]}")
        
        # Build sample command
        settings = EncodingSettings(
            codec=VideoCodec.H264,
            crf=20,
            preset='medium'
        )
        
        cmd = ffmpeg.build_command(
            input_path="frames/%05d.png",
            output_path="output.mp4",
            settings=settings,
            fps=24
        )
        
        print(f"\nSample command:")
        print(' '.join(cmd))
        
    except RuntimeError as e:
        print(f"FFmpeg not found: {e}")


if __name__ == "__main__":
    main()
