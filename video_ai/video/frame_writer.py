"""
Frame Writer - Handles writing individual frames to disk

This module handles:
- Frame format conversion
- Quality optimization
- Sequential frame naming
- Batch writing
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Any
import numpy as np

logger = logging.getLogger(__name__)


class FrameWriter:
    """
    Handles writing video frames to disk in various formats.
    
    Supports PNG, JPEG, and other image formats through
    PIL or OpenCV backends.
    """
    
    SUPPORTED_FORMATS = {
        'png': {'extension': '.png', 'quality': None, 'lossless': True},
        'jpg': {'extension': '.jpg', 'quality': 95, 'lossless': False},
        'jpeg': {'extension': '.jpg', 'quality': 95, 'lossless': False},
        'webp': {'extension': '.webp', 'quality': 90, 'lossless': False},
        'bmp': {'extension': '.bmp', 'quality': None, 'lossless': True},
        'tiff': {'extension': '.tiff', 'quality': None, 'lossless': True},
    }
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        format: str = "png",
        quality: Optional[int] = None,
        prefix: str = "frame",
        start_index: int = 0,
        zero_pad: int = 5
    ):
        """
        Initialize the frame writer.
        
        Args:
            output_dir: Directory to write frames to
            format: Output format (png, jpg, webp, etc.)
            quality: Quality for lossy formats (1-100)
            prefix: Filename prefix
            start_index: Starting frame index
            zero_pad: Number of digits for frame numbering
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        format = format.lower()
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        self.format = format
        self.format_info = self.SUPPORTED_FORMATS[format]
        self.quality = quality or self.format_info['quality']
        self.prefix = prefix
        self.current_index = start_index
        self.zero_pad = zero_pad
        
        # Track written frames
        self.written_frames: List[Path] = []
        
        # Detect available backend
        self._backend = self._detect_backend()
        logger.info(f"FrameWriter initialized with {self._backend} backend")
    
    def _detect_backend(self) -> str:
        """Detect available image writing backend"""
        try:
            from PIL import Image
            return 'pil'
        except ImportError:
            pass
        
        try:
            import cv2
            return 'cv2'
        except ImportError:
            pass
        
        raise ImportError(
            "No image backend available. Install PIL or OpenCV: "
            "pip install Pillow opencv-python"
        )
    
    def _get_filename(self, index: Optional[int] = None) -> Path:
        """Generate filename for frame"""
        if index is None:
            index = self.current_index
        
        filename = f"{self.prefix}_{str(index).zfill(self.zero_pad)}{self.format_info['extension']}"
        return self.output_dir / filename
    
    def write_frame(
        self,
        frame: np.ndarray,
        index: Optional[int] = None,
        auto_increment: bool = True
    ) -> Path:
        """
        Write a single frame to disk.
        
        Args:
            frame: Frame data as numpy array [H, W, C] or [C, H, W]
            index: Optional frame index (uses auto-increment if None)
            auto_increment: Whether to auto-increment index
            
        Returns:
            Path to written frame
        """
        if index is None:
            index = self.current_index
        
        filepath = self._get_filename(index)
        
        # Ensure correct shape [H, W, C]
        if frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:
            # Likely [C, H, W] format
            frame = np.transpose(frame, (1, 2, 0))
        
        # Handle single channel
        if frame.ndim == 2:
            frame = np.expand_dims(frame, axis=-1)
            frame = np.repeat(frame, 3, axis=-1)
        
        # Convert to uint8 if needed
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        
        # Write using appropriate backend
        if self._backend == 'pil':
            self._write_pil(frame, filepath)
        else:
            self._write_cv2(frame, filepath)
        
        self.written_frames.append(filepath)
        
        if auto_increment:
            self.current_index += 1
        
        return filepath
    
    def _write_pil(self, frame: np.ndarray, filepath: Path):
        """Write frame using PIL"""
        from PIL import Image
        
        img = Image.fromarray(frame)
        
        save_kwargs = {}
        if self.quality is not None and not self.format_info['lossless']:
            if self.format == 'webp':
                save_kwargs['quality'] = self.quality
            else:
                save_kwargs['quality'] = self.quality
        
        if self.format == 'png':
            save_kwargs['compress_level'] = 6  # Balanced compression
        
        img.save(filepath, **save_kwargs)
    
    def _write_cv2(self, frame: np.ndarray, filepath: Path):
        """Write frame using OpenCV"""
        import cv2
        
        # OpenCV uses BGR
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        params = []
        if self.format in ['jpg', 'jpeg'] and self.quality:
            params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        elif self.format == 'png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
        elif self.format == 'webp' and self.quality:
            params = [cv2.IMWRITE_WEBP_QUALITY, self.quality]
        
        cv2.imwrite(str(filepath), frame, params)
    
    def write_frames(
        self,
        frames: Union[np.ndarray, List[np.ndarray]],
        start_index: Optional[int] = None
    ) -> List[Path]:
        """
        Write multiple frames to disk.
        
        Args:
            frames: Array of frames [N, H, W, C] or list of frames
            start_index: Optional starting index
            
        Returns:
            List of paths to written frames
        """
        if start_index is not None:
            self.current_index = start_index
        
        paths = []
        
        if isinstance(frames, np.ndarray):
            if frames.ndim == 4:
                # [N, H, W, C] or [N, C, H, W]
                for i in range(frames.shape[0]):
                    path = self.write_frame(frames[i])
                    paths.append(path)
            else:
                # Single frame
                path = self.write_frame(frames)
                paths.append(path)
        else:
            # List of frames
            for frame in frames:
                path = self.write_frame(frame)
                paths.append(path)
        
        logger.info(f"Wrote {len(paths)} frames to {self.output_dir}")
        return paths
    
    def get_frame_pattern(self) -> str:
        """
        Get the frame pattern for FFmpeg.
        
        Returns:
            Pattern string like "frame_%05d.png"
        """
        return f"{self.prefix}_%0{self.zero_pad}d{self.format_info['extension']}"
    
    def get_frame_glob(self) -> str:
        """
        Get glob pattern for finding frames.
        
        Returns:
            Glob pattern like "frame_*.png"
        """
        return f"{self.prefix}_*{self.format_info['extension']}"
    
    def get_written_frames(self) -> List[Path]:
        """Get list of all written frames"""
        return self.written_frames.copy()
    
    def get_frame_count(self) -> int:
        """Get number of written frames"""
        return len(self.written_frames)
    
    def clear_frames(self, delete_files: bool = False):
        """
        Clear frame tracking, optionally deleting files.
        
        Args:
            delete_files: Whether to delete the frame files
        """
        if delete_files:
            for frame_path in self.written_frames:
                if frame_path.exists():
                    frame_path.unlink()
            logger.info(f"Deleted {len(self.written_frames)} frame files")
        
        self.written_frames.clear()
        self.current_index = 0
    
    def list_existing_frames(self) -> List[Path]:
        """List existing frames in output directory matching pattern"""
        pattern = self.get_frame_glob()
        frames = sorted(self.output_dir.glob(pattern))
        return frames


class FrameBuffer:
    """
    In-memory frame buffer for batch processing.
    
    Accumulates frames in memory before writing to disk,
    useful for memory-constrained scenarios.
    """
    
    def __init__(self, max_frames: int = 100):
        """
        Initialize frame buffer.
        
        Args:
            max_frames: Maximum frames to buffer before auto-flush
        """
        self.max_frames = max_frames
        self.frames: List[np.ndarray] = []
        self.metadata: List[dict] = []
    
    def add(self, frame: np.ndarray, metadata: Optional[dict] = None):
        """
        Add a frame to the buffer.
        
        Args:
            frame: Frame data
            metadata: Optional metadata for the frame
        """
        self.frames.append(frame)
        self.metadata.append(metadata or {})
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity"""
        return len(self.frames) >= self.max_frames
    
    def flush(self, writer: FrameWriter) -> List[Path]:
        """
        Write all buffered frames to disk.
        
        Args:
            writer: FrameWriter to use
            
        Returns:
            List of written frame paths
        """
        paths = []
        
        for frame, meta in zip(self.frames, self.metadata):
            path = writer.write_frame(frame)
            paths.append(path)
        
        self.clear()
        return paths
    
    def clear(self):
        """Clear the buffer without writing"""
        self.frames.clear()
        self.metadata.clear()
    
    def __len__(self) -> int:
        return len(self.frames)


def main():
    """Test frame writer"""
    logging.basicConfig(level=logging.INFO)
    
    # Create test frames
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = FrameWriter(tmpdir, format='png')
        
        # Create dummy frames
        frames = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
        
        # Write frames
        paths = writer.write_frames(frames)
        
        print(f"Wrote {len(paths)} frames")
        print(f"Frame pattern: {writer.get_frame_pattern()}")
        print(f"Frame paths: {paths[:3]}...")


if __name__ == "__main__":
    main()
