"""
Video AI Python SDK

High-level Python SDK for interacting with Video AI Enterprise Platform.

Usage:
    from video_ai.sdk import VideoAIClient
    
    # Initialize client
    client = VideoAIClient(api_url="http://localhost:8000")
    
    # Generate video
    result = await client.generate(
        prompt="A sunset over the ocean",
        duration=10,
        quality="high"
    )
    
    # Download result
    await client.download(result.job_id, "output.mp4")
"""

import os
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import HTTP clients
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

class JobStatus(Enum):
    """Job status enumeration"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationConfig:
    """Video generation configuration"""
    prompt: str
    negative_prompt: str = ""
    width: int = 1280
    height: int = 720
    fps: int = 24
    duration_seconds: float = 6.0
    model_name: Optional[str] = None
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    quality_preset: str = "balanced"
    style_preset: Optional[str] = None
    use_preview: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class GenerationJob:
    """Represents a generation job"""
    job_id: str
    status: JobStatus
    progress: float = 0.0
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_url: Optional[str] = None
    preview_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationJob':
        return cls(
            job_id=data.get('job_id', ''),
            status=JobStatus(data.get('status', 'queued')),
            progress=data.get('progress', 0.0),
            created_at=data.get('created_at'),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            output_url=data.get('output_url'),
            preview_url=data.get('preview_url'),
            error=data.get('error'),
            metadata=data.get('metadata', {})
        )


@dataclass
class ModelInfo:
    """Model information"""
    name: str
    type: str
    parameters: str
    max_resolution: List[int]
    max_frames: int
    vram_required_gb: float
    supported_modes: List[str]


# =============================================================================
# Video AI Client
# =============================================================================

class VideoAIClient:
    """
    Python SDK client for Video AI Enterprise Platform.
    
    Provides high-level methods for:
    - Video generation (text-to-video, image-to-video)
    - Job management
    - Real-time progress tracking via WebSocket
    - Model information
    - System status
    
    Usage:
        # Async usage
        async with VideoAIClient("http://localhost:8000") as client:
            job = await client.generate(prompt="A sunset over the ocean")
            result = await client.wait_for_completion(job.job_id)
            await client.download(result.job_id, "video.mp4")
        
        # Sync usage
        client = VideoAIClient("http://localhost:8000")
        job = client.generate_sync(prompt="A sunset")
        result = client.wait_sync(job.job_id)
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        verify_ssl: bool = True
    ):
        """
        Initialize Video AI client.
        
        Args:
            api_url: Base URL of the API server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            verify_ssl: Verify SSL certificates
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")
        
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        
        # WebSocket connection
        self._ws_connection = None
        
        # Callbacks
        self._progress_callbacks: Dict[str, Callable] = {}
    
    async def __aenter__(self):
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized"""
        if self._client is None:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            self._client = httpx.AsyncClient(
                base_url=self.api_url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
    
    async def close(self):
        """Close client connections"""
        if self._client:
            await self._client.aclose()
            self._client = None
        
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    # =========================================================================
    # Generation Methods
    # =========================================================================
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
        duration_seconds: float = 6.0,
        model_name: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        quality_preset: str = "balanced",
        style_preset: Optional[str] = None,
        use_preview: bool = False,
        on_progress: Optional[Callable[[float, str], None]] = None
    ) -> GenerationJob:
        """
        Generate a video from a text prompt.
        
        Args:
            prompt: Text description of the video
            negative_prompt: What to avoid in generation
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            duration_seconds: Video duration
            model_name: Model to use (auto-selected if None)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            seed: Random seed for reproducibility
            quality_preset: Quality preset (fast, balanced, quality, ultra)
            style_preset: Visual style preset
            use_preview: Generate fast preview first
            on_progress: Progress callback function
            
        Returns:
            GenerationJob with job details
        """
        await self._ensure_client()
        
        config = GenerationConfig(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            fps=fps,
            duration_seconds=duration_seconds,
            model_name=model_name,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            quality_preset=quality_preset,
            style_preset=style_preset,
            use_preview=use_preview
        )
        
        response = await self._client.post(
            "/generate",
            json=config.to_dict()
        )
        response.raise_for_status()
        
        data = response.json()
        job = GenerationJob(
            job_id=data['job_id'],
            status=JobStatus(data['status']),
            metadata={'estimated_time': data.get('estimated_time_seconds')}
        )
        
        # Set up progress callback if provided
        if on_progress:
            self._progress_callbacks[job.job_id] = on_progress
        
        return job
    
    async def generate_from_config(
        self,
        config: GenerationConfig,
        on_progress: Optional[Callable[[float, str], None]] = None
    ) -> GenerationJob:
        """Generate video from a configuration object"""
        return await self.generate(**config.to_dict(), on_progress=on_progress)
    
    async def generate_image_to_video(
        self,
        prompt: str,
        image_path: str,
        motion_strength: float = 0.5,
        camera_motion: Optional[str] = None,
        fps: int = 24,
        duration_seconds: float = 4.0,
        **kwargs
    ) -> GenerationJob:
        """
        Generate video from an image.
        
        Args:
            prompt: Motion/action description
            image_path: Path to input image
            motion_strength: Motion intensity (0-1)
            camera_motion: Camera motion type
            fps: Frames per second
            duration_seconds: Video duration
            **kwargs: Additional parameters
            
        Returns:
            GenerationJob with job details
        """
        await self._ensure_client()
        
        # Read and encode image
        import base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        request_data = {
            'prompt': prompt,
            'image': image_data,
            'motion_strength': motion_strength,
            'camera_motion': camera_motion,
            'fps': fps,
            'duration_seconds': duration_seconds,
            **kwargs
        }
        
        response = await self._client.post(
            "/generate/image-to-video",
            json=request_data
        )
        response.raise_for_status()
        
        data = response.json()
        return GenerationJob(
            job_id=data['job_id'],
            status=JobStatus(data['status'])
        )
    
    # =========================================================================
    # Job Management Methods
    # =========================================================================
    
    async def get_job(self, job_id: str) -> GenerationJob:
        """
        Get job status and details.
        
        Args:
            job_id: Job identifier
            
        Returns:
            GenerationJob with current status
        """
        await self._ensure_client()
        
        response = await self._client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        
        return GenerationJob.from_dict(response.json())
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or queued job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancelled successfully
        """
        await self._ensure_client()
        
        response = await self._client.delete(f"/jobs/{job_id}")
        return response.status_code == 200
    
    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[GenerationJob]:
        """
        List generation jobs.
        
        Args:
            status: Filter by status
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of GenerationJob objects
        """
        await self._ensure_client()
        
        params = {'limit': limit, 'offset': offset}
        if status:
            params['status'] = status
        
        response = await self._client.get("/jobs", params=params)
        response.raise_for_status()
        
        data = response.json()
        return [GenerationJob.from_dict(j) for j in data.get('jobs', [])]
    
    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        on_progress: Optional[Callable[[float, str], None]] = None
    ) -> GenerationJob:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job identifier
            poll_interval: Polling interval in seconds
            timeout: Maximum wait time (None = no timeout)
            on_progress: Progress callback
            
        Returns:
            Completed GenerationJob
            
        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If job failed
        """
        start_time = time.time()
        
        while True:
            job = await self.get_job(job_id)
            
            if on_progress:
                on_progress(job.progress, f"Status: {job.status.value}")
            
            if job.status == JobStatus.COMPLETED:
                return job
            
            if job.status == JobStatus.FAILED:
                raise RuntimeError(f"Job failed: {job.error}")
            
            if job.status == JobStatus.CANCELLED:
                raise RuntimeError("Job was cancelled")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
            
            await asyncio.sleep(poll_interval)
    
    async def download(
        self,
        job_id: str,
        output_path: str,
        download_preview: bool = False
    ) -> str:
        """
        Download generated video.
        
        Args:
            job_id: Job identifier
            output_path: Path to save the video
            download_preview: Download preview instead of full video
            
        Returns:
            Path to downloaded file
        """
        await self._ensure_client()
        
        endpoint = f"/jobs/{job_id}/{'preview' if download_preview else 'output'}"
        
        async with self._client.stream('GET', endpoint) as response:
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)
        
        return output_path
    
    # =========================================================================
    # WebSocket Methods
    # =========================================================================
    
    async def connect_websocket(self, client_id: Optional[str] = None):
        """
        Connect to WebSocket for real-time updates.
        
        Args:
            client_id: Client identifier (auto-generated if None)
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets is required. Install with: pip install websockets")
        
        import websockets
        
        client_id = client_id or f"sdk_{int(time.time())}"
        ws_url = self.api_url.replace('http', 'ws') + f"/ws/{client_id}"
        
        self._ws_connection = await websockets.connect(ws_url)
    
    async def subscribe_to_job(self, job_id: str):
        """Subscribe to job updates via WebSocket"""
        if not self._ws_connection:
            raise RuntimeError("WebSocket not connected. Call connect_websocket() first.")
        
        await self._ws_connection.send(json.dumps({
            'type': 'subscribe',
            'job_id': job_id
        }))
    
    async def listen_for_updates(self) -> Dict[str, Any]:
        """Listen for WebSocket messages"""
        if not self._ws_connection:
            raise RuntimeError("WebSocket not connected")
        
        message = await self._ws_connection.recv()
        return json.loads(message)
    
    # =========================================================================
    # System Methods
    # =========================================================================
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        await self._ensure_client()
        
        response = await self._client.get("/status")
        response.raise_for_status()
        return response.json()
    
    async def get_models(self) -> List[ModelInfo]:
        """Get available models"""
        await self._ensure_client()
        
        response = await self._client.get("/models")
        response.raise_for_status()
        
        return [
            ModelInfo(
                name=m['name'],
                type=m['type'],
                parameters=m['parameters'],
                max_resolution=m['max_resolution'],
                max_frames=m['max_frames'],
                vram_required_gb=m['vram_required_gb'],
                supported_modes=m['supported_modes']
            )
            for m in response.json()
        ]
    
    async def health_check(self) -> bool:
        """Check if API is healthy"""
        await self._ensure_client()
        
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
    
    # =========================================================================
    # Synchronous Wrappers
    # =========================================================================
    
    def generate_sync(self, **kwargs) -> GenerationJob:
        """Synchronous wrapper for generate()"""
        return asyncio.get_event_loop().run_until_complete(
            self.generate(**kwargs)
        )
    
    def wait_sync(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None
    ) -> GenerationJob:
        """Synchronous wrapper for wait_for_completion()"""
        return asyncio.get_event_loop().run_until_complete(
            self.wait_for_completion(job_id, poll_interval, timeout)
        )
    
    def download_sync(self, job_id: str, output_path: str) -> str:
        """Synchronous wrapper for download()"""
        return asyncio.get_event_loop().run_until_complete(
            self.download(job_id, output_path)
        )


# =============================================================================
# Convenience Functions
# =============================================================================

async def generate_video(
    prompt: str,
    api_url: str = "http://localhost:8000",
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Quick function to generate and download a video.
    
    Args:
        prompt: Video description
        api_url: API server URL
        output_path: Where to save video (auto-generated if None)
        **kwargs: Additional generation parameters
        
    Returns:
        Path to generated video
    """
    async with VideoAIClient(api_url) as client:
        # Generate
        job = await client.generate(prompt=prompt, **kwargs)
        
        # Wait for completion
        result = await client.wait_for_completion(job.job_id)
        
        # Download
        if output_path is None:
            output_path = f"video_{job.job_id}.mp4"
        
        await client.download(result.job_id, output_path)
        
        return output_path


def generate_video_sync(
    prompt: str,
    api_url: str = "http://localhost:8000",
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """Synchronous version of generate_video()"""
    return asyncio.get_event_loop().run_until_complete(
        generate_video(prompt, api_url, output_path, **kwargs)
    )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video AI SDK CLI')
    parser.add_argument('command', choices=['generate', 'status', 'models'])
    parser.add_argument('--prompt', '-p', help='Generation prompt')
    parser.add_argument('--api-url', default='http://localhost:8000')
    parser.add_argument('--output', '-o', help='Output path')
    parser.add_argument('--job-id', help='Job ID for status check')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        if not args.prompt:
            print("Error: --prompt is required")
            return
        
        result = generate_video_sync(
            prompt=args.prompt,
            api_url=args.api_url,
            output_path=args.output
        )
        print(f"Video saved to: {result}")
    
    elif args.command == 'status':
        async def check_status():
            async with VideoAIClient(args.api_url) as client:
                if args.job_id:
                    job = await client.get_job(args.job_id)
                    print(f"Job: {job.job_id}")
                    print(f"Status: {job.status.value}")
                    print(f"Progress: {job.progress * 100:.1f}%")
                else:
                    status = await client.get_status()
                    print(json.dumps(status, indent=2))
        
        asyncio.get_event_loop().run_until_complete(check_status())
    
    elif args.command == 'models':
        async def list_models():
            async with VideoAIClient(args.api_url) as client:
                models = await client.get_models()
                for m in models:
                    print(f"- {m.name} ({m.parameters}): {m.supported_modes}")
        
        asyncio.get_event_loop().run_until_complete(list_models())


if __name__ == '__main__':
    main()
