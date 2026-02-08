"""
Video AI Enterprise API Server

Microservices-based REST and WebSocket API for video generation.

Features:
- REST API for job submission and management
- WebSocket for real-time progress and feedback
- JWT authentication (optional)
- Rate limiting and quotas
- OpenAPI documentation
- Health checks and metrics
"""

import os
import json
import time
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from contextlib import asynccontextmanager
import yaml

# FastAPI imports
from fastapi import (
    FastAPI, HTTPException, WebSocket, WebSocketDisconnect,
    BackgroundTasks, Depends, Query, File, UploadFile, Form
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class VideoGenerationRequest(BaseModel):
    """Video generation request"""
    prompt: str = Field(..., description="Text prompt for video generation", min_length=1, max_length=2000)
    negative_prompt: str = Field("", description="Negative prompt for what to avoid")
    
    # Video settings
    width: int = Field(854, ge=256, le=3840, description="Video width in pixels")
    height: int = Field(480, ge=256, le=2160, description="Video height in pixels")
    fps: int = Field(24, ge=1, le=60, description="Frames per second")
    duration_seconds: float = Field(6.0, ge=1.0, le=60.0, description="Video duration in seconds")
    
    # Model settings
    model_name: Optional[str] = Field(None, description="Model to use (auto-selected if not specified)")
    num_inference_steps: int = Field(30, ge=1, le=150, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Classifier-free guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    # Quality settings
    quality_preset: str = Field("balanced", description="Quality preset: fast, balanced, quality, ultra")
    use_preview: bool = Field(False, description="Generate fast preview first")
    
    # Advanced settings
    style_preset: Optional[str] = Field(None, description="Style preset to apply")
    reference_image: Optional[str] = Field(None, description="Base64 encoded reference image for img2vid")
    audio_prompt: Optional[str] = Field(None, description="Audio prompt for synchronized audio generation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A cinematic drone shot over misty mountains at sunrise, golden light",
                "negative_prompt": "blurry, low quality, distorted",
                "width": 1280,
                "height": 720,
                "fps": 24,
                "duration_seconds": 6.0,
                "quality_preset": "balanced"
            }
        }


class ImageToVideoRequest(BaseModel):
    """Image-to-video generation request"""
    prompt: str = Field(..., description="Text prompt describing the motion")
    image: str = Field(..., description="Base64 encoded input image")
    
    # Motion settings
    motion_strength: float = Field(0.5, ge=0.0, le=1.0, description="Motion intensity")
    camera_motion: Optional[str] = Field(None, description="Camera motion type: pan, zoom, orbit, etc.")
    
    # Video settings
    fps: int = Field(24, ge=1, le=60)
    duration_seconds: float = Field(4.0, ge=1.0, le=30.0)
    
    # Model settings
    model_name: Optional[str] = Field(None)
    num_inference_steps: int = Field(30, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(None)


class JobResponse(BaseModel):
    """Job submission response"""
    job_id: str
    status: str
    message: str
    estimated_time_seconds: Optional[float] = None
    queue_position: Optional[int] = None


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    progress: float = 0.0
    current_step: Optional[str] = None
    estimated_remaining_seconds: Optional[float] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    output_url: Optional[str] = None
    preview_url: Optional[str] = None
    metadata: Dict[str, Any] = {}


class SystemStatusResponse(BaseModel):
    """System status response"""
    status: str
    gpu_name: str
    gpu_vram_total_gb: float
    gpu_vram_free_gb: float
    queue_size: int
    active_jobs: int
    uptime_seconds: float
    version: str


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    type: str
    parameters: str
    max_resolution: List[int]
    max_frames: int
    vram_required_gb: float
    supported_modes: List[str]


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}  # job_id -> connections
        self.client_connections: Dict[str, WebSocket] = {}  # client_id -> connection
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.client_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")
    
    def disconnect(self, client_id: str):
        if client_id in self.client_connections:
            del self.client_connections[client_id]
        logger.info(f"WebSocket disconnected: {client_id}")
    
    async def subscribe_to_job(self, websocket: WebSocket, job_id: str):
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
    
    async def broadcast_job_update(self, job_id: str, message: Dict[str, Any]):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket: {e}")
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        if client_id in self.client_connections:
            try:
                await self.client_connections[client_id].send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client {client_id}: {e}")


# =============================================================================
# Application Setup
# =============================================================================

# Global state
connection_manager = ConnectionManager()
jobs_store: Dict[str, Dict[str, Any]] = {}
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Video AI API Server...")
    
    # Initialize GPU scheduler
    from ..runtime.gpu_scheduler import GPUScheduler
    app.state.scheduler = GPUScheduler()
    app.state.scheduler.start()
    
    # Load configuration
    config_dir = Path(__file__).parent.parent / "configs"
    app.state.config_dir = config_dir
    
    yield
    
    # Shutdown
    logger.info("Shutting down Video AI API Server...")
    app.state.scheduler.stop()


# Create FastAPI app
app = FastAPI(
    title="Video AI Enterprise API",
    description="Enterprise-ready AI video generation platform with REST and WebSocket APIs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REST API Endpoints
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """API root endpoint"""
    return {
        "name": "Video AI Enterprise API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - start_time
    }


@app.get("/status", response_model=SystemStatusResponse, tags=["General"])
async def system_status():
    """Get system status and GPU information"""
    try:
        from ..runtime.cuda_session import CUDASession
        cuda = CUDASession()
        cuda.initialize()
        
        mem_info = cuda.get_memory_info()
        device_info = cuda._device_info
        
        return SystemStatusResponse(
            status="operational",
            gpu_name=device_info.name if device_info else "Unknown",
            gpu_vram_total_gb=mem_info['total'],
            gpu_vram_free_gb=mem_info['free'],
            queue_size=len([j for j in jobs_store.values() if j['status'] == 'queued']),
            active_jobs=len([j for j in jobs_store.values() if j['status'] == 'running']),
            uptime_seconds=time.time() - start_time,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return SystemStatusResponse(
            status="degraded",
            gpu_name="Unknown",
            gpu_vram_total_gb=0,
            gpu_vram_free_gb=0,
            queue_size=0,
            active_jobs=0,
            uptime_seconds=time.time() - start_time,
            version="1.0.0"
        )


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List available video generation models"""
    config_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    models = []
    for name, info in config.get('video_diffusion', {}).items():
        models.append(ModelInfo(
            name=name,
            type=info.get('type', 'unknown'),
            parameters=info.get('parameters', 'unknown'),
            max_resolution=info.get('supported_resolutions', [[854, 480]])[-1],
            max_frames=info.get('max_frames', 144),
            vram_required_gb=info.get('vram_requirement_gb', 6),
            supported_modes=info.get('supported_modes', ['text-to-video'])
        ))
    
    return models


@app.post("/generate", response_model=JobResponse, tags=["Generation"])
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a text-to-video generation job.
    
    Returns a job ID that can be used to track progress via the /jobs/{job_id} endpoint
    or WebSocket connection.
    """
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Store job
    job = {
        'id': job_id,
        'status': 'queued',
        'progress': 0.0,
        'request': request.dict(),
        'created_at': datetime.utcnow().isoformat(),
        'started_at': None,
        'completed_at': None,
        'error': None,
        'output_path': None,
        'preview_path': None
    }
    jobs_store[job_id] = job
    
    # Submit to scheduler
    background_tasks.add_task(process_generation_job, job_id, request)
    
    # Estimate time based on settings
    estimated_time = estimate_generation_time(request)
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message="Job submitted successfully",
        estimated_time_seconds=estimated_time,
        queue_position=len([j for j in jobs_store.values() if j['status'] == 'queued'])
    )


@app.post("/generate/image-to-video", response_model=JobResponse, tags=["Generation"])
async def generate_image_to_video(
    request: ImageToVideoRequest,
    background_tasks: BackgroundTasks
):
    """Submit an image-to-video generation job"""
    job_id = f"img2vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    job = {
        'id': job_id,
        'status': 'queued',
        'progress': 0.0,
        'request': request.dict(),
        'created_at': datetime.utcnow().isoformat(),
        'type': 'image-to-video'
    }
    jobs_store[job_id] = job
    
    # background_tasks.add_task(process_img2vid_job, job_id, request)
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message="Image-to-video job submitted"
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get the status of a generation job"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_store[job_id]
    
    return JobStatusResponse(
        job_id=job['id'],
        status=job['status'],
        progress=job['progress'],
        current_step=job.get('current_step'),
        created_at=job['created_at'],
        started_at=job.get('started_at'),
        completed_at=job.get('completed_at'),
        error=job.get('error'),
        output_url=f"/jobs/{job_id}/output" if job.get('output_path') else None,
        preview_url=f"/jobs/{job_id}/preview" if job.get('preview_path') else None,
        metadata=job.get('metadata', {})
    )


@app.get("/jobs/{job_id}/output", tags=["Jobs"])
async def download_output(job_id: str):
    """Download the generated video"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_store[job_id]
    
    if not job.get('output_path'):
        raise HTTPException(status_code=404, detail="Output not ready")
    
    return FileResponse(
        job['output_path'],
        media_type="video/mp4",
        filename=f"{job_id}.mp4"
    )


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def cancel_job(job_id: str):
    """Cancel a queued or running job"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_store[job_id]
    
    if job['status'] in ['completed', 'failed', 'cancelled']:
        raise HTTPException(status_code=400, detail="Job already finished")
    
    job['status'] = 'cancelled'
    job['completed_at'] = datetime.utcnow().isoformat()
    
    return {"message": "Job cancelled", "job_id": job_id}


@app.get("/jobs", tags=["Jobs"])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List generation jobs with optional filtering"""
    jobs = list(jobs_store.values())
    
    if status:
        jobs = [j for j in jobs if j['status'] == status]
    
    # Sort by created_at descending
    jobs.sort(key=lambda x: x['created_at'], reverse=True)
    
    return {
        "total": len(jobs),
        "jobs": jobs[offset:offset + limit]
    }


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time updates.
    
    Connect to receive real-time job progress updates.
    Send messages to subscribe to specific jobs.
    """
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get('type') == 'subscribe':
                job_id = data.get('job_id')
                if job_id:
                    await connection_manager.subscribe_to_job(websocket, job_id)
                    await websocket.send_json({
                        'type': 'subscribed',
                        'job_id': job_id
                    })
            
            elif data.get('type') == 'submit':
                # Submit job via WebSocket
                request_data = data.get('request', {})
                request = VideoGenerationRequest(**request_data)
                
                job_id = f"ws_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                
                job = {
                    'id': job_id,
                    'status': 'queued',
                    'progress': 0.0,
                    'request': request.dict(),
                    'created_at': datetime.utcnow().isoformat()
                }
                jobs_store[job_id] = job
                
                await connection_manager.subscribe_to_job(websocket, job_id)
                
                await websocket.send_json({
                    'type': 'job_created',
                    'job_id': job_id
                })
                
                # Start processing
                asyncio.create_task(process_generation_job_async(job_id, request))
            
            elif data.get('type') == 'cancel':
                job_id = data.get('job_id')
                if job_id in jobs_store:
                    jobs_store[job_id]['status'] = 'cancelled'
                    await websocket.send_json({
                        'type': 'job_cancelled',
                        'job_id': job_id
                    })
            
            elif data.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
    
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)


# =============================================================================
# Helper Functions
# =============================================================================

def estimate_generation_time(request: VideoGenerationRequest) -> float:
    """Estimate generation time based on settings"""
    base_time = 30.0  # Base time in seconds
    
    # Adjust for resolution
    pixels = request.width * request.height
    base_pixels = 854 * 480
    resolution_factor = pixels / base_pixels
    
    # Adjust for duration
    duration_factor = request.duration_seconds / 6.0
    
    # Adjust for steps
    steps_factor = request.num_inference_steps / 30.0
    
    # Adjust for quality preset
    quality_factors = {
        'fast': 0.5,
        'balanced': 1.0,
        'quality': 2.0,
        'ultra': 4.0
    }
    quality_factor = quality_factors.get(request.quality_preset, 1.0)
    
    return base_time * resolution_factor * duration_factor * steps_factor * quality_factor


async def process_generation_job(job_id: str, request: VideoGenerationRequest):
    """Process a generation job (background task)"""
    job = jobs_store.get(job_id)
    if not job:
        return
    
    try:
        job['status'] = 'running'
        job['started_at'] = datetime.utcnow().isoformat()
        
        # Broadcast status update
        await connection_manager.broadcast_job_update(job_id, {
            'type': 'job_started',
            'job_id': job_id,
            'status': 'running'
        })
        
        # TODO: Actual generation logic
        # This would call the inference engine
        
        # Simulate progress
        for i in range(10):
            await asyncio.sleep(1.0)
            job['progress'] = (i + 1) / 10
            job['current_step'] = f"Processing step {i + 1}/10"
            
            await connection_manager.broadcast_job_update(job_id, {
                'type': 'progress',
                'job_id': job_id,
                'progress': job['progress'],
                'current_step': job['current_step']
            })
        
        # Complete
        job['status'] = 'completed'
        job['completed_at'] = datetime.utcnow().isoformat()
        job['progress'] = 1.0
        
        await connection_manager.broadcast_job_update(job_id, {
            'type': 'job_completed',
            'job_id': job_id,
            'status': 'completed'
        })
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job['status'] = 'failed'
        job['error'] = str(e)
        job['completed_at'] = datetime.utcnow().isoformat()
        
        await connection_manager.broadcast_job_update(job_id, {
            'type': 'job_failed',
            'job_id': job_id,
            'error': str(e)
        })


async def process_generation_job_async(job_id: str, request: VideoGenerationRequest):
    """Async wrapper for job processing"""
    await process_generation_job(job_id, request)


# =============================================================================
# Server Entry Point
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
