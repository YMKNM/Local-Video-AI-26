# =============================================================================
# Video AI Platform - Production Dockerfile
# =============================================================================
# Multi-stage build optimized for NVIDIA CUDA with TensorRT support
# Target: RTX 3080 (Ampere, Compute Capability 8.6)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base CUDA Runtime
# -----------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS cuda-base

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# -----------------------------------------------------------------------------
# Stage 2: TensorRT Installation
# -----------------------------------------------------------------------------
FROM cuda-base AS tensorrt-base

# Install TensorRT 8.6
ARG TENSORRT_VERSION=8.6.1.6
RUN pip install --upgrade pip setuptools wheel && \
    pip install tensorrt==${TENSORRT_VERSION}

# Install PyTorch with CUDA 11.8 support
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# -----------------------------------------------------------------------------
# Stage 3: Dependencies
# -----------------------------------------------------------------------------
FROM tensorrt-base AS dependencies

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    onnxruntime-gpu==1.16.0 \
    diffusers>=0.24.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    safetensors>=0.4.0 \
    huggingface-hub>=0.19.0 \
    xformers==0.0.22.post7 \
    triton>=2.1.0

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install additional enterprise dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    websockets>=12.0 \
    pydantic>=2.5.0 \
    python-multipart>=0.0.6 \
    aiofiles>=23.2.0 \
    prometheus-client>=0.19.0 \
    opentelemetry-api>=1.21.0 \
    opentelemetry-sdk>=1.21.0 \
    redis>=5.0.0 \
    celery>=5.3.0 \
    boto3>=1.33.0

# -----------------------------------------------------------------------------
# Stage 4: Application
# -----------------------------------------------------------------------------
FROM dependencies AS application

WORKDIR /app

# Create non-root user for security
RUN groupadd -r videoai && useradd -r -g videoai videoai

# Copy application code
COPY --chown=videoai:videoai . .

# Create necessary directories
RUN mkdir -p /app/outputs /app/cache /app/logs /app/models \
    && chown -R videoai:videoai /app

# Set permissions
RUN chmod +x /app/setup.py 2>/dev/null || true

# Switch to non-root user
USER videoai

# Expose ports
EXPOSE 8000 8001 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 5: Development
# -----------------------------------------------------------------------------
FROM application AS development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.4.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.1.0 \
    black>=23.0.0 \
    isort>=5.12.0 \
    mypy>=1.7.0 \
    ipython>=8.18.0

USER videoai

# Development command with auto-reload
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# -----------------------------------------------------------------------------
# Stage 6: Production (Optimized)
# -----------------------------------------------------------------------------
FROM application AS production

# Production environment
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Pre-compile Python files
RUN python -m compileall /app

# Production command with workers
CMD ["python", "-m", "uvicorn", "api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--limit-concurrency", "100", \
     "--timeout-keep-alive", "30"]
