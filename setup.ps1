#!/usr/bin/env pwsh
# Video AI - PowerShell Setup Script for Windows
# This script sets up the complete environment for Video AI

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "                    VIDEO AI SETUP SCRIPT" -ForegroundColor Cyan
Write-Host "           Local AI Video Generation for AMD GPUs" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.10 or later from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host ""
Write-Host "[1/6] Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
Write-Host "Virtual environment ready" -ForegroundColor Green

# Activate virtual environment
Write-Host ""
Write-Host "[2/6] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "[3/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install requirements
Write-Host ""
Write-Host "[4/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Dependencies installed" -ForegroundColor Green

# Create model directories
Write-Host ""
Write-Host "[5/6] Creating directories..." -ForegroundColor Yellow
$directories = @(
    "models\text_encoder\clip-vit-large",
    "models\video_diffusion\ltx-video-2b",
    "models\video_diffusion\cogvideox-2b",
    "models\video_diffusion\hummingbird-0.9b",
    "models\vae\ltx-vae",
    "models\vae\sd-vae",
    "models\schedulers",
    "models\.cache",
    "outputs",
    "logs"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "Directories created" -ForegroundColor Green

# Check FFmpeg
Write-Host ""
Write-Host "[6/6] Checking FFmpeg..." -ForegroundColor Yellow
$ffmpegVersion = ffmpeg -version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "WARNING: FFmpeg not found!" -ForegroundColor Yellow
    Write-Host "Please install FFmpeg:" -ForegroundColor Yellow
    Write-Host "  1. Download from https://ffmpeg.org/download.html" -ForegroundColor White
    Write-Host "  2. Extract to C:\ffmpeg" -ForegroundColor White
    Write-Host "  3. Add C:\ffmpeg\bin to your PATH" -ForegroundColor White
    Write-Host ""
    Write-Host "Or install via Chocolatey: choco install ffmpeg" -ForegroundColor Gray
    Write-Host "Or install via Scoop: scoop install ffmpeg" -ForegroundColor Gray
} else {
    $ffmpegFirstLine = ($ffmpegVersion -split "`n")[0]
    Write-Host "FFmpeg found: $ffmpegFirstLine" -ForegroundColor Green
}

# Check DirectML
Write-Host ""
Write-Host "Checking DirectML support..." -ForegroundColor Yellow
$dmlCheck = python -c "import onnxruntime as ort; print('DmlExecutionProvider' in ort.get_available_providers())" 2>&1
if ($dmlCheck -eq "True") {
    Write-Host "DirectML is available!" -ForegroundColor Green
} else {
    Write-Host "DirectML not detected. Make sure onnxruntime-directml is installed." -ForegroundColor Yellow
}

# System info
Write-Host ""
Write-Host "Checking system..." -ForegroundColor Yellow
python generate.py --info

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "                     SETUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Download models (see README.md for instructions)" -ForegroundColor White
Write-Host "  2. Generate video:" -ForegroundColor White
Write-Host "     python generate.py --prompt ""A sunset over the ocean""" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment in the future:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
