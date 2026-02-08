#!/usr/bin/env pwsh
# ============================================================================
# Video AI - Complete Setup and Run Script (PowerShell)
# ============================================================================
# This script will:
# 1. Create a virtual environment (if needed)
# 2. Install all requirements
# 3. Launch the application
# ============================================================================

$ErrorActionPreference = "Stop"

# Change to script directory
Set-Location $PSScriptRoot

Write-Host ""
Write-Host " =============================================" -ForegroundColor Cyan
Write-Host "  Video AI - Auto Setup and Launch" -ForegroundColor Cyan
Write-Host " =============================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Step 1: Check for Python
# ============================================================================
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion found" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.11 or higher." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# ============================================================================
# Step 2: Create Virtual Environment
# ============================================================================
Write-Host "[2/4] Setting up virtual environment..." -ForegroundColor Yellow

if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "[OK] Virtual environment already exists" -ForegroundColor Green
} else {
    Write-Host "[INFO] Creating new virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[OK] Virtual environment created" -ForegroundColor Green
}

Write-Host ""

# ============================================================================
# Step 3: Activate and Install Requirements
# ============================================================================
Write-Host "[3/4] Installing requirements..." -ForegroundColor Yellow

# Activate virtual environment
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "[INFO] Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

# Install PyTorch with CUDA
Write-Host "[INFO] Installing PyTorch with CUDA 11.8..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] PyTorch installation had issues, continuing..." -ForegroundColor Yellow
}

# Install remaining requirements
Write-Host "[INFO] Installing remaining packages (this may take a few minutes)..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Some packages may have failed to install" -ForegroundColor Yellow
    Write-Host "[INFO] Continuing with available packages..." -ForegroundColor Cyan
}

Write-Host "[OK] Requirements installed" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Step 4: Launch Application
# ============================================================================
Write-Host "[4/4] Launching Video AI Platform..." -ForegroundColor Yellow
Write-Host ""
Write-Host " =============================================" -ForegroundColor Cyan
Write-Host "  Starting in 3 seconds..." -ForegroundColor Cyan
Write-Host " =============================================" -ForegroundColor Cyan
Write-Host ""

Start-Sleep -Seconds 3

# Check for command line arguments
$runArgs = if ($args.Count -gt 0) { $args } else { @("--all") }

# Run the application
& ".\run.bat" @runArgs

# Deactivate when done
deactivate

Write-Host ""
Write-Host " =============================================" -ForegroundColor Cyan
Write-Host "  Video AI Platform Stopped" -ForegroundColor Cyan
Write-Host " =============================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
