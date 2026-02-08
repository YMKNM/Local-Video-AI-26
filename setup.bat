@echo off
REM Video AI - Bootstrap Setup Script for Windows
REM This script sets up the complete environment for Video AI

echo.
echo ================================================================
echo                    VIDEO AI SETUP SCRIPT
echo           Local AI Video Generation for AMD GPUs
echo ================================================================
echo.

REM Check Python version
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or later from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo [1/6] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo [2/6] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [3/6] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo [4/6] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Create model directories
echo [5/6] Creating model directories...
if not exist "models\text_encoder\clip-vit-large" mkdir "models\text_encoder\clip-vit-large"
if not exist "models\video_diffusion\ltx-video-2b" mkdir "models\video_diffusion\ltx-video-2b"
if not exist "models\vae\ltx-vae" mkdir "models\vae\ltx-vae"
if not exist "models\schedulers" mkdir "models\schedulers"
if not exist "outputs" mkdir "outputs"
if not exist "logs" mkdir "logs"

REM Check FFmpeg
echo [6/6] Checking FFmpeg...
ffmpeg -version 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: FFmpeg not found!
    echo Please install FFmpeg:
    echo   1. Download from https://ffmpeg.org/download.html
    echo   2. Extract to C:\ffmpeg
    echo   3. Add C:\ffmpeg\bin to your PATH
    echo.
) else (
    echo FFmpeg found!
)

echo.
echo ================================================================
echo                     SETUP COMPLETE!
echo ================================================================
echo.
echo Next steps:
echo   1. Download models (see README.md for instructions)
echo   2. Verify setup: python generate.py --info
echo   3. Generate video: python generate.py --prompt "Your prompt"
echo.
echo To activate the environment in the future:
echo   venv\Scripts\activate.bat
echo.
pause
