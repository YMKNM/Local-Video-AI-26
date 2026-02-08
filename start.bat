@echo off
REM ============================================================================
REM Video AI - Complete Setup and Run Script
REM ============================================================================
REM This script will:
REM 1. Create a virtual environment (if needed)
REM 2. Install all requirements
REM 3. Launch the application
REM ============================================================================

setlocal enabledelayedexpansion

REM Change to script directory
cd /d "%~dp0"

echo.
echo  =============================================
echo   Video AI - Auto Setup and Launch
echo  =============================================
echo.

REM ============================================================================
REM Step 1: Check for Python
REM ============================================================================
echo [1/4] Checking Python installation...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.11 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found
echo.

REM ============================================================================
REM Step 2: Create Virtual Environment
REM ============================================================================
echo [2/4] Setting up virtual environment...

if exist "venv\Scripts\activate.bat" (
    echo [OK] Virtual environment already exists
) else (
    echo [INFO] Creating new virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM ============================================================================
REM Step 3: Activate and Install Requirements
REM ============================================================================
echo [3/4] Installing requirements...

call venv\Scripts\activate.bat

REM Upgrade pip first
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install PyTorch with CUDA
echo [INFO] Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
if %errorlevel% neq 0 (
    echo [WARN] PyTorch installation had issues, continuing...
)

REM Install remaining requirements
echo [INFO] Installing remaining packages (this may take a few minutes)...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARN] Some packages may have failed to install
    echo [INFO] Continuing with available packages...
)

echo [OK] Requirements installed
echo.

REM ============================================================================
REM Step 4: Launch Application
REM ============================================================================
echo [4/4] Launching Video AI Platform...
echo.
echo  =============================================
echo   Starting in 3 seconds...
echo  =============================================
echo.

timeout /t 3 /nobreak >nul

REM Check for command line arguments
set "ARGS=--all"
if not "%~1"=="" set "ARGS=%*"

REM Run the application
call run.bat %ARGS%

REM Deactivate when done
deactivate

echo.
echo  =============================================
echo   Video AI Platform Stopped
echo  =============================================
echo.
pause
