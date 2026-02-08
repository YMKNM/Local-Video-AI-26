@echo off
REM =============================================================================
REM Video AI Platform - Startup Script
REM =============================================================================
REM Starts the complete Video AI platform including:
REM - API Server (FastAPI)
REM - Web UI (Gradio)
REM - Background Services
REM =============================================================================

echo.
echo  ██╗   ██╗██╗██████╗ ███████╗ ██████╗      █████╗ ██╗
echo  ██║   ██║██║██╔══██╗██╔════╝██╔═══██╗    ██╔══██╗██║
echo  ██║   ██║██║██║  ██║█████╗  ██║   ██║    ███████║██║
echo  ╚██╗ ██╔╝██║██║  ██║██╔══╝  ██║   ██║    ██╔══██║██║
echo   ╚████╔╝ ██║██████╔╝███████╗╚██████╔╝    ██║  ██║██║
echo    ╚═══╝  ╚═╝╚═════╝ ╚══════╝ ╚═════╝     ╚═╝  ╚═╝╚═╝
echo.
echo  Enterprise AI Video Generation Platform
echo  ========================================
echo.

REM Check for Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.10+
    pause
    exit /b 1
)

REM Set working directory
cd /d "%~dp0"

REM Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARN] No virtual environment found. Using system Python.
    echo [HINT] Run: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
)

REM Create output directories
if not exist "outputs" mkdir outputs
if not exist "outputs\images" mkdir outputs\images
if not exist "outputs\videos" mkdir outputs\videos
if not exist "outputs\animations" mkdir outputs\animations
if not exist "outputs\aggressive" mkdir outputs\aggressive
if not exist "logs" mkdir logs
if not exist "cache" mkdir cache

REM Set environment variables
set PYTHONPATH=%CD%
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set HF_HOME=%CD%\cache\huggingface

REM ── Free ports before starting ──────────────────────────────────
echo [INFO] Clearing ports 7860 and 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7860 " ^| findstr "LISTENING"') do (
    echo [INFO] Killing process %%a on port 7860
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    echo [INFO] Killing process %%a on port 8000
    taskkill /F /PID %%a >nul 2>&1
)
echo [OK] Ports cleared
echo.

REM Parse command line arguments
set MODE=all
set API_PORT=8000
set UI_PORT=7860
set DEBUG=0
set SHARE=0

:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--api" set MODE=api
if /i "%~1"=="--ui" set MODE=ui
if /i "%~1"=="--all" set MODE=all
if /i "%~1"=="--api-port" set API_PORT=%~2& shift
if /i "%~1"=="--ui-port" set UI_PORT=%~2& shift
if /i "%~1"=="--debug" set DEBUG=1
if /i "%~1"=="--share" set SHARE=1
if /i "%~1"=="--help" goto show_help
shift
goto parse_args
:end_parse

REM Check CUDA availability
echo [INFO] Checking GPU availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Could not check CUDA. Make sure PyTorch is installed.
)
echo.

REM Start services based on mode
if "%MODE%"=="api" goto start_api
if "%MODE%"=="ui" goto start_ui
if "%MODE%"=="all" goto start_all
goto start_all

:start_api
echo [INFO] Starting API Server on port %API_PORT%...
echo.
if "%DEBUG%"=="1" (
    python -m uvicorn api.server:app --host 0.0.0.0 --port %API_PORT% --reload
) else (
    python -m uvicorn api.server:app --host 0.0.0.0 --port %API_PORT%
)
goto end

:start_ui
echo [INFO] Starting Web UI on port %UI_PORT%...
echo.
if "%SHARE%"=="1" (
    python run_ui.py --port %UI_PORT% --share
) else (
    python run_ui.py --port %UI_PORT%
)
goto end

:start_all
echo [INFO] Starting Full Platform...
echo [INFO] API Server: http://localhost:%API_PORT%
echo [INFO] Web UI: http://localhost:%UI_PORT%
echo.

REM Start API server in background
start "Video AI - API Server" cmd /c "python -m uvicorn api.server:app --host 0.0.0.0 --port %API_PORT%"

REM Wait for API to start
timeout /t 3 /nobreak >nul

REM Start UI in foreground
echo [INFO] Starting Web UI...
if "%SHARE%"=="1" (
    python run_ui.py --port %UI_PORT% --share
) else (
    python run_ui.py --port %UI_PORT%
)
goto end

:show_help
echo.
echo Usage: run.bat [OPTIONS]
echo.
echo Options:
echo   --api          Start only the API server
echo   --ui           Start only the Web UI
echo   --all          Start both API and UI (default)
echo   --api-port N   Set API server port (default: 8000)
echo   --ui-port N    Set Web UI port (default: 7860)
echo   --debug        Enable debug mode with auto-reload
echo   --share        Create public Gradio link
echo   --help         Show this help message
echo.
echo Examples:
echo   run.bat                      # Start everything
echo   run.bat --ui --share         # Start UI with public link
echo   run.bat --api --debug        # Start API in debug mode
echo   run.bat --api-port 9000      # Start API on port 9000
echo.
goto end

:end
echo.
echo [INFO] Video AI Platform stopped.
pause
