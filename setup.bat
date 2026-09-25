@echo off
REM ============================================================
REM  VERMA-FL-UL - Windows setup
REM  Double-click this file, or run:  setup.bat
REM ============================================================
setlocal

cd /d "%~dp0"
echo Project folder: %CD%
echo.

REM ---------- 1. Python 3.11 ----------
echo [1/6] Checking Python 3.11 ...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo   FAIL: Python 3.11 not found.
    echo   Install from python.org and tick "Add python.exe to PATH".
    goto :fail
)
py -3.11 --version
echo.

REM ---------- 2. ffmpeg ----------
echo [2/6] Checking ffmpeg ...
where ffprobe >nul 2>&1
if errorlevel 1 (
    echo   FAIL: ffprobe not on PATH.
    echo   Run: winget install Gyan.FFmpeg
    echo   Then CLOSE this window and open a new one.
    goto :fail
)
echo   ffprobe found.
echo.

REM ---------- 3. NVIDIA driver ----------
echo [3/6] Checking NVIDIA driver ...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   WARNING: nvidia-smi not found. Phase 2 will not run on GPU.
) else (
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
)
echo.

REM ---------- 4. venv ----------
echo [4/6] Creating virtual environment ...
if exist venv\Scripts\activate.bat (
    echo   venv already exists, reusing it.
) else (
    py -3.11 -m venv venv
    if errorlevel 1 goto :fail
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
echo   venv ready.
echo.

REM ---------- 5. torch FIRST ----------
echo [5/6] Installing torch for CUDA 12.8 (large download, be patient) ...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :fail

python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch',torch.__version__,'cuda',torch.version.cuda,'available',ok); sys.exit(0 if ok else 1)"
if errorlevel 1 (
    echo.
    echo   WARNING: CUDA not available.
    echo   Either the CPU build got installed, or your driver is older than 12.8.
    echo   Continuing anyway - CPU will work but slowly.
)
echo.

REM ---------- 6. everything else ----------
echo [6/6] Installing remaining requirements ...
pip install -r requirements.txt
if errorlevel 1 goto :fail

python -c "import torch; print('CUDA still available:', torch.cuda.is_available())"
echo.

REM ---------- .env ----------
if not exist .env (
    copy .env.example .env >nul
    echo Created .env from template.
)

echo ============================================================
echo  Setup complete.
echo.
echo  NEXT STEPS:
echo    1. notepad .env
echo       - put your DASHSCOPE_API_KEY
echo       - set PVS_DIR using FORWARD slashes, e.g.
echo         PVS_DIR=C:/Users/You/Downloads/archive/PVS 2
echo.
echo    2. Copy your models into weights\ as:
echo         yolo11s_pothole.pt
echo         yolo11s_sign.pt
echo.
echo    3. Run the smoke test:
echo         venv\Scripts\activate.bat
echo         python -m src.pipeline.run --side left --smoke
echo ============================================================
goto :end

:fail
echo.
echo Setup stopped. Fix the error above, then run setup.bat again.

:end
pause
endlocal
