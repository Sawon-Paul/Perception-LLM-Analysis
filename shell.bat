@echo off
REM Double-click to open a working shell with the venv active.
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo venv not found. Run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo ============================================================
echo  VERMA-FL-UL  -  %CD%
echo ============================================================
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 echo torch not installed yet - run setup.bat
echo.
echo  Common commands:
echo    python -m src.utils.check_setup                  preflight check
echo    python -m src.pipeline.run --side left --smoke   tiny test run
echo    python -m src.pipeline.run --side left           full run
echo    python -m src.eval.metrics --side left           results table
echo.

cmd /k
