@echo off
REM Tiny end-to-end run. Always do this before a full run.
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
    echo venv not found. Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python -m src.utils.check_setup
if errorlevel 1 (
    echo.
    echo Preflight failed. Fix the items above before running the pipeline.
    pause
    exit /b 1
)
echo.
echo Starting smoke run...
python -m src.pipeline.run --side left --smoke
pause
