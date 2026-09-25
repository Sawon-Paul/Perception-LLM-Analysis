@echo off
REM Preflight check. Double-click any time to see what is missing.
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
    echo venv not found. Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python -m src.utils.check_setup
pause
