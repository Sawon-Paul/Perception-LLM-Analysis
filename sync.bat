@echo off
REM Finds the true video/sensor offset for every trace.
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m src.detect.sync_finder --all
pause
