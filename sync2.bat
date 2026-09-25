@echo off
REM Optical-flow sync. Slower than the first attempt - a few minutes per trace.
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m src.detect.sync_flow --all
pause
