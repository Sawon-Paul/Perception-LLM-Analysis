@echo off
REM Builds ONE self-contained timeline.jpg you can upload.
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m src.detect.sync_image --trace "PVS 2"
echo.
echo Upload this file:
echo   C:\Sawon\LLM\data\sync_timeline\PVS2\timeline.jpg
echo.
start "" "data\sync_timeline\PVS2"
pause
