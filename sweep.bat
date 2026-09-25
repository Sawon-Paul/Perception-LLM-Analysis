@echo off
REM Visual sync sweep using surface labels. Orange = unpaved, blue = asphalt.
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m src.detect.sync_sweep2 --trace "PVS 2"
echo.
start "" "data\sync_sweep\PVS2\sweep.html"
pause
