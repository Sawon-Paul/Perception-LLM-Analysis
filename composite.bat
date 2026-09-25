@echo off
REM Compares the composite video against the raw video to recover the authors' alignment.
REM Slow - several minutes.
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m src.detect.sync_image --trace "PVS 2" --composite
pause
