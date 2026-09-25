@echo off
REM Works out how video time maps to sensor time for every trace.
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m src.detect.sync_diagnose --all
pause
