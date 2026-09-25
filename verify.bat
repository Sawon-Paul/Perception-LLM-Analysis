@echo off
cd /d "%~dp0"
findstr /M /C:"crops -> " src\context\builder.py >nul 2>&1
if errorlevel 1 (echo   [OLD] builder.py - copy failed) else (echo   [OK]  builder.py)
pause
