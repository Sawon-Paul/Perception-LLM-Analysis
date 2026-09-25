@echo off
REM Runs every complete PVS trace one after another.
REM Do this only AFTER a single trace has run clean end to end.
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
    echo venv not found. Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

for %%T in ("PVS 1" "PVS 2" "PVS 3" "PVS 4" "PVS 5" "PVS 6" "PVS 7" "PVS 8" "PVS 9") do (
    echo.
    echo ============================================================
    echo  TRACE %%T
    echo ============================================================
    python -m src.pipeline.run --side left --trace %%T
    if errorlevel 1 echo    %%T FAILED - continuing with the next trace
)
echo.
echo All traces attempted.
pause
