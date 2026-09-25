@echo off
REM Full pipeline for one trace.
REM   run_full.bat              -> DEFAULT_TRACE from .env, left side
REM   run_full.bat "PVS 5"      -> that trace, left side
REM   run_full.bat "PVS 5" right
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
    echo venv not found. Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

set SIDE=left
if not "%~2"=="" set SIDE=%~2

if "%~1"=="" (
    echo Running full pipeline, default trace, side: %SIDE%
    python -m src.pipeline.run --side %SIDE%
) else (
    echo Running full pipeline, trace: %~1, side: %SIDE%
    python -m src.pipeline.run --side %SIDE% --trace "%~1"
)
pause
