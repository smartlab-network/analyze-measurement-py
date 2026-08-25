@echo off
setlocal

set SCRIPT_DIR=%~dp0
set FOC48_EXE=%SCRIPT_DIR%FOC48.exe

set MODE=%1
if "%MODE%"=="" set MODE=basler

if "%MODE%"=="basler" (
    "%FOC48_EXE%" --mode basler --duration 60 --fps 60 --output-dir "%SCRIPT_DIR%analyze_measurement"
) else if "%MODE%"=="avi" (
    if "%2"=="" (
        echo Usage: run_foc48.bat avi ^<path_to_avi^>
        pause
        exit /b 1
    )
    "%FOC48_EXE%" "%2" --mode avi --output-dir "%SCRIPT_DIR%analyze_measurement"
) else (
    echo Unknown mode: %MODE%
    pause
    exit /b 1
)

pause

endlocal