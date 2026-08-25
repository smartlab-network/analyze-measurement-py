@echo off
setlocal

set PYTHON_MIN_MAJOR=3
set PYTHON_MIN_MINOR=10

REM --- Check Python is available and meets the minimum version ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VERSION=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VERSION%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

if %PY_MAJOR% LSS %PYTHON_MIN_MAJOR% (
    echo ERROR: Python %PY_VERSION% found, but %PYTHON_MIN_MAJOR%.%PYTHON_MIN_MINOR%+ is required.
    exit /b 1
)
if %PY_MAJOR% EQU %PYTHON_MIN_MAJOR% if %PY_MINOR% LSS %PYTHON_MIN_MINOR% (
    echo ERROR: Python %PY_VERSION% found, but %PYTHON_MIN_MAJOR%.%PYTHON_MIN_MINOR%+ is required.
    exit /b 1
)

REM --- Ensure PyInstaller is installed ---
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller.
        exit /b 1
    )
)

REM --- Clean previous build artifacts ---
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "FOC48.spec" del "FOC48.spec"

pyinstaller ^
    --name FOC48 ^
    --onedir ^
    --console ^
    --add-data "config.yaml;." ^
    --hidden-import=processing ^
    --hidden-import=calibration ^
    --hidden-import=stream_source ^
    --collect-all pypylon ^
    main.py

if errorlevel 1 (
    echo Build FAILED. See errors above.
    exit /b 1
)

echo Build complete: dist\FOC48\FOC48.exe

endlocal
pause