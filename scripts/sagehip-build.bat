@echo off
setlocal EnableDelayedExpansion
:: ---------------------------------------------------------------------------
::  scripts\sagehip-build.bat -- build this checkout for one AMD arch and install
::  it into the Python you point it at (e.g. ComfyUI's python_embeded).
::
::  Run it from anywhere; the repo root is derived from this script's location.
::  Edit PY / GPU_ARCH below for your machine (defaults: RX 7800 XT, rocm313).
:: ---------------------------------------------------------------------------

set "REPO=%~dp0.."
set "PY=H:\rocm313\python_embeded\python.exe"
set "GPU_ARCH=gfx1101"
set "FULLCLEAN=0"

echo ========================================================
echo   SageAttention-HIP build
echo   source : %REPO%
echo   python : %PY%
echo   arch   : %GPU_ARCH%
echo ========================================================

if not exist "%PY%" (
    echo [ERROR] %PY% not found. Set PY at the top of this script.
    pause & exit /b 1
)
if not exist "%REPO%\setup.py" (
    echo [ERROR] %REPO% does not look like the repo root.
    pause & exit /b 1
)

cd /d "%REPO%"

set "GPU_ARCHS=%GPU_ARCH%"
set "PYTORCH_ROCM_ARCH=%GPU_ARCH%"
set "MAX_JOBS=%NUMBER_OF_PROCESSORS%"
set "CMAKE_BUILD_TYPE=Release"
set "CMAKE_GENERATOR=Ninja"

where ninja >nul 2>nul
if errorlevel 1 (
    echo       installing ninja...
    "%PY%" -m pip install -q ninja
)

if "%FULLCLEAN%"=="1" (
    echo [1/3] FULL CLEAN - removing build artifacts...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    for /d %%i in (*.egg-info) do rmdir /s /q "%%i"
) else (
    echo [1/3] Incremental build ^(build\ kept^)
)

echo.
echo [2/3] Compiling for %GPU_ARCH%...
"%PY%" -m pip install . --no-build-isolation --no-deps --force-reinstall
if errorlevel 1 (
    echo.
    echo [FAILED] compilation/installation failed.
    echo   - after a small change: set FULLCLEAN=1 and rerun
    echo   - "cl.exe not found": call vcvars64.bat in this shell first
    pause & exit /b 1
)

echo.
echo [3/3] Clearing import cache and verifying...
for /d /r "%PY%\..\Lib\site-packages\sageattention" %%D in (__pycache__) do (
    rmdir /s /q "%%D" 2>nul
)
"%PY%" -c "import sageattention, torch; from sageattention import gfx10x; print('  sageattention :', sageattention.__file__); print('  torch        :', torch.__version__); print('  device       :', torch.cuda.get_device_name(0)); print('  backend      :', gfx10x._BACKEND); print('  native ext ok:', gfx10x._get_native_ops() is not None)"

echo.
echo ========================================================
echo   DONE. Restart ComfyUI (or your process) to load the new .pyd.
echo ========================================================
pause
