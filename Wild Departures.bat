@echo off
setlocal
title Wild Departures
rem Ammonix "Wild Departures": the animal airport on the 1.75-bit ternary model, live on this GPU.
rem   Double-click               live airport (first start: creates .venv, downloads about 7 GB)
rem   Wild Departures --replay   replay of the measured laptop run, no GPU needed
rem Needs Python 3.10 or newer on this computer (python.org) and an NVIDIA GPU with 8 GB or more.
set "DEMO=%~dp0"
set "DEMO=%DEMO:~0,-1%"
set "URL=http://127.0.0.1:8767/"
cd /d "%DEMO%"

netstat -an | find ":8767 " | find "LISTENING" >nul
if not errorlevel 1 (
    echo Wild Departures is already running. Opening %URL%
    start "" %URL%
    exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python environment of Wild Departures ...
    set "BASE="
    if defined AMMONIX_PYTHON set "BASE=%AMMONIX_PYTHON%"
    if not defined BASE for %%P in (py.exe) do if not "%%~$PATH:P"=="" set "BASE=py -3"
    if not defined BASE for %%P in (python.exe) do if not "%%~$PATH:P"=="" set "BASE=python"
    if not defined BASE (
        echo Python 3 was not found. Install it from https://www.python.org/downloads/ and run this again.
        pause
        exit /b 1
    )
    %BASE% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
    ".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r requirements.txt || (echo pip install failed & pause & exit /b 1)
)

if /i "%~1"=="--replay" (
    echo Starting the replay of the measured run ...
    ".venv\Scripts\python.exe" serve_ternary.py
    goto end
)

set "HAVE_MODEL="
if exist "models\Ternary-Bonsai-2-27B-PTQ1_0.gguf" set "HAVE_MODEL=1"
if exist "..\runtime\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf" set "HAVE_MODEL=1"
if defined AMMONIX_TERNARY_MODEL if exist "%AMMONIX_TERNARY_MODEL%" set "HAVE_MODEL=1"
if not defined HAVE_MODEL (
    echo The ternary model is not here yet. Downloading about 7 GB, this takes a while ...
    ".venv\Scripts\python.exe" setup.py || (echo Download failed; run "python setup.py" again to resume. & pause & exit /b 1)
)

echo Starting Wild Departures live. Loading the model takes about half a minute ...
echo The airport opens at %URL%   Close this window to stop it.
".venv\Scripts\python.exe" serve_ternary.py --live

:end
if errorlevel 1 (
    echo.
    echo Wild Departures could not start. The error is shown above.
    pause
    exit /b 1
)
endlocal
