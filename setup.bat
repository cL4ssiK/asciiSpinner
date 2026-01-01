@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM ===============================
REM Check if pyenv is installed
REM ===============================
where pyenv >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo pyenv not found! Install pyenv-win first:
    echo https://github.com/pyenv-win/pyenv-win
    exit /b 1
)

REM ===============================
REM Install Python 3.10.5 if missing
REM ===============================
CALL pyenv install -s 3.10.5
CALL pyenv local 3.10.5

REM ===============================
REM Remove old virtual environment
REM ===============================
IF EXIST .venv (
    echo Removing old virtual environment...
    rmdir /s /q .venv
)

REM ===============================
REM Create venv using pyenv Python
REM ===============================
SET PYTHON_PATH=%USERPROFILE%\.pyenv\pyenv-win\versions\3.10.5\python.exe
"%PYTHON_PATH%" -m venv .venv

REM ===============================
REM Activate venv
REM ===============================
CALL .venv\Scripts\activate.bat

REM ===============================
REM Upgrade pip and install requirements
REM ===============================
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Setup complete!
echo To activate the venv later, run:
echo .venv\Scripts\activate.bat
ENDLOCAL
