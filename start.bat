@echo off
title Chatterbox Turbo Local
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 py -3.11 -m venv .venv
 if errorlevel 1 (echo Python 3.11 required.^&pause^&exit /b 1)
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
REM Pinned requirements: this also repairs venvs that grabbed newer/broken
REM versions before (downgrades them back to the known-good set).
python -m pip install -r requirements.txt
if errorlevel 1 (
 echo.
 echo INSTALL FAILED. Try deleting the ".venv" folder and run start.bat again.
 pause
 exit /b 1
)
start "" "http://127.0.0.1:7860"
python app.py
pause
