@echo off
cd /d "%~dp0"
python -m pip install -e .
if errorlevel 1 exit /b 1
python -m demandcast.cli serve --open
