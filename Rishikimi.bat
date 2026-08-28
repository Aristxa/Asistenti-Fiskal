@echo off
REM Launcher for the benchmark review screen.
REM
REM Started from a shell it lives and dies with that shell; run this instead and
REM it stays up until the window is closed. Double-click, or run from anywhere.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore
echo Po hapet rishikimi i bazes se testimit...
echo Mbylle kete dritare per ta ndalur.
echo.
".venv\Scripts\python.exe" scripts\review_app.py
pause
