@echo off
rem One-time setup on Windows: create .venv and install everything (Python 3.12+ required).
cd /d "%~dp0"
py -3 -m venv .venv || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
rem Download the demo's browser files now, so the demo also works offline later.
python prepare_demo_assets.py
echo.
echo Setup complete. Next: run_demo.bat, run_tests.bat or run_experiments.bat
