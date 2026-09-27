@echo off
rem Full study: 6 scenarios x 3 strategies x 30 seeds, then the supporting studies (~25 min).
rem Pass --quick for a 30-second check of the main experiment only.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if "%1"=="--quick" (
  python run_experiments.py --quick
) else (
  python run_experiments.py --scenarios all --runs 30 && python run_studies.py
)
