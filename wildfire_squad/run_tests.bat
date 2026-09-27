@echo off
rem All unit, integration and regression tests.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m pytest %*
