@echo off
rem Live demo in the browser at http://localhost:8765
cd /d "%~dp0"
call .venv\Scripts\activate.bat
solara run app.py
