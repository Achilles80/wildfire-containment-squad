@echo off
rem Live demo in the browser at http://localhost:8765
cd /d "%~dp0"
call .venv\Scripts\activate.bat
rem Make sure the demo's browser files are cached (downloads any that are missing).
python prepare_demo_assets.py
rem Serve those files locally even when started from a VS Code terminal (Solara turns this off there).
set SOLARA_ASSETS_PROXY=true
solara run app.py
