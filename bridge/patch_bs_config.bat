@echo off
rem Patch Bambu Studio config: point Helio integration to the local engine.
rem IMPORTANT: close Bambu Studio completely BEFORE running this.
set "PY=python"
where python >nul 2>nul
if errorlevel 1 set "PY=%~dp0..\.venv\Scripts\python.exe"
"%PY%" "%~dp0patch_bs_config.py"
pause
