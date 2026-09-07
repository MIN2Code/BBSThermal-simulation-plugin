@echo off
rem Thermal Assess 本地启动脚本 → http://127.0.0.1:8760/
cd /d %~dp0
.venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8760
