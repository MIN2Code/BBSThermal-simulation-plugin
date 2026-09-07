"""Thermal Assess 本地服务入口。

启动：  .venv/Scripts/python -m uvicorn backend.app:app --port 8760
访问：  http://127.0.0.1:8760/
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.routes import router  # noqa: E402
from backend.api.calibration import router as cal_router  # noqa: E402
from backend.helio_api.emulator import router as helio_router  # noqa: E402

app = FastAPI(title="Thermal Assess — FDM 热仿真分析", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(helio_router)
app.include_router(cal_router)

_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@app.get("/")
def index():
    return FileResponse(os.path.join(_FRONTEND, "index.html"))


@app.get("/calibration")
def calibration_page():
    return FileResponse(os.path.join(_FRONTEND, "calibration.html"))


app.mount("/static", StaticFiles(directory=_FRONTEND), name="static")
