"""校准芯片 API：下载芯片、提交掰断结果、后台拟合、档案管理。"""
from __future__ import annotations

import secrets
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from ..calibration.chip import generate_chip
from ..calibration.fit import fit_from_outcomes
from ..profiles import list_profiles, save_profile
from ..thermal.materials import get_material

router = APIRouter(prefix="/api/calibration")

# 拟合任务（内存态）
FITS: dict[str, dict] = {}
_LOCK = threading.Lock()


class FitRequest(BaseModel):
    material: str = "PLA"
    chamber_temp: float | None = None
    section_layers: int = 30
    outcomes: dict[str, str]  # {"1": "strong", "2": "weak", ...}（段号 1 基）


@router.get("/chip")
def get_chip(material: str = "PLA"):
    text, manifest = generate_chip(material)
    return {"gcode": text, "manifest": manifest}


@router.get("/chip/download")
def download_chip(material: str = "PLA"):
    text, manifest = generate_chip(material)
    from fastapi.responses import Response

    return Response(
        content=text.encode("utf-8"),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="cal_chip_{material}.gcode"'},
    )


@router.post("/fit")
def start_fit(req: FitRequest):
    outcomes = {}
    for k, v in req.outcomes.items():
        try:
            outcomes[int(k)] = v
        except ValueError:
            raise HTTPException(400, f"段号必须是数字: {k}") from None
        if v not in ("weak", "strong"):
            raise HTTPException(400, f"结果只能是 weak/strong: {k}={v}")
    if len(outcomes) < 3:
        raise HTTPException(400, "至少报告 3 段结果才有拟合意义")

    fit_id = secrets.token_hex(6)
    with _LOCK:
        FITS[fit_id] = {"status": "fitting", "progress": 0.0, "report": None,
                        "material": req.material, "outcomes": req.outcomes}

    def worker():
        fit = FITS[fit_id]
        try:
            def prog(p):
                fit["progress"] = min(p, 0.99)
            report = fit_from_outcomes(
                outcomes, get_material(req.material),
                section_layers=req.section_layers,
                chamber_temp=req.chamber_temp,
                progress_cb=prog,
            )
            if report.get("ok"):
                profile = {
                    "material": req.material,
                    "iface_reheat": report["kappa"],
                    "nozzle_heat": report["eta"],
                    "hfan": report["hfan"],
                    "cold_below": report["cold_below"],
                    "ideal_lo": report["ideal_lo"],
                    "bond_threshold": report["bond_threshold"],
                    "agreement": report["agreement"],
                    "section_iface": report["section_iface"],
                    "source": "calibration-chip",
                }
                save_profile(profile)
                report["profile_saved"] = True
            fit["report"] = report
            fit["progress"] = 1.0
            fit["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            fit["status"] = "error"
            fit["error"] = str(exc)

    threading.Thread(target=worker, daemon=True).start()
    return {"fit_id": fit_id}


@router.get("/fit/{fit_id}")
def fit_status(fit_id: str):
    fit = FITS.get(fit_id)
    if not fit:
        raise HTTPException(404, "拟合任务不存在")
    return {"status": fit["status"], "progress": fit["progress"],
            "report": fit["report"], "error": fit.get("error")}


@router.get("/profiles")
def profiles():
    return list_profiles()


# ---------------------------------------------------------------------------
# 标准量化测试件（温度塔 / VFA 速度塔）
# ---------------------------------------------------------------------------
@router.get("/tower/download")
def download_tower(material: str = "PLA"):
    from ..calibration.standard import generate_temp_tower

    text, manifest = generate_temp_tower(material)
    return Response(
        content=text.encode("utf-8"), media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="temp_tower_{material}.gcode"'},
    )


@router.get("/vfa/download")
def download_vfa(material: str = "PLA"):
    from ..calibration.standard import generate_vfa

    text, manifest = generate_vfa(material)
    return Response(
        content=text.encode("utf-8"), media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="vfa_{material}.gcode"'},
    )


class TowerFitRequest(BaseModel):
    material: str = "PLA"
    outcomes: dict[str, str]  # {"1": "weak", "2": "ok", "3": "hot", ...}（块号 1 基）


@router.post("/tower/fit")
def start_tower_fit(req: TowerFitRequest):
    from ..calibration.standard_fit import fit_temp_tower

    outcomes = {}
    for k, v in req.outcomes.items():
        try:
            bi = int(k)
        except ValueError:
            raise HTTPException(400, f"块号必须是数字: {k}") from None
        if v not in ("weak", "ok", "hot"):
            raise HTTPException(400, f"块评价只能是 weak/ok/hot: {k}={v}")
        outcomes[bi] = v
    if len(outcomes) < 3:
        raise HTTPException(400, "至少报告 3 块结果")
    material = get_material(req.material)

    fit_id = secrets.token_hex(6)
    with _LOCK:
        FITS[fit_id] = {"status": "fitting", "progress": 0.0, "report": None,
                        "material": req.material}

    def worker():
        fit = FITS[fit_id]
        try:
            report = fit_temp_tower(outcomes, material,
                                    progress_cb=lambda p: setattr(fit, "progress", min(p, 0.99)))
            if report.get("ok"):
                save_profile({
                    "material": req.material,
                    "iface_reheat": report["kappa"],
                    "nozzle_heat": report["eta"],
                    "cold_below": report["cold_below"],
                    "ideal_lo": report["ideal_lo"],
                    "ideal_hi": report["ideal_hi"],
                    "hot_above": report["hot_above"],
                    "bond_threshold": report["t_bond"],
                    "agreement": report["agreement"],
                    "section_iface": report["block_iface"],
                    "source": "temp-tower",
                })
                report["profile_saved"] = True
            fit["report"] = report
            fit["progress"] = 1.0
            fit["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            fit["status"] = "error"
            fit["error"] = str(exc)

    threading.Thread(target=worker, daemon=True).start()
    return {"fit_id": fit_id}


class VfaFitRequest(BaseModel):
    material: str = "PLA"
    onset_band: int  # 起始瑕疵档（1 基）


@router.post("/vfa/fit")
def vfa_fit(req: VfaFitRequest):
    from ..calibration.standard import generate_vfa
    from ..calibration.standard_fit import fit_vfa

    _, manifest = generate_vfa(req.material)
    if not 1 <= req.onset_band <= len(manifest["bands"]):
        raise HTTPException(400, f"档位超出范围 1~{len(manifest['bands'])}")
    report = fit_vfa(req.onset_band, get_material(req.material), manifest)
    profile = {
        "material": req.material,
        "flow_ref": report["flow_ref"],
        "flow_span": report["flow_span"],
        "max_flow_mm3s": report["onset_flow"],
        "source": "vfa",
    }
    save_profile(profile)
    report["profile_saved"] = True
    return report
