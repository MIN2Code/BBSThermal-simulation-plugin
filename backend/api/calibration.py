"""校准芯片 API：下载芯片、提交掰断结果、后台拟合、档案管理。"""
from __future__ import annotations

import secrets
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from ..calibration.chip import generate_chip
from ..calibration.detect import detect_nozzle_blocks, detect_speed_bands
from ..calibration.fit import fit_from_outcomes
from ..calibration.standard_fit import fit_temp_tower_gcode, fit_vfa_onset
from ..profiles import list_profiles, save_profile
from ..thermal.materials import get_material
from ..thermal.voxel import SimConfig, ThermalSimulator

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
# ---------------------------------------------------------------------------
# 标准量化测试件（基于已上传任务：自动识别温度块/速度带）
# ---------------------------------------------------------------------------
def _resolve_job(job_id: str):
    """统一解析任务：网页上传或 Bambu Studio 直连。返回 (parsed, sim_result)。"""
    from .. import pipeline
    from ..helio_api import emulator

    job = pipeline.JOBS.get(job_id)
    if job and job.parsed is not None:
        return job.parsed, job.result
    g = emulator.GCODES.get(job_id)
    if g and g.get("parsed") is not None:
        return g["parsed"], None
    return None, None


@router.get("/jobs")
def list_jobs():
    """列出两个来源的任务（网页上传 + Bambu Studio 直连）。"""
    from .. import pipeline
    from ..helio_api import emulator

    out = []
    for jid, job in pipeline.JOBS.items():
        if job.parsed is None:
            continue
        out.append({"job_id": jid, "name": job.name or jid, "source": "web",
                    "layers": job.parsed.num_layers,
                    "segments": job.parsed.num_segments,
                    "material": job.parsed.info.detected_material,
                    "sim_done": job.result is not None})
    for gid, g in emulator.GCODES.items():
        if g.get("parsed") is None:
            continue
        out.append({"job_id": gid, "name": g["name"], "source": "bambu",
                    "layers": g["parsed"].num_layers,
                    "segments": g["parsed"].num_segments,
                    "material": g["parsed"].info.detected_material,
                    "sim_done": False})
    out.sort(key=lambda x: x["job_id"])
    return out


@router.get("/structure/{job_id}")
def structure(job_id: str):
    """识别任务里的温度块 / 速度带结构。"""
    from ..calibration.detect import detect_nozzle_blocks, detect_speed_bands

    parsed, _ = _resolve_job(job_id)
    if parsed is None:
        raise HTTPException(404, "任务不存在")
    blocks = detect_nozzle_blocks(parsed)
    if len(blocks) >= 2:
        return {"kind": "temp_tower",
                "blocks": [{"index": i + 1, "temp": round(b["temp"], 1)} for i, b in enumerate(blocks)]}
    bands = detect_speed_bands(parsed)
    if len(bands) >= 2:
        return {"kind": "vfa",
                "bands": [{"band": b["band"], "speed": b["speed"], "flow_mm3s": b["flow_mm3s"]}
                          for b in bands]}
    return {"kind": "unknown"}


@router.post("/tower/fit")
def tower_fit(req: dict):
    """{job_id, outcomes: {块号: weak|ok|hot}} → 拟合结合窗口并写入档案。"""
    import threading

    from ..calibration.detect import detect_nozzle_blocks
    from ..calibration.standard_fit import fit_temp_tower_gcode
    from ..profiles import save_profile

    job_id = req.get("job_id", "")
    outcomes_raw = req.get("outcomes", {})
    parsed, result = _resolve_job(job_id)
    if parsed is None:
        raise HTTPException(404, "任务不存在")
    outcomes = {}
    for k, v in outcomes_raw.items():
        try:
            bi = int(k)
        except ValueError:
            raise HTTPException(400, f"块号必须是数字: {k}") from None
        if v not in ("weak", "ok", "hot"):
            raise HTTPException(400, f"块评价只能是 weak/ok/hot: {k}={v}")
        outcomes[bi] = v

    fit_id = secrets.token_hex(6)
    with _LOCK:
        FITS[fit_id] = {"status": "fitting", "progress": 0.0, "report": None,
                        "material": req.get("material", "PLA")}

    def worker():
        fit = FITS[fit_id]
        try:
            material = get_material(req.get("material", parsed.info.detected_material))
            if result is None:
                def prog(p):
                    fit["progress"] = min(p, 0.99)
                result = ThermalSimulator(parsed, material, SimConfig(voxel_mm=1.0)).run()
            report = fit_temp_tower_gcode(parsed, result, outcomes)
            if report.get("ok"):
                save_profile({
                    "material": material.name,
                    "iface_reheat": float(req.get("kappa", report.get("kappa", 0.65))),
                    "nozzle_heat": float(req.get("eta", report.get("eta", 0.35))),
                    "cold_below": report["cold_below"],
                    "ideal_lo": report["ideal_lo"],
                    "ideal_hi": report["ideal_hi"],
                    "hot_above": report["hot_above"],
                    "bond_threshold": report["t_bond"],
                    "agreement": report.get("agreement"),
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


@router.post("/vfa/fit")
def vfa_fit(req: dict):
    """{job_id, onset_speed} → 流量悬崖参数并写入档案。"""
    from ..calibration.detect import detect_speed_bands
    from ..calibration.standard_fit import fit_vfa_onset
    from ..profiles import save_profile

    job_id = req.get("job_id", "")
    onset_speed = float(req.get("onset_speed") or 0)
    parsed, _ = _resolve_job(job_id)
    if parsed is None:
        raise HTTPException(404, "任务不存在")
    report = fit_vfa_onset(parsed, onset_speed)
    if report.get("ok"):
        save_profile({"material": req.get("material", parsed.info.detected_material),
                      "flow_ref": report["flow_ref"], "flow_span": report["flow_span"],
                      "max_flow_mm3s": report["onset_flow"], "source": "vfa"})
        report["profile_saved"] = True
    return report


@router.get("/profiles")
def profiles():
    return list_profiles()
