"""Helio API 本地仿真层：实现 HelioDragon.cpp 所需的 GraphQL 协议子集，
后端挂本地引擎（解析器 / ThermalSimulator / optimize_speeds）。

协议要点（逆向自 Helio-Additive/BambuStudio 分叉的 src/slic3r/Utils/HelioDragon.cpp
与 helio-api-cookbook）：
- POST {helio_api_url}，JSON {"query", "variables"}，Authorization: Bearer <PAT>
- 流程：getPresignedUrl → PUT 上传 → createGcodeV2 → 轮询 gcodeV2 至 READY →
  createSimulationV2 → 轮询 simulation 至 FINISHED → 下载 thermalIndexGcodeUrl
- 热指数 G-code：在原 G-code 每条挤出移动行尾追加
  ;helioadditive=(ti.max=X,ti.min=X,ti.mean=X,element.index=N)，ti ∈ [-1,1]
"""
from __future__ import annotations

import json
import re
import secrets
import threading
import time
import uuid

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from ..gcode.parser import parse_gcode
from ..thermal.materials import get_material
from ..thermal.optimize import OptimizeConfig, _retime, optimize_speeds, rewrite_gcode
from ..thermal.voxel import SimConfig, ThermalSimulator

router = APIRouter()

# ---------------------------------------------------------------------------
# 内存态
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
GCODES: dict[str, dict] = {}      # id -> {name, key, size_kb, status, progress, text, parsed}
SIMS: dict[str, dict] = {}        # id -> {name, status, progress, gcode_id, result, ti_gcode, report, cfg}
OPTS: dict[str, dict] = {}        # id -> {name, status, progress, gcode_id, result, opt_gcode, opt_meta}
UPLOADS: dict[str, bytes] = {}    # key -> bytes（presigned PUT 目标）
ASSETS: dict[str, tuple[bytes, str]] = {}  # asset_id -> (bytes, media_type)

PRINTERS = [
    # (id, name, heated_chamber, bambustudio 精确档案名)
    ("bambu-p2s", "Bambu Lab P2S", True, "Bambu Lab P2S"),
    ("bambu-x1c", "Bambu Lab X1 Carbon", True, "Bambu Lab X1 Carbon"),
    ("bambu-x1e", "Bambu Lab X1E", True, "Bambu Lab X1E"),
    ("bambu-p1s", "Bambu Lab P1S", True, "Bambu Lab P1S"),
    ("bambu-p1p", "Bambu Lab P1P", False, "Bambu Lab P1P"),
    ("bambu-a1", "Bambu Lab A1", False, "Bambu Lab A1"),
    ("bambu-a1mini", "Bambu Lab A1 mini", False, "Bambu Lab A1 mini"),
    ("bambu-h2d", "Bambu Lab H2D", True, "Bambu Lab H2D"),
]
MATERIALS = [
    # (id, name, bambustudio 档案类型名)
    ("mat-pla", "PLA", "PLA"),
    ("mat-petg", "PETG", "PETG"),
    ("mat-abs", "ABS", "ABS"),
    ("mat-asa", "ASA", "ASA"),
    ("mat-tpu", "TPU", "TPU"),
    ("mat-pc", "PC", "PC"),
    ("mat-pa", "PA", "PA"),
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# ti 标注生成
# ---------------------------------------------------------------------------
def annotate_gcode_with_ti(text: str, parsed, res) -> tuple[str, int]:
    """按 Helio 格式在每条挤出移动行尾追加热指数注释。返回 (新文本, 标注行数)。"""
    ti = np.clip(res.tqi.astype(np.float64) / 100.0, -1.0, 1.0)
    labels = res.element_labels if res.element_labels is not None else np.zeros(len(ti), dtype=np.int32)
    lines = text.splitlines()
    src = parsed.src_line
    count = 0
    # 段按行分组（段序即文件序）
    by_line: dict[int, list[int]] = {}
    for i in range(len(src)):
        by_line.setdefault(int(src[i]), []).append(i)
    for ln, segs in by_line.items():
        idx = ln - 1
        if idx < 0 or idx >= len(lines):
            continue
        line = lines[idx]
        if " E" not in line and " E\t" not in line:
            continue  # 只标注挤出移动
        ti_vals = ti[segs]
        m = float(np.mean(ti_vals))
        # 空段（无 ti 信息的辅助段）跳过
        if not np.isfinite(ti_vals).all():
            continue
        el = int(labels[segs[0]])
        lines[idx] = (line.rstrip() +
                      f" ;helioadditive=(ti.max={m:.2f},ti.min={m:.2f},ti.mean={m:.2f},element.index={el})")
        count += 1
    return "\n".join(lines), count


# ---------------------------------------------------------------------------
# 后台仿真 / 优化
# ---------------------------------------------------------------------------
def _run_sim_async(sim_id: str) -> None:
    sim = SIMS[sim_id]
    try:
        g = GCODES[sim["gcode_id"]]
        text = g["text"]
        parsed = g["parsed"]          # 复用 createGcodeV2 时的解析结果（省 5~10s）
        material = get_material(sim["material_name"])
        cfg = SimConfig(
            voxel_mm=float(sim["voxel_mm"]),
            chamber_temp=sim["chamber_temp"],
            iface_reheat=0.5, nozzle_heat=0.35,
        )
        def prog(p):
            sim["progress"] = int(min(p, 0.99) * 100)
        simulator = ThermalSimulator(parsed, material, cfg, progress_cb=prog)
        res = simulator.run()
        sim["result"] = res
        sim["parsed"] = parsed
        ti_text, _ = annotate_gcode_with_ti(text, parsed, res)
        asset_id = uuid.uuid4().hex
        ASSETS[asset_id] = (ti_text.encode("utf-8"), "text/plain")
        report = {
            "printOutcome": "SUCCESS",
            "printOutcomeDescription": "本地引擎仿真完成",
            "temperatureDirection": "COLD" if float(np.mean(res.tqi)) < -20 else (
                "HOT" if float(np.mean(res.tqi)) > 20 else "BALANCED"),
            "temperatureDirectionDescription": (
                f"平均 TQI {float(np.mean(res.tqi)):.1f}；偏冷段 {(res.tqi[res.tqi_valid] < -50).mean()*100:.0f}%，"
                f"偏热段 {(res.tqi[res.tqi_valid] > 50).mean()*100:.0f}%"),
            "caveats": [],
        }
        sim["report"] = report
        report_id = uuid.uuid4().hex
        ASSETS[report_id] = (json.dumps(report, ensure_ascii=False).encode("utf-8"), "application/json")
        sim["thermal_url"] = f"{sim['base']}/helio-assets/{asset_id}/thermal_index.gcode"
        sim["report_url"] = f"{sim['base']}/helio-assets/{report_id}/report.json"
        sim["layers"] = parsed.num_layers
        sim["slicer"] = parsed.info.slicer or "bambu"
        sim["progress"] = 100
        sim["status"] = "FINISHED"
    except Exception as exc:  # noqa: BLE001
        sim["status"] = "FAILED"
        sim["error"] = str(exc)


def _run_opt_async(opt_id: str) -> None:
    opt = OPTS[opt_id]
    try:
        g = GCODES[opt["gcode_id"]]
        text = g["text"]
        parsed = g["parsed"]          # 复用 createGcodeV2 的解析结果（省 5~10s）
        from dataclasses import replace

        from ..profiles import load_profile
        material = get_material(g.get("material_name", "PLA"))
        profile = load_profile(material.name)
        if profile:
            material = replace(material,
                               h_conv_on=material.h_conv_on * float(profile.get("hfan", 1.0)),
                               cold_below=float(profile.get("cold_below", material.cold_below)),
                               ideal_lo=float(profile.get("ideal_lo", material.ideal_lo)))
        # 大文件自动减轮次：保证在 BS 的 120s 轮询窗口内完成
        rounds = 2 if parsed.num_segments > 150_000 else 3
        cfg = SimConfig(
            voxel_mm=1.5, iface_reheat=0.5, nozzle_heat=0.35,
        )
        opt_cfg = OptimizeConfig(
            rounds=rounds,
            min_speed=float(opt.get("min_velocity") or 15.0),
            max_speed=float(opt.get("max_velocity") or 300.0),
            layers_from=opt.get("layers_from"),
            layers_to=opt.get("layers_to"),
            max_flow_mm3s=float(opt["max_flow"]) if opt.get("max_flow") else None,
            priority=opt.get("priority", "speed_strength"),
        )
        def prog(p):
            opt["progress"] = int(min(p, 0.99) * 100)
        result = optimize_speeds(parsed, material, cfg, opt_cfg, progress_cb=prog)
        opt["result"] = result
        # optimize_speeds 结束时会把 parsed 时间轴恢复为原速——
        # ti 标注必须反映优化后的速度，先按新速度重计时
        _retime(parsed, result.new_feed)
        opt_text, _ = rewrite_gcode(text, parsed, result.new_feed)
        # 优化结果同样附带 ti 标注（optimizedGcodeWithThermalIndexes）
        final_sim = ThermalSimulator(parsed, material, cfg)
        final_res = final_sim.run()
        ti_text, _ = annotate_gcode_with_ti(opt_text, parsed, final_res)
        asset_id = uuid.uuid4().hex
        ASSETS[asset_id] = (ti_text.encode("utf-8"), "text/plain")
        opt["opt_url"] = f"{opt['base']}/helio-assets/{asset_id}/optimized_thermal_index.gcode"
        b, f = result.baseline, result.final
        opt["quality_mean_improvement"] = (f["mean_tqi"] - b["mean_tqi"]) if (
            b.get("mean_tqi") is not None and f.get("mean_tqi") is not None) else 0.0
        opt["progress"] = 100
        opt["status"] = "FINISHED"
    except Exception as exc:  # noqa: BLE001
        opt["status"] = "FAILED"
        opt["error"] = str(exc)


# ---------------------------------------------------------------------------
# GraphQL 分发
# ---------------------------------------------------------------------------
def _g(msg: dict) -> JSONResponse:
    return JSONResponse(msg)


@router.post("/graphql/helio")
async def graphql(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return _g({"errors": [{"message": "invalid JSON"}]})
    query = payload.get("query", "")
    variables = payload.get("variables") or {}
    # C++ Http 客户端需要绝对 URL
    base = str(request.base_url).rstrip("/")

    if "getPresignedUrl" in query:
        upload_id = uuid.uuid4().hex
        key = f"{upload_id}/{variables.get('fileName', 'upload.gcode')}"
        UPLOADS[upload_id] = b""
        return _g({"data": {"getPresignedUrl": {
            "mimeType": "application/octet-stream",
            "url": f"{base}/helio-upload/{upload_id}",
            "key": key,
        }}})

    if "createGcodeV2" in query:
        inp = variables.get("input", {})
        key = inp.get("gcodeKey", "")
        upload_id = key.split("/")[0] if "/" in key else key
        data = UPLOADS.get(upload_id, b"")
        try:
            parsed = parse_gcode(data.decode("utf-8-sig", errors="replace"))
            status, progress = "READY", 100
        except Exception:
            parsed, status, progress = None, "ERROR", 0
        gid = uuid.uuid4().hex
        mat_name = next((m_name for mid, m_name, _ in MATERIALS
                         if mid == inp.get("materialId")), "PLA")
        GCODES[gid] = {
            "name": inp.get("name", "gcode"), "key": key, "size_kb": len(data) // 1024,
            "status": status, "progress": progress, "text": data.decode("utf-8-sig", errors="replace"),
            "parsed": parsed,
            "printer_id": inp.get("printerId", ""), "material_id": inp.get("materialId", ""),
            "material_name": mat_name,
        }
        return _g({"data": {"createGcodeV2": {
            "id": gid, "name": inp.get("name", "gcode"), "sizeKb": len(data) // 1024,
            "status": status, "progress": progress}}})

    if "gcodeV2" in query or 'GcodeV2($id' in query:
        gid = variables.get("id", "")
        g = GCODES.get(gid)
        if not g:
            return _g({"errors": [{"message": "gcode not found"}]})
        return _g({"data": {"gcodeV2": {
            "id": gid, "name": g["name"], "sizeKb": g["size_kb"],
            "status": g["status"], "progress": g["progress"],
            "errors": None, "errorsV2": None}}})

    if "createSimulation" in query:
        inp = variables.get("input", {})
        settings = inp.get("simulationSettings", {}) or {}
        gid = inp.get("gcodeId", "")
        g = GCODES.get(gid)
        if not g or g["parsed"] is None:
            return _g({"errors": [{"message": "gcode not ready"}]})
        sid = uuid.uuid4().hex
        chamber = settings.get("stabilizedAirTemperature") or settings.get("airTemperatureAboveBuildPlate")
        SIMS[sid] = {
            "name": inp.get("name", "sim"), "status": "PROCESSING", "progress": 0,
            "gcode_id": gid, "text": g["text"], "material_name": "PLA",
            "voxel_mm": 1.5, "chamber_temp": chamber, "base": base,
        }
        threading.Thread(target=_run_sim_async, args=(sid,), daemon=True).start()
        sim_obj = {
            "id": sid, "name": inp.get("name", "sim"), "progress": 0, "status": "PROCESSING",
            "gcode": {"id": gid, "name": g["name"]},
            "printer": {"id": g["printer_id"], "name": g["printer_id"]},
            "material": {"id": g["material_id"], "name": g["material_id"]},
            "reportJsonUrl": None, "thermalIndexGcodeUrl": None,
            "estimatedSimulationDurationSeconds": 60,
            "insertedAt": _now(), "updatedAt": _now()}
        # BS 分叉用 V1 变异名 createSimulation，cookbook 用 createSimulationV2——两者都给
        return _g({"data": {"createSimulation": sim_obj, "createSimulationV2": sim_obj}})

    if "simulation(" in query.replace(" ", "") or "query Simulation" in query:
        sid = variables.get("id", "")
        s = SIMS.get(sid)
        if not s:
            return _g({"errors": [{"message": "simulation not found"}]})
        data = {"id": sid, "name": s["name"], "progress": s["progress"], "status": s["status"]}
        if s["status"] == "FINISHED":
            data.update({
                "thermalIndexGcodeUrl": s["thermal_url"],
                "printInfo": s["report"],
                "speedFactor": 1.0,
                "suggestedFixes": [],
            })
        return _g({"data": {"simulation": data}})

    if "createOptimization" in query:
        inp = variables.get("input", {})
        gid = inp.get("gcodeId", "")
        g = GCODES.get(gid)
        if not g or g["parsed"] is None:
            return _g({"errors": [{"message": "gcode not ready"}]})
        oid = uuid.uuid4().hex
        layers = inp.get("layersToOptimize", {}) or {}
        OPTS[oid] = {
            "name": inp.get("name", "opt"), "status": "PROCESSING", "progress": 0,
            "gcode_id": gid, "min_velocity": inp.get("minVelocity"),
            "max_velocity": inp.get("maxVelocity"),
            "max_flow": inp.get("maxExtruderFlowRate"),
            "layers_from": layers.get("fromLayer"),
            "layers_to": layers.get("toLayer"),
            "priority": inp.get("printPriority", "speed_strength"),
            "base": base,
        }
        threading.Thread(target=_run_opt_async, args=(oid,), daemon=True).start()
        opt_obj = {
            "id": oid, "name": inp.get("name", "opt"), "progress": 0, "status": "PROCESSING",
            "gcode": {"id": gid, "name": g["name"]},
            "printer": {"id": g["printer_id"], "name": g["printer_id"]},
            "material": {"id": g["material_id"], "name": g["material_id"]},
            "insertedAt": _now(), "updatedAt": _now()}
        return _g({"data": {"createOptimization": opt_obj, "createOptimizationV2": opt_obj}})

    if "optimization(" in query.replace(" ", "") or "query Optimization" in query:
        oid = variables.get("id", "")
        o = OPTS.get(oid)
        if not o:
            return _g({"errors": [{"message": "optimization not found"}]})
        data = {"id": oid, "name": o["name"], "progress": o["progress"], "status": o["status"]}
        if o.get("error"):
            data["error"] = o["error"]
        if o["status"] == "FINISHED":
            data.update({
                "optimizedGcodeWithThermalIndexesUrl": o["opt_url"],
                # C++ CheckOptimizationResult 里这两个字段是 std::string——必须回字符串
                "qualityStdImprovement": "0.00",
                "qualityMeanImprovement": f"{float(o.get('quality_mean_improvement') or 0.0):.2f}",
            })
        return _g({"data": {"optimization": data}})

    if "GetPrinters" in query or "printers(" in query.replace(" ", ""):
        return _g({"data": {"printers": {
            "pages": 1, "pageInfo": {"hasNextPage": False},
            "objects": [{"id": pid, "name": name, "heatedChamber": heated,
                         "alternativeNames": {"bambustudio": bs_name}}
                        for pid, name, heated, bs_name in PRINTERS]}}})

    if "GetMaterials" in query or "materials(" in query.replace(" ", ""):
        return _g({"data": {"materials": {
            "pages": 1, "pageInfo": {"hasNextPage": False},
            "objects": [{"id": mid, "name": name, "feedstock": "FILAMENT",
                         "alternativeNames": {"bambustudio": bs_name}}
                        for mid, name, bs_name in MATERIALS]}}})

    if "printPriorityOptions" in query:
        return _g({"data": {"printPriorityOptions": [
            {"value": "speed_strength", "label": "Speed & Strength", "isAvailable": True,
             "description": "最大性能与层结合"},
            {"value": "surface", "label": "Preserve Surface Finish", "isAvailable": True,
             "description": "保留外墙壁速"},
        ]}})

    if "GetUserRemainingOpts" in query or "remainingOptsThisMonth" in query:
        return _g({"data": {
            "user": {"remainingOptsThisMonth": 9999, "addOnOptimizations": 0,
                     "isFreeTrialActive": True, "isFreeTrialClaimed": True,
                     "subscription": {"name": "local"}},
            "freeTrialEligibility": False}})

    if "defaultOptimizationSettings" in query:
        return _g({"data": {"defaultOptimizationSettings": {
            "minVelocity": 15.0, "maxVelocity": 300.0, "minVelocityIncrement": 5.0,
            "minExtruderFlowRate": 0.5, "maxExtruderFlowRate": 20.0,
            "tolerance": 5.0, "maxIterations": 5,
            "reductionStrategySettings": {"strategy": "AUTO", "autolinearDoCriticality": True,
                                          "autolinearDoFitness": True, "autolinearDoInterpolation": True,
                                          "autolinearCriticalityMaxNodesDensity": 0.5,
                                          "autolinearCriticalityThreshold": -30.0,
                                          "autolinearFitnessMaxNodesDensity": 0.5,
                                          "autolinearFitnessThreshold": 0.0,
                                          "autolinearInterpolationLevels": 3,
                                          "linearNodesLimit": 100},
            "residualStrategySettings": {"strategy": "AUTO", "exponentialPenaltyHigh": 1.0,
                                         "exponentialPenaltyLow": 0.1},
            "layersToOptimize": {"fromLayer": 0, "toLayer": 999999},
            "optimizer": "LOCAL"}}})

    if "GetRecentRuns" in query:
        sims = [{"id": sid, "name": s["name"], "status": s["status"],
                 "thermalIndexGcodeUrl": s.get("thermal_url"),
                 "qualityMeanImprovement": None, "qualityStdImprovement": None,
                 "gcode": {"gcodeUrl": None, "gcodeKey": s["gcode_id"],
                           "material": {"id": "", "name": ""}, "printer": {"id": "", "name": ""},
                           "numberOfLayers": s.get("layers", 0), "slicer": s.get("slicer", "")},
                 "printInfo": {"printOutcome": s.get("report", {}).get("printOutcome", "SUCCESS")}}
                for sid, s in SIMS.items()]
        opts = [{"id": oid, "name": o["name"], "status": o["status"],
                 "optimizedGcodeWithThermalIndexesUrl": o.get("opt_url"),
                 "qualityMeanImprovement": f"{float(o.get('quality_mean_improvement') or 0.0):.2f}",
                 "qualityStdImprovement": "0.00",
                 "gcode": {"gcodeUrl": None, "gcodeKey": o["gcode_id"],
                           "material": {"id": "", "name": ""}, "printer": {"id": "", "name": ""},
                           "numberOfLayers": 0, "slicer": ""}}
                for oid, o in OPTS.items()]
        return _g({"data": {"optimizations": {"objects": opts}, "simulations": {"objects": sims}}})

    return _g({"errors": [{"message": f"unsupported operation: {query[:120]}"}]})


# ---------------------------------------------------------------------------
# presigned PUT 目标 & 资产下载
# ---------------------------------------------------------------------------
@router.put("/helio-upload/{upload_id}")
async def helio_upload(upload_id: str, request: Request):
    body = await request.body()
    UPLOADS[upload_id] = body
    return Response(status_code=200)


@router.get("/helio-assets/{asset_id}/{filename}")
def helio_asset(asset_id: str, filename: str):
    asset = ASSETS.get(asset_id)
    if not asset:
        return Response(status_code=404)
    content, media = asset
    return Response(content=content, media_type=media)
