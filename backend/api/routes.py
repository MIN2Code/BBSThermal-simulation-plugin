"""REST API 路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import Response

from .. import bs_switch, pipeline

router = APIRouter(prefix="/api")


@router.get("/bsconfig")
def bs_config_state():
    """Bambu Studio Helio 端点当前指向（local/helio）与运行状态。"""
    return bs_switch.read_state()


@router.post("/bsconfig")
def bs_config_switch(payload: dict):
    """切换 Bambu Studio 的 Helio 端点：mode = 'local' | 'helio'。"""
    mode = payload.get("mode")
    try:
        result = bs_switch.apply_mode(mode, pat=payload.get("pat"))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return result


@router.get("/materials")
def get_materials():
    return pipeline.materials_payload()


@router.post("/upload")
async def upload_gcode(file: UploadFile):
    raw = await file.read()
    try:
        job_id, summary = pipeline.create_job_from_bytes(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"解析失败：{exc}") from exc
    return {"job_id": job_id, "summary": summary}


@router.post("/job/{job_id}/simulate")
def simulate(job_id: str, params: dict = None):
    try:
        pipeline.start_simulation(job_id, params or {})
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@router.get("/job/{job_id}")
def job_status(job_id: str):
    try:
        job = pipeline.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "summary": job.parsed.summary() if job.parsed else None,
    }


@router.post("/job/{job_id}/cancel")
def cancel(job_id: str):
    try:
        accepted = pipeline.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True, "accepted": accepted}


@router.get("/preview/{job_id}/meta")
def preview_meta(job_id: str):
    try:
        job = pipeline.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job.meta is None:
        raise HTTPException(409, "预览数据不存在")
    return job.meta


@router.get("/preview/{job_id}/binary")
def preview_binary(job_id: str):
    try:
        job = pipeline.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job.payload is None:
        raise HTTPException(409, "预览数据不存在")
    return Response(
        content=job.payload,
        media_type="application/octet-stream",
        headers={"Content-Length": str(len(job.payload))},
    )


@router.post("/job/{job_id}/optimize")
def optimize(job_id: str, params: dict = None):
    try:
        pipeline.start_optimize(job_id, params or {})
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@router.get("/optimize/{job_id}/meta")
def optimize_meta(job_id: str):
    try:
        job = pipeline.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job.opt_meta is None:
        raise HTTPException(409, "优化尚未完成")
    return job.opt_meta


@router.get("/job/{job_id}/tempbias/{delta}")
def temp_bias_download(job_id: str, delta: float):
    """整件喷嘴温度偏置（±15°C clamp）下载。S<100 关温指令不受影响。"""
    try:
        data, changed = pipeline.apply_temp_bias(job_id, delta)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    job = pipeline.get_job(job_id)
    name = f"tempbias_{delta:+.0f}_{job_id}." + ("gcode.3mf" if job.source_zip else "gcode")
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{name}"',
                 "X-Bias-Changed": str(changed)},
    )


@router.get("/optimize/{job_id}/download")
def optimize_download(job_id: str):
    try:
        job = pipeline.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job.opt_gcode is None:
        raise HTTPException(409, "优化尚未完成")
    if job.source_zip:
        # 源文件是 .gcode.3mf：优化结果回包为同名容器（Bambu Studio 原生识别）
        from backend.gcode.bambu3mf import repack_3mf

        content = repack_3mf(job.source_zip, job.opt_gcode)
        name = f"optimized_{job_id}.gcode.3mf"
    else:
        content = job.opt_gcode
        name = f"optimized_{job_id}.gcode"
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.get("/result/{job_id}/meta")
def result_meta(job_id: str):
    try:
        job = pipeline.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job.status != "done" or job.meta is None:
        raise HTTPException(409, "仿真尚未完成")
    return job.meta


@router.get("/result/{job_id}/binary")
def result_binary(job_id: str):
    try:
        job = pipeline.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job.status != "done" or job.payload is None:
        raise HTTPException(409, "仿真尚未完成")
    return Response(
        content=job.payload,
        media_type="application/octet-stream",
        headers={"Content-Length": str(len(job.payload))},
    )
