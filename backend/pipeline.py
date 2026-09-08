"""任务管线：上传解析 → 后台仿真 → 打包二进制结果。

自用工具，任务保存在内存中；结果二进制按 (层号, 时间) 排序交错存储，
前端可直接用 drawRange 做逐层显示。
"""
from __future__ import annotations

import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from .gcode.model import ParsedGcode
from .gcode.parser import parse_gcode
from .thermal.materials import get_material, list_materials
from .thermal.optimize import rewrite_gcode
from .thermal.tqi import tqi_from_interface_temp
from .thermal.voxel import SimConfig, SimResult, SimulationCancelled, ThermalSimulator

# 每段 40 字节：6×f32 几何 + f32 tqi + f32 iface + i32 层号 + u8 特性 + u8 有效 + 2 pad
SEG_STRIDE = 40


@dataclass
class Job:
    id: str
    name: str = ""
    status: str = "parsed"          # parsed | simulating | optimizing | done | error | cancelled
    progress: float = 0.0
    error: str | None = None
    cancel_requested: bool = False  # 前端停止按钮置位，worker 在检查点响应
    created: float = field(default_factory=time.time)
    parsed: ParsedGcode | None = None
    result: SimResult | None = None
    payload: bytes | None = None
    meta: dict | None = None
    printer_settings: dict = field(default_factory=dict)  # 3MF 工程配置摘要
    raw_text: str = ""              # 原始 G-code 文本（优化回写用）
    source_zip: bytes | None = None # 上传为 .gcode.3mf 时保留原始容器（回包用）
    sim_config: SimConfig | None = None  # 最近一次仿真配置（优化沿用）
    opt_result: object | None = None     # OptimizeResult
    opt_gcode: bytes | None = None       # 优化后 G-code
    opt_meta: dict | None = None
    profile: dict | None = None          # 已应用的耗材档案


JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def ingest_bytes(raw: bytes) -> tuple[str, dict, ParsedGcode]:
    """统一入口：识别纯 G-code 或 .gcode.3mf 容器，返回 (text, settings, parsed)。"""
    from .gcode.bambu3mf import extract_3mf, is_zip_bytes

    if is_zip_bytes(raw):
        text, settings = extract_3mf(raw)
    else:
        text = raw.decode("utf-8-sig", errors="replace")
        settings = {}
    parsed = parse_gcode(text)
    info = parsed.info
    if settings.get("printer_model"):
        info.printer_model = settings["printer_model"]
    if settings.get("printer_variant"):
        info.printer_variant = settings["printer_variant"]
    if settings.get("filament_type"):
        info.detected_material = settings["filament_type"].upper()
    if settings.get("chamber_temp") and settings["chamber_temp"] > 0:
        info.chamber_temp = float(settings["chamber_temp"])
    return text, settings, parsed


def create_job_from_bytes(raw: bytes, name: str = "") -> tuple[str, dict]:
    from .gcode.bambu3mf import is_zip_bytes

    text, settings, parsed = ingest_bytes(raw)
    job_id = secrets.token_hex(6)
    job = Job(id=job_id, parsed=parsed, raw_text=text, name=name or job_id)
    if is_zip_bytes(raw):
        job.source_zip = raw  # 优化结果将回包为 .gcode.3mf
    job.payload, job.meta = pack_preview(parsed)  # 解析即可预览
    job.printer_settings = settings
    with _LOCK:
        JOBS[job_id] = job
        if len(JOBS) > 6:
            oldest = min(JOBS, key=lambda k: JOBS[k].created)
            if oldest != job_id:
                del JOBS[oldest]
    return job_id, parsed.summary()


def create_job_from_text(text: str) -> tuple[str, dict]:
    parsed = parse_gcode(text)
    job_id = secrets.token_hex(6)
    job = Job(id=job_id, parsed=parsed)
    job.payload, job.meta = pack_preview(parsed)  # 解析即可预览
    with _LOCK:
        JOBS[job_id] = job
        # 只保留最近 6 个任务，防内存膨胀
        if len(JOBS) > 6:
            oldest = min(JOBS, key=lambda k: JOBS[k].created)
            if oldest != job_id:
                del JOBS[oldest]
    return job_id, parsed.summary()


def get_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise KeyError(f"任务 {job_id} 不存在或已过期")
    return job


def start_simulation(job_id: str, params: dict) -> None:
    job = get_job(job_id)
    if job.status == "simulating":
        raise RuntimeError("该任务正在仿真中")
    with _LOCK:
        job.status = "simulating"
        job.progress = 0.0
        job.error = None
        job.result = None
        job.payload = None
        job.meta = None
        job.cancel_requested = False
    t = threading.Thread(target=_worker, args=(job, params), daemon=True)
    t.start()


def _worker(job: Job, params: dict) -> None:
    try:
        from .profiles import apply_profile, load_profile

        material = get_material(params.get("material", job.parsed.info.detected_material))
        cfg = SimConfig(
            voxel_mm=float(params.get("voxel_mm", 1.5)),
            bucket_s=float(params.get("bucket_s", 0.3)),
            chamber_temp=params.get("chamber_temp"),
            iface_reheat=float(params.get("iface_reheat", 0.8)),
        )
        # 已校准档案优先（显式传入的 iface_reheat 仍以用户为准）
        profile = load_profile(material.name)
        material, cfg = apply_profile(material, cfg, profile)
        if "iface_reheat" in params:
            cfg.iface_reheat = float(params["iface_reheat"])
        job.sim_config = cfg
        job.profile = profile
        sim = ThermalSimulator(
            job.parsed, material, cfg,
            progress_cb=lambda p: setattr(job, "progress", min(float(p), 0.99)),
            cancel_check=lambda: job.cancel_requested,
        )
        job.result = sim.run()
        job.payload, job.meta = pack_result(job.parsed, job.result, material.name)
        job.progress = 1.0
        job.status = "done"
    except SimulationCancelled:
        job.status = "cancelled"
        job.error = None
    except Exception as exc:  # noqa: BLE001
        job.error = f"{exc}\n{traceback.format_exc(limit=3)}"
        job.status = "error"


def cancel_job(job_id: str) -> bool:
    """请求中断进行中的仿真/优化。返回是否已受理。"""
    job = get_job(job_id)
    if job.status in ("simulating", "optimizing"):
        job.cancel_requested = True
        return True
    return False


def start_optimize(job_id: str, params: dict) -> None:
    from .thermal.optimize import OptimizeConfig, optimize_speeds

    job = get_job(job_id)
    if job.status == "simulating" or job.status == "optimizing":
        raise RuntimeError("任务正在计算中")
    if job.result is None:
        raise RuntimeError("请先完成热仿真再优化")
    with _LOCK:
        job.status = "optimizing"
        job.progress = 0.0
        job.error = None
        job.cancel_requested = False
    t = threading.Thread(target=_opt_worker, args=(job, params), daemon=True)
    t.start()


def _opt_worker(job: Job, params: dict) -> None:
    try:
        from .thermal.optimize import OptimizeConfig, optimize_speeds

        material = get_material(params.get("material") or job.meta.get("config", {}).get("material", "PLA"))
        cfg = job.sim_config or SimConfig()
        opt_cfg = OptimizeConfig(
            mode=params.get("mode", "quality"),
            rounds=int(params.get("rounds", 3)),
            cold_gain=float(params.get("cold_gain", 1.25)),
            hot_gain=float(params.get("hot_gain", 0.80)),
        )
        mode = opt_cfg.mode
        if mode == "surface":
            from .thermal.optimize import optimize_surface
            job.opt_result = optimize_surface(
                job.parsed, material, cfg, job.result, opt_cfg,
                progress_cb=lambda p: setattr(job, "progress", min(float(p) * 0.97, 0.97)),
                cancel_check=lambda: job.cancel_requested,
            )
        else:
            job.opt_result = optimize_speeds(
                job.parsed, material, cfg, opt_cfg,
                progress_cb=lambda p: setattr(job, "progress", min(float(p) * 0.97, 0.97)),
                cancel_check=lambda: job.cancel_requested,
            )
        lt_before = (job.parsed.layer_t1.astype(float) - job.parsed.layer_t0.astype(float))

        def _ds(arr, cap=600):
            a = np.asarray(arr, dtype=float)
            if a.size == 0:
                return []
            if a.size > cap:
                a = a[np.linspace(0, a.size - 1, cap).astype(int)]
            return [round(float(x), 2) for x in a]

        lta = getattr(job.opt_result, "layer_times_after", None)
        layer_times = None if lta is None else {"before": _ds(lt_before), "after": _ds(lta)}
        out_text, changed = rewrite_gcode(job.raw_text, job.parsed, job.opt_result.new_feed)
        job.opt_gcode = out_text.encode("utf-8")
        base = job.opt_result.baseline
        fin = job.opt_result.final
        job.opt_meta = {
            "baseline": base,
            "final": fin,
            "rounds": job.opt_result.round_stats,
            "changed_lines": changed,
            "mode": mode,
            "layer_times": layer_times,
            "params": {"rounds": opt_cfg.rounds, "cold_gain": opt_cfg.cold_gain,
                       "hot_gain": opt_cfg.hot_gain},
            "material": material.name,
            "output_format": "gcode.3mf" if job.source_zip else "gcode",
        }
        job.progress = 1.0
        job.status = "done"
    except SimulationCancelled:
        job.status = "cancelled"
        job.error = None
    except Exception as exc:  # noqa: BLE001
        job.error = f"{exc}\n{traceback.format_exc(limit=3)}"
        job.status = "error"


# ---------------------------------------------------------------------------
def _layer_ranges(layer: np.ndarray) -> list[dict]:
    uniq, start_idx, counts = np.unique(layer, return_index=True, return_counts=True)
    return [
        {"layer": int(u), "start": int(s), "count": int(c)}
        for u, s, c in zip(uniq, start_idx, counts)
    ]


def pack_preview(parsed: ParsedGcode) -> tuple[bytes, dict]:
    """解析后即可渲染的预览包（TQI 全 0、valid=0 → 前端按特性着色）。"""
    order = np.lexsort((parsed.t_mid, parsed.layer_idx))
    geo = parsed.geometry[order].astype(np.float32, copy=False)
    layer = parsed.layer_idx[order].astype(np.int32, copy=False)
    feat = parsed.feature_id[order]
    n = geo.shape[0]

    payload = np.zeros((n, SEG_STRIDE), dtype=np.uint8)
    f32 = payload.view(np.float32).reshape(n, 10)
    f32[:, 0:6] = geo
    payload.view(np.int32).reshape(n, 10)[:, 8] = layer
    payload[:, 36] = feat
    payload[:, 37] = 0

    meta = {
        "summary": parsed.summary(),
        "config": {},
        "runtime_s": 0.0,
        "grid_shape": [],
        "voxel_mm": 0.0,
        "layer_ranges": _layer_ranges(layer),
        "layer_stats": [],
        "num_segments": int(n),
        "seg_stride": SEG_STRIDE,
        "material": parsed.info.detected_material,
        "preview": True,
    }
    return payload.tobytes(), meta


def pack_result(parsed: ParsedGcode, res: SimResult, material_name: str) -> tuple[bytes, dict]:
    """按 (层号, 时间) 稳定排序后交错打包为二进制。"""
    order = np.lexsort((parsed.t_mid, parsed.layer_idx))
    geo = parsed.geometry[order].astype(np.float32, copy=False)
    tqi = res.tqi[order].astype(np.float32, copy=False)
    iface = res.iface_temp[order].astype(np.float32, copy=False)
    layer = parsed.layer_idx[order].astype(np.int32, copy=False)
    feat = parsed.feature_id[order]
    valid = res.tqi_valid[order].astype(np.uint8, copy=False)

    n = geo.shape[0]
    payload = np.zeros((n, SEG_STRIDE), dtype=np.uint8)
    f32 = payload.view(np.float32).reshape(n, 10)   # 与 payload 共享内存
    f32[:, 0:6] = geo
    f32[:, 6] = tqi
    f32[:, 7] = iface
    i32 = payload.view(np.int32).reshape(n, 10)
    i32[:, 8] = layer
    payload[:, 36] = feat.astype(np.uint8)
    payload[:, 37] = valid

    meta = {
        "summary": parsed.summary(),
        "config": res.config,
        "runtime_s": res.runtime_s,
        "grid_shape": list(res.grid_shape),
        "voxel_mm": res.voxel_mm,
        "layer_ranges": _layer_ranges(layer),
        "layer_stats": res.layer_stats,
        "num_segments": int(n),
        "seg_stride": SEG_STRIDE,
        "material": material_name,
    }
    return payload.tobytes(), meta


def materials_payload() -> list[dict]:
    out = []
    for name in list_materials():
        m = get_material(name)
        out.append({
            "name": m.name, "nozzle": m.nozzle, "bed": m.bed, "tg": m.tg,
            "cold_below": m.cold_below, "ideal_lo": m.ideal_lo,
            "ideal_hi": m.ideal_hi, "hot_above": m.hot_above,
        })
    return out
