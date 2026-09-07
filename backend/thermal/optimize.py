"""Enhance 式打印速度优化器。

思路（对齐 Helio 的公开行为——只调速度、不改路径）：
- 界面温度的物理主导项是「层时间」：某层的持续时间决定了它覆盖下层时
  下层材料已冷却多久。层越快 → 界面越暖（结合越好）；层越慢 → 越冷。
- 每轮：仿真 → 按逐层平均 TQI 分类（冷/热/合格）→ 对该层所有段施加
  速度因子（冷层提速、热层降速）→ 重计时 → 再仿真，迭代收敛。
- 输出：新速度表 + 重写 F 值的 G-code。

速度因子按层施加而非按段：段自身的速度几乎不影响自己这一段的界面温度
（基材冷热在其沉积前已定），只影响下一层覆盖时的温度。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace as dc_replace

import numpy as np

from ..gcode.model import ParsedGcode
from .materials import Material
from .voxel import SimConfig, ThermalSimulator


@dataclass
class OptimizeConfig:
    rounds: int = 3
    cold_gain: float = 1.15     # 冷层提速系数（阻尼版：过强会与层间耦合共振震荡）
    hot_gain: float = 0.90      # 热层降速系数
    min_speed: float = 15.0
    max_speed: float = 300.0
    cold_threshold: float = -30.0
    hot_threshold: float = 30.0
    layers_from: int | None = None   # 只优化该层号范围（None=全部）
    layers_to: int | None = None
    max_flow_mm3s: float | None = None  # 体积流量上限（按段珠截面折算限速）
    priority: str = "speed_strength"    # speed_strength | surface（表面优先：外墙不提速）


@dataclass
class OptimizeResult:
    new_feed: np.ndarray            # (N,) 优化后速度 mm/s（按原始段序）
    round_stats: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)   # 末轮 TQI 概览
    baseline: dict = field(default_factory=dict)  # 首轮（=原速度）概览


# ---------------------------------------------------------------------------
def _retime(parsed: ParsedGcode, feed: np.ndarray) -> None:
    """按新速度重算时间轴（就地更新 parsed 的时间字段）。"""
    g = parsed.geometry
    dist = np.hypot(g[:, 3] - g[:, 0], g[:, 4] - g[:, 1])
    dur = dist / np.maximum(feed, 1e-6)
    ends = np.cumsum(parsed.travel_before + dur)
    t_mid = ends - dur * 0.5
    n_layers = parsed.num_layers
    layer_t0 = np.full(n_layers, np.inf, dtype=np.float64)
    layer_t1 = np.full(n_layers, -np.inf, dtype=np.float64)
    np.fmin.at(layer_t0, parsed.layer_idx, ends - dur)
    np.fmax.at(layer_t1, parsed.layer_idx, ends)
    parsed.duration[:] = dur.astype(np.float32)
    parsed.t_mid[:] = t_mid.astype(np.float32)
    parsed.layer_t0[:] = layer_t0.astype(np.float32)
    parsed.layer_t1[:] = np.minimum(layer_t1, ends[-1]).astype(np.float32)


def _tqi_overview(res, parsed) -> dict:
    v = res.tqi[res.tqi_valid]
    t = res.iface_temp[res.tqi_valid]
    if v.size == 0:
        return {"mean_tqi": None}
    return {
        "mean_tqi": float(v.mean()),
        "cold_pct": float((v < -50).mean() * 100),
        "hot_pct": float((v > 50).mean() * 100),
        "ok_pct": float((np.abs(v) <= 50).mean() * 100),
        "iface_median": float(np.median(t)),
    }


def optimize_speeds(
    parsed: ParsedGcode,
    material: Material,
    sim_config: SimConfig,
    opt_config: OptimizeConfig,
    progress_cb=None,
) -> OptimizeResult:
    n_layers = parsed.num_layers
    orig_feed = parsed.feedrate.copy()
    orig_times = (parsed.t_mid.copy(), parsed.duration.copy(),
                  parsed.layer_t0.copy(), parsed.layer_t1.copy())
    feed = orig_feed.copy()

    result = OptimizeResult(new_feed=feed)
    # 粗细结合提速：中间轮用粗体素/大桶快速定方向，基线与终验用用户精度
    cfg_coarse = dc_replace(sim_config,
                            voxel_mm=min(sim_config.voxel_mm * 1.7, 3.0),
                            bucket_s=sim_config.bucket_s * 1.6)
    try:
        for rnd in range(opt_config.rounds):
            use_cfg = sim_config if rnd == 0 else cfg_coarse
            _retime(parsed, feed)
            frac = (rnd + 1) / opt_config.rounds
            sim = ThermalSimulator(
                parsed, material, use_cfg,
                progress_cb=(lambda p, f=frac: progress_cb and progress_cb(
                    (f - 1.0 / opt_config.rounds) + p / opt_config.rounds)),
            )
            res = sim.run()
            overview = _tqi_overview(res, parsed)
            overview["est_time_s"] = float(parsed.t_mid[-1] + parsed.duration[-1])
            if rnd == 0:
                result.baseline = overview
                best_mean = overview["mean_tqi"]
            result.round_stats.append({"round": rnd + 1, **overview})

            # 逐层速度因子（仅优化范围内的层）
            lf = opt_config.layers_from
            lt = opt_config.layers_to
            layer_factor = np.ones(n_layers)
            for stat in res.layer_stats:
                mt = stat["mean_tqi"]
                li = stat["layer"]
                if mt is None:
                    continue
                if lf is not None and li < lf:
                    continue
                if lt is not None and li > lt:
                    continue
                if mt < opt_config.cold_threshold:
                    layer_factor[li] = opt_config.cold_gain
                elif mt > opt_config.hot_threshold:
                    layer_factor[li] = opt_config.hot_gain
            new_feed = np.clip(feed * layer_factor[parsed.layer_idx],
                               opt_config.min_speed, opt_config.max_speed)
            # 表面优先：外墙段只降不升
            if opt_config.priority == "surface":
                outer = parsed.feature_id == 8  # Feature.OUTER_WALL
                new_feed = np.where(outer & (new_feed > feed), feed, new_feed)
            # 体积流量上限：feed ≤ max_flow / (mm³/mm)
            if opt_config.max_flow_mm3s:
                g = parsed.geometry
                seg_len = np.maximum(np.hypot(g[:, 3] - g[:, 0], g[:, 4] - g[:, 1]), 1e-6)
                vol_per_mm = np.maximum(parsed.extrusion_mm3 / seg_len, 1e-6)
                feed_cap = opt_config.max_flow_mm3s / vol_per_mm
                new_feed = np.minimum(new_feed, np.maximum(feed_cap, opt_config.min_speed))
            changed = int((np.abs(new_feed - feed) > 0.5).sum())
            feed = new_feed
            if progress_cb:
                progress_cb((rnd + 1) / opt_config.rounds)
            if changed == 0:
                break

        # 防回退：若优化后均值反而变差，回滚为原速（保底：至少不变差）
        _retime(parsed, feed)
        sim = ThermalSimulator(parsed, material, sim_config)
        res = sim.run()
        result.final = _tqi_overview(res, parsed)
        result.final["est_time_s"] = float(parsed.t_mid[-1] + parsed.duration[-1])
        if (result.baseline.get("mean_tqi") is not None
                and result.final.get("mean_tqi") is not None
                and result.final["mean_tqi"] < result.baseline["mean_tqi"]):
            # 优化反而变差 → 回滚原速（保底：至少不变差）
            feed = orig_feed.copy()
            _retime(parsed, feed)
            sim = ThermalSimulator(parsed, material, sim_config)
            res = sim.run()
            result.final = _tqi_overview(res, parsed)
            result.final["est_time_s"] = float(parsed.t_mid[-1] + parsed.duration[-1])
            result.final["rolled_back"] = True
    finally:
        # 还原 parsed 到原始状态（任务仍需原数据做对比渲染）
        parsed.feedrate[:] = orig_feed
        (parsed.t_mid[:], parsed.duration[:], parsed.layer_t0[:], parsed.layer_t1[:]) = orig_times
        _retime(parsed, orig_feed)

    result.new_feed = feed
    return result


# ---------------------------------------------------------------------------
_F_RE = re.compile(r"F([0-9.]+)")


def rewrite_gcode(text: str, parsed: ParsedGcode, new_feed: np.ndarray,
                  min_delta: float = 0.5) -> tuple[str, int]:
    """按 new_feed 重写 G-code 中挤出移动的 F 值（只动速度，不动路径）。"""
    # 行号 → 该行产生的段区间
    line_of_seg = parsed.src_line
    order = np.argsort(line_of_seg, kind="stable")
    lines_out = text.splitlines()
    changed = 0
    i = 0
    n = order.size
    while i < n:
        j = i
        ln = line_of_seg[order[i]]
        while j < n and line_of_seg[order[j]] == ln:
            j += 1
        segs = order[i:j]
        i = j
        orig = float(parsed.feedrate[segs[0]])
        desired = float(np.mean(new_feed[segs]))
        if abs(desired - orig) < min_delta:
            continue
        idx = ln - 1
        if idx < 0 or idx >= len(lines_out):
            continue
        line = lines_out[idx]
        f_mm_min = desired * 60.0
        if _F_RE.search(line):
            new_line = _F_RE.sub(lambda m: f"F{f_mm_min:.0f}", line, count=1)
        else:
            # 无显式 F 的挤出行：在行尾注入
            new_line = f"{line} F{f_mm_min:.0f}"
        lines_out[idx] = new_line
        changed += 1
    return "\n".join(lines_out), changed
