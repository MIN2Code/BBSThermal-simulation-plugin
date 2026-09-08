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
from .voxel import SimConfig, SimulationCancelled, ThermalSimulator


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
    mode: str = "quality"           # quality=TQI 窗口迭代 | surface=层时平滑（表面一致）
    smooth_target: float = 0.5      # 层时向邻域中位收缩系数 ρ（0=不动，1=完全压平）
    lt_window: int = 25             # 层时邻域窗口（层，奇数化）
    smooth_min_factor: float = 0.6  # 单层速度因子范围
    smooth_max_factor: float = 1.8


@dataclass
class OptimizeResult:
    new_feed: np.ndarray            # (N,) 优化后速度 mm/s（按原始段序）
    round_stats: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)   # 末轮 TQI 概览
    baseline: dict = field(default_factory=dict)  # 首轮（=原速度）概览
    layer_times_after: list[float] | None = None  # surface 模式：优化后逐层层时（回滚=None）


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
        "std_tqi": float(v.std()),
        "cold_pct": float((v < -50).mean() * 100),
        "hot_pct": float((v > 50).mean() * 100),
        "ok_pct": float((np.abs(v) <= 50).mean() * 100),
        "iface_median": float(np.median(t)),
    }


def _score(overview: dict) -> float | None:
    """综合质量分：均值 − 0.25×标准差（越好越热学均衡）。"""
    m, s = overview.get("mean_tqi"), overview.get("std_tqi")
    if m is None:
        return None
    return m - 0.25 * (s or 0.0)


def _smooth_factor(factor: np.ndarray, nodes: int = 9) -> np.ndarray:
    """节点平滑：把逐层因子压缩为 K 个锚点的线性插值曲线。

    对齐 Helio 的 autolinear 策略——自由度受限的调速曲线更平滑、
    不会逐层震荡；锚点值取邻近层的中位数抗野值。
    """
    n = len(factor)
    if n <= nodes:
        return factor
    ax = np.linspace(0, n - 1, nodes)
    vals = []
    for a in ax:
        lo, hi = int(max(0, a - 2)), int(min(n, a + 3))
        vals.append(np.median(factor[lo:hi]))
    return np.interp(np.arange(n), ax, np.asarray(vals))


def optimize_speeds(
    parsed: ParsedGcode,
    material: Material,
    sim_config: SimConfig,
    opt_config: OptimizeConfig,
    progress_cb=None,
    cancel_check=None,
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
            if cancel_check is not None and cancel_check():
                raise SimulationCancelled("优化已被用户中断")
            use_cfg = sim_config if rnd == 0 else cfg_coarse
            _retime(parsed, feed)
            frac = (rnd + 1) / opt_config.rounds
            sim = ThermalSimulator(
                parsed, material, use_cfg,
                progress_cb=(lambda p, f=frac: progress_cb and progress_cb(
                    (f - 1.0 / opt_config.rounds) + p / opt_config.rounds)),
                cancel_check=cancel_check,
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
            # 节点平滑：K 锚点插值约束调速曲线（Helio autolinear 思路）
            layer_factor = _smooth_factor(layer_factor, nodes=9)
            new_feed = np.clip(feed * layer_factor[parsed.layer_idx],
                               opt_config.min_speed, opt_config.max_speed)            # 表面优先：外墙段只降不升
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

        # 防回退：综合分（均值 − 0.25×均匀性罚）若反而变差，回滚为原速
        _retime(parsed, feed)
        sim = ThermalSimulator(parsed, material, sim_config, cancel_check=cancel_check)
        res = sim.run()
        result.final = _tqi_overview(res, parsed)
        result.final["est_time_s"] = float(parsed.t_mid[-1] + parsed.duration[-1])
        base_score = _score(result.baseline)
        final_score = _score(result.final)
        if (base_score is not None and final_score is not None
                and final_score < base_score):
            # 优化反而变差 → 回滚原速（保底：至少不变差）
            feed = orig_feed.copy()
            _retime(parsed, feed)
            sim = ThermalSimulator(parsed, material, sim_config, cancel_check=cancel_check)
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


def _lt_smoothness(layer_times: np.ndarray) -> float:
    """层时平滑度：对数层时的二阶差分 L1（越小越平滑）。"""
    t = np.asarray(layer_times, dtype=np.float64)
    t = t[np.isfinite(t) & (t > 0.05)]
    if t.size < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.log(t), 2))))


def _rolling_median(a: np.ndarray, w: int) -> np.ndarray:
    pad = w // 2
    ap = np.pad(a, pad, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(ap, w), axis=1)


def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    pad = w // 2
    ap = np.pad(a, pad, mode="edge")
    return np.convolve(ap, np.ones(w) / w, mode="valid")[: a.size]


def compute_layer_time_factors(
    layer_times: np.ndarray,
    layer_tqi: np.ndarray | None,
    cfg: OptimizeConfig,
) -> np.ndarray:
    """表面一致模式核心：层时平滑速度因子（纯函数）。

    目标层时 = 原层时与「低通趋势」按 ρ 混合，f = t / target：
    - 两级滤波（中值去窄尖峰 → 均值提取大趋势）保证宽窄突变都能检测
      （纯滑动中位在脉冲宽度≥半窗口时会被自身污染而失效）；
    - 尖峰被压回邻域、台阶跳变被摊成渐变，缓慢爬坡（≈趋势）保持不动
      ——对应用户需求：层时允许线性渐变，消除突变；
    - TQI 方向约束：偏冷层只许提速（f≥1），偏热层只许降速（f≤1）。
    """
    t = np.asarray(layer_times, dtype=np.float64)
    n = t.size
    f = np.ones(n)
    valid = np.isfinite(t) & (t > 0.05)
    if valid.sum() < 9:
        return f
    tv = t.copy()
    tv[~valid] = np.median(t[valid])
    wm = max(3, int(cfg.lt_window * 0.5) | 1)
    we = max(5, int(cfg.lt_window * 1.6) | 1)
    trend = _rolling_mean(_rolling_median(tv, wm), we)
    target = np.maximum(tv + cfg.smooth_target * (trend - tv), 0.05)
    f = np.where(valid, tv / target, 1.0)
    f = np.clip(f, cfg.smooth_min_factor, cfg.smooth_max_factor)
    if layer_tqi is not None:
        tq = np.asarray(layer_tqi, dtype=np.float64)
        cold = np.isfinite(tq) & (tq < cfg.cold_threshold)
        hot = np.isfinite(tq) & (tq > cfg.hot_threshold)
        f = np.where(cold, np.maximum(f, 1.0), f)
        f = np.where(hot, np.minimum(f, 1.0), f)
    return f


def optimize_surface(
    parsed: ParsedGcode,
    material: Material,
    sim_config: SimConfig,
    baseline_result,
    opt_config: OptimizeConfig,
    progress_cb=None,
    cancel_check=None,
) -> OptimizeResult:
    """表面一致模式：平滑逐层层时（冷却纹理的根源），TQI 仅作安全约束。

    层时突变 = 表面光泽突变。策略：每层速度因子 = 原层时 / 平滑目标层时
    （单次解析计算，不迭代），随后跑一次验证仿真确认热质量不崩
    （mean TQI 掉超 8 分则回滚原速）。需要基线仿真结果提供逐层 TQI。
    """
    orig_feed = parsed.feedrate.copy()
    orig_times = (parsed.t_mid.copy(), parsed.duration.copy(),
                  parsed.layer_t0.copy(), parsed.layer_t1.copy())
    result = OptimizeResult(new_feed=orig_feed.copy())

    lt = parsed.layer_t1.astype(np.float64) - parsed.layer_t0.astype(np.float64)
    tq = None
    base_ov = None
    if baseline_result is not None and getattr(baseline_result, "layer_stats", None):
        tq = np.full(parsed.num_layers, np.nan)
        for st in baseline_result.layer_stats:
            if st.get("mean_tqi") is not None and 0 <= st["layer"] < parsed.num_layers:
                tq[st["layer"]] = st["mean_tqi"]
        base_ov = _tqi_overview(baseline_result, parsed)
        base_ov["est_time_s"] = float(parsed.t_mid[-1] + parsed.duration[-1])
        result.baseline = base_ov

    factors = compute_layer_time_factors(lt, tq, opt_config)
    lf, ltf = opt_config.layers_from, opt_config.layers_to
    if lf is not None:
        factors[:max(int(lf), 0)] = 1.0
    if ltf is not None:
        factors[int(ltf) + 1:] = 1.0

    feed = np.clip(orig_feed * factors[parsed.layer_idx],
                   opt_config.min_speed, opt_config.max_speed)
    if opt_config.max_flow_mm3s:
        g = parsed.geometry
        seg_len = np.maximum(np.hypot(g[:, 3] - g[:, 0], g[:, 4] - g[:, 1]), 1e-6)
        vol_per_mm = np.maximum(parsed.extrusion_mm3 / seg_len, 1e-6)
        feed = np.minimum(feed, np.maximum(opt_config.max_flow_mm3s / vol_per_mm,
                                           opt_config.min_speed))
    _retime(parsed, feed)
    lt_after = parsed.layer_t1.astype(np.float64) - parsed.layer_t0.astype(np.float64)
    result.round_stats.append({
        "round": 1,
        "smoothness_before": _lt_smoothness(lt),
        "smoothness_after": _lt_smoothness(lt_after),
    })
    if progress_cb:
        progress_cb(0.6)

    sim = ThermalSimulator(parsed, material, sim_config, cancel_check=cancel_check)
    res = sim.run()
    overview = _tqi_overview(res, parsed)
    overview["est_time_s"] = float(parsed.t_mid[-1] + parsed.duration[-1])

    # 热质量保底：mean TQI 掉超 8 分 → 回滚原速（表面一致不能以热崩为代价）
    if base_ov is not None and base_ov.get("mean_tqi") is not None:
        if (overview.get("mean_tqi") is not None
                and overview["mean_tqi"] < base_ov["mean_tqi"] - 8.0):
            feed = orig_feed.copy()
            _retime(parsed, feed)
            overview = dict(base_ov)
            overview["rolled_back"] = True
        else:
            result.layer_times_after = [round(float(x), 2) for x in lt_after]
    else:
        result.layer_times_after = [round(float(x), 2) for x in lt_after]

    result.final = overview
    result.final["est_time_s"] = float(parsed.t_mid[-1] + parsed.duration[-1])
    result.new_feed = feed
    # 恢复 parsed 到原始状态（与 quality 模式 finally 语义一致；优化速度在 new_feed）
    parsed.feedrate[:] = orig_feed
    (parsed.t_mid[:], parsed.duration[:], parsed.layer_t0[:], parsed.layer_t1[:]) = orig_times
    _retime(parsed, orig_feed)
    return result





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
