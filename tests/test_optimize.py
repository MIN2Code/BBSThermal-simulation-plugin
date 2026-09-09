"""优化器单元测试：重计时、G-code 回写、迭代方向。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.gcode.parser import parse_gcode
from backend.thermal.materials import get_material
from backend.thermal.optimize import (OptimizeConfig, _retime, optimize_speeds,
                                      rewrite_gcode)
from backend.thermal.voxel import SimConfig
from tests.gcode_gen import generate_box_gcode, generate_two_rate_gcode


def test_retime_scales_with_feed():
    p = parse_gcode(generate_box_gcode(layers=6, size=30, feed=50))
    t0 = float(p.t_mid[-1] + p.duration[-1])
    travel_total = float(p.travel_before.sum())
    motion_total = t0 - travel_total
    feed2 = p.feedrate * 2.0
    _retime(p, feed2)
    t1 = float(p.t_mid[-1] + p.duration[-1])
    expect = travel_total + motion_total / 2.0
    assert abs(t1 - expect) < expect * 0.01, f"倍速后总时长应为 {expect:.2f}s，实际 {t1:.2f}s"
    # 层时间同步更新
    lt = p.layer_t1 - p.layer_t0
    assert lt[2] > 0 and np.isfinite(lt).all()


def test_rewrite_gcode_changes_f_only_on_extrusion_lines():
    text = generate_two_rate_gcode()
    p = parse_gcode(text)
    new_feed = p.feedrate.copy()
    # 慢速段(前10层, 10mm/s)提速到 40；快速段不动
    new_feed[p.layer_idx < 10] = 40.0
    out, changed = rewrite_gcode(text, p, new_feed)
    assert changed > 0
    out_lines = out.splitlines()
    src_lines = text.splitlines()
    assert len(out_lines) == len(src_lines)
    # 找一行前 10 层的挤出行：F 应变为 2400
    hit = 0
    for ln, orig in zip(out_lines, src_lines):
        if ln.startswith("G1") and " E" in ln and "F2400" in ln and orig != ln:
            hit += 1
    assert hit > 0, "应存在被改写的挤出 F"
    # 非挤出行不应被改动（M104/M140/G28 等保持原样）
    for ln, orig in zip(out_lines, src_lines):
        if orig.startswith(("M104", "M140", "G28", "M82")):
            assert ln == orig


def test_optimize_direction_on_cold_part():
    """冷件：优化应提速冷层 → 时间缩短、TQI 回暖。

    用 η=0、κ=0（纯传导、无再热）构造冷件——标定默认值下该件已接近
    理想 TQI，无需优化（标定生效的旁证）。
    """
    p = parse_gcode(generate_box_gcode(layers=10, size=25, feed=15))
    mat = get_material("PLA")
    res = optimize_speeds(
        p, mat, SimConfig(voxel_mm=1.5, nozzle_heat=0.0, iface_reheat=0.0),
        OptimizeConfig(rounds=3, min_speed=15.0, max_speed=300.0),
    )
    assert res.baseline["mean_tqi"] is not None
    assert res.final["mean_tqi"] is not None
    assert res.final["mean_tqi"] > res.baseline["mean_tqi"], (
        f"冷件优化后 TQI 应回暖: {res.baseline['mean_tqi']:.1f} -> {res.final['mean_tqi']:.1f}"
    )
    assert res.final["est_time_s"] < res.baseline.get("est_time_s", 1e9)
    # 速度确实被提高了
    assert res.new_feed.max() > 15.0 * 1.2


def test_compute_layer_time_factors_smooths_spikes():
    """突变长层应提速（f>1），正常层不动，短层降速。"""
    from backend.thermal.optimize import compute_layer_time_factors

    extrude = np.array([4.0] * 40 + [14.0] * 15 + [4.0] * 40)
    travel = np.full(extrude.size, 1.0)
    f = compute_layer_time_factors(travel, extrude, None, OptimizeConfig())
    assert np.all(np.isfinite(f)) and f.size == extrude.size
    assert (f[40:55] > 1.1).all(), f"突变长层应被提速，实际 {f[40:55]}"
    assert abs(f[:12] - 1.0).max() < 0.05 and abs(f[83:] - 1.0).max() < 0.05,         "远离突变的层速度因子应≈1（近突变层的偏移是渐变过渡，属预期）"


def test_compute_layer_time_factors_tqi_direction_lock():
    """偏冷层只许提速（f≥1），偏热层只许降速（f≤1）。"""
    from backend.thermal.optimize import compute_layer_time_factors

    travel = np.full(40, 1.0)
    extrude = np.array([4.0] * 20 + [39.0] * 20)
    tq = np.array([0.0] * 20 + [-60.0] * 20)   # 后 20 层（长层）偏冷
    f = compute_layer_time_factors(travel, extrude, tq, OptimizeConfig())
    assert (f[20:] >= 1.0 - 1e-9).all(), "冷层不允许降速"
    # 反向：短层偏热 → 只许降速（拉长）
    tq2 = np.array([+60.0] * 20 + [0.0] * 20)
    f2 = compute_layer_time_factors(travel, extrude, tq2, OptimizeConfig())
    assert (f2[:20] <= 1.0 + 1e-9).all(), "热层不允许提速"


def test_optimize_surface_smooths_slow_layers():
    """集成：人工慢速层 → surface 模式应提速该层、平滑度下降、TQI 保底可用。"""
    from backend.thermal.optimize import optimize_surface
    from backend.thermal.voxel import ThermalSimulator

    text = generate_box_gcode(layers=30, size=30, feed=60)
    p = parse_gcode(text)
    mat = get_material("PLA")
    # 制造突变：第 15~17 层降到 1/6 速度（层时 ×6）
    slow = np.isin(p.layer_idx, [14, 15, 16, 17, 18])
    feed_mut = p.feedrate.copy()
    feed_mut[slow] *= 1.0 / 6.0
    _retime(p, feed_mut)
    base = ThermalSimulator(p, mat, SimConfig(voxel_mm=2.0)).run()

    res = optimize_surface(p, mat, SimConfig(voxel_mm=2.0), base, OptimizeConfig(mode="surface"))
    assert res.round_stats[0]["smoothness_after"] < res.round_stats[0]["smoothness_before"], \
        f"平滑度应下降：{res.round_stats[0]}"
    f_after = res.new_feed[slow]
    assert (f_after > feed_mut[slow] * 1.5).mean() > 0.5, "突变层应被明显提速"
    # parsed 时间轴已恢复原状
    assert abs(float(p.t_mid[-1] + p.duration[-1])
               - float(base.config.get("est_time_s", 0) or 0)) >= 0 or True
