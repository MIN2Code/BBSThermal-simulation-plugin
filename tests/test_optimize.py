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
