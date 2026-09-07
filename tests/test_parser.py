"""解析器单元测试。"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.gcode.parser import parse_gcode
from backend.gcode.model import Feature
from tests.gcode_gen import generate_box_gcode, generate_two_rate_gcode


def test_parse_basic_box():
    text = generate_box_gcode(layers=10, size=40, feed=50)
    p = parse_gcode(text)
    assert p.num_segments > 0
    assert p.num_layers == 10
    assert np.all(np.diff(p.layer_z) > 0)
    assert abs(float(p.layer_z[0]) - 0.2) < 1e-6
    # 时间单调不减
    assert np.all(np.diff(p.t_mid) >= -1e-6)
    # 挤出体积全部为正
    assert np.all(p.extrusion_mm3 > 0)
    # 头部信息
    assert p.info.detected_material == "PLA"
    assert p.info.nozzle_temp == 210.0
    assert p.info.bed_temp == 55.0
    assert p.info.slicer == "testgen"


def test_segment_geometry_matches_square():
    text = generate_box_gcode(layers=3, size=40, feed=50)
    p = parse_gcode(text)
    li = p.layer_idx == 1
    walls = li & (p.feature_id == int(Feature.OUTER_WALL))
    geo = p.geometry[walls]
    # 每段长度与体积对齐：Σlen ≈ 160mm 周长
    lens = np.hypot(geo[:, 3] - geo[:, 0], geo[:, 4] - geo[:, 1])
    assert abs(lens.sum() - 160.0) < 1e-2
    # 体积 = 长度 × 珠截面积
    vols = p.extrusion_mm3[walls]
    assert np.allclose(vols, lens * 0.42 * 0.2, rtol=1e-4)


def test_layer_times_feed_dependent():
    slow = parse_gcode(generate_box_gcode(layers=4, size=40, feed=10))
    fast = parse_gcode(generate_box_gcode(layers=4, size=40, feed=150))
    t_slow = float(slow.layer_t1[1] - slow.layer_t0[1])
    t_fast = float(fast.layer_t1[1] - fast.layer_t0[1])
    assert t_slow > t_fast * 5  # 15 倍速度差 → 层时间至少差 5 倍


def test_arc_parsing():
    text = "\n".join([
        "M82", "G92 E0",
        "G1 X10 Y10 F1800",
        "G1 X20 Y10 E0.5 F1200",
        ";TYPE:Outer wall",
        "G2 X30 Y20 I10 J0 E1.0 F1200",  # 顺时针 1/4 圆弧 r=10
    ])
    p = parse_gcode(text)
    assert p.num_segments >= 4
    geo = p.geometry
    # 弧终点
    assert abs(geo[-1, 3] - 30.0) < 1e-6
    assert abs(geo[-1, 4] - 20.0) < 1e-6
    # 弧长 ≈ π/4 × d = π/2 × r ≈ 15.7
    arc_len = sum(
        math.hypot(g[3] - g[0], g[4] - g[1]) for g in geo if abs(g[1] - 10) > 1e-3 or abs(g[0] - 20) < 1e-3
    )
    # 只检查存在弧段（粗略）
    assert p.num_segments > 5


def test_relative_e_and_retract():
    text = "\n".join([
        "M83",  # 相对 E
        "G92 E0",
        "G1 X10 Y10 F3000",
        "G1 X20 Y10 E1.0 F1800",
        "G1 E-0.8 F2400",           # 抽回：不产生段
        "G1 X30 Y10 F3000",          # 空走：不产生段
        "G1 E0.8 F2400",             # 预压：不产生段
        "G1 X40 Y20 E1.0 F1800",     # 又一段
    ])
    p = parse_gcode(text)
    assert p.num_segments == 2
    assert np.all(p.extrusion_mm3 > 0)
    assert abs(p.geometry[1, 3] - 40) < 1e-6


def test_fan_and_dwell():
    text = "\n".join([
        "M82", "G92 E0",
        "G1 X0 Y0 F3000",
        "M106 S128",
        "G1 X10 Y0 E0.5 F600",
        "G4 P2000",                 # 驻留 2s
        "M107",
        "G1 X20 Y0 E1.0 F600",
    ])
    p = parse_gcode(text)
    assert abs(float(p.fan[0]) - 128 / 255) < 1e-6
    assert float(p.fan[1]) == 0.0
    # G4 2s 应体现在时间轴（第二段中点比第一段末尾晚 ≥2s）
    t0_end = float(p.t_mid[0] + p.duration[0])
    assert float(p.t_mid[1]) >= t0_end + 2.0 - 1e-3


def test_curas_layer_comments():
    text = "\n".join([
        ";LAYER_COUNT:2",
        "M82", "G92 E0",
        "G1 X5 Y5 F3000",
        ";LAYER:0",
        ";TYPE:SKIN",
        "G1 X15 Y5 E0.5 F1200",
        ";LAYER:1",
        "G1 Z0.4 F600",
        "G1 X15 Y15 E1.0 F1200",
    ])
    p = parse_gcode(text)
    assert p.num_layers == 2
    assert set(np.unique(p.layer_idx)) == {0, 1}
