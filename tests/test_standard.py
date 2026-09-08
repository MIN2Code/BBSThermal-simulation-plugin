"""标准量化测试件测试：温度塔/VFA 生成、喷嘴跟踪、拟合回代。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.calibration.detect import detect_nozzle_blocks
from backend.calibration.standard import generate_temp_tower, generate_vfa
from backend.calibration.standard_fit import block_iface_medians, fit_temp_tower_gcode
from backend.gcode.parser import parse_gcode
from backend.thermal.materials import get_material
from backend.thermal.voxel import SimConfig, ThermalSimulator


def test_temp_tower_generator_and_nozzle_tracking():
    text, manifest = generate_temp_tower("PLA")
    assert len(manifest["blocks"]) == 7
    temps = [b["temp"] for b in manifest["blocks"]]
    assert temps == sorted(temps, reverse=True)          # 温度递减
    assert manifest["total_layers"] == 7 * 40

    p = parse_gcode(text)
    assert p.nozzle_seg is not None, "温度塔必须产生逐段喷嘴温度"
    assert p.num_layers == manifest["total_layers"]
    # 每块中段的喷嘴温度应等于该块设定值
    for blk in manifest["blocks"]:
        mid_layer = blk["layers"] * (blk["block"] - 1) + blk["layers"] // 2
        sel = p.layer_idx == mid_layer
        assert sel.any()
        vals = np.unique(np.round(p.nozzle_seg[sel], 1))
        assert len(vals) == 1 and abs(vals[0] - blk["temp"]) < 0.01, (
            f"块 {blk['block']} 喷嘴温度应为 {blk['temp']}，实际 {vals}"
        )


def test_vfa_generator():
    text, manifest = generate_vfa("PLA")
    speeds = [b["speed"] for b in manifest["bands"]]
    assert speeds == sorted(speeds) and speeds[0] == 20 and speeds[-1] == 300
    assert manifest["total_layers"] == len(speeds) * 50
    p = parse_gcode(text)
    assert p.num_layers == manifest["total_layers"]
    # 逐带挤出速度：抽查首带（20mm/s → F1200）与末带（300 → F18000）
    assert "F1200" in text and "F18000" in text


def test_tower_fit_roundtrip():
    """合成真值回代：以 θ_true 仿真 → 按阈值导出三级评价 → 拟合应可行且一致。"""
    text, manifest = generate_temp_tower("PLA")
    p = parse_gcode(text)
    mat = get_material("PLA")
    theta = {"kappa": 0.65, "eta": 0.35}
    cfg = SimConfig(voxel_mm=1.0, iface_reheat=theta["kappa"], nozzle_heat=theta["eta"])
    res = ThermalSimulator(p, mat, cfg).run()

    blocks = detect_nozzle_blocks(p)
    block_iface = block_iface_medians(p, res, blocks)
    bi_sorted = np.argsort(block_iface)
    # 阈值取最小/最大 iface 的中点 → 产生弱/可用/过热三级的观测
    weak_iface = block_iface[bi_sorted[0]]
    hot_iface = block_iface[bi_sorted[-1]]
    t_bond = weak_iface + 3.0
    t_sag = hot_iface - 3.0
    outcomes = {}
    for bi, iface in enumerate(block_iface, 1):
        if iface <= t_bond:
            outcomes[bi] = "weak"
        elif iface >= t_sag:
            outcomes[bi] = "hot"
        else:
            outcomes[bi] = "ok"

    report = fit_temp_tower_gcode(p, res, outcomes)
    assert report["ok"], f"真值参数应可行: {report.get('diagnosis', '')}"
    assert report["agreement"] >= 0.8
    # 拟合出的窗口应把弱/过热块分开
    assert report["t_bond"] >= block_iface[bi_sorted[0]] - 5
    assert report["t_sag"] <= block_iface[bi_sorted[-1]] + 5
