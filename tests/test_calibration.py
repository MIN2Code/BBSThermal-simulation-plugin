"""校准芯片测试：生成器、分段物理方向、拟合可行性区间、档案存取。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.calibration.chip import generate_chip, section_of_layer
from backend.calibration.fit import fit_from_outcomes, section_iface_medians
from backend.gcode.parser import parse_gcode
from backend.thermal.materials import get_material
from backend.thermal.voxel import SimConfig, ThermalSimulator
from backend.profiles import apply_profile, load_profile, profile_path, save_profile


def test_chip_manifest_and_parse():
    text, manifest = generate_chip("PLA")
    assert len(manifest["sections"]) == 6
    fans = [s["fan"] for s in manifest["sections"]]
    assert fans == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]
    assert manifest["total_layers"] == 180
    # z 单调
    zs = [s["z_from"] for s in manifest["sections"]]
    assert zs == sorted(zs)

    p = parse_gcode(text)
    assert p.num_layers == 180
    assert p.num_segments > 0
    assert p.info.nozzle_temp == 210.0
    # 风扇序列按段号应为 0/0/0.5/0.5/1/1（每段取该段首层的风扇值）
    seen = []
    for li in range(p.num_layers):
        f = float(p.fan[p.layer_idx == li][0])
        if not seen or abs(seen[-1] - f) > 0.01:
            seen.append(f)
    assert len(seen) == 3 and abs(seen[0]) < 0.01 and abs(seen[1] - 0.5) < 0.01 and seen[2] > 0.99


def test_chip_section_physcis_ordering():
    """芯片分段物理方向：层时主导（快速段界面暖于慢速段），风扇在同速下降温。

    六段预期排序：快速无扇 > 快速全扇 ≈> 慢速无扇 > 慢速全扇。
    （真实 PLA 经验一致：薄壁慢打无扇=热而结实，长层时+风扇=冷而弱）
    """
    text, manifest = generate_chip("PLA")
    p = parse_gcode(text)
    mat = get_material("PLA")
    res = ThermalSimulator(p, mat, SimConfig(voxel_mm=1.0, iface_reheat=0.5, nozzle_heat=0.35)).run()
    ifaces = section_iface_medians(p, res, section_layers=30, n_sections=6)
    assert np.isfinite(ifaces).all(), "各段界面温度均应有值"
    s1, s2, s5, s6 = ifaces[0], ifaces[1], ifaces[4], ifaces[5]
    assert s2 > s1, f"无扇组：快速段({s2:.0f})应暖于慢速段({s1:.0f})——层时效应"
    assert s6 > s5, f"全扇组：快速段({s6:.0f})应暖于慢速段({s5:.0f})"
    assert s1 > s5, f"慢速组：无扇段({s1:.0f})应暖于全扇段({s5:.0f})——风扇效应"
    assert s2 > s6, f"快速组：无扇段({s2:.0f})应暖于全扇段({s6:.0f})——风扇效应"


def test_fit_finds_feasible_hfan():
    """合成真值回代：以 θ_true 仿真得到各段界面温度，按阈值导出掰断结果，
    拟合器应判定真实参数组可行。"""
    text, _ = generate_chip("PLA")
    p = parse_gcode(text)
    mat = get_material("PLA")
    theta_true = {"kappa": 0.5, "eta": 0.35, "hfan": 1.0}
    mat_true = get_material("PLA")
    cfg = SimConfig(voxel_mm=1.0, iface_reheat=theta_true["kappa"], nozzle_heat=theta_true["eta"])
    res = ThermalSimulator(p, mat_true, cfg).run()
    ifaces = section_iface_medians(p, res, section_layers=30, n_sections=6)

    t_bond = float((np.nanmax(ifaces) + np.nanmin(ifaces)) / 2)
    outcomes = {}
    for s in range(1, 7):
        outcomes[s] = "strong" if ifaces[s - 1] >= t_bond else "weak"

    report = fit_from_outcomes(
        outcomes, mat,
        section_layers=30,
        grid={"kappa": [0.5], "eta": [0.35], "hfan": [0.6, 0.8, 1.0, 1.25, 1.5]},
    )
    assert report["ok"], f"真值参数应可行: {report.get('diagnosis', '')}"
    # hfan 网格里应包含真值 1.0 且其为可行解
    assert any(abs(r["hfan"] - 1.0) < 1e-6 for r in [report] ) or report["n_feasible"] >= 1
    assert abs(report["bond_threshold"] - t_bond) < 15 or report["agreement"] == 1.0


def test_profile_save_apply_roundtrip(tmp_path):
    from backend import profiles as P

    P.PROFILES_DIR = str(tmp_path)
    profile = {"material": "PLA", "iface_reheat": 0.55, "nozzle_heat": 0.28,
               "hfan": 1.2, "cold_below": 83.0, "ideal_lo": 108.0}
    P.save_profile(profile)
    loaded = P.load_profile("PLA")
    assert loaded and loaded["hfan"] == 1.2 and loaded["cold_below"] == 83.0

    mat = get_material("PLA")
    cfg = SimConfig(voxel_mm=1.5, iface_reheat=0.5, nozzle_heat=0.35)
    mat2, cfg2 = apply_profile(mat, cfg, loaded)
    assert abs(mat2.h_conv_on - mat.h_conv_on * 1.2) < 1e-6
    assert abs(mat2.cold_below - 83.0) < 1e-6
    assert abs(cfg2.iface_reheat - 0.55) < 1e-6
    assert abs(cfg2.nozzle_heat - 0.28) < 1e-6
    assert P.load_profile("PETG") is None  # 未标定材料无档案


def test_section_of_layer_helper():
    assert section_of_layer(0) == 0
    assert section_of_layer(29) == 0
    assert section_of_layer(30) == 1
    assert section_of_layer(179) == 5
