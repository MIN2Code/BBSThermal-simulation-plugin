"""校准拟合引擎：从芯片打印的掰断结果反推机器/耗材参数。

流程：
1. 生成芯片 G-code（固定几何），在参数网格 (κ, η, h_fan_scale) 上逐一仿真，
   得到每段预测界面温度；
2. 对每组参数，用「可行区间法」求结合阈值 T_bond：
   报告"结实"的段 → T_bond ≤ 该段界面温度；"易断"的段 → T_bond ≥ 界面温度。
   所有观测的交集非空 → 该组参数与观测完全一致，区间宽度即置信余量；
3. 取余量最大的参数组，T_bond 取区间中点 → 生成耗材档案。

自诊断：若所有参数组都无法一致解释观测，说明打印过程有异常（受潮、
风道改装、层间 loyalty 破坏等），如实报告而不是硬拟合。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..gcode.parser import parse_gcode
from ..thermal.materials import Material, get_material
from ..thermal.voxel import SimConfig, ThermalSimulator
from .chip import generate_chip

# 网格：围绕当前默认值的邻域
GRID_KAPPA = (0.3, 0.4, 0.5, 0.6, 0.7)
GRID_ETA = (0.15, 0.25, 0.35, 0.45)
GRID_HFAN = (0.6, 0.8, 1.0, 1.2, 1.5)


def _scaled_material(material: Material, h_fan_scale: float) -> Material:
    return replace(material, h_conv_on=material.h_conv_on * h_fan_scale)


def section_iface_medians(parsed, res, section_layers: int, n_sections: int) -> list[float]:
    """按段号分组取界面温度中位数（0 基段号）。"""
    medians: list[float] = []
    layers = parsed.layer_idx
    iface = res.iface_temp
    valid = res.tqi_valid | np.ones(len(iface), dtype=bool)  # 芯片段全部计入
    for s in range(n_sections):
        sel = (layers // section_layers == s) & valid
        medians.append(float(np.median(iface[sel])) if sel.any() else float("nan"))
    return medians


def fit_from_outcomes(
    outcomes: dict[int, str],
    material: Material,
    *,
    section_layers: int = 30,
    chamber_temp: float | None = None,
    grid: dict | None = None,
    progress_cb=None,
) -> dict:
    """outcomes: {段号(1基): "weak"|"strong"}。返回拟合报告 dict。"""
    text, manifest = generate_chip(material.name, section_layers=section_layers)
    parsed = parse_gcode(text)
    n_sections = len(manifest["sections"])

    grid = grid or {}
    kappas = grid.get("kappa", GRID_KAPPA)
    etas = grid.get("eta", GRID_ETA)
    hfans = grid.get("hfan", GRID_HFAN)

    strong_ifaces: list[float] = []
    weak_ifaces: list[float] = []
    results: list[dict] = []
    total = len(kappas) * len(etas) * len(hfans)
    done = 0

    for kappa in kappas:
        for eta in etas:
            for hfan in hfans:
                mat2 = _scaled_material(material, hfan)
                cfg = SimConfig(
                    voxel_mm=1.0, bucket_s=0.3,
                    iface_reheat=float(kappa), nozzle_heat=float(eta),
                    chamber_temp=chamber_temp,
                )
                sim = ThermalSimulator(parsed, mat2, cfg)
                res = sim.run()
                ifaces = section_iface_medians(parsed, res, section_layers, n_sections)

                # 可行区间：strong → T_bond ≤ iface；weak → T_bond ≥ iface
                lo = 0.0   # 下界来自 weak（T_bond 至少这么高）
                hi = 999.0  # 上界来自 strong（T_bond 至多这么高）
                feasible = True
                for s_idx, outcome in outcomes.items():
                    iface = ifaces[s_idx - 1]
                    if not np.isfinite(iface):
                        continue
                    if outcome == "strong":
                        hi = min(hi, iface)
                    else:
                        lo = max(lo, iface)
                if lo > hi + 1e-6:
                    feasible = False
                margin = (hi - lo) if feasible else 0.0
                pred = [("strong" if (np.isfinite(ifaces[s - 1]) and ifaces[s - 1] >= (lo + hi) / 2) else "weak")
                        for s in range(1, n_sections + 1)]
                agree = sum(1 for s in range(1, n_sections + 1)
                            if s in outcomes and pred[s - 1] == outcomes[s]) / max(len(outcomes), 1)
                results.append({
                    "kappa": float(kappa), "eta": float(eta), "hfan": float(hfan),
                    "ifaces": ifaces, "feasible": feasible, "margin": margin,
                    "t_bond": (lo + hi) / 2 if feasible else None,
                    "agreement": agree,
                })
                done += 1
                if progress_cb:
                    progress_cb(done / total)

    feasible_results = [r for r in results if r["feasible"]]
    feasible_results.sort(key=lambda r: -r["margin"])
    report: dict = {
        "n_combinations": total,
        "n_feasible": len(feasible_results),
        "sections_reported": {str(k): v for k, v in outcomes.items()},
    }

    if feasible_results:
        best = feasible_results[0]
        t_bond = round(best["t_bond"], 1)
        report.update({
            "ok": True,
            "kappa": best["kappa"],
            "eta": best["eta"],
            "hfan": best["hfan"],
            "bond_threshold": t_bond,
            "margin": round(best["margin"], 1),
            "agreement": best["agreement"],
            "section_iface": [round(v, 1) for v in best["ifaces"]],
            # 阈值映射进材料窗口：cold_below=T_bond，ideal_lo=T_bond+25
            "cold_below": t_bond,
            "ideal_lo": round(t_bond + 25, 1),
        })
        alt = [f"κ={r['kappa']} η={r['eta']} 扇×{r['hfan']}" for r in feasible_results[1:4]]
        report["alternatives"] = alt
    else:
        # 自诊断：无一致解
        results.sort(key=lambda r: -r["agreement"])
        report.update({
            "ok": False,
            "diagnosis": "所有参数组合都无法一致解释掰断结果。常见原因："
                         "耗材受潮（强度普遍偏低）、风道被改装（实际风量与预期差异过大）、"
                         "或掰断判断与段顺序相反。建议烘干耗材后重打一次再试。",
            "best_guess": results[0] if results else None,
        })
    return report
