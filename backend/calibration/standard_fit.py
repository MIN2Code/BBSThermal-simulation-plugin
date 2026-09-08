"""标准测试件的拟合：温度塔 → 结合窗口两端；VFA → 流量悬崖参数。

温度塔（每块三级评价 弱/可用/过热）：
  仿真给出每块的预测界面温度（对块内喷嘴温度敏感，经 κ 再热进入块间）。
  - 弱块：iface ≤ T_bond
  - 可用块：T_bond < iface < T_sag
  - 过热块：iface ≥ T_sag
  二维可行域 (T_bond, T_sag) 取最大内接中心。

VFA（报告出现瑕疵的起始档 + 最佳档）：
  起始档流量 = 流量悬崖 → flow_ref = 0.5×onset，flow_span = 0.6×onset
  （降额幅度沿用默认；温度塔可交叉验证）。
"""
from __future__ import annotations

import numpy as np

from ..gcode.parser import parse_gcode
from ..thermal.materials import Material, get_material
from ..thermal.voxel import SimConfig, ThermalSimulator
from .standard import generate_temp_tower, generate_vfa

_GRID = {"kappa": (0.5, 0.65, 0.8), "eta": (0.25, 0.35, 0.45)}


def tower_iface_per_block(parsed, res, manifest) -> list[float]:
    """每块（1 基）的界面温度中位数，块内按 manifest 的层号范围取。"""
    out = []
    for blk in manifest["blocks"]:
        z0, z1 = blk["z_from"], blk["z_to"]
        z = parsed.geometry[:, 5]
        sel = (z >= z0) & (z < z1) & res.tqi_valid
        # 测量层取块内后 80%（跳过块首的温度过渡层）
        idx = np.where(sel)[0]
        keep = idx[int(len(idx) * 0.2):] if len(idx) > 5 else idx
        out.append(float(np.median(res.iface_temp[keep])) if len(keep) else float("nan"))
    return out


def fit_temp_tower(outcomes: dict[int, str], material: Material, *,
                   block_layers: int = 40, progress_cb=None) -> dict:
    """outcomes: {块号(1基): "weak"|"ok"|"hot"}。返回拟合报告（含窗口参数）。"""
    text, manifest = generate_temp_tower(material.name, block_layers=block_layers)
    parsed = parse_gcode(text)
    n_blocks = len(manifest["blocks"])

    kappas = _GRID["kappa"]
    etas = _GRID["eta"]
    results = []
    done = 0
    for kappa in kappas:
        for eta in etas:
            cfg = SimConfig(voxel_mm=1.0, iface_reheat=float(kappa), nozzle_heat=float(eta))
            sim = ThermalSimulator(parsed, material, cfg)
            res = sim.run()
            ifaces = tower_iface_per_block(parsed, res, manifest)

            lo_b = 0.0     # T_bond 下界（弱块的 iface）
            hi_b = 999.0   # T_bond 上界（可用块的 iface）
            lo_s = 0.0     # T_sag 下界（可用块的 iface）
            hi_s = 999.0   # T_sag 上界（过热块的 iface）
            feasible = True
            agree = 0
            reported = 0
            for bi in range(1, n_blocks + 1):
                iface = ifaces[bi - 1]
                cat = outcomes.get(bi)
                if not cat or not np.isfinite(iface):
                    continue
                reported += 1
                if cat == "weak":
                    lo_b = max(lo_b, iface)
                elif cat == "hot":
                    hi_s = min(hi_s, iface)
                else:  # ok
                    hi_b = min(hi_b, iface)
                    lo_s = max(lo_s, iface)
            if lo_b > hi_b + 1e-6 or lo_s > hi_s + 1e-6:
                feasible = False
            margin = min(hi_b - lo_b, hi_s - lo_s) if feasible else 0.0
            # 用区间中点复评一致率
            tb, ts = (lo_b + hi_b) / 2, (lo_s + hi_s) / 2
            for bi in range(1, n_blocks + 1):
                iface = ifaces[bi - 1]
                cat = outcomes.get(bi)
                if not cat:
                    continue
                pred = "weak" if iface <= tb else ("hot" if iface >= ts else "ok")
                if pred == cat:
                    agree += 1
            results.append({
                "kappa": float(kappa), "eta": float(eta), "ifaces": ifaces,
                "feasible": feasible, "margin": margin,
                "t_bond": tb if feasible else None, "t_sag": ts if feasible else None,
                "agreement": agree / max(reported, 1),
            })
            done += 1
            if progress_cb:
                progress_cb(done / (len(kappas) * len(etas)))

    feas = [r for r in results if r["feasible"]]
    feas.sort(key=lambda r: -r["margin"])
    report: dict = {"n_combinations": len(results), "n_feasible": len(feas),
                    "sections_reported": {str(k): v for k, v in outcomes.items()}}
    if feas:
        best = feas[0]
        t_bond, t_sag = round(best["t_bond"], 1), round(best["t_sag"], 1)
        report.update({
            "ok": True, "kappa": best["kappa"], "eta": best["eta"],
            "t_bond": t_bond, "t_sag": t_sag, "margin": round(best["margin"], 1),
            "agreement": best["agreement"],
            "block_iface": [round(v, 1) for v in best["ifaces"]],
            "cold_below": t_bond, "ideal_lo": round(t_bond + 0.3 * (t_sag - t_bond), 1),
            "ideal_hi": round(t_sag - 0.3 * (t_sag - t_bond), 1), "hot_above": t_sag,
        })
    else:
        results.sort(key=lambda r: -r["agreement"])
        report.update({"ok": False,
                       "diagnosis": "温度塔观测无法与任何参数组合一致。常见原因："
                                    "块级判断与温度顺序相反、耗材受潮、或每块打印时间过短"
                                    "导致温度未稳定。"})
    return report


def fit_vfa(onset_band: int, material: Material, manifest: dict) -> dict:
    """VFA：起始瑕疵档 → 流量悬崖参数。

    onset_band 为 1 基；该档的 flow 视为质量悬崖。
    """
    band = manifest["bands"][max(0, min(onset_band - 1, len(manifest["bands"]) - 1))]
    onset_flow = float(band["flow_mm3s"])
    return {
        "ok": True,
        "onset_speed": float(band["speed"]),
        "onset_flow": onset_flow,
        "flow_ref": round(onset_flow * 0.5, 2),
        "flow_span": round(onset_flow * 0.6, 2),
        "note": "流量降额幅度（flow_derate）默认 8°C；与温度塔联用时以塔为准交叉校验",
    }
