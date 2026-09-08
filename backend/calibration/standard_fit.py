"""标准测试件的拟合：直接作用于上传的 G-code（自动识别块/带结构）。

温度塔（每块三级评价 弱/可用/过热）：
  仿真给出每块的预测界面温度（对块内喷嘴温度敏感，经 κ 再热进入块间）。
  - 弱块：iface ≤ T_bond
  - 可用块：T_bond < iface < T_sag
  - 过热块：iface ≥ T_sag
  二维可行域 (T_bond, T_sag) 取最大内接中心。

VFA（报告出现瑕疵的起始速度）：
  起始档流量 = 流量悬崖 → flow_ref = 0.5×onset，flow_span = 0.6×onset
  （降额幅度沿用默认；温度塔可交叉验证）。
"""
from __future__ import annotations

import numpy as np

from .detect import detect_nozzle_blocks, detect_speed_bands


def block_iface_medians(parsed, res, blocks) -> list[float]:
    """每块的界面温度中位数：块内后 80% 的段（跳过块首温度过渡）。"""
    out = []
    for blk in blocks:
        idx = np.arange(blk["seg_start"], blk["seg_end"] + 1)
        keep = idx[int(len(idx) * 0.2):] if len(idx) > 5 else idx
        vals = res.iface_temp[keep]
        vals = vals[np.isfinite(vals)]
        out.append(float(np.median(vals)) if len(vals) else float("nan"))
    return out


def fit_temp_tower_gcode(parsed, res, outcomes: dict[int, str]) -> dict:
    """outcomes: {块号(1基): "weak"|"ok"|"hot"}。返回拟合报告（含窗口参数）。"""
    blocks = detect_nozzle_blocks(parsed)
    ifaces = block_iface_medians(parsed, res, blocks)
    reported = [bi for bi in outcomes if bi <= len(blocks)]
    if len(reported) < 2:
        return {"ok": False, "diagnosis": "有效报告的块数不足 2 个，无法拟合窗口"}

    lo_b, hi_b = -np.inf, np.inf   # T_bond 区间（弱块 ≤ T_bond < 可用块）
    lo_s, hi_s = -np.inf, np.inf   # T_sag 区间（可用块 ≤ T_sag < 过热块）
    feasible = True
    agree = 0
    reported_n = 0
    for bi, cat in outcomes.items():
        iface = float(ifaces[bi - 1])
        if not np.isfinite(iface):
            continue
        reported_n += 1
        if cat == "weak":
            lo_b = max(lo_b, iface)
        elif cat == "hot":
            hi_s = min(hi_s, iface)
        elif cat == "ok":
            hi_b = min(hi_b, iface)
            lo_s = max(lo_s, iface)
    if lo_b > hi_b + 1e-6 or lo_s > hi_s + 1e-6:
        feasible = False
    margin = min(hi_b - lo_b, hi_s - lo_s) if feasible else 0.0

    # 用区间中点复评一致率
    t_bond = (lo_b + hi_b) / 2 if np.isfinite(hi_b) else lo_b
    t_sag = (lo_s + hi_s) / 2 if np.isfinite(hi_s) else hi_s
    for bi, cat in outcomes.items():
        iface = float(ifaces[bi - 1])
        pred = "weak" if iface <= t_bond else ("hot" if iface >= t_sag else "ok")
        if pred == cat:
            agree += 1

    report: dict = {
        "blocks": [{"index": b["index"], "temp": b["temp"], "iface": round(ifaces[b["index"] - 1], 1)}
                   for b in blocks if np.isfinite(ifaces[b["index"] - 1])],
        "reported_n": reported_n, "feasible": feasible, "margin": round(margin, 1),
        "agreement": agree / max(reported_n, 1),
    }
    if feasible:
        report.update({
            "ok": True,
            "t_bond": round(t_bond, 1), "t_sag": round(t_sag, 1),
            "margin": round(margin, 1),
            "cold_below": round(t_bond, 1),
            "ideal_lo": round(t_bond + 0.3 * (t_sag - t_bond), 1),
            "ideal_hi": round(t_sag - 0.3 * (t_sag - t_bond), 1),
            "hot_above": round(t_sag, 1),
        })
    else:
        report.update({"ok": False,
                       "diagnosis": "块级观测与任何 (T_bond, T_sag) 组合矛盾——请检查块的顺序判断"
                                    "（弱/过热是否填反）后重试。"})
    return report


def fit_vfa_onset(parsed, onset_speed: float) -> dict:
    """起始瑕疵速度 → 流量悬崖参数。"""
    bands = detect_speed_bands(parsed)
    onset = None
    for b in bands:
        if b["speed"] >= onset_speed:
            onset = b
            break
    if onset is None:
        onset = bands[-1] if bands else None
    if onset is None:
        return {"ok": False, "diagnosis": "未能从该 G-code 识别出速度带"}
    onset_flow = onset["flow_mm3s"]
    return {
        "ok": True,
        "onset_speed": float(onset["speed"]),
        "onset_flow": onset_flow,
        "flow_ref": round(onset_flow * 0.5, 2),
        "flow_span": round(onset_flow * 0.6, 2),
        "note": "流量降额幅度（flow_derate）默认 8°C；与温度塔联用时以塔为准交叉校验",
    }
