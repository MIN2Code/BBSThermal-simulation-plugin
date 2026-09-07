"""热质量指数（TQI）：把界面温度映射为 -100 ~ +100 的质量分。

语义对齐 Helio：
-  -100 太冷 → 层间结合弱（-50 对应约一半强度）
-     0 理想 → 强度与精度最佳
-  +100 太热 → 下垂 / 变形风险
"""
from __future__ import annotations

import numpy as np

from .materials import Material


def tqi_from_interface_temp(t_iface: np.ndarray, m: Material) -> np.ndarray:
    """分段线性映射（向量化）。

    cold_below → -100；cold_below→ideal_lo 线性升至 -5；
    [ideal_lo, ideal_hi] 平坦小 ±5 内；ideal_lo 前后线性穿越 0；
    ideal_hi→hot_above 升至 +100；hot_above 以上封顶 +100。
    """
    t = np.asarray(t_iface, dtype=np.float64)
    out = np.empty_like(t)

    cold = t <= m.cold_below
    ramp_up = (t > m.cold_below) & (t < m.ideal_lo)
    ideal = (t >= m.ideal_lo) & (t <= m.ideal_hi)
    ramp_hot = (t > m.ideal_hi) & (t < m.hot_above)
    hot = t >= m.hot_above

    out[cold] = -100.0
    out[ramp_up] = -100.0 + 95.0 * (t[ramp_up] - m.cold_below) / (m.ideal_lo - m.cold_below)
    out[ideal] = -5.0 + 10.0 * (t[ideal] - m.ideal_lo) / max(m.ideal_hi - m.ideal_lo, 1e-6)
    out[ramp_hot] = 5.0 + 95.0 * (t[ramp_hot] - m.ideal_hi) / (m.hot_above - m.ideal_hi)
    out[hot] = 100.0
    return out


def tqi_color(tqi: np.ndarray) -> np.ndarray:
    """TQI → RGB（0~1），蓝(冷)→绿(理想)→红(热)，供前端着色参考/后端出图。"""
    t = np.clip(np.asarray(tqi, dtype=np.float64), -100, 100) / 100.0
    r = np.clip(1.5 * t + 0.5, 0, 1)
    g = np.clip(1.0 - 2.0 * np.abs(t), 0, 1)
    b = np.clip(0.5 - 1.5 * t, 0, 1)
    return np.stack([r, g, b], axis=-1)
