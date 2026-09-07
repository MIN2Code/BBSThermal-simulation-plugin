"""材料热参数库（近似手册值，可经 JSON 覆盖/自定义）。

参数含义：
- k          导热系数 W/(m·K)
- rho        密度 kg/m³
- cp         比热容 J/(kg·K)
- tg         玻璃化转变温度 °C（层间结合的冷端参考）
- nozzle/bed 推荐打印温度 °C
- tqi_*      界面温度 → TQI 映射窗口（见 tqi.py）：
             cold_below 之下为 -100（弱结合），[ideal_lo, ideal_hi] 为理想区，
             hot_above 之上为 +100（过热下垂风险）
- h_conv_*   表面对流系数 W/(m²·K)：自然对流 / 风扇全开
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class Material:
    name: str
    k: float
    rho: float
    cp: float
    tg: float
    nozzle: float
    bed: float
    cold_below: float
    ideal_lo: float
    ideal_hi: float
    hot_above: float
    h_conv_off: float = 12.0     # 自然对流
    h_conv_on: float = 180.0     # 部分冷却风扇全开
    h_bed: float = 600.0         # 与热床的接触换热（第一层底面）

    @property
    def alpha(self) -> float:
        """热扩散率 mm²/s = k/(rho·cp) × 1e6。"""
        return self.k / (self.rho * self.cp) * 1e6


_LIBRARY: dict[str, Material] = {
    m.name: m
    for m in [
        # tg 以下约 15°C 仍偏弱；ideal 窗口取「充分高于 tg 且远离喷嘴温度」的经验带
        Material("PLA",  k=0.13,  rho=1240, cp=2100, tg=60,
                 nozzle=210, bed=55,
                 cold_below=72, ideal_lo=95, ideal_hi=150, hot_above=180),
        Material("PETG", k=0.15,  rho=1270, cp=1900, tg=80,
                 nozzle=240, bed=75,
                 cold_below=95, ideal_lo=120, ideal_hi=185, hot_above=215),
        Material("ABS",  k=0.10,  rho=1050, cp=1900, tg=105,
                 nozzle=250, bed=95,
                 cold_below=120, ideal_lo=150, ideal_hi=215, hot_above=240),
        Material("ASA",  k=0.10,  rho=1070, cp=1900, tg=100,
                 nozzle=250, bed=95,
                 cold_below=115, ideal_lo=145, ideal_hi=210, hot_above=235),
        Material("TPU",  k=0.13,  rho=1200, cp=1800, tg=-30,
                 nozzle=225, bed=45,
                 # TPU 无明显冷结合问题，窗口整体放宽松
                 cold_below=60, ideal_lo=90, ideal_hi=180, hot_above=215),
        Material("PC",   k=0.20,  rho=1200, cp=1250, tg=145,
                 nozzle=270, bed=105,
                 cold_below=165, ideal_lo=195, ideal_hi=245, hot_above=262),
        Material("PA",   k=0.18,  rho=1140, cp=1700, tg=55,  # PA6 半结晶，tg 取参考值
                 nozzle=280, bed=100,
                 cold_below=120, ideal_lo=170, ideal_hi=245, hot_above=268),
    ]
}


def get_material(name: str) -> Material:
    return _LIBRARY.get(name.upper().strip(), _LIBRARY["PLA"])


def list_materials() -> list[str]:
    return list(_LIBRARY.keys())


def load_user_materials(path: str) -> None:
    """从 JSON 加载用户自定义材料（合并进库，同名覆盖）。"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("materials", []):
        base = asdict(get_material(item.get("base", "PLA")))
        base.update({k: v for k, v in item.items() if k in base})
        m = Material(**base)
        _LIBRARY[m.name] = m
