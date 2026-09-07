"""耗材档案：校准拟合结果的存储与应用。

档案按材料名保存（backend/profiles/<材料>.json），内容为拟合出的
机器/耗材耦合参数。仿真与优化时通过 apply_profile 自动套用。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace

PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles")

# 档案里可覆盖到 SimConfig 的键
_CFG_KEYS = ("iface_reheat", "nozzle_heat")


def profile_path(name: str) -> str:
    safe = "".join(c for c in name.upper() if c.isalnum() or c in "-_@ ")
    return os.path.join(PROFILES_DIR, f"{safe.strip() or 'DEFAULT'}.json")


def save_profile(profile: dict) -> str:
    os.makedirs(PROFILES_DIR, exist_ok=True)
    profile = dict(profile)
    profile["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = profile_path(profile.get("material", "DEFAULT"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return path


def load_profile(name: str) -> dict | None:
    path = profile_path(name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_profiles() -> list[dict]:
    out = []
    if not os.path.isdir(PROFILES_DIR):
        return out
    for fn in sorted(os.listdir(PROFILES_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(PROFILES_DIR, fn), "r", encoding="utf-8") as f:
                p = json.load(f)
            out.append({
                "material": p.get("material", fn[:-5]),
                "saved_at": p.get("saved_at"),
                "kappa": p.get("iface_reheat"),
                "eta": p.get("nozzle_heat"),
                "hfan": p.get("hfan"),
                "cold_below": p.get("cold_below"),
                "bond_threshold": p.get("bond_threshold"),
                "agreement": p.get("agreement"),
            })
        except Exception:  # noqa: BLE001
            continue
    return out


def apply_profile(material, cfg, profile: dict | None):
    """把档案参数套用到 (Material, SimConfig)，返回 (material2, cfg2)。

    档案键：iface_reheat/nozzle_heat → SimConfig；hfan → h_conv_on 缩放；
    cold_below/ideal_lo → 材料结合窗口。
    """
    if not profile:
        return material, cfg
    material = replace(material,
                       h_conv_on=material.h_conv_on * float(profile.get("hfan", 1.0)),
                       cold_below=float(profile.get("cold_below", material.cold_below)),
                       ideal_lo=float(profile.get("ideal_lo", material.ideal_lo)))
    cfg = replace(cfg,
                  iface_reheat=float(profile.get("iface_reheat", cfg.iface_reheat)),
                  nozzle_heat=float(profile.get("nozzle_heat", cfg.nozzle_heat)))
    return material, cfg
