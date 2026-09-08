"""从上传的 G-code 中自动识别校准块结构（温度块 / 速度带）。

温度塔：喷嘴温度 (nozzle_seg) 的连续同值段 = 一个块；
VFA 速度塔：挤出段的进给速度连续同值段 = 一个带。
均为自下而上（时间顺序）排列。
"""
from __future__ import annotations

import numpy as np


def detect_nozzle_blocks(parsed) -> list[dict]:
    """按喷嘴温度的连续段划分块（1 基块号，自下而上）。"""
    noz = parsed.nozzle_seg
    lay = parsed.layer_idx
    blocks: list[dict] = []
    cur: dict | None = None
    for i in range(len(noz)):
        t = float(noz[i])
        if cur is None or abs(t - cur["temp"]) > 0.5:
            if cur is not None:
                cur["layer_to"] = int(lay[i - 1]) if i > 0 else int(lay[0])
                blocks.append(cur)
            cur = {"temp": t, "layer_from": int(lay[i]), "seg_start": i}
        cur["seg_end"] = i
    if cur is not None:
        cur["layer_to"] = int(lay[-1])
        blocks.append(cur)
    for n, blk in enumerate(blocks, 1):
        blk["index"] = n
    return blocks


def detect_speed_bands(parsed, *, min_band_layers: int = 3) -> list[dict]:
    """按层的挤出主导速度聚类成带（相邻层速度差 <3% 合并）。

    适配 BS 自带 VFA/最大流速测试件：速度按段阶梯变化，墙/填充的微差
    不应碎裂分带。跨度小于 min_band_layers 的过渡带被丢弃。
    返回按速度升序的带列表（含层范围与平均流量）。
    """
    ext = parsed.extrusion_mm3 > 0
    lay = parsed.layer_idx[ext]
    feed = parsed.feedrate[ext].astype(float)
    flow = parsed.extrusion_mm3[ext] / np.maximum(parsed.duration[ext], 1e-3)

    # 每层主导速度 = 层内挤出段速度中位数
    layers = np.unique(lay)
    layer_speed = {}
    for L in layers:
        layer_speed[int(L)] = float(np.median(feed[lay == L]))

    # 相邻层速度差 <3% 合并为带
    bands: list[dict] = []
    cur: dict | None = None
    for L in sorted(layer_speed):
        v = layer_speed[L]
        if cur is None or abs(v - cur["speed"]) / max(cur["speed"], 1e-6) > 0.03:
            cur = {"speed": v, "layer_from": L, "layer_to": L, "flows": [flow[lay == L]]}
            bands.append(cur)
        else:
            cur["layer_to"] = L
            cur["flows"].append(flow[lay == L])
    for b in bands:
        b["flow_mm3s"] = round(float(np.mean(np.concatenate(b["flows"]))), 2)
        b["layers"] = f"{b['layer_from']}-{b['layer_to']}"
        b["span"] = b["layer_to"] - b["layer_from"] + 1
        del b["flows"]
    bands = [b for b in bands if b["span"] >= min_band_layers]
    bands.sort(key=lambda b: b["speed"])
    for n, b in enumerate(bands, 1):
        b["band"] = n
    return bands
