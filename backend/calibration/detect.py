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


def detect_speed_bands(parsed) -> list[dict]:
    """按挤出速度的连续段划分带（1 基带号）。返回按速度升序排序的带列表。"""
    ext = parsed.extrusion_mm3 > 0
    feed = parsed.feedrate[ext].astype(float)
    lay = parsed.layer_idx[ext]
    flow = parsed.extrusion_mm3[ext] / np.maximum(parsed.duration[ext], 1e-3)
    bands: list[dict] = []
    cur: dict | None = None
    for i in range(len(feed)):
        v = float(round(feed[i], 0))
        if cur is None or abs(v - cur["speed"]) > 1.0:
            if cur is not None:
                bands.append(cur)
            cur = {"speed": v, "count": 0, "flow_sum": 0.0,
                   "layer_from": int(lay[i]), "layer_to": int(lay[i])}
        cur["count"] += 1
        cur["flow_sum"] += float(flow[i])
        cur["layer_to"] = int(lay[i])
    if cur is not None:
        bands.append(cur)
    for b in bands:
        b["flow_mm3s"] = round(b["flow_sum"] / max(b["count"], 1), 2)
    bands.sort(key=lambda b: b["speed"])
    for n, b in enumerate(bands, 1):
        b["band"] = n
    return bands
