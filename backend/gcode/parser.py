"""G-code 解析器：把文本 G-code 还原为时间有序的挤出段序列。

支持 Slic3r 系（Bambu Studio / OrcaSlicer / PrusaSlicer）与 Cura 风格注释，
处理 G0/G1/G2/G3/G4/G92、M82/M83（E 模式）、M106/M107（风扇）、
;TYPE: 特性注释、;LAYER_CHANGE / ;LAYER:n 层标记（缺失时按 Z 变化兜底）。
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from .model import Feature, ParsedGcode, PrintInfo

_META_KEYS = {
    "filament_type": "filament_type",
    "filament_diameter": "filament_diameter",
    "nozzle_temperature": "nozzle_temp",
    "nozzle_temperature_initial": "nozzle_temp",
    "hot_plate_temp": "bed_temp",
    "heatbed_temp": "bed_temp",
    "bed_temp": "bed_temp",
    "layer_height": "layer_height",
    "estimated printing time (normal mode)": "est_time",
    "estimated printing time": "est_time",
    "total filament used [g]": None,
}

_FEATURE_MAP = {
    "outer wall": Feature.OUTER_WALL,
    "wall-outer": Feature.OUTER_WALL,
    "inner wall": Feature.INNER_WALL,
    "wall-inner": Feature.INNER_WALL,
    "sparse infill": Feature.SPARSE_INFILL,
    "fill": Feature.SPARSE_INFILL,
    "internal sparse infill": Feature.SPARSE_INFILL,
    "solid infill": Feature.SOLID_INFILL,
    "skin": Feature.SOLID_INFILL,
    "internal solid infill": Feature.SOLID_INFILL,
    "top solid infill": Feature.SOLID_INFILL,
    "gap infill": Feature.GAP_INFILL,
    "bridge": Feature.BRIDGE,
    "internal bridge": Feature.INTERNAL_BRIDGE,
    "support material": Feature.SUPPORT,
    "support": Feature.SUPPORT,
    "support material interface": Feature.SUPPORT_INTERFACE,
    "support interface": Feature.SUPPORT_INTERFACE,
    "skirt": Feature.SKIRT,
    "brim": Feature.BRIM,
    "prime line": Feature.PRIME_LINE,
    "purge line": Feature.PRIME_LINE,
    "custom": Feature.CUSTOM,
    "ironing": Feature.IRONING,
}

_WORD_RE = re.compile(r"([A-Za-z])\s*(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_TIME_PART_RE = re.compile(r"(\d+)\s*h|(\d+)\s*m|(\d+)\s*s")


def _parse_est_time(value: str) -> float:
    total = 0.0
    for m in _TIME_PART_RE.finditer(value):
        if m.group(1):
            total += int(m.group(1)) * 3600
        if m.group(2):
            total += int(m.group(2)) * 60
        if m.group(3):
            total += int(m.group(3))
    return total


class _State:
    """解码器运行状态。"""

    __slots__ = (
        "x", "y", "z", "e", "feed", "abs_xyz", "abs_e", "fan",
        "layer_idx", "layer_z", "t", "feature", "expect_layer_z",
        "pending_z_update", "helio_ti",
    )

    def __init__(self) -> None:
        self.x = self.y = self.z = 0.0
        self.e = 0.0
        self.feed = 1500.0 / 60.0  # mm/s，兜底初值
        self.abs_xyz = True
        self.abs_e = True
        self.fan = 0.0
        self.layer_idx = -1
        self.layer_z = -1e9
        self.t = 0.0
        self.feature = Feature.UNKNOWN
        self.expect_layer_z = False
        self.pending_z_update = False
        self.helio_ti = None


def parse_gcode(text: str) -> ParsedGcode:
    info = PrintInfo()
    st = _State()

    geo: list[tuple[float, float, float, float, float, float]] = []
    t_mid: list[float] = []
    dur: list[float] = []
    feed_list: list[float] = []
    vol_list: list[float] = []
    fan_list: list[float] = []
    layer_list: list[int] = []
    feat_list: list[Feature] = []
    helio_list: list[float] = []
    src_line_list: list[int] = []
    travel_list: list[float] = []

    layer_z_seq: list[float] = []
    layer_t0_seq: list[float] = []
    layer_t1_seq: list[float] = []
    layer_last_t: list[float] = []

    seen_gcode = False
    fil_d = info.filament_diameter
    fil_area = math.pi / 4.0 * fil_d * fil_d

    def start_layer(z: float) -> None:
        if st.layer_idx >= 0 and len(layer_t1_seq) < len(layer_z_seq):
            layer_t1_seq.append(st.t)  # 上一层的结束时刻（close 未先行时兜底）
        st.layer_idx += 1
        st.layer_z = z
        layer_z_seq.append(z)
        layer_t0_seq.append(st.t)
        layer_last_t.append(st.t)

    def close_last_layer() -> None:
        if st.layer_idx >= 0 and len(layer_t1_seq) < len(layer_z_seq):
            layer_t1_seq.append(layer_last_t[-1])

    last_end = [0.0]   # 上一挤出段的结束时刻（计算段前空走用）
    cur_line = [0]     # 当前源文件行号（G-code 回写用）

    def emit(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float,
             de: float, feed: float) -> None:
        if st.layer_idx < 0:
            start_layer(z1)
        dist = math.hypot(x1 - x0, y1 - y0)
        move_t = dist / feed if feed > 0 else 0.0
        travel = st.t - last_end[0]  # 段前空走/驻留时间
        st.t += move_t
        last_end[0] = st.t
        vol = de * fil_area
        geo.append((x0, y0, z0, x1, y1, z1))
        t_mid.append(st.t - move_t * 0.5)
        dur.append(move_t)
        feed_list.append(feed)
        vol_list.append(vol)
        fan_list.append(st.fan)
        layer_list.append(st.layer_idx)
        feat_list.append(st.feature)
        helio_list.append(st.helio_ti if st.helio_ti is not None else float("nan"))
        src_line_list.append(cur_line[0])
        travel_list.append(max(travel, 0.0))
        if st.layer_idx >= len(layer_last_t):
            # 兜底：段所属层未被记录（理论上不应发生）
            while len(layer_last_t) <= st.layer_idx:
                layer_last_t.append(st.t)
        layer_last_t[st.layer_idx] = st.t

    def arc_points(cx: float, cy: float, x1: float, y1: float, cw: bool,
                   r: float | None) -> list[tuple[float, float]]:
        # 返回从当前位置到 (x1,y1) 的圆弧插值点（不含起点，含终点）
        x0, y0 = st.x, st.y
        if r is not None:
            # R 形式：由弦长与半径求圆心（两解，取劣弧侧）
            dx, dy = x1 - x0, y1 - y0
            d = math.hypot(dx, dy)
            if d < 1e-9:
                return [(x1, y1)]
            h = math.sqrt(max(r * r - d * d / 4.0, 0.0))
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ux, uy = -dy / d, dx / d
            if cw:
                h = -h
            cx, cy = mx + ux * h, my + uy * h
        a0 = math.atan2(y0 - cy, x0 - cx)
        a1 = math.atan2(y1 - cy, x1 - cx)
        if cw:
            while a1 >= a0 - 1e-9:
                a1 -= 2 * math.pi
        else:
            while a1 <= a0 + 1e-9:
                a1 += 2 * math.pi
        sweep = abs(a1 - a0)
        radius = math.hypot(x0 - cx, y0 - cy)
        n = max(2, int(math.ceil(sweep / (2 * math.acos(max(0.0, 1 - 0.05 / max(radius, 0.05)))))))
        pts = []
        for i in range(1, n + 1):
            a = a0 + (a1 - a0) * i / n
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        return pts

    for raw_line_no, raw in enumerate(text.splitlines(), 1):
        cur_line[0] = raw_line_no
        line = raw.strip()
        if not line or len(line) > 2000:  # 超长行=二进制杂质（如误喂 ZIP）
            continue
        st.helio_ti = None  # 每行重置，Helio 标注只属于本行指令
        # 行内注释剥离（如 G1 X.. E.. ;helioadditive=(...)）
        trailing = None
        ci = line.find(";")
        if ci >= 0:
            trailing = line[ci + 1:].strip()
            line = line[:ci].strip()
        if not line:
            # 纯注释行
            body = trailing or ""
            low = body.lower()
        else:
            # 代码行：捕获 Helio 逐段热指数标注
            if trailing and "helioadditive" in trailing:
                m_ti = re.search(r"ti\.mean=(-?\d*\.?\d+)", trailing)
                st.helio_ti = float(m_ti.group(1)) if m_ti else None
            if trailing and trailing.lower().startswith("type:"):
                feat = _FEATURE_MAP.get(trailing[5:].strip().lower())
                if feat is not None:
                    st.feature = feat
                    info.has_feature_comments = True
            body = None
            low = ""

        if low.startswith("type:"):
            feat = _FEATURE_MAP.get(body[5:].strip().lower())
            if feat is not None:
                st.feature = feat
                info.has_feature_comments = True
            continue
        if low.startswith("layer_change"):
            # 层高由紧随其后的 ;Z: 注释给出（该注释出现在 Z 升移之前）
            st.expect_layer_z = True
            continue
        if low.startswith("z:"):
            try:
                z = float(body[2:].strip())
            except ValueError:
                continue
            if st.expect_layer_z:
                st.expect_layer_z = False
                start_layer(z)
            elif st.layer_idx >= 0:
                st.layer_z = z
            continue
        if low.startswith("layer:"):
            try:
                n = int(body[6:].strip())
            except ValueError:
                continue
            if n != st.layer_idx:
                close_last_layer()
                st.layer_idx = n
                while len(layer_z_seq) <= n:
                    layer_z_seq.append(st.z)
                    layer_t0_seq.append(st.t)
                    layer_last_t.append(st.t)
                st.layer_z = st.z
                st.pending_z_update = True  # Cura：层号先到，Z 稍后才升
            continue
        if body is not None and not seen_gcode:
            low_b = body.lower()
            if low_b.startswith("model printing time"):
                # Bambu 头：'; model printing time: 2h 10m 13s; total estimated time: ...'
                seg = body.split(";")[0].split(":", 1)[-1]
                info.est_print_time_s = _parse_est_time(seg)
                continue
        if body is not None and not seen_gcode and "=" in body:
            key, _, val = body.partition("=")
            key_l = key.strip().lower()
            val_s = val.strip().strip('"')
            if key_l == "filament_type":
                info.detected_material = val_s.split(",")[0].strip().upper()
            elif key_l == "filament_diameter":
                try:
                    info.filament_diameter = float(val_s.split(",")[0])
                    fil_d = info.filament_diameter
                    fil_area = math.pi / 4.0 * fil_d * fil_d
                except ValueError:
                    pass
            elif key_l == "nozzle_temperature":
                try:
                    info.nozzle_temp = float(val_s.split(",")[0])
                except ValueError:
                    pass
            elif key_l in ("hot_plate_temp", "heatbed_temp", "bed_temp"):
                try:
                    info.bed_temp = float(val_s.split(",")[0])
                except ValueError:
                    pass
            elif key_l == "layer_height":
                try:
                    info.layer_height = float(val_s)
                except ValueError:
                    pass
            elif key_l.startswith("estimated printing time"):
                info.est_print_time_s = _parse_est_time(val_s)
            elif key_l == "slicer":
                info.slicer = val_s
            continue
        if body is not None:
            # 其余纯注释行忽略
            continue

        seen_gcode = True
        words: dict[str, float] = {}
        ok = True
        first_tok = True
        for tok in line.split():
            key = tok[0].upper()
            if not ("A" <= key <= "Z"):
                continue  # 非字母开头 = 二进制杂质/异常 token，忽略
            try:
                words[key] = float(tok[1:])
            except ValueError:
                m = _WORD_RE.match(tok)
                if m:
                    words[m.group(1).upper()] = float(m.group(2))
                elif first_tok:
                    ok = False  # 命令词本身无法解析 → 放弃本行
                    break
                else:
                    # 无参数值的旗标参数（如 M1006 W / M620 M）：不废掉整行
                    words.setdefault(key, 0.0)
            first_tok = False
        if not ok or not words:
            continue

        cmd = words.get("G") if "G" in words else (words.get("M") if "M" in words else None)
        if cmd is None:
            continue

        if cmd == 0 or cmd == 1:  # G0/G1 直线移动
            nx = st.x + (words["X"] - st.x) if ("X" in words and not st.abs_xyz) else words.get("X", st.x)
            ny = st.y + (words["Y"] - st.y) if ("Y" in words and not st.abs_xyz) else words.get("Y", st.y)
            nz = st.z + (words["Z"] - st.z) if ("Z" in words and not st.abs_xyz) else words.get("Z", st.z)
            if "F" in words:
                st.feed = max(words["F"] / 60.0, 0.1)
            if "E" in words:
                ne = words["E"] if st.abs_e else st.e + words["E"]
                de = ne - st.e
                st.e = ne
            else:
                de = 0.0
            dist_xy = math.hypot(nx - st.x, ny - st.y)
            z_move = abs(nz - st.z) > 1e-6
            if de > 1e-8 and dist_xy > 1e-6:
                # 挤出段：处理层归属
                if st.pending_z_update and abs(nz - st.layer_z) > 1e-4:
                    # 已由 ;LAYER:n 开层，此处用实际挤出高度修正该层 Z
                    st.layer_z = nz
                    layer_z_seq[st.layer_idx] = nz
                    st.pending_z_update = False
                elif nz - st.layer_z > max(0.02, info.layer_height * 0.6):
                    st.expect_layer_z = False
                    close_last_layer()
                    start_layer(nz)
                emit(st.x, st.y, st.z, nx, ny, nz, de, st.feed)
            elif de > 1e-8 and dist_xy <= 1e-6 and abs(nz - st.z) > 1e-6:
                # 螺旋升程（无 XY 的 E+Z 少见）：按时间消耗处理，不产生段
                st.t += abs(nz - st.z) / st.feed
            else:
                # 空走 / 抽回 / 预压：只计时间
                dist3 = math.sqrt(dist_xy * dist_xy + (nz - st.z) ** 2)
                st.t += dist3 / st.feed if st.feed > 0 else 0.0
            st.x, st.y, st.z = nx, ny, nz

        elif cmd == 2 or cmd == 3:  # G2 顺弧 / G3 逆弧
            nx = words.get("X", st.x)
            ny = words.get("Y", st.y)
            nz = words.get("Z", st.z)
            if "F" in words:
                st.feed = max(words["F"] / 60.0, 0.1)
            if "E" in words:
                ne = words["E"] if st.abs_e else st.e + words["E"]
                de = ne - st.e
                st.e = ne
            else:
                de = 0.0
            cw = cmd == 2
            pts = arc_points(
                st.x + words.get("I", 0.0), st.y + words.get("J", 0.0),
                nx, ny, cw, words.get("R"),
            )
            total_len = 0.0
            seg_pts = [(st.x, st.y)] + pts
            for (ax, ay), (bx, by) in zip(seg_pts, seg_pts[1:]):
                total_len += math.hypot(bx - ax, by - ay)
            st.t += total_len / st.feed if st.feed > 0 else 0.0
            if de > 1e-8 and total_len > 1e-6:
                if nz - st.layer_z > max(0.02, info.layer_height * 0.6):
                    close_last_layer()
                    start_layer(nz)
                per_de = de / len(pts)
                for (ax, ay), (bx, by) in zip(seg_pts, seg_pts[1:]):
                    emit(ax, ay, st.z, bx, by, st.z, per_de, st.feed)
            st.x, st.y, st.z = nx, ny, nz

        elif cmd == 4:  # G4 驻留
            st.t += words.get("S", words.get("P", 0.0) / 1000.0)

        elif cmd == 92:  # G92 设定坐标
            if "X" in words:
                st.x = words["X"]
            if "Y" in words:
                st.y = words["Y"]
            if "Z" in words:
                st.z = words["Z"]
            if "E" in words:
                st.e = words["E"]

        elif cmd in (90, 91):
            st.abs_xyz = cmd == 90

        elif cmd in (82, 83):
            st.abs_e = cmd == 82

        elif cmd == 106:  # M106 风扇开
            # Bambu: P1=部分冷却风扇（缺省），P2=辅助风扇，P3=排气风扇——后两者不计入
            fan_idx = int(words.get("P", 1))
            if fan_idx == 1:
                if "S" in words:
                    st.fan = min(max(words["S"], 0.0), 255.0) / 255.0
                else:
                    st.fan = 1.0
        elif cmd == 107:  # M107 风扇关
            fan_idx = int(words.get("P", 1))
            if fan_idx == 1:
                st.fan = 0.0
        elif cmd == 622:  # Bambu/Marlin 层开始标记（M622 J<层号>）
            st.expect_layer_z = True
        elif cmd == 623:  # 层结束标记
            close_last_layer()

    close_last_layer()

    if not geo:
        raise ValueError("未能从 G-code 中解析出任何挤出段：请确认这是切片后的 G-code 文件。")

    geometry = np.asarray(geo, dtype=np.float32)
    bbox_min = geometry[:, :3].min(axis=0)
    bbox_max = geometry[:, 3:].max(axis=0)

    layer_z_arr = np.asarray(layer_z_seq, dtype=np.float32)
    layer_t0_arr = np.asarray(layer_t0_seq, dtype=np.float32)
    if len(layer_t1_seq) < len(layer_z_seq):
        layer_t1_arr = np.concatenate(
            [np.asarray(layer_t1_seq, dtype=np.float32),
             np.full(len(layer_z_seq) - len(layer_t1_seq), st.t, dtype=np.float32)]
        )
    else:
        layer_t1_arr = np.asarray(layer_t1_seq[: len(layer_z_seq)], dtype=np.float32)

    helio_arr = np.asarray(helio_list, dtype=np.float32)
    if not np.isfinite(helio_arr).any():
        helio_arr = None

    return ParsedGcode(        info=info,
        geometry=geometry,
        t_mid=np.asarray(t_mid, dtype=np.float32),
        duration=np.asarray(dur, dtype=np.float32),
        feedrate=np.asarray(feed_list, dtype=np.float32),
        extrusion_mm3=np.asarray(vol_list, dtype=np.float32),
        fan=np.asarray(fan_list, dtype=np.float32),
        layer_idx=np.asarray(layer_list, dtype=np.int32),
        feature_id=np.asarray([int(f) for f in feat_list], dtype=np.uint8),
        layer_z=layer_z_arr,
        layer_t0=layer_t0_arr,
        layer_t1=layer_t1_arr,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        helio_ti=helio_arr,
        src_line=np.asarray(src_line_list, dtype=np.int32),
        travel_before=np.asarray(travel_list, dtype=np.float32),
    )


def parse_gcode_file(path: str) -> ParsedGcode:
    # Bambu/Orca 输出为 UTF-8（可能带 BOM），容错读取
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return parse_gcode(f.read())
