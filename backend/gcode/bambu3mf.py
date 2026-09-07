"""Bambu Studio .gcode.3mf 容器支持。

.gcode.3mf 是 ZIP 包：
- Metadata/plate_*.gcode   真正的 G-code（可能多板，取最大的一个）
- Metadata/project_settings.config  完整切片参数（JSON），含打印机型号、
  耗材、喷嘴/热床温度、舱温字段等——用于自动匹配打印机/耗材档案。
- Metadata/plate_*.gcode.md5  G-code 的 MD5（重打包时必须同步更新）
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile


def is_zip_bytes(raw: bytes) -> bool:
    return raw[:2] == b"PK"


def _pick_gcode_member(zf: zipfile.ZipFile):
    gcodes = [i for i in zf.infolist() if i.filename.lower().endswith(".gcode")]
    if not gcodes:
        raise ValueError("3MF 容器内未找到 G-code（.gcode）成员——请确认这是切片输出的 .gcode.3mf")
    return max(gcodes, key=lambda i: i.file_size)


def extract_3mf(raw: bytes) -> tuple[str, dict]:
    """返回 (gcode_text, 项目设置摘要)。非 3mf 或缺少 G-code 时抛 ValueError。"""
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ValueError("文件是 ZIP 容器但无法打开（可能已损坏）") from exc

    target = _pick_gcode_member(zf)
    text = zf.read(target.filename).decode("utf-8-sig", errors="replace")

    settings: dict = {}
    try:
        cfg = json.loads(zf.read("Metadata/project_settings.config").decode("utf-8-sig"))
    except (KeyError, json.JSONDecodeError):
        cfg = {}

    def first(key: str) -> str:
        v = cfg.get(key)
        if isinstance(v, list):
            v = v[0] if v else ""
        return str(v).strip() if v is not None else ""

    if first("printer_model"):
        settings["printer_model"] = first("printer_model")
    if first("printer_variant"):
        settings["printer_variant"] = first("printer_variant")
    if first("filament_settings_id"):
        settings["filament_profile"] = first("filament_settings_id")
    if first("filament_type"):
        settings["filament_type"] = first("filament_type")
    if first("nozzle_temperature"):
        settings["nozzle_temp"] = first("nozzle_temperature")
    if first("hot_plate_temp"):
        settings["bed_temp"] = first("hot_plate_temp")
    if first("chamber_temperatures"):
        try:
            settings["chamber_temp"] = float(first("chamber_temperatures"))
        except ValueError:
            pass
    if first("layer_height"):
        try:
            settings["layer_height"] = float(first("layer_height"))
        except ValueError:
            pass
    return text, settings


def repack_3mf(src_zip_bytes: bytes, new_gcode: bytes) -> bytes:
    """把优化后的 G-code 替换回原 3MF 容器（保留全部元数据，同步更新 md5）。"""
    try:
        zin = zipfile.ZipFile(io.BytesIO(src_zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("源 3MF 无法打开") from exc
    target = _pick_gcode_member(zin)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            name = item.filename
            if name == target.filename:
                zout.writestr(name, new_gcode)
            elif name.lower().endswith(".gcode.md5") and name.lower().startswith(target.filename.lower()):
                zout.writestr(name, hashlib.md5(new_gcode).hexdigest())
            else:
                zout.writestr(item, zin.read(name))
    return buf.getvalue()
