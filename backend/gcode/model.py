"""G-code 解析结果的数据结构。

核心思想：把 G-code 还原成按时间排序的挤出段（Segment）序列，
每段记录几何、时间、风扇、特性类型，供 3D 渲染与热仿真共同消费。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Feature(IntEnum):
    """刀路特性类型（归一化自各家切片器的 ;TYPE: 注释）。"""

    UNKNOWN = 0
    SKIRT = 1
    PRIME_LINE = 2
    PURGE = 3
    BRIM = 4
    SUPPORT = 5
    SUPPORT_INTERFACE = 6
    INNER_WALL = 7
    OUTER_WALL = 8
    GAP_INFILL = 9
    SOLID_INFILL = 10
    SPARSE_INFILL = 11
    BRIDGE = 12
    INTERNAL_BRIDGE = 13
    IRONING = 14
    CUSTOM = 15


# 参与热仿真与 TQI 统计的特性（排除裙边/支撑等辅助结构）
TQI_FEATURES = {
    Feature.INNER_WALL,
    Feature.OUTER_WALL,
    Feature.GAP_INFILL,
    Feature.SOLID_INFILL,
    Feature.SPARSE_INFILL,
    Feature.BRIDGE,
    Feature.INTERNAL_BRIDGE,
    Feature.IRONING,
}


@dataclass
class PrintInfo:
    """打印件元信息（解析头部得出）。"""

    detected_material: str = "PLA"
    nozzle_temp: float = 210.0
    bed_temp: float = 55.0
    chamber_temp: float | None = None
    filament_diameter: float = 1.75
    layer_height: float = 0.2
    slicer: str = ""
    est_print_time_s: float = 0.0
    has_feature_comments: bool = False  # 文件是否带 ;TYPE: 特性注释
    printer_model: str = ""  # 来自 3MF 工程配置（如 Bambu Lab P2S）
    printer_variant: str = ""


@dataclass
class ParsedGcode:
    """一次解析的完整产出。"""

    info: PrintInfo
    # 每段 6 个 float32: x0,y0,z0,x1,y1,z1（毫米，绝对坐标）
    geometry: np.ndarray  # (N, 6) float32
    # 每段附属标量
    t_mid: np.ndarray  # (N,) float32 段中点时刻（秒）
    duration: np.ndarray  # (N,) float32 段耗时（秒）
    feedrate: np.ndarray  # (N,) float32 mm/s
    extrusion_mm3: np.ndarray  # (N,) float32 挤出体积 mm³
    fan: np.ndarray  # (N,) float32 0~1 部分冷却风扇
    layer_idx: np.ndarray  # (N,) int32 层号
    feature_id: np.ndarray  # (N,) uint8 Feature
    layer_z: np.ndarray  # (L,) float32 每层 Z 高度
    layer_t0: np.ndarray  # (L,) float32 每层起始时刻
    layer_t1: np.ndarray  # (L,) float32 每层结束时刻
    bbox_min: np.ndarray  # (3,) float32
    bbox_max: np.ndarray  # (3,) float32
    helio_ti: np.ndarray | None = None  # (N,) float32 Helio 逐段热指数（无标注为 NaN）
    src_line: np.ndarray | None = None  # (N,) int32 每段来源行号（G-code 回写用）
    travel_before: np.ndarray | None = None  # (N,) float32 段前空走/驻留时间（重计时用）

    @property
    def num_segments(self) -> int:
        return int(self.geometry.shape[0])

    @property
    def num_layers(self) -> int:
        return int(self.layer_z.shape[0])

    def summary(self) -> dict:
        ext = float(self.extrusion_mm3.sum())
        return {
            "segments": self.num_segments,
            "layers": self.num_layers,
            "print_time_s": float(self.t_mid[-1] + self.duration[-1]) if self.num_segments else 0.0,
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
            "extrusion_mm3": ext,
            "material": self.info.detected_material,
            "nozzle_temp": self.info.nozzle_temp,
            "bed_temp": self.info.bed_temp,
            "layer_height": self.info.layer_height,
            "slicer": self.info.slicer,
            "printer_model": self.info.printer_model,
            "has_type_comments": self.info.has_feature_comments,
        }
