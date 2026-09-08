"""κ 敏感性权衡实验：接触再热 κ 对「绝对均值」与「速度敏感度」的权衡。

对每个 κ：跑基准 / 全局 0.6× / 全局 1.5× 三次仿真，输出
  - 基准 mean TQI（对齐目标：官方 ≈ -47.6）
  - 敏感度 Δiface(1.5× vs 0.6×)（优化器赖以工作的信号强度）
"""
import sys
sys.path.insert(0, '.')
import numpy as np

from backend.gcode.parser import parse_gcode
from backend.gcode.bambu3mf import extract_3mf
from backend.thermal.voxel import ThermalSimulator, SimConfig
from backend.thermal.materials import get_material
from backend.thermal.optimize import _retime

text, _ = extract_3mf(open(r'测试模型/佩里卡_plate_10_2.gcode.3mf', 'rb').read())
parsed = parse_gcode(text)
mat = get_material('PLA')
ORIG_FEED = parsed.feedrate.copy()

print(f'{"κ":>4} {"基准TQI":>8} {"iface(0.6×)":>11} {"iface(基准)":>11} {"iface(1.5×)":>11} {"敏感度Δ°C":>9}')
for k in (0.8, 0.6, 0.4, 0.2, 0.0):
    cfg = SimConfig(chamber_temp=None, iface_reheat=k, nozzle_heat=0.35)
    row = []
    for f in (0.6, 1.0, 1.5):
        _retime(parsed, ORIG_FEED * f)
        res = ThermalSimulator(parsed, mat, cfg).run()
        v = res.tqi_valid
        row.append((float(np.mean(res.tqi[v])), float(np.mean(res.iface_temp[v]))))
    _retime(parsed, ORIG_FEED)
    tqi_base = row[1][0]
    sens = row[2][1] - row[0][1]
    print(f'{k:>4} {tqi_base:>+8.1f} {row[0][1]:>11.1f} {row[1][1]:>11.1f} {row[2][1]:>11.1f} {sens:>+9.1f}')
