"""层时保持项验证：交替块模式下慢块/快块的层均 T_eff 差。

判据：官方截面标定的层时敏感度 ≈2.2°C 等效温度/ln(倍层时)。
交替块 0.6/1.6 的层时比 ≈1.6/0.6 → 官方量级的块差应 ≈ 2.2×ln(1.6/0.6) ≈ 2.2°C。
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
L = parsed.layer_idx
NL = parsed.num_layers
Lrng = np.arange(NL)
alt = np.where((Lrng // 25) % 2 == 0, 0.6, 1.6)

print(f'{"gain":>5} {"慢块T_eff":>10} {"快块T_eff":>10} {"块差°C":>7}  (官方目标 ≈ +2.2)')
for gain in (0.0, 1.5):
    cfg = SimConfig(chamber_temp=None, iface_reheat=0.8, nozzle_heat=0.35,
                    layer_time_gain=gain)
    _retime(parsed, ORIG_FEED * alt[L])
    res = ThermalSimulator(parsed, mat, cfg).run()
    v = res.tqi_valid
    k = L[v]
    iface = res.iface_temp[v]
    slow = (k // 25) % 2 == 0
    lt = (parsed.layer_t1.astype(float) - parsed.layer_t0.astype(float))[k]
    # 慢块 = 层时长的块（factor 0.6）
    a, b = float(iface[slow].mean()), float(iface[~slow].mean())
    print(f'{gain:>5} {a:>10.1f} {b:>10.1f} {b - a:>+7.2f}')

_retime(parsed, ORIG_FEED)
