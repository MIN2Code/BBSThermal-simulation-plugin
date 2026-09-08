"""敏感性诊断：真仿真对速度的响应到底有多大？卡在哪一环？

输出逐级对比：层时(时间轴) → 界面温度 → TQI，全局 f ∈ {0.6, 0.8, 1.0, 1.5}
+ 大反差交替块(0.6/1.6)。若 iface 不动 → 仿真内部问题（κ 再热/重计时）；
若 iface 动了但 TQI 不动 → 统计/映射问题。
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
cfg = SimConfig(chamber_temp=None, iface_reheat=0.8, nozzle_heat=0.35)
L = parsed.layer_idx

def run(feed):
    _retime(parsed, feed)
    res = ThermalSimulator(parsed, mat, cfg).run()
    v = res.tqi_valid
    return (float(np.mean(res.tqi[v])), float(np.mean(res.iface_temp[v])),
            float(parsed.t_mid[-1] + parsed.duration[-1]))

print(f'{"模式":<18} {"总时长s":>8} {"mean iface":>10} {"mean TQI":>9}')
for f in (0.6, 0.8, 1.0, 1.5):
    tqi, iface, dur = run(ORIG_FEED * f)
    print(f'{"全局 x"+str(f):<18} {dur:>8.0f} {iface:>10.1f} {tqi:>+9.1f}')

NL = parsed.num_layers
Lrng = np.arange(NL)
alt = np.where((Lrng // 25) % 2 == 0, 0.6, 1.6)
tqi, iface, dur = run(ORIG_FEED * alt[L])
print(f'{"交替块 0.6/1.6":<18} {dur:>8.0f} {iface:>10.1f} {tqi:>+9.1f}')

# 逐层 iface 空间相关性：交替块下相邻块_iface 应有阶跃
_retime(parsed, ORIG_FEED * alt[L])
res = ThermalSimulator(parsed, mat, cfg).run()
v = res.tqi_valid
layer_iface = np.array([res.iface_temp[(L == k) & v].mean()
                        if ((L == k) & v).any() else np.nan for k in range(NL)])
a = np.nanmean([layer_iface[k] for k in range(0, min(NL, 200), 50) if (k // 25) % 2 == 0])
b = np.nanmean([layer_iface[k] for k in range(0, min(NL, 200), 50) if (k // 25) % 2 == 1])
print(f'\n交替块前 200 层 iface：慢块均值 {a:.1f}°C  快块均值 {b:.1f}°C  差 {b - a:+.1f}°C')
_retime(parsed, ORIG_FEED)
