"""层集总热代理（LTLM）可行性验证 v2。

模型：层 i+1 覆盖时界面温度 = T_env + (T_dep-T_env)·exp(-冷却时长/τ_i)，
冷却时长 = 层 i+1 开始 − 层 i 沉积时刻（取层 i 时间中点）。
验证：逐层速度因子模式（正弦波 / 交替块）下，代理预测 vs 真重仿真。
"""
import sys
sys.path.insert(0, '.')
import numpy as np

from backend.gcode.parser import parse_gcode
from backend.gcode.bambu3mf import extract_3mf
from backend.thermal.voxel import ThermalSimulator, SimConfig
from backend.thermal.materials import get_material
from backend.thermal.tqi import tqi_from_interface_temp
from backend.thermal.optimize import _retime, _smooth_factor

text, _ = extract_3mf(open(r'测试模型/佩里卡_plate_10_2.gcode.3mf', 'rb').read())
parsed = parse_gcode(text)
mat = get_material('PLA')
ORIG_FEED = parsed.feedrate.copy()

cfg = SimConfig(chamber_temp=None, iface_reheat=0.8, nozzle_heat=0.35)
res0 = ThermalSimulator(parsed, mat, cfg).run()
T_ENV = float(cfg.chamber_temp)
t0, t1 = parsed.layer_t0.astype(float).copy(), parsed.layer_t1.astype(float).copy()
tmid = 0.5 * (t0 + t1)
L = parsed.layer_idx
valid = res0.tqi_valid
NL = len(t0)

iface_layer = np.array([
    float(res0.iface_temp[(L == k) & valid].mean())
    if ((L == k) & valid).any() else np.nan for k in range(NL)])

cool = t0[1:] - tmid[:-1]
iface_next = iface_layer[1:]
ok = np.isfinite(iface_next) & (cool > 1.0) & (iface_next > T_ENV + 1.5)
print(f'可标定层: {ok.sum()}/{NL-1}  舱温={T_ENV:.0f}°C  冷却时长中位={np.median(cool):.1f}s')

best = None
for T_dep in np.arange(120.0, 216.0, 2.0):
    ratio = (T_dep - T_ENV) / np.maximum(iface_next[ok] - T_ENV, 1e-3)
    if np.any(ratio <= 1.01):
        continue
    tau = cool[ok] / np.log(ratio)
    tau = tau[(tau > 0.5) & (tau < 3000)]
    if len(tau) < NL // 4:
        continue
    lt = np.log(tau)
    tau_med = float(np.exp(np.median(lt)))
    pred = T_ENV + (T_dep - T_ENV) * np.exp(-cool[ok] / tau_med)
    rmse = float(np.sqrt(np.mean((pred - iface_next[ok]) ** 2)))
    score = float(lt.std()) + rmse / 10.0
    if best is None or score < best[0]:
        best = (score, T_dep, tau_med, float(lt.std()), rmse)
_, T_DEP, TAU_MED, SPREAD, RMSE = best
print(f'拟合: T_dep={T_DEP:.0f}°C  中位τ={TAU_MED:.1f}s  τ对数σ={SPREAD:.2f}  中位τ预测RMSE={RMSE:.1f}°C')

ratio_l = (T_DEP - T_ENV) / np.maximum(iface_next - T_ENV, 1e-3)
tau_l = np.where(ratio_l > 1.01, cool / np.log(np.maximum(ratio_l, 1.01)), np.nan)
tau_l = np.where(np.isfinite(tau_l) & (tau_l > 0.5) & (tau_l < 3000), tau_l, TAU_MED)
tau_s = _smooth_factor(tau_l, nodes=11)

def new_axis(factors):
    """逐层速度因子 → 近似新时间轴（层时长按 1/f 缩放累积，层间空走同比例）。"""
    dur = (t1 - t0) / factors
    gaps = np.maximum(np.diff(t0) - (t1[:-1] - t0[:-1]), 0.0) / factors[1:]
    nt0 = np.concatenate([[0.0], np.cumsum(dur[:-1] + gaps)])
    return nt0, nt0 + dur

def proxy_mean_tqi(factors):
    nt0, nt1 = new_axis(factors)
    ncool = nt0[1:] - (0.5 * (nt0[:-1] + nt1[:-1]))
    niface = T_ENV + (T_DEP - T_ENV) * np.exp(-np.maximum(ncool, 0.0) / tau_s)
    return float(np.mean(tqi_from_interface_temp(niface, mat)))

def true_mean_tqi(factors):
    _retime(parsed, ORIG_FEED * factors[L])
    res = ThermalSimulator(parsed, mat, cfg).run()
    return float(np.mean(res.tqi[res.tqi_valid]))

def report(name, factors):
    t = true_mean_tqi(factors)
    p = proxy_mean_tqi(factors)
    print(f'{name:<22} f∈[{factors.min():.2f},{factors.max():.2f}] | 代理 {p:+7.1f} | 真 {t:+7.1f} | 误差 {abs(p - t):5.1f}')

Lrng = np.arange(NL, dtype=float)
report('基准(全1)', np.ones(NL))
report('正弦波 ±15%', 1.0 + 0.15 * np.sin(Lrng / 25.0))
report('正弦波 ±30%', 1.0 + 0.30 * np.sin(Lrng / 40.0))
report('交替块 0.8/1.25', np.where((Lrng // 40) % 2 == 0, 0.8, 1.25))
report('前慢后快', np.where(Lrng < NL / 2, 0.85, 1.2))

_retime(parsed, ORIG_FEED)
