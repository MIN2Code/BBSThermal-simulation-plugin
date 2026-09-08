"""从 Helio 官方优化后 G-code 的 ti 标注提取官方引擎的速度敏感度。

问题：我们的体素仿真对速度几乎不敏感（2.5 倍速差 → iface 仅 0.8°C），
而优化器的全部价值建立在"速度影响界面热历史"上。官方自己的 ti 标注
直接编码了它在优化后速度下的热判定——分析 ti 与当地速度/层时的截面
关系，得到官方的真实敏感度 d(ti)/d(log F)，作为我们模型的校准目标。
"""
import sys, re, zipfile
import numpy as np

path = r'测试模型/佩里卡_plate_10_2.gcode.3mf'
z = zipfile.ZipFile(path)
name = [n for n in z.namelist() if n.endswith('.gcode')][0]

pat_ti = re.compile(r'ti\.mean=([-+0-9.eE]+).*element\.index=(\d+)')
pat_f = re.compile(r'(?:^|\s)F([-+0-9.]+)')

rows = []           # (z_layer, element, ti, F, seg_len)
x = y = zcur = 0.0
cur_f = 0.0
EX = EY = None
prev = None
with z.open(name) as f:
    for raw in f:
        line = raw.decode('utf-8', 'replace')
        if not line or line[0] == ';':
            continue
        # 位置/速度跟踪
        m = pat_f.search(line)
        parts = line.split()
        cmd = parts[0] if parts else ''
        px = py = pz = None
        for t in parts[1:]:
            try:
                if t[0] == 'X':
                    px = float(t[1:])
                elif t[0] == 'Y':
                    py = float(t[1:])
                elif t[0] == 'Z':
                    pz = float(t[1:])
            except ValueError:
                break
        if cmd == 'G0' or cmd == 'G1':
            if m:
                cur_f = float(m.group(1))
            mti = pat_ti.search(line)
            has_e = any(t[0] == 'E' for t in parts[1:])
            if mti and has_e and px is not None and py is not None and prev is not None:
                seg_len = float(np.hypot(px - prev[0], py - prev[1]))
                if seg_len > 0.05:
                    rows.append((zcur, int(mti.group(2)), float(mti.group(1)), cur_f, seg_len))
            if px is not None or py is not None or pz is not None:
                prev = ((px if px is not None else (prev[0] if prev else 0.0)),
                        (py if py is not None else (prev[1] if prev else 0.0)))
        if pz is not None and abs(pz - zcur) > 1e-6:
            zcur = pz
            prev = None

arr = np.array(rows)
print(f'标注行样本: {len(arr)}  层数: {len(np.unique(arr[:,0]))}  element 数: {len(np.unique(arr[:,1]))}')

zl, el, ti, F, seg = arr.T
logF = np.log(F)

def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return float('nan')
    return float(np.corrcoef(a[m], b[m])[0, 1])

print(f'\n全样本: corr(ti, logF) = {corr(ti, logF):+.3f}')

# 层级聚合：层时 ≈ Σ seg/F；层均 ti（按段长加权）
layers = np.unique(zl)
lay_t, lay_ti, lay_F = [], [], []
for zl_v in layers:
    m = zl == zl_v
    t = float(np.sum(seg[m] / F[m]))
    w = seg[m]
    lay_t.append(t)
    lay_ti.append(float(np.average(ti[m], weights=w)))
    lay_F.append(float(np.average(F[m], weights=w)))
lay_t, lay_ti, lay_F = map(np.array, (lay_t, lay_ti, lay_F))
print(f'层级: corr(层均ti, log层均F) = {corr(lay_ti, np.log(lay_F)):+.3f}   corr(层均ti, 层时) = {corr(lay_ti, lay_t):+.3f}')

# 控制 element 后：每个大 element 内部 corr(ti, logF)
els, counts = np.unique(el, return_counts=True)
print(f'element 样本分布: 数量={len(els)}  最大={counts.max()}  中位={np.median(counts):.0f}')
big = els[counts > 800]
cs = [corr(ti[el == e], logF[el == e]) for e in big]
cs = [c for c in cs if np.isfinite(c)]
if cs:
    print(f'element 内截面相关（{len(cs)} 个大 element）: 中位 {np.nanmedian(cs):+.3f}  四分位 [{np.percentile(cs,25):+.3f}, {np.percentile(cs,75):+.3f}]')
else:
    print('element 内截面相关: 无足够样本')

# 敏感度：多元回归 lay_ti ~ log(路径长) + log(F)，控制几何后 F 的纯效应
lay_len = np.array([float(np.sum(seg[zl == zl_v])) for zl_v in layers])
A = np.vstack([np.log(lay_len), np.log(lay_F), np.ones(len(lay_F))]).T
coef, *_ = np.linalg.lstsq(A, lay_ti, rcond=None)
pred = A @ coef
r2 = 1 - np.var(lay_ti - pred) / np.var(lay_ti)
print(f'\n多元回归 ti ~ log(路径长) + log(F):')
print(f'  β(logF) = {coef[1]:+.3f}   β(log路径长) = {coef[0]:+.3f}   R² = {r2:.3f}')
print(f'  纯速度敏感度: 速度 ×1.3 → Δti = {coef[1] * np.log(1.3):+.3f}（×2 → {coef[1] * np.log(2):+.3f}）')
print(f'  层时敏感度: corr(层均ti, 层时) = {corr(lay_ti, lay_t):+.3f}')
print(f'ti 总体标准差(层级): {lay_ti.std():.2f}')
