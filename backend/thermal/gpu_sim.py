"""GPU（torch/CUDA）体素热仿真：与 voxel.py 的 CPU 模型逐项对应的 GPU 实现。

设计：
- 整场 T/frac/last_dep 常驻显存（float32），全程无逐桶主机-显存传输
- 掩码导热/外露暴露次数用 3×3×3 卷积一次算出（2 次 conv3d/子步）
- 沉积/驻热/界面采样全部向量化，点级索引预计算一次上传
- 无 CUDA/torch 时不可导入；调用方应以 gpu_available() 探测
"""
from __future__ import annotations

import math
import time

import numpy as np

from .voxel import _OCC_EPS, SimConfig, SimResult
from .tqi import tqi_from_interface_temp

try:
    import torch
    _TORCH_OK = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    _TORCH_OK = False


def gpu_available() -> bool:
    return _TORCH_OK


def simulate_gpu(parsed, material, cfg: SimConfig, progress_cb=None) -> SimResult:
    import torch

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    if cfg.chamber_temp is None:
        cfg.chamber_temp = min(0.3 * material.bed + cfg.ambient_temp, 45.0)
    amb = float(cfg.chamber_temp)
    nozzle_t = float(material.nozzle)
    bed_t = float(material.bed)

    # ---- 网格（与 CPU 同规则）----
    ext = np.maximum(parsed.bbox_max - parsed.bbox_min, 1e-3)
    voxel = float(cfg.voxel_mm)
    margin = cfg.margin_cells
    while True:
        shape = np.ceil(ext / voxel).astype(np.int64) + 2 * margin + 1
        if int(np.prod(shape)) <= cfg.max_cells or voxel >= 8.0:
            break
        voxel = min(voxel * 1.3, 8.0)
    nx, ny, nz = (int(v) for v in shape)
    dx = voxel

    origin = parsed.bbox_min - margin * dx
    T = torch.full((nx, ny, nz), float(cfg.chamber_temp), device=dev, dtype=torch.float32)
    frac = torch.zeros((nx, ny, nz), device=dev, dtype=torch.float32)
    last_dep = torch.full((nx, ny, nz), -np.inf, device=dev, dtype=torch.float32)

    cell_v = dx ** 3
    cell_cap = material.rho * material.cp * cell_v * 1e-9          # J/K
    face_over_cap = (dx * dx * 1e-6) / cell_cap                    # m²/(J/K)
    alpha_dt_base = material.alpha / (dx * dx)                     # 1/(s·mm²) 系数
    tau_min = 1.0 / max(material.h_conv_on * face_over_cap, 1e-12)
    dt_max = min(0.45 / (6.0 * alpha_dt_base), 0.2 * tau_min)
    bed_iz = margin
    frac_lo = 0.2
    eps = _OCC_EPS

    # ---- 段级预计算（CPU numpy 一次，整块上传）----
    order = np.argsort(parsed.t_mid, kind="stable")
    seg_t = parsed.t_mid[order].astype(np.float64)
    seg_vol = parsed.extrusion_mm3[order].astype(np.float64)
    seg_dur = parsed.duration[order].astype(np.float64)
    seg_fan = parsed.fan[order].astype(np.float32)
    seg_layer = parsed.layer_idx[order].astype(np.int64)
    seg_layer_t0 = parsed.layer_t0[seg_layer].astype(np.float64)
    layer0 = torch.from_numpy((seg_layer == 0))

    g = parsed.geometry[order].astype(np.float64)
    gx0, gy0, gz0 = g[:, 0], g[:, 1], g[:, 2]
    gx1, gy1, gz1 = g[:, 3], g[:, 4], g[:, 5]
    gmid_x = (gx0 + gx1) * 0.5
    gmid_y = (gy0 + gy1) * 0.5

    def cell_of(xa, ya, za):
        hi = shape - 2
        ix = np.clip(((xa - origin[0]) / voxel).astype(np.int64), 1, hi[0])
        iy = np.clip(((ya - origin[1]) / voxel).astype(np.int64), 1, hi[1])
        iz = np.clip(((za - origin[2]) / voxel).astype(np.int64), 1, hi[2])
        return ix, iy, iz

    cix, ciy, ciz = cell_of(gmid_x, gmid_y, gz1)
    seg_cell_flat = torch.from_numpy(((cix * ny + ciy) * nz + ciz)).to(dev)

    # 沉积点级展开（沿段路径每 dx/2 采样一点）
    seg_len = np.hypot(gx1 - gx0, gy1 - gy0)
    n_pts = np.maximum((seg_len / (dx * 0.5)).astype(np.int64) + 1, 1)
    total_pts = int(n_pts.sum())
    seg_of_pt = np.repeat(np.arange(len(seg_t)), n_pts)
    starts = np.concatenate([[0], np.cumsum(n_pts)[:-1]])
    within = np.arange(total_pts) - starts[seg_of_pt]
    u = within / n_pts[seg_of_pt]
    px = gx0[seg_of_pt] + (gx1[seg_of_pt] - gx0[seg_of_pt]) * u
    py = gy0[seg_of_pt] + (gy1[seg_of_pt] - gy0[seg_of_pt]) * u
    pz = gz1[seg_of_pt]
    pix, piy, piz = cell_of(px, py, pz)
    pt_cell = torch.from_numpy(((pix * ny + piy) * nz + piz)).to(dev)
    pt_vol = torch.from_numpy((seg_vol / n_pts)[seg_of_pt].astype(np.float32)).to(dev)
    # 每点有效沉积温度（流量降额）
    flow = seg_vol / np.maximum(seg_dur, 1e-3)
    t_dep_seg = nozzle_t - cfg.flow_derate * np.clip(
        (flow - cfg.flow_ref) / max(cfg.flow_span, 1e-6), 0.0, 1.0)
    pt_tdep = torch.from_numpy(t_dep_seg[seg_of_pt].astype(np.float32)).to(dev)
    pt_iz_cpu = piz  # CPU 侧推进范围计算用
    seg_pt_start = np.concatenate([[0], np.cumsum(n_pts)]).astype(np.int64)
    pt_z_cpu = pz

    # 有效熔温 / 驻留
    v_seg = seg_vol / (dx ** 3)
    dwell_seg = 1.0 - np.exp(-seg_dur / cfg.dwell_tau)

    # 界面采样偏移（18 邻域）
    s1, s2 = ny, nz
    neigh_off = torch.tensor(
        [dxx * s1 * s2 + dyy * s2 + dzz
         for dzz in (0, -1) for dyy in (-1, 0, 1) for dxx in (-1, 0, 1)],
        device=dev, dtype=torch.int64)
    side_off = torch.tensor([s1 * s2, -s1 * s2, s2, -s2], device=dev, dtype=torch.int64)

    seg_layer_t0_t = torch.from_numpy(seg_layer_t0.astype(np.float32)).to(dev)
    layer0_t = layer0.to(dev)
    fan_t = torch.from_numpy(seg_fan).to(dev)
    t_dep_t = torch.from_numpy(t_dep_seg.astype(np.float32)).to(dev)
    dwell_t = torch.from_numpy(dwell_seg.astype(np.float32)).to(dev)
    v_seg_t = torch.from_numpy((v_seg).astype(np.float32)).to(dev)

    iface_gpu = torch.zeros(len(seg_t), device=dev, dtype=torch.float32)
    Tflat = T.view(-1)
    fracflat = frac.view(-1)
    lastdep_flat = last_dep.view(-1)

    # conv3d 两个核：6 邻域值和 / 6 邻域占用计数
    K = torch.zeros(1, 1, 3, 3, 3, device=dev)
    for a, b, c in ((1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (0, 1, 1), (2, 1, 1)):
        K[0, 0, a, b, c] = 1.0

    def conv6(x):
        return torch.nn.functional.conv3d(x.view(1, 1, *x.shape), K).view(x.shape)

    amb_t = amb
    bed_t = float(material.bed)
    nozzle_tf = nozzle_t

    n_seg = len(seg_t)
    iface_np = np.zeros(n_seg, dtype=np.float64)

    i = 0
    t_clock = float(seg_t[0])
    t_start_wall = time.perf_counter()
    n_bucket = 0

    while i < n_seg:
        j = max(int(np.searchsorted(seg_t, seg_t[i] + cfg.bucket_s, side="left")), i + 1)
        top = int(min(pt_iz_cpu[seg_pt_start[i]:seg_pt_start[j]].max() + 3, nz)) if j > i else nz
        top = max(top, bed_iz + 2)

        # ---- 1) 界面温度采样（只认本层开始前沉积的基材）----
        base_flat = seg_cell_flat[i:j]
        neigh = base_flat[:, None] + neigh_off[None, :]
        f_all = fracflat[neigh]
        t_all = Tflat[neigh]
        tl0 = seg_layer_t0_t[i:j][:, None]
        ok_m = (f_all > eps) & (lastdep_flat[neigh] < tl0)
        any_mat = ok_m.any(dim=1)
        cand = torch.where(ok_m, t_all, torch.tensor(-np.inf, device=dev)).max(dim=1).values
        base = torch.where(any_mat, cand, torch.tensor(amb_t, device=dev))
        base = torch.where(layer0_t[i:j], torch.tensor(bed_t, device=dev), base)

        # ---- 2) κ 接触再热 ----
        kappa = float(cfg.iface_reheat)
        if kappa > 0.0:
            v_seg_b = v_seg_t[i:j]
            f_c = torch.clamp(v_seg_b, min=0.02)
            t_dep_b = t_dep_t[i:j]
            mixed = (v_seg_b * t_dep_b + f_c * base) / torch.clamp(v_seg_b + f_c, min=1e-9)
            base = torch.where(any_mat, (1.0 - kappa) * base + kappa * mixed,
                               torch.tensor(amb_t, device=dev))
        iface_gpu[i:j] = base.float()

        # ---- 3) 喷嘴驻留加热（本格 + 4 侧邻，0.5 权重）----
        if cfg.nozzle_heat > 0:
            f_self = fracflat[base_flat]
            t_self = Tflat[base_flat]
            t_dep_b = t_dep_t[i:j]
            dT = cfg.nozzle_heat * (t_dep_b - t_self) * dwell_t[i:j] / torch.clamp(f_self, min=0.05)
            dT = torch.where((f_self > eps) & (dT > 0),
                             torch.clamp(dT, max=torch.clamp(t_dep_b - t_self, min=0.0).clamp(max=cfg.dwell_max_dt)),
                             torch.zeros_like(dT))
            Tflat.index_put_((base_flat,), dT, accumulate=True)
            for off in side_off.tolist():
                nb = base_flat + off
                f_n = fracflat[nb]
                t_n = Tflat[nb]
                dTn = 0.5 * cfg.nozzle_heat * (t_dep_b - t_n) * dwell_t[i:j] / torch.clamp(f_n, min=0.05)
                dTn = torch.where((f_n > eps) & (dTn > 0),
                                  torch.clamp(dTn, max=torch.clamp(t_dep_b - t_n, min=0.0).clamp(max=0.5 * cfg.dwell_max_dt)),
                                  torch.zeros_like(dTn))
                Tflat.index_put_((nb,), dTn, accumulate=True)

        # ---- 4) 沉积（体积加权混入，last_dep 更新）----
        p0, p1 = seg_pt_start[i], seg_pt_start[j]
        if p1 > p0:
            pc = pt_cell[p0:p1]
            f0_pt = fracflat[pc]
            t0_pt = Tflat[pc]
            v_pt = pt_vol[p0:p1]
            acc_v = torch.zeros_like(fracflat)
            acc_vt = torch.zeros_like(fracflat)
            acc_v.index_put_((pc,), v_pt, accumulate=True)
            acc_vt.index_put_((pc,), v_pt * pt_tdep[p0:p1], accumulate=True)
            av = acc_v[pc]
            avt = acc_vt[pc]
            new_f = torch.clamp(f0_pt + av, max=1.0)
            new_t = (t0_pt * f0_pt + avt) / torch.clamp(f0_pt + av, min=1e-12)
            fracflat.index_put_((pc,), new_f)
            Tflat.index_put_((pc,), new_t)
            t_bucket = float(seg_t[i:j].mean())
            lastdep_flat.index_put_((pc,), torch.full_like(new_f, t_bucket))

        # ---- 5) 温度场推进（conv3d 掩码拉普拉斯 + 对流 + 热床 + 钉空气）----
        end_t = float(seg_t[j - 1] + seg_dur[j - 1])
        bucket_dt = max(end_t - t_clock, 1e-3)
        t_clock = end_t
        n_sub = max(1, int(math.ceil(bucket_dt / dt_max)))
        dt = bucket_dt / n_sub
        fan = float(seg_fan[i:j].max())

        slab = T[:, :, :top]
        fracs = frac[:, :, :top]
        occ = (fracs > eps).float()
        occ4 = occ.view(1, 1, *occ.shape)
        T4 = slab.view(1, 1, *slab.shape)

        occ_nb = torch.nn.functional.conv3d(occ4, K, padding=1).view(slab.shape)   # 占用邻居数 0~6
        lap_T = torch.nn.functional.conv3d((T4 * occ4).view(1, 1, *slab.shape), K, padding=1).view(slab.shape)
        # 掩码拉普拉斯：Σ_occ(T_n − T_c) = conv(T·occ) − T_c·conv(occ)
        lap = lap_T - slab * occ_nb
        expo = 6.0 - occ_nb                                              # 外露面数
        T4v = T4.view(slab.shape)
        T4v.add_(lap, alpha=alpha_dt_base * dt)
        # 对流：外露面 × 系数，顶面全额、侧面 40%（与 CPU 一致的简化）
        top_expo = torch.zeros_like(slab)
        top_expo[:, :, :-1] = 1.0 - occ[:, :, 1:]
        fan_w = top_expo + 0.4 * (expo - top_expo)
        conv_base = material.h_conv_off * face_over_cap * dt
        fan_extra = max(fan * (material.h_conv_on - material.h_conv_off), 0.0) * face_over_cap * dt
        f_eff = torch.clamp(fracs, min=frac_lo)
        T4v.sub_((conv_base * expo + fan_extra * fan_w) / f_eff * (T4v - amb))
        # 首层底面热床换热
        if top > bed_iz:
            T4v[:, :, bed_iz] -= material.h_bed * face_over_cap * dt * (T4v[:, :, bed_iz] - bed_t)
        # 空气格钉在腔温
        T4v[occ == 0] = amb

        n_bucket += 1
        if progress_cb and (n_bucket % 16 == 0 or j >= n_seg):
            progress_cb(j / n_seg)
        i = j

    # ---- 结果回收 ----
    iface_ordered = iface_gpu.cpu().numpy().astype(np.float64)
    iface = np.empty(n_seg, dtype=np.float64)
    iface[order] = iface_ordered
    tqi = tqi_from_interface_temp(iface, material)

    from ..gcode.model import TQI_FEATURES
    if parsed.info.has_feature_comments:
        feats = np.isin(parsed.feature_id, [int(f) for f in TQI_FEATURES])
    else:
        # 无特性注释（如 Bambu 机打 G-code）：全部主体段计入（与 CPU _valid_mask 一致）
        feats = np.ones(parsed.num_segments, dtype=bool)
    valid = (parsed.layer_idx > 0) & feats

    frac_cpu = frac.cpu().numpy()
    from scipy import ndimage
    labels_grid, _ = ndimage.label(frac_cpu > _OCC_EPS)
    cix_o, ciy_o, ciz_o = cell_of(gmid_x, gmid_y, gz1)
    seg_labels = labels_grid[cix_o, ciy_o, ciz_o].astype(np.int32)

    layer_stats = []
    for li in range(parsed.num_layers):
        sel = parsed.layer_idx == li
        v = valid & sel
        tq = tqi[v]
        has = tq.size > 0
        layer_stats.append({
            "layer": li, "z": float(parsed.layer_z[li]),
            "t0": float(parsed.layer_t0[li]), "t1": float(parsed.layer_t1[li]),
            "mean_tqi": float(tq.mean()) if has else None,
            "min_tqi": float(tq.min()) if has else None,
            "cold_frac": float((tq < -50).mean()) if has else None,
            "hot_frac": float((tq > 50).mean()) if has else None,
        })

    return SimResult(
        iface_temp=iface.astype(np.float32),
        tqi=tqi.astype(np.float32),
        tqi_valid=valid,
        layer_stats=layer_stats,
        config={
            "voxel_mm": float(voxel), "bucket_s": cfg.bucket_s,
            "chamber_temp": float(cfg.chamber_temp), "ambient_temp": cfg.ambient_temp,
            "material": material.name, "nozzle_temp": material.nozzle, "bed_temp": material.bed,
            "backend": "gpu-torch",
        },
        runtime_s=time.perf_counter() - t_start_wall,
        grid_shape=(nx, ny, nz), voxel_mm=float(voxel),
        element_labels=seg_labels,
    )
