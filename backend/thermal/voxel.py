"""体素级 FDM 热仿真引擎。

模型概述（对应 Helio「voxel 级热历史仿真」的简化复刻）：
1. 打印空间离散为体素网格；按 G-code 时间顺序逐段“沉积”材料
   （体积按热容加权混入体素，温度向喷嘴温度靠拢）。
2. 显式有限差分推进温度场：仅发生在已沉积体素之间的热传导
   （邻居未沉积则不导热），外露表面对流散热（风扇提高对流系数，
   顶面全额、侧面四成），底面与热床换热，薄壁（材料占比低）降温更快。
3. 每段沉积时采样其正下方材料的瞬时温度 = 界面温度——层间结合强度与
   过热下垂的物理决定因素，交由 tqi.py 映射为 TQI。

时间推进采用“时间桶”批处理：桶内所有段先采样界面温度、再统一沉积，
然后按稳定性限制分若干子步推进整场。子步内核用 numba 编译（无 numba
时自动回退到等价的 numpy 向量化实现）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

from ..gcode.model import ParsedGcode
from .materials import Material
from .tqi import tqi_from_interface_temp

# 体素"含材料"判定阈值：0.08mm 薄层的单珠在 1.5mm 体素中占比约 1.5%，
# 阈值必须低于该值，否则新沉积热量会被当空气清零
_OCC_EPS = 0.002


@dataclass
class SimConfig:
    voxel_mm: float = 1.5        # 体素边长
    bucket_s: float = 0.3        # 时间桶长度（秒）
    chamber_temp: float | None = None  # None → 自动估计
    ambient_temp: float = 25.0
    margin_cells: int = 3        # 网格外扩格数（保证沉积格都在动力内核内）
    max_cells: int = 8_000_000   # 网格单元上限（超出自动加大体素）
    # κ/η 默认值：对 Helio 官方标定显示拟合（P2S 手办件，均值差 +2.5°C、层相关 0.40）
    iface_reheat: float = 0.5    # 界面再热系数 κ∈[0,1)：新珠对本格基面的接触再热权重
    # —— 流量 → 有效熔温：流速越快，熔体在喷嘴内吸热不足，出口温度低于设定值
    flow_derate: float = 8.0     # 满档降额 °C
    flow_ref: float = 2.0        # 起降流量 mm³/s（低于此不降）
    flow_span: float = 10.0      # 达到满档的流量跨度
    # —— 喷嘴驻留加热：打印中的喷嘴是移动热源，慢速=在某处停留久=烤暖下方基材
    nozzle_heat: float = 0.35    # 加热系数（每驻留时间常数可升温比例）
    dwell_tau: float = 1.2       # 驻留时间常数 s
    dwell_max_dt: float = 20.0   # 单次驻留升温硬上限 °C（防饱和失真）


@dataclass
class SimResult:
    iface_temp: np.ndarray       # (N,) 每段界面温度 °C（按原始段序）
    tqi: np.ndarray              # (N,) 每段 TQI（按原始段序）
    tqi_valid: np.ndarray        # (N,) bool，首层/辅助结构不计入统计
    layer_stats: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    runtime_s: float = 0.0
    grid_shape: tuple = ()
    voxel_mm: float = 0.0
    element_labels: np.ndarray | None = None  # (N,) int32 每段所属连通体（用于 Helio element.index）


# ---------------------------------------------------------------------------
# numba 内核：一个时间子步的温度场推进（Gauss-Seidel 式就地更新）
# ---------------------------------------------------------------------------
if _HAS_NUMBA:

    @njit(parallel=True, cache=True, fastmath=True)
    def _step_kernel(T, frac, occ, nz_top, conduction, alpha_dt_inv_dx2,
                     conv_base_dt, fan_extra_dt, amb, bed_t, bed_iz,
                     frac_lo, dt_bed):
        """就地推进一个子步。conv_base_dt = h_off*face/cap*dt，
        fan_extra_dt = fan*(h_on-h_off)*face/cap*dt。"""
        nx, ny, nz = T.shape
        for k in prange(1, nz_top - 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    if not occ[i, j, k]:
                        T[i, j, k] = amb
                        continue
                    t = T[i, j, k]
                    lap = 0.0
                    if occ[i + 1, j, k]:
                        lap += T[i + 1, j, k] - t
                    if occ[i - 1, j, k]:
                        lap += T[i - 1, j, k] - t
                    if occ[i, j + 1, k]:
                        lap += T[i, j + 1, k] - t
                    if occ[i, j - 1, k]:
                        lap += T[i, j - 1, k] - t
                    if occ[i, j, k + 1]:
                        lap += T[i, j, k + 1] - t
                    if occ[i, j, k - 1]:
                        lap += T[i, j, k - 1] - t
                    t += conduction * alpha_dt_inv_dx2 * lap
                    # 外露面数（0~6）
                    expo = 0.0
                    if not occ[i + 1, j, k]:
                        expo += 1.0
                    if not occ[i - 1, j, k]:
                        expo += 1.0
                    if not occ[i, j + 1, k]:
                        expo += 1.0
                    if not occ[i, j - 1, k]:
                        expo += 1.0
                    if not occ[i, j, k + 1]:
                        expo += 1.0
                    if not occ[i, j, k - 1]:
                        expo += 1.0
                    if expo > 0.0:
                        top_expo = 1.0 if not occ[i, j, k + 1] else 0.0
                        fan_w = top_expo + 0.4 * (expo - top_expo)
                        f = frac[i, j, k]
                        f_eff = f if f > frac_lo else frac_lo
                        loss = (conv_base_dt * expo + fan_extra_dt * fan_w) / f_eff
                        t -= loss * (t - amb)
                    if k == bed_iz:
                        t -= dt_bed * (t - bed_t)
                    T[i, j, k] = t


# ---------------------------------------------------------------------------
class ThermalSimulator:
    def __init__(
        self,
        parsed: ParsedGcode,
        material: Material,
        config: SimConfig | None = None,
        progress_cb: Callable[[float], None] | None = None,
    ) -> None:
        self.p = parsed
        self.m = material
        self.cfg = config or SimConfig()
        self.progress_cb = progress_cb
        if self.cfg.chamber_temp is None:
            # 腔温估计：热床对腔体加热的保守折中
            self.cfg.chamber_temp = min(0.3 * material.bed + self.cfg.ambient_temp, 45.0)

    # ------------------------------------------------------------------
    def _build_grid(self) -> tuple[float, np.ndarray]:
        p, cfg = self.p, self.cfg
        ext = np.maximum(p.bbox_max - p.bbox_min, 1e-3)
        voxel = float(cfg.voxel_mm)
        while True:
            shape = np.ceil(ext / voxel).astype(np.int64) + 2 * cfg.margin_cells + 1
            if int(np.prod(shape)) <= cfg.max_cells or voxel >= 8.0:
                break
            voxel = min(voxel * 1.3, 8.0)
        return voxel, shape

    def _cell_of(self, x, y, z):
        o = self.origin
        hi = self.shape - 2  # 动力内核上界（含 margin 的外壳不动）
        ix = np.clip(((x - o[0]) / self.voxel).astype(np.int64), 1, hi[0])
        iy = np.clip(((y - o[1]) / self.voxel).astype(np.int64), 1, hi[1])
        iz = np.clip(((z - o[2]) / self.voxel).astype(np.int64), 1, hi[2])
        return ix, iy, iz

    # ------------------------------------------------------------------
    def run(self) -> SimResult:
        import time

        t0 = time.perf_counter()
        self._prepare()
        iface_ordered = self._run_buckets()

        # 结果映射回原始段序
        iface = np.empty(iface_ordered.shape, dtype=np.float64)
        iface[self._order] = iface_ordered
        tqi = tqi_from_interface_temp(iface, self.m)
        valid = self._valid_mask()
        labels = self._element_labels()

        result = SimResult(
            iface_temp=iface.astype(np.float32),
            tqi=tqi.astype(np.float32),
            tqi_valid=valid,
            layer_stats=self._layer_stats(tqi, valid),
            config={
                "voxel_mm": float(self.voxel),
                "bucket_s": self.cfg.bucket_s,
                "chamber_temp": float(self.cfg.chamber_temp),
                "ambient_temp": self.cfg.ambient_temp,
                "material": self.m.name,
                "nozzle_temp": self.m.nozzle,
                "bed_temp": self.m.bed,
                "backend": "numba" if _HAS_NUMBA else "numpy",
            },
            runtime_s=time.perf_counter() - t0,
            grid_shape=tuple(int(s) for s in self.shape),
            voxel_mm=float(self.voxel),
            element_labels=labels,
        )
        return result

    def _element_labels(self) -> np.ndarray:
        """每段所属连通沉积体（6 连通），供 Helio element.index 语义使用。"""
        from scipy import ndimage

        occ = self.frac > _OCC_EPS
        labels_grid, n = ndimage.label(occ)  # 6 连通默认结构
        if n == 0:
            return np.zeros(self.p.num_segments, dtype=np.int32)
        # 段中点格 → 连通体标签
        g = self.p.geometry
        mx = (g[:, 0] + g[:, 3]) * 0.5
        my = (g[:, 1] + g[:, 4]) * 0.5
        ix, iy, iz = self._cell_of(mx, my, g[:, 5])
        seg_labels = labels_grid[ix, iy, iz]
        # 落在空气格的段（桥接等）：借最近邻已沉积格，找不到给 0
        empty = seg_labels == 0
        if empty.any():
            for dx in (-1, 1):
                jx = np.clip(ix[empty] + dx, 0, self.shape[0] - 1)
                seg_labels[empty] = np.where(
                    seg_labels[empty] == 0, labels_grid[jx, iy[empty], iz[empty]], seg_labels[empty]
                )
        return seg_labels.astype(np.int32)

    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        p, m, cfg = self.p, self.m, self.cfg
        self.voxel, self.shape = self._build_grid()
        dx = self.voxel

        self.origin = p.bbox_min - cfg.margin_cells * dx
        self.T = np.full(tuple(self.shape), float(self.cfg.chamber_temp), dtype=np.float64)
        self.frac = np.zeros(tuple(self.shape), dtype=np.float64)
        self.top_iz = cfg.margin_cells + 1

        cell_v = dx ** 3                                    # mm³
        self._cell_heat_cap = m.rho * m.cp * cell_v * 1e-9  # J/K
        self._alpha_dt = m.alpha / (dx * dx)                # mm²/s / mm²
        self._face_over_cap = (dx * dx * 1e-6) / self._cell_heat_cap  # m²/J·K
        tau_min = 1.0 / max(m.h_conv_on * self._face_over_cap, 1e-12)
        self._dt_max = min(0.45 / (6.0 * self._alpha_dt), 0.2 * tau_min)
        self._bed_iz = cfg.margin_cells  # 热床表面所在阵列 z

        # 台阶邻域采样用：18 个邻居的展平偏移（iz∈{0,-1} × 3×3）
        s1, s2 = int(self.shape[1]), int(self.shape[2])
        self._neigh_off = np.array(
            [dx * s1 * s2 + dy * s2 + dz
             for dz in (0, -1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
            dtype=np.int64)
        self._T_flat = self.T.ravel()
        self._frac_flat = self.frac.ravel()

        order = np.argsort(p.t_mid, kind="stable")
        self._order = order

    def _valid_mask(self) -> np.ndarray:
        from ..gcode.model import TQI_FEATURES

        p = self.p
        if p.info.has_feature_comments:
            feats = np.isin(p.feature_id, [int(f) for f in TQI_FEATURES])
        else:
            # 无特性注释的文件（如 Helio 优化后输出）：全部主体段计入，
            # 仅排除首层（裙边/支撑无法区分，接受其混入）
            feats = np.ones(p.num_segments, dtype=bool)
        return (p.layer_idx > 0) & feats

    # ------------------------------------------------------------------
    def _run_buckets(self) -> np.ndarray:
        p, m, cfg = self.p, self.m, self.cfg
        n_seg = self._order.size
        iface_out = np.zeros(n_seg, dtype=np.float64)

        g = p.geometry[self._order]
        gx0, gy0 = g[:, 0], g[:, 1]
        gx1, gy1, gz1 = g[:, 3], g[:, 4], g[:, 5]
        gmid = ((gx0 + gx1) * 0.5, (gy0 + gy1) * 0.5)
        cell_ix, cell_iy, cell_iz = self._cell_of(gmid[0], gmid[1], gz1)
        seg_len = np.hypot(gx1 - gx0, gy1 - gy0)
        n_pts = np.maximum((seg_len / (self.voxel * 0.5)).astype(np.int64) + 1, 1)
        seg_t = p.t_mid[self._order]
        seg_vol = p.extrusion_mm3[self._order]
        seg_fan = p.fan[self._order]
        seg_end = p.t_mid[self._order] + p.duration[self._order]
        seg_dur = p.duration[self._order]
        layer0 = p.layer_idx[self._order] == 0
        # 有效熔温：体积流量大 → 喷嘴吸热不足 → 出口温度低于设定值
        cfg = self.cfg
        flow = seg_vol / np.maximum(seg_dur, 1e-3)
        t_dep_all = float(m.nozzle) - cfg.flow_derate * np.clip(
            (flow - cfg.flow_ref) / max(cfg.flow_span, 1e-6), 0.0, 1.0
        )

        amb = float(self.cfg.chamber_temp)
        nozzle_t, bed_t = float(m.nozzle), float(m.bed)
        h_off, h_on = float(m.h_conv_off), float(m.h_conv_on)
        frac_lo = 0.2

        i = 0
        n_bucket = 0
        sim_clock = float(seg_t[0])
        while i < n_seg:
            j = max(int(np.searchsorted(seg_t, seg_t[i] + cfg.bucket_s, side="left")), i + 1)

            ix, iy, iz = cell_ix[i:j], cell_iy[i:j], cell_iz[i:j]

            # ---- 1) 沉积前：喷嘴驻留加热 + 界面温度采样 ----
            # 喷嘴是移动热源：打印本段时烤暖本格与同级邻格中已有的材料
            # （正是下一颗珠子要结合的台阶），慢速=停留久=更暖。
            t_dep = t_dep_all[i:j]
            if cfg.nozzle_heat > 0:
                f_self = self.frac[ix, iy, iz]
                t_self = self.T[ix, iy, iz]
                dwell = 1.0 - np.exp(-seg_dur[i:j] / cfg.dwell_tau)
                dT_self = np.where(f_self > _OCC_EPS,
                                   np.clip(cfg.nozzle_heat * (t_dep - t_self) * dwell
                                           / np.maximum(f_self, 0.05),
                                           0.0, np.minimum(t_dep - t_self, cfg.dwell_max_dt)), 0.0)
                self.T[ix, iy, iz] += dT_self
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    jx = np.clip(ix + dx, 1, self.shape[0] - 2)
                    jy = np.clip(iy + dy, 1, self.shape[1] - 2)
                    f_n = self.frac[jx, jy, iz]
                    t_n = self.T[jx, jy, iz]
                    dT_n = np.where(f_n > _OCC_EPS,
                                    np.clip(0.5 * cfg.nozzle_heat * (t_dep - t_n) * dwell
                                            / np.maximum(f_n, 0.05),
                                            0.0, np.minimum(t_dep - t_n, 0.5 * cfg.dwell_max_dt)), 0.0)
                    self.T[jx, jy, iz] += dT_n

            # ---- 1.1) 界面温度采样（v3：展平大矩阵化，一次采集 18 邻域）----
            base_flat = (ix * self.shape[1] + iy) * self.shape[2] + iz
            neigh = base_flat[:, None] + self._neigh_off[None, :]
            f_all = self._frac_flat[neigh]
            t_all = self._T_flat[neigh]
            ok_m = f_all > _OCC_EPS
            any_mat = ok_m.any(axis=1)
            cand = np.where(ok_m, t_all, -np.inf).max(axis=1)
            base = np.where(any_mat, cand, amb)
            base[layer0[i:j]] = bed_t

            # 接触再热 κ：新珠热量对基面表层的重熔加温（0=关闭）
            kappa = float(self.cfg.iface_reheat)
            if kappa > 0.0:
                # v_norm 体积占比按段计算（沉积总量/体素容积）
                v_seg = (seg_vol[i:j] / (self.voxel ** 3))
                f_c = np.maximum(v_seg, 0.02)
                mixed = (v_seg * t_dep + f_c * base) / np.maximum(v_seg + f_c, 1e-9)
                # 空气格（桥接）不再热：界面仍是腔温
                base = np.where(any_mat, (1.0 - kappa) * base + kappa * mixed, amb)
            ti = base
            iface_out[i:j] = ti

            # ---- 2) 沉积本桶全部段 ----
            reps = n_pts[i:j]
            total = int(reps.sum())
            if total:
                starts = np.concatenate([[0], np.cumsum(reps)[:-1]])
                seg_of_pt = np.repeat(np.arange(j - i, dtype=np.int64), reps)
                within = np.arange(total, dtype=np.int64) - starts[seg_of_pt]
                u = within / reps[seg_of_pt]
                a = np.arange(i, j)
                px = gx0[a][seg_of_pt] + (gx1[a][seg_of_pt] - gx0[a][seg_of_pt]) * u
                py = gy0[a][seg_of_pt] + (gy1[a][seg_of_pt] - gy0[a][seg_of_pt]) * u
                pz = gz1[a][seg_of_pt]
                dix, diy, diz = self._cell_of(px, py, pz)
                vol_each = np.repeat(seg_vol[i:j] / reps, reps)
                t_each = np.repeat(t_dep, reps)
                self._deposit(dix, diy, diz, vol_each, t_each)

            # ---- 3) 温度场推进（覆盖到本桶末段结束，含桶间空走/暂停）----
            end_t = float(seg_end[j - 1])
            bucket_dt = max(end_t - sim_clock, 1e-3)
            sim_clock = end_t
            self._advance(bucket_dt, float(seg_fan[i:j].max()), h_off, h_on, amb, bed_t, frac_lo)

            n_bucket += 1
            if self.progress_cb and (n_bucket % 8 == 0 or j >= n_seg):
                self.progress_cb(j / n_seg)
            i = j

        return iface_out

    # ------------------------------------------------------------------
    def _deposit(self, ix, iy, iz, vol, t_dep: np.ndarray) -> None:
        """闭式沉积混入：同一格多次沉积按体积加权合并（每点自带沉积温度）。"""
        dx3 = self.voxel ** 3
        v_norm = vol / dx3
        flat = (ix * self.shape[1] + iy) * self.shape[2] + iz

        uniq, inv = np.unique(flat, return_inverse=True)
        n = uniq.size
        f0 = np.zeros(n)
        t0 = np.zeros(n)
        f0[inv] = self.frac[ix, iy, iz]   # 同格取值相同，任意次覆盖无碍
        t0[inv] = self.T[ix, iy, iz]
        vsum = np.zeros(n)
        vtsum = np.zeros(n)
        np.add.at(vsum, inv, v_norm)
        np.add.at(vtsum, inv, v_norm * t_dep)

        new_f = np.minimum(f0 + vsum, 1.0)
        new_t = (t0 * f0 + vtsum) / np.maximum(f0 + vsum, 1e-12)

        ux = uniq // (self.shape[1] * self.shape[2])
        uy = (uniq // self.shape[2]) % self.shape[1]
        uz = uniq % self.shape[2]
        self.T[ux, uy, uz] = new_t
        self.frac[ux, uy, uz] = new_f
        if uz.size:
            self.top_iz = max(self.top_iz, int(uz.max()))

    # ------------------------------------------------------------------
    def _advance(self, dt_total: float, fan: float, h_off: float, h_on: float,
                 amb: float, bed_t: float, frac_lo: float) -> None:
        cfg = self.cfg
        top = int(min(self.top_iz + 2, self.shape[2]))
        if top <= cfg.margin_cells + 1:
            return
        n_sub = max(1, int(math.ceil(dt_total / self._dt_max)))
        dt = dt_total / n_sub

        conv_base = h_off * self._face_over_cap * dt
        fan_extra = max(fan * (h_on - h_off), 0.0) * self._face_over_cap * dt
        bed_loss = self.m.h_bed * self._face_over_cap * dt
        conduction = 1.0
        alpha_dt = self._alpha_dt * dt

        if _HAS_NUMBA:
            occ_grid = self.frac > _OCC_EPS
            for _ in range(n_sub):
                _step_kernel(self.T, self.frac, occ_grid, top,
                             conduction, alpha_dt, conv_base, fan_extra,
                             amb, bed_t, self._bed_iz, frac_lo, bed_loss)
        else:
            for _ in range(n_sub):
                self._advance_numpy(top, alpha_dt, conv_base, fan_extra,
                                    amb, bed_t, frac_lo, bed_loss)

    def _advance_numpy(self, top, alpha_dt, conv_base, fan_extra,
                       amb, bed_t, frac_lo, bed_loss):
        """numba 不可用时的等价向量化实现（单步；由 _advance 循环调用）。"""
        T = self.T
        occ = self.frac[:, :, :top] > _OCC_EPS
        occ_i = occ.astype(np.float32)
        core = (slice(1, -1), slice(1, -1), slice(1, -1))
        masks = (occ[1:, :, :], occ[:-1, :, :], occ[:, 1:, :],
                 occ[:, :-1, :], occ[:, :, 1:], occ[:, :, :-1])
        exp6 = np.zeros(occ.shape, dtype=np.float32)
        exp6[:-1] += 1 - occ_i[1:]
        exp6[1:] += 1 - occ_i[:-1]
        exp6[:, :-1] += 1 - occ_i[:, 1:]
        exp6[:, 1:] += 1 - occ_i[:, :-1]
        exp6[:, :, :-1] += 1 - occ_i[:, :, 1:]
        exp6[:, :, 1:] += 1 - occ_i[:, :, :-1]
        top_expo = np.zeros(occ.shape, dtype=np.float32)
        top_expo[:, :, :-1] = 1 - occ_i[:, :, 1:]
        f_eff = np.maximum(self.frac[:, :, :top], frac_lo)[core]

        Tc = T[core]
        lap = (
            masks[0][core] * (T[2:, 1:-1, 1:-1] - Tc)
            + masks[1][core] * (T[:-2, 1:-1, 1:-1] - Tc)
            + masks[2][core] * (T[1:-1, 2:, 1:-1] - Tc)
            + masks[3][core] * (T[1:-1, :-2, 1:-1] - Tc)
            + masks[4][core] * (T[1:-1, 1:-1, 2:] - Tc)
            + masks[5][core] * (T[1:-1, 1:-1, :-2] - Tc)
        )
        Tc += alpha_dt * lap
        expo = exp6[core]
        texp = top_expo[core]
        loss = (conv_base * expo + fan_extra * (texp + 0.4 * (expo - texp))) / f_eff
        Tc -= loss * (Tc - amb)
        biz = self._bed_iz
        if top > biz:
            T[1:-1, 1:-1, biz] -= bed_loss * (T[1:-1, 1:-1, biz] - bed_t)
        T[~occ] = amb

    # ------------------------------------------------------------------
    def _layer_stats(self, tqi: np.ndarray, valid: np.ndarray) -> list[dict]:
        p = self.p
        stats = []
        for li in range(p.num_layers):
            sel = p.layer_idx == li
            if not sel.any():
                continue
            v = valid & sel
            tq = tqi[v]
            has = tq.size > 0
            stats.append({
                "layer": li,
                "z": float(p.layer_z[li]),
                "t0": float(p.layer_t0[li]),
                "t1": float(p.layer_t1[li]),
                "mean_tqi": float(tq.mean()) if has else None,
                "min_tqi": float(tq.min()) if has else None,
                "cold_frac": float((tq < -50).mean()) if has else None,
                "hot_frac": float((tq > 50).mean()) if has else None,
            })
        return stats
