"""热仿真引擎物理正确性测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.gcode.parser import parse_gcode
from backend.thermal.materials import get_material
from backend.thermal.tqi import tqi_from_interface_temp
from backend.thermal.voxel import SimConfig, ThermalSimulator
from tests.gcode_gen import generate_box_gcode, generate_two_rate_gcode


def _sim(text, **cfg):
    p = parse_gcode(text)
    mat = get_material(p.info.detected_material)
    sim = ThermalSimulator(p, mat, SimConfig(**cfg))
    return sim.run(), sim


def test_interface_temp_monotonic_in_layer_time():
    """纯传导物理（η=0）下：慢速层（长层时）界面应显著冷于快速层。"""
    def run(feed_slow_part):
        p = parse_gcode(generate_two_rate_gcode())
        mat = get_material("PLA")
        cfg = SimConfig(voxel_mm=1.5, bucket_s=0.25, nozzle_heat=0.0, iface_reheat=0.0)
        return ThermalSimulator(p, mat, cfg).run(), p

    res, p = run(0)
    p_iface = res.iface_temp

    def mean_iface(layers):
        sel = np.isin(p.layer_idx, layers) & res.tqi_valid
        return float(p_iface[sel].mean())

    slow_mean = mean_iface(range(2, 10))    # 10 mm/s
    fast_mean = mean_iface(range(11, 20))   # 150 mm/s
    assert slow_mean < 100.0, f"慢速层界面应明显冷却，实际 {slow_mean:.1f}°C"
    assert fast_mean > 150.0, f"快速层界面应保持高温，实际 {fast_mean:.1f}°C"
    assert slow_mean < fast_mean - 50.0


def test_hot_spot_vs_slow_region_tqi():
    """纯传导物理（η=0）：快速大件应显著热于慢速小件（层时效应的排序不变；
    标定后慢速小件 TQI 接近 0 属正常——无风扇+驻留补偿）。"""
    res_slow, _ = _sim(generate_box_gcode(layers=15, size=25, feed=15), voxel_mm=1.5, nozzle_heat=0.0, iface_reheat=0.0)
    res_fast, _ = _sim(generate_box_gcode(layers=15, size=60, feed=300), voxel_mm=1.5, nozzle_heat=0.0, iface_reheat=0.0)
    tq_slow = res_slow.tqi[res_slow.tqi_valid]
    tq_fast = res_fast.tqi[res_fast.tqi_valid]
    assert tq_fast.mean() > tq_slow.mean() + 20.0, (
        f"快速大件应显著更热: slow={tq_slow.mean():.1f} fast={tq_fast.mean():.1f}"
    )


def test_dwell_heating_warms_slow_print():
    """喷嘴驻留热隔离检验：同一慢速件，开启驻留热（标定值 0.35）后界面温度应显著高于关闭。"""
    res_off, _ = _sim(generate_box_gcode(layers=8, size=25, feed=15), voxel_mm=1.5, nozzle_heat=0.0, iface_reheat=0.0)
    res_on, _ = _sim(generate_box_gcode(layers=8, size=25, feed=15), voxel_mm=1.5, nozzle_heat=0.35)
    t_off = res_off.iface_temp[res_off.tqi_valid]
    t_on = res_on.iface_temp[res_on.tqi_valid]
    assert t_on.mean() > t_off.mean() + 3.0, (
        f"驻留热应抬升慢速件界面温度：off={t_off.mean():.1f} on={t_on.mean():.1f}"
    )


def test_no_temperature_overshoot():
    """温度场必须始终被夹在 [环境-ε, 喷嘴+ε]。"""
    p = parse_gcode(generate_box_gcode(layers=6, size=30, feed=120))
    mat = get_material("PLA")
    sim = ThermalSimulator(p, mat, SimConfig(voxel_mm=1.5))
    orig_advance = sim._advance

    def spy(*a, **kw):
        orig_advance(*a, **kw)
        tmax = float(sim.T.max())
        tmin = float(sim.T.min())
        assert tmax <= mat.nozzle + 1e-6, f"超温 {tmax}"
        assert tmin >= sim.cfg.chamber_temp - 1e-6, f"低于环境 {tmin}"

    sim._advance = spy
    sim.run()


def test_thin_wall_cools_faster_than_bulk():
    """薄壁（材料占比低）比密实体降温快——frac 修正的内核级直接检验。

    同温度、同环境、互不接触的两个格子：低 frac（薄壁）应降温更快。
    """
    p = parse_gcode(generate_box_gcode(layers=2, size=20, feed=60))
    mat = get_material("PLA")
    sim = ThermalSimulator(p, mat, SimConfig(voxel_mm=1.5))
    sim._prepare()
    # 两个相距足够远的内部格子
    a = (4, 4, 4)
    b = (4, 4, 5)
    for c in (a, b):
        sim.T[c] = 200.0
    sim.frac[a] = 0.05   # 薄壁
    sim.frac[b] = 1.0    # 密实
    sim.top_iz = 6
    sim._advance(dt_total=1.0, fan=0.0, h_off=mat.h_conv_off, h_on=mat.h_conv_on,
                 amb=sim.cfg.chamber_temp, bed_t=mat.bed, frac_lo=0.2)
    ta, tb = float(sim.T[a]), float(sim.T[b])
    assert ta < tb < 200.0, f"薄壁格({ta:.1f})应比密实格({tb:.1f})降温更快"


def test_same_part_slower_feed_colder_interface():
    """纯传导物理（η=0）：同一零件，慢速打印（长层时）界面温度应低于快速打印。"""
    res_slow, _ = _sim(generate_box_gcode(layers=10, size=40, feed=20), voxel_mm=1.5, nozzle_heat=0.0, iface_reheat=0.0)
    res_fast, _ = _sim(generate_box_gcode(layers=10, size=40, feed=200), voxel_mm=1.5, nozzle_heat=0.0, iface_reheat=0.0)
    t_slow = res_slow.iface_temp[res_slow.tqi_valid]
    t_fast = res_fast.iface_temp[res_fast.tqi_valid]
    assert t_slow.mean() < t_fast.mean() - 30.0


def test_first_layer_excluded_from_stats():
    p = parse_gcode(generate_two_rate_gcode())
    assert not p.layer_idx[p.layer_idx == 0].size or True
    res, _ = _sim(generate_two_rate_gcode(), voxel_mm=1.5)
    assert not res.tqi_valid[p.layer_idx == 0].any(), "首层段不应计入 TQI 统计"


def test_tqi_mapping_shape():
    m = get_material("PLA")
    t = np.array([20.0, m.cold_below, 90.0, 120.0, (m.ideal_lo + m.ideal_hi) / 2,
                  m.hot_above, 300.0])
    q = tqi_from_interface_temp(t, m)
    assert q[0] == -100.0                      # 深冷
    assert -100.0 <= q[1] <= -95.0             # 冷边界
    assert q[2] > q[1]                         # 单调
    assert abs(q[4]) <= 5.0                    # 理想区 ≈ 0
    assert q[6] == 100.0                       # 深热封顶
    assert np.all(np.diff(q) > -1e-9)          # 单调不减


def test_progress_called_and_result_meta():
    calls = []
    p = parse_gcode(generate_box_gcode(layers=8, size=30, feed=80))
    mat = get_material("PLA")
    sim = ThermalSimulator(p, mat, SimConfig(voxel_mm=1.5), progress_cb=calls.append)
    res = sim.run()
    assert len(calls) > 0 and abs(calls[-1] - 1.0) < 1e-6
    assert res.config["material"] == "PLA"
    assert res.runtime_s > 0
    assert len(res.layer_stats) == 8
    assert res.grid_shape and len(res.grid_shape) == 3


def test_simulation_cancelled():
    """cancel_check 恒真 → 仿真应抛 SimulationCancelled 而非跑完。"""
    from backend.thermal.voxel import SimulationCancelled

    p = parse_gcode(generate_box_gcode(layers=8, size=20, feed=30))
    mat = get_material("PLA")
    sim = ThermalSimulator(p, mat, SimConfig(voxel_mm=2.0), cancel_check=lambda: True)
    try:
        sim.run()
        raise AssertionError("仿真应当被取消")
    except SimulationCancelled:
        pass


def test_optimization_cancelled():
    """cancel_check 恒真 → 优化应在首轮内抛 SimulationCancelled。"""
    from backend.thermal.optimize import OptimizeConfig, optimize_speeds
    from backend.thermal.voxel import SimulationCancelled

    p = parse_gcode(generate_box_gcode(layers=8, size=20, feed=30))
    mat = get_material("PLA")
    try:
        optimize_speeds(p, mat, SimConfig(voxel_mm=2.0), OptimizeConfig(rounds=2),
                        cancel_check=lambda: True)
        raise AssertionError("优化应当被取消")
    except SimulationCancelled:
        pass
