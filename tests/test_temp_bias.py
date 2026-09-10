"""温度偏置后处理测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.gcode.temp_bias import apply_nozzle_temp_bias


def test_bias_shifts_print_temps_only():
    text = "\n".join([
        "M104 S210 ; set hotend",
        "M109 S210",
        "G1 X10 Y10 E1 F1200",
        "M104 S220",
        "M104 S0 ; turn off",
        "M106 S255",
        "M140 S55",
        "",
    ])
    out, changed = apply_nozzle_temp_bias(text, 5.0)
    lines = out.splitlines()
    assert changed == 3, f"应改 3 条打印温度指令，实际 {changed}"
    assert "M104 S215" in lines[0]
    assert "M109 S215" in lines[1]
    assert "M104 S225" in lines[3]
    assert "M104 S0" in lines[4], "关温指令不得偏移"
    assert "M106 S255" in lines[5] and "M140 S55" in lines[6], "风扇/热床不动"
    # 行尾注释保留
    assert "; set hotend" in lines[0]


def test_bias_negative_and_clamp_range():
    out, changed = apply_nozzle_temp_bias("M104 S210\n", -8.0)
    assert "M104 S202" in out and changed == 1
    # 偏移到 S<100 的结果仍保留（不做二次过滤——API 层负责 clamp ±15）
    out2, _ = apply_nozzle_temp_bias("M104 S105\n", -15.0)
    assert "M104 S90" in out2


def test_temp_tower_whole_curve_shifts():
    """变温打印（温度塔）：所有档位整体平移 = 温度曲线平移。"""
    blocks = "\n".join(f"M104 S{t}\nM109 S{t}\nG1 X1 Y1 E1" for t in (190, 200, 210, 220))
    out, changed = apply_nozzle_temp_bias(blocks, 7.0)
    assert changed == 8
    for t in (197, 207, 217, 227):
        assert f"M104 S{t}" in out
