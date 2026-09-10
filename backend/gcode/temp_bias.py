"""整件喷嘴温度偏置（表面一致/结合质量的温度旋钮·台阶一）。

把 G-code 中所有打印温度指令（M104/M109，S≥100 的实设定值）统一平移
ΔT——常温打印是单一偏置；变温打印（温度塔）是整条温度曲线平移。
S<100 的关温/待机指令不动。

设计约定（台阶二再扩展）：
- 只改既有指令的 S 值，不插入新行——无热端延迟时机问题，变温件天然生效；
- 偏置范围由 API 层 clamp（±15°C），材料窗口安全边界来自温度塔档案（后续接入）。
"""
from __future__ import annotations

import re

_CMD_RE = re.compile(r"^(\s*M10[49]\b)")
_S_RE = re.compile(r"\bS(-?[0-9.]+)")


def apply_nozzle_temp_bias(text: str, delta_t: float) -> tuple[str, int]:
    """对 G-code 文本施加喷嘴温度偏置。返回 (新文本, 修改的指令数)。"""
    if not delta_t:
        return text, 0
    out = []
    changed = 0
    for line in text.splitlines(keepends=True):
        if _CMD_RE.match(line):
            m = _S_RE.search(line)
            if m:
                v = float(m.group(1))
                if v >= 100.0:
                    nv = v + delta_t
                    tag = f"{nv:.1f}".rstrip("0").rstrip(".")
                    line = line[: m.start()] + "S" + tag + line[m.end():]
                    changed += 1
        out.append(line)
    return "".join(out), changed
