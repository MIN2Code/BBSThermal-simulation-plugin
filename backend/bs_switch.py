"""Bambu Studio Helio 端点切换器：本地引擎 ⇄ 官方 Helio 云。

原理：Bambu Studio 从 %APPDATA%\\BambuStudio\\BambuStudio.conf 读取
helio_api_china / helio_api_other（端点）与 helio_pat_china / helio_pat_other（PAT）。
切换 = 改写这些键 + helio_enable。

切回官方时 PAT 的处理：首次切到本地时会把当时的真实 PAT 备份到本地文件；
切回官方时自动还原备份；若没有备份则需用户提供。

安全：Bambu Studio 运行中时拒绝切换（退出时会用内存里的旧配置覆盖我们的修改）。
"""
from __future__ import annotations

import os
import subprocess

CONF_PATH = os.path.join(os.environ.get("APPDATA", ""), "BambuStudio", "BambuStudio.conf")
PAT_BACKUP = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".helio_pat_backup")

LOCAL_URL = "http://127.0.0.1:8760/graphql/helio"
OFFICIAL_URLS = {
    "helio_api_china": "https://api.helioam.cn/graphql",
    "helio_api_other": "https://api.helioadditive.com/graphql",
}
HELIO_KEYS = ("helio_api_china", "helio_api_other",
              "helio_pat_china", "helio_pat_other")


def bs_running() -> bool:
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq BambuStudio.exe"],
            capture_output=True, text=True, encoding="gbk", errors="replace", timeout=15,
        ).stdout
        return "BambuStudio.exe" in out
    except Exception:  # noqa: BLE001
        return False  # 探测失败按未运行处理（让用户自己确认）


def _backup_pat(pat: str) -> None:
    if pat and pat != "local-pat":
        with open(PAT_BACKUP, "w", encoding="utf-8") as f:
            f.write(pat)


def _restore_pat() -> str | None:
    if os.path.exists(PAT_BACKUP):
        with open(PAT_BACKUP, "r", encoding="utf-8") as f:
            pat = f.read().strip()
        return pat or None
    return None


def read_state(conf_path: str = CONF_PATH) -> dict:
    if not os.path.exists(conf_path):
        return {"exists": False, "mode": None, "bs_running": bs_running()}
    mode = None
    keys: dict[str, str] = {}
    with open(conf_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if '":"' not in s and '":' not in s:
                continue
            head, _, tail = s.partition('":')
            key = head.lstrip('"').strip()
            val = tail.strip().rstrip(",").strip('"')
            if key in HELIO_KEYS or key == "helio_enable":
                keys[key] = val
            if key == "helio_api_china" or key == "helio_api_other":
                if "127.0.0.1" in val:
                    mode = "local"
                elif mode is None:
                    mode = "helio"
    return {
        "exists": True, "mode": mode, "keys": keys,
        "helio_enable": keys.get("helio_enable") == "true",
        "pat_is_local": "local-pat" in (keys.get("helio_pat_china", "") + keys.get("helio_pat_other", "")),
        "pat_backup_exists": os.path.exists(PAT_BACKUP),
        "bs_running": bs_running(),
    }


def apply_mode(mode: str, conf_path: str = CONF_PATH, pat: str | None = None) -> dict:
    """mode: 'local' | 'helio'。返回结果 dict（成功/警告）。"""
    if mode not in ("local", "helio"):
        raise ValueError("mode 必须是 local 或 helio")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"未找到 Bambu Studio 配置: {conf_path}")
    if bs_running():
        return {"applied": False, "warning": "Bambu Studio 正在运行——请先完全关闭它再切换，"
                                             "否则退出时会把旧配置写回。"}

    with open(conf_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if mode == "local":
        # 备份现有真实 PAT（若有）
        for line in lines:
            s = line.strip()
            if '":' not in s:
                continue
            head, _, tail = s.partition('":')
            key = head.lstrip('"').strip()
            val = tail.strip().rstrip(",").strip('"')
            if key in ("helio_pat_china", "helio_pat_other"):
                _backup_pat(val)
        new_vals = {
            "helio_api_china": LOCAL_URL,
            "helio_api_other": LOCAL_URL,
            "helio_pat_china": "local-pat",
            "helio_pat_other": "local-pat",
            "helio_enable": "true",
        }
    else:
        restore = pat or _restore_pat()
        new_vals = {
            "helio_api_china": OFFICIAL_URLS["helio_api_china"],
            "helio_api_other": OFFICIAL_URLS["helio_api_other"],
            "helio_enable": "true",
        }
        if restore:
            new_vals["helio_pat_china"] = restore
            new_vals["helio_pat_other"] = restore
        pat_missing = restore is None

    changed = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if '":' not in s:
            continue
        head, _, _ = s.partition('":')
        key = head.lstrip('"').strip()
        if key in new_vals:
            indent = line[: len(line) - len(line.lstrip())]
            val = new_vals[key]
            formatted = val if val in ("true", "false") else f'"{val}"'
            lines[i] = f'{indent}"{key}": {formatted},'
            changed += 1

    with open(conf_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    out = {"applied": True, "mode": mode, "changed_keys": changed,
           "bs_running": bs_running()}
    if mode == "helio" and pat_missing:
        out["warning"] = "未找到备份的官方 PAT——切到官方 Helio 后请在 BS 偏好设置里重新生成。"
    return out
