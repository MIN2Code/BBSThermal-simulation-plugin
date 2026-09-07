"""Bambu Studio 配置修补：把 Helio 集成指向本地引擎并注入本地 PAT。

用法：先完全关闭 Bambu Studio，然后运行：
    python patch_bs_config.py
会修改 %APPDATA%\\BambuStudio\\BambuStudio.conf：
    helio_api_china / helio_api_other → http://127.0.0.1:8760/graphql/helio
    helio_pat_china  / helio_pat_other → local-pat（本地引擎不校验，非空即可）
    helio_enable → true
幂等：重复运行无副作用。
"""
import os

CONF = os.path.join(os.environ.get("APPDATA", ""), "BambuStudio", "BambuStudio.conf")
LOCAL_URL = "http://127.0.0.1:8760/graphql/helio"
PAT = "local-pat"

UPDATES = {
    "helio_api_china": LOCAL_URL,
    "helio_api_other": LOCAL_URL,
    "helio_pat_china": PAT,
    "helio_pat_other": PAT,
    "helio_enable": "true",
}


def main() -> int:
    if not os.path.exists(CONF):
        print(f"未找到配置文件: {CONF}")
        return 1
    with open(CONF, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    pending = dict(UPDATES)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if '":"' not in stripped and '":' not in stripped:
            continue
        head, _, tail = line.strip().partition('":')
        key = head.lstrip('"').strip()
        if key in pending:
            indent = line[: len(line) - len(line.lstrip())]
            val = pending.pop(key)
            formatted = val if val in ("true", "false") else f'"{val}"'
            lines[i] = f'{indent}"{key}": {formatted},'
    # 未存在的键：插入到 helio_api_china 行之后（或文件首个键附近）
    if pending:
        insert_at = None
        for i, line in enumerate(lines):
            if '"helio_api_china"' in line:
                insert_at = i + 1
                break
        if insert_at is None:
            for i, line in enumerate(lines):
                if line.strip().startswith('"'):
                    insert_at = i
                    break
        for key, val in reversed(list(pending.items())):
            indent = '        '
            formatted = val if val in ("true", "false") else f'"{val}"'
            lines.insert(insert_at, f'{indent}"{key}": {formatted},')

    with open(CONF, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("已写入以下配置：")
    for key, val in UPDATES.items():
        print(f"  {key} = {val}")
    print(f"\n配置文件: {CONF}")
    print("现在可以启动 Bambu Studio 了。")
    return 0


if __name__ == "__main__":
    sys_exit = main()
    import sys
    sys.exit(sys_exit)
