"""Bambu Studio Helio 端点切换器（命令行版）。

用法：
    python patch_bs_config.py            # 切到本地引擎
    python patch_bs_config.py local      # 同上
    python patch_bs_config.py helio      # 切回官方 Helio 云（自动还原备份的官方 PAT）
    python patch_bs_config.py status     # 查看当前指向

也可在网页主界面顶栏的「BS Helio 端点」开关直接切换。
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from backend.bs_switch import apply_mode, read_state  # noqa: E402


def main() -> int:
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "local"
    if mode == "status":
        print(json.dumps(read_state(), ensure_ascii=False, indent=1))
        return 0
    if mode not in ("local", "helio"):
        print(f"未知模式: {mode}（可用: local / helio / status）", file=sys.stderr)
        return 1
    result = apply_mode(mode)
    if not result.get("applied"):
        print(f"未应用: {result.get('warning')}", file=sys.stderr)
        return 2
    state = read_state()
    print(f"已切换到: {mode}")
    print(f"  helio_api_china = {state['keys'].get('helio_api_china')}")
    print(f"  helio_api_other = {state['keys'].get('helio_api_other')}")
    print(f"  helio_pat = {'local-pat' if mode == 'local' else '（已还原备份的官方 PAT）'}")
    if result.get("warning"):
        print(f"注意: {result['warning']}")
    print("请启动（或重启）Bambu Studio 生效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
