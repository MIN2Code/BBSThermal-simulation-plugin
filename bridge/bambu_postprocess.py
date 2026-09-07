"""Bambu Studio 后处理脚本桥接：切片完成后自动把 G-code 送入 Thermal Assess。

用法（Bambu Studio → 打印设置 → 其他 → 后处理脚本，添加本文件）：
    Windows:  <系统Python> J:\\claudebox\\thermal-assess\\bridge\\bambu_postprocess.py
    （脚本仅用标准库，任何 Python 3.8+ 均可运行）

行为：切片完成后，把 G-code 副本 POST 到本地引擎 http://127.0.0.1:8760，
并自动用默认浏览器打开该任务的预览页。不修改 G-code 本身。
"""
import json
import sys
import urllib.request
import urllib.error
import uuid
import webbrowser

SERVER = "http://127.0.0.1:8760"


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: bambu_postprocess.py <gcode文件路径>", file=sys.stderr)
        return 1
    path = sys.argv[1]
    try:
        with open(path, "rb") as f:
            data = f.read()
        boundary = uuid.uuid4().hex
        body = b"".join([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{path}"\r\n'.encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            data,
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        req = urllib.request.Request(
            f"{SERVER}/api/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            job = json.loads(resp.read().decode())
        job_id = job.get("job_id", "")
        webbrowser.open(f"{SERVER}/?job={job_id}")
        print(f"[thermal-assess] 已提交分析: {SERVER}/?job={job_id}")
        return 0
    except urllib.error.URLError:
        print("[thermal-assess] 本地引擎未启动（http://127.0.0.1:8760），跳过分析", file=sys.stderr)
        return 0  # 引擎没开不应阻塞切片流程


if __name__ == "__main__":
    sys.exit(main())
