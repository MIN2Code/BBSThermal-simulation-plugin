"""管线打包与 API 集成测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.gcode.parser import parse_gcode
from backend.thermal.materials import get_material
from backend.thermal.voxel import SimConfig, ThermalSimulator
from backend.pipeline import SEG_STRIDE, create_job_from_text, pack_preview, pack_result
from tests.gcode_gen import generate_box_gcode


def test_pack_preview_roundtrip():
    p = parse_gcode(generate_box_gcode(layers=4, size=20, feed=60))
    payload, meta = pack_preview(p)
    assert len(payload) == p.num_segments * SEG_STRIDE
    assert meta["num_segments"] == p.num_segments
    f32 = np.frombuffer(payload, dtype=np.float32).reshape(p.num_segments, 10)
    u8 = np.frombuffer(payload, dtype=np.uint8).reshape(p.num_segments, SEG_STRIDE)
    # 有效标志全 0 → 前端按特性着色
    assert (u8[:, 37] == 0).all()
    # 几何与解析结果一致（lexsort 后仍是同一集合：按层号+时间的排序键）
    assert np.allclose(np.sort(f32[:, 0]), np.sort(p.geometry[:, 0]), rtol=1e-6)
    assert meta["layer_ranges"], "层区间不能为空"
    total = sum(r["count"] for r in meta["layer_ranges"])
    assert total == p.num_segments


def test_pack_result_roundtrip():
    p = parse_gcode(generate_box_gcode(layers=5, size=20, feed=60))
    mat = get_material("PLA")
    res = ThermalSimulator(p, mat, SimConfig(voxel_mm=1.5)).run()
    payload, meta = pack_result(p, res, "PLA")
    assert len(payload) == p.num_segments * SEG_STRIDE
    f32 = np.frombuffer(payload, dtype=np.float32).reshape(p.num_segments, 10)
    i32 = np.frombuffer(payload, dtype=np.int32).reshape(p.num_segments, 10)
    u8 = np.frombuffer(payload, dtype=np.uint8).reshape(p.num_segments, SEG_STRIDE)
    # TQI 与界面温度逐段存在
    assert np.all(np.isfinite(f32[:, 6])) and np.all(np.isfinite(f32[:, 7]))
    assert f32[:, 6].min() >= -100.0 and f32[:, 6].max() <= 100.0
    # 层号区间单调且覆盖
    layers = i32[:, 8]
    assert (np.diff(layers) >= 0).all()
    assert meta["layer_ranges"][0]["start"] == 0
    assert u8[:, 37].sum() == int(res.tqi_valid.sum())


def test_create_job_and_preview_endpoints():
    job_id, summary = create_job_from_text(generate_box_gcode(layers=3, size=15, feed=50))
    assert summary["layers"] == 3
    from backend.pipeline import get_job
    job = get_job(job_id)
    assert job.status == "parsed"
    assert job.payload is not None and job.meta is not None
    assert job.meta["preview"] is True
    assert job.meta["summary"]["segments"] == summary["segments"]


def test_repack_3mf_preserves_metadata_and_updates_content():
    import zipfile
    from backend.gcode.bambu3mf import repack_3mf

    src = (Path(__file__).resolve().parents[1] / "测试模型" / "佩里卡_plate_10.gcode.3mf")
    if not src.exists():
        return  # 样例缺失则跳过（CI 环境）
    raw = src.read_bytes()
    new_gcode = b"G1 X1 Y2 E1 F1200\n"
    out = repack_3mf(raw, new_gcode)

    zsrc = zipfile.ZipFile(src)
    zout = zipfile.ZipFile(__import__("io").BytesIO(out))
    names_src = set(zsrc.namelist())
    names_out = set(zout.namelist())
    assert names_src == names_out, "成员集合应完全一致"

    # G-code 成员已替换、md5 已更新
    gcode_member = "Metadata/plate_10.gcode"
    assert zout.read(gcode_member) == new_gcode
    md5_member = gcode_member + ".md5"
    import hashlib
    assert zout.read(md5_member) == hashlib.md5(new_gcode).hexdigest().encode()
    # 工程配置等元数据原样保留
    assert zout.read("Metadata/project_settings.config") == zsrc.read("Metadata/project_settings.config")
