"""测试统一强制 CPU 仿真路径（GPU 有专门的独立测试）。"""
import pytest


@pytest.fixture(autouse=True)
def _force_cpu_sim(monkeypatch):
    try:
        from backend.thermal import gpu_sim
        monkeypatch.setattr(gpu_sim, "gpu_available", lambda: False)
    except ImportError:
        pass
    yield
