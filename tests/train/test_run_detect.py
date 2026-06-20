"""检测训练入口（#14）：run_detect 经 TrainerBackend(NanodetBackend),非 shell 直调。

mock run_nanodet 的子进程函数 → 验证路由 + 句柄 + 导出,无需真 NanoDet env。"""

from __future__ import annotations

from omegaconf import OmegaConf

from edge_cam.train.backends import NanodetRef
from edge_cam.train.detect import run as detect_run
from edge_cam.train.detect import run_nanodet


def _cfg(export=True):
    return OmegaConf.create(
        {
            "backend": "nanodet",
            "detect": {
                "config_path": "cfg.yml",
                "checkpoint": "model_best.ckpt",
                "nanodet_repo": "third_party/nanodet",
                "nanodet_python": "python",
            },
            "export": {"enabled": export, "out_path": "/tmp/det.onnx", "input_size": 416},
        }
    )


def test_run_detect_routes_through_backend(monkeypatch) -> None:
    calls = {}

    def fake_train(*a, **k):
        calls["train"] = a
        return 0  # 退出码 0 = 成功

    def fake_export(*a, **k):
        calls["export"] = k
        return 0

    monkeypatch.setattr(run_nanodet, "train_nanodet", fake_train)
    monkeypatch.setattr(run_nanodet, "export_nanodet_onnx", fake_export)
    ref, onnx = detect_run.run_detect(_cfg(export=True))
    assert isinstance(ref, NanodetRef)  # 经 NanodetBackend 而非 shell 直调
    assert ref.checkpoint == "model_best.ckpt"
    assert "train" in calls and "export" in calls  # 子进程经 backend 被调
    assert str(onnx) == "/tmp/det.onnx"
    assert calls["export"]["input_shape"] == (416, 416)


def test_run_detect_train_failure_raises(monkeypatch) -> None:
    monkeypatch.setattr(run_nanodet, "train_nanodet", lambda *a, **k: 1)  # 非零退出
    import pytest

    with pytest.raises(RuntimeError, match="NanoDet 训练失败"):
        detect_run.run_detect(_cfg(export=False))
