"""板子相关桩：接口在位、未配 SDK 时干净抛 NotImplementedError。"""

from __future__ import annotations

import pytest

from edge_cam.deploy.packager.acuity_packager import AcuityPackager
from edge_cam.deploy.packager.base import PackagerBackend


def test_acuity_satisfies_protocol() -> None:
    assert isinstance(AcuityPackager(), PackagerBackend)  # runtime_checkable Protocol


def test_acuity_pack_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="pegasus"):
        AcuityPackager().pack("m.onnx", "m.nb", "dataset.txt")


def test_viplite_runner_stub_raises() -> None:
    from edge_cam.edge.viplite_runner.runner import VIPLiteRunner

    with pytest.raises(NotImplementedError, match="VIPLite"):
        VIPLiteRunner("m.nb")
