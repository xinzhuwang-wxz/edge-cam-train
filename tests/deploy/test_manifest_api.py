"""OTA manifest API：healthz + 按 platform/channel 返回 stable 模型。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from edge_cam.contracts.schemas.model_card import ModelCard
from edge_cam.deploy.manifest_api.app import create_app
from edge_cam.registry.store import ModelRegistry


def _client(tmp_path: Path) -> TestClient:
    index = tmp_path / "models.yaml"
    reg = ModelRegistry(index)
    reg.register(
        ModelCard(
            name="birds525",
            task="classify",
            backbone="efficientnet_lite0",
            num_classes=525,
            input_size=224,
            platform="v85x",
            channel="stable",
            gate_pass=True,
        )
    )
    reg.register(  # candidate，不应出现在 stable manifest
        ModelCard(
            name="cand",
            task="classify",
            backbone="x",
            num_classes=2,
            input_size=224,
            platform="v85x",
            channel="candidate",
        )
    )
    return TestClient(create_app(index))


def test_healthz(tmp_path: Path) -> None:
    assert _client(tmp_path).get("/healthz").json() == {"status": "ok"}


def test_manifest_returns_only_stable_for_platform(tmp_path: Path) -> None:
    resp = _client(tmp_path).get("/v1/manifest/v85x/stable")
    assert resp.status_code == 200
    body = resp.json()
    names = [m["name"] for m in body["models"]]
    assert names == ["birds525"]  # 只 stable + v85x


def test_manifest_empty_for_unknown_platform(tmp_path: Path) -> None:
    body = _client(tmp_path).get("/v1/manifest/dev/stable").json()
    assert body["models"] == []
