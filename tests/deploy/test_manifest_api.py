"""OTA manifest API：healthz + 按 platform/channel 返回 stable 模型。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from edge_cam.contracts.schemas.model_card import ModelCard
from edge_cam.core.paths import CONFIGS_DIR
from edge_cam.deploy.manifest_api.app import create_app
from edge_cam.registry.store import ModelRegistry


def _index(tmp_path: Path) -> Path:
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
    return index


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(_index(tmp_path)))


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


def test_manifest_without_channels_dir_omits_fallback(tmp_path: Path) -> None:
    """向后兼容：未给 channels_dir → 响应不含 cloud_fallback。"""
    body = _client(tmp_path).get("/v1/manifest/v85x/stable").json()
    assert "cloud_fallback" not in body


def test_manifest_attaches_channel_policy(tmp_path: Path) -> None:
    """给 channels_dir（用仓内真实配置）→ 响应附 cloud_fallback + min_runtime_abi。"""
    client = TestClient(create_app(_index(tmp_path), channels_dir=CONFIGS_DIR / "channels"))
    body = client.get("/v1/manifest/v85x/stable").json()
    assert body["min_runtime_abi"] == "viplite-1.0"
    assert body["rollout_percent"] == 100
    assert body["cloud_fallback"]["enabled"] is True
    assert body["cloud_fallback"]["endpoint"].startswith("https://")
