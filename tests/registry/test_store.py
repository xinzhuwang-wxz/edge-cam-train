"""ModelRegistry：register/list/get/promote（gate 门）+ sha256。"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_cam.contracts.schemas.model_card import ModelCard
from edge_cam.registry.store import ModelRegistry, sha256_file


def _card(name="m", version="v0", gate_pass=False, channel="candidate") -> ModelCard:
    return ModelCard(
        name=name,
        version=version,
        task="classify",
        backbone="efficientnet_lite0",
        num_classes=525,
        input_size=224,
        gate_pass=gate_pass,
        channel=channel,
    )


def test_register_and_get(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "models.yaml")
    reg.register(_card())
    got = reg.get("m")
    assert got is not None and got.name == "m"


def test_register_overwrites_same_version(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "models.yaml")
    reg.register(_card())
    reg.register(_card(gate_pass=True))  # 同名同版本 → 覆盖
    assert len(reg.list()) == 1
    assert reg.get("m").gate_pass is True


def test_promote_requires_gate(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "models.yaml")
    reg.register(_card(gate_pass=False))
    with pytest.raises(ValueError, match="gate"):
        reg.promote("m", "v0")
    reg.register(_card(gate_pass=True))
    promoted = reg.promote("m", "v0")
    assert promoted.channel == "stable"
    assert len(reg.list(channel="stable")) == 1


def test_sha256_file(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    # sha256("hello")
    assert sha256_file(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
