"""ChannelPolicy 契约：默认值、yaml 往返、仓内配置加载、rollout 边界。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from edge_cam.contracts.schemas.channel import ChannelPolicy
from edge_cam.core.paths import CONFIGS_DIR


def test_defaults() -> None:
    p = ChannelPolicy()
    assert p.platform == "v85x"
    assert p.rollout_percent == 0
    assert p.cloud_fallback.enabled is False


def test_repo_configs_load() -> None:
    stable = ChannelPolicy.from_yaml(CONFIGS_DIR / "channels" / "stable.yaml")
    assert stable.channel == "stable"
    assert stable.rollout_percent == 100
    assert stable.cloud_fallback.enabled is True
    cand = ChannelPolicy.from_yaml(CONFIGS_DIR / "channels" / "candidate.yaml")
    assert cand.channel == "candidate"
    assert cand.rollout_percent == 10


def test_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("platform: v85x\nchannel: candidate\nrollout_percent: 25\n", encoding="utf-8")
    p = ChannelPolicy.from_yaml(cfg)
    assert p.rollout_percent == 25


def test_rollout_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        ChannelPolicy(rollout_percent=150)
