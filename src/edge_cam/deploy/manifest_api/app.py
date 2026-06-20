"""OTA manifest API（engineering §6：FastAPI OTA routes）。

端侧按 platform+channel 拉「该升到哪个模型」。承载 .nb 灰度（candidate/stable）+
cloud-fallback 锚点。OTA bundle 见 plan §C.5（net.nb + taxonomy + mask + min_runtime_abi
原子下发）—— 这里返回 ModelCard 索引，二进制由对象存储分发。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from edge_cam.contracts.schemas.channel import ChannelPolicy
from edge_cam.registry.store import ModelRegistry


def create_app(index_path: str | Path, channels_dir: str | Path | None = None) -> FastAPI:
    """OTA manifest 服务。

    channels_dir 给定时，按 <channels_dir>/<channel>.yaml 加载 ChannelPolicy，
    在 manifest 响应里附 cloud_fallback + min_runtime_abi（§C.5 原子下发）；
    缺省/缺文件时省略这些字段，保持向后兼容。
    """
    app = FastAPI(title="edge-cam OTA manifest API", version="0.1.0")
    registry = ModelRegistry(index_path)
    channels_dir = Path(channels_dir) if channels_dir else None

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/manifest/{platform}/{channel}")
    def manifest(platform: str, channel: str) -> dict:
        cards = [c for c in registry.list(channel=channel) if c.platform == platform]
        out: dict = {
            "platform": platform,
            "channel": channel,
            "models": [c.model_dump() for c in cards],
        }
        if channels_dir:
            policy_file = channels_dir / f"{channel}.yaml"
            if policy_file.exists():
                policy = ChannelPolicy.from_yaml(policy_file)
                out["min_runtime_abi"] = policy.min_runtime_abi
                out["rollout_percent"] = policy.rollout_percent
                out["cloud_fallback"] = policy.cloud_fallback.model_dump()
        return out

    return app
