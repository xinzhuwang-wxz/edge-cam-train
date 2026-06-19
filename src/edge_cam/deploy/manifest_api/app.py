"""OTA manifest API（engineering §6：FastAPI OTA routes）。

端侧按 platform+channel 拉「该升到哪个模型」。承载 .nb 灰度（candidate/stable）+
cloud-fallback 锚点。OTA bundle 见 plan §C.5（net.nb + taxonomy + mask + min_runtime_abi
原子下发）—— 这里返回 ModelCard 索引，二进制由对象存储分发。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from edge_cam.registry.store import ModelRegistry


def create_app(index_path: str | Path) -> FastAPI:
    app = FastAPI(title="edge-cam OTA manifest API", version="0.1.0")
    registry = ModelRegistry(index_path)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/manifest/{platform}/{channel}")
    def manifest(platform: str, channel: str) -> dict:
        cards = [c for c in registry.list(channel=channel) if c.platform == platform]
        return {
            "platform": platform,
            "channel": channel,
            "models": [c.model_dump() for c in cards],
        }

    return app
