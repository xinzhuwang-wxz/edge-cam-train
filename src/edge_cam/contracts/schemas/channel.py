"""OTA 通道策略契约（engineering §6 configs/channels；plan §C.5 OTA 契约 / §C.10 云锚点）。

每个 channel（candidate/stable）一份策略：灰度比例 + 运行时 ABI 下限 + 云锚点回退。
ModelCard 承载「升到哪个模型」，ChannelPolicy 承载「这个通道怎么发、回退去哪」——
manifest_api 把二者合并下发给端侧（§C.5 原子下发：net.nb + taxonomy + mask + min_runtime_abi）。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from edge_cam.contracts.schemas.model_card import Channel, Platform


class CloudFallback(BaseModel):
    """云 API 锚点（plan §C.10：端侧低置信/稀有种/不在地域清单 → 回退云，本仓不实现云端）。"""

    enabled: bool = False
    endpoint: str = ""  # 云识别 API url（占位，端侧契约用）


class ChannelPolicy(BaseModel):
    """单个 OTA 通道的发布策略。"""

    platform: Platform = "v85x"
    channel: Channel = "candidate"
    rollout_percent: int = Field(default=0, ge=0, le=100)  # 灰度比例
    min_runtime_abi: str = ""  # VIPLite/awnn ABI 下限（版本不匹配端侧拒绝升级）
    cloud_fallback: CloudFallback = Field(default_factory=CloudFallback)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChannelPolicy:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)
