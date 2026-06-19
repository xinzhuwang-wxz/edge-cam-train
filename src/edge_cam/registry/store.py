"""模型 registry 薄层（engineering §3：git-yaml 索引 + sha256 + channel）。

成熟模式、格式无关：ModelCard 索引进一个 git 跟踪的 models.yaml；channel(candidate/stable)
承载灰度；**gate_pass=True 才能 promote 到 stable**。单人项目用这条腿即可，无需 MLflow。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from edge_cam.contracts.schemas.model_card import Channel, ModelCard


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """流式分块算 sha256（大 .nb 友好，engineering §3）。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class ModelRegistry:
    """models.yaml 索引（一组 ModelCard）。register/list/get/promote。"""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)

    def _load(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        return yaml.safe_load(self.index_path.read_text(encoding="utf-8")) or []

    def _save(self, cards: list[dict]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            yaml.safe_dump(cards, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    def register(self, card: ModelCard) -> ModelCard:
        """登记一张卡片（同名同版本则覆盖）。若 artifact_path 存在则补算 sha256。"""
        if card.artifact_path and Path(card.artifact_path).exists() and not card.sha256:
            card = card.model_copy(update={"sha256": sha256_file(card.artifact_path)})
        cards = [
            c for c in self._load() if not (c["name"] == card.name and c["version"] == card.version)
        ]
        cards.append(card.model_dump())
        self._save(cards)
        return card

    def list(self, *, channel: Channel | None = None, task: str | None = None) -> list[ModelCard]:
        out = [ModelCard.model_validate(c) for c in self._load()]
        if channel is not None:
            out = [c for c in out if c.channel == channel]
        if task is not None:
            out = [c for c in out if c.task == task]
        return out

    def get(
        self, name: str, *, version: str | None = None, channel: Channel | None = None
    ) -> ModelCard | None:
        matches = [
            c
            for c in self.list(channel=channel)
            if c.name == name and (version is None or c.version == version)
        ]
        return matches[-1] if matches else None

    def promote(self, name: str, version: str) -> ModelCard:
        """candidate → stable；**必须 gate_pass**（engineering §3 质量门）。"""
        cards = self._load()
        for c in cards:
            if c["name"] == name and c["version"] == version:
                if not c.get("gate_pass"):
                    raise ValueError(
                        f"promote 拒绝：{name} v{version} 未过 gate（gate_pass=False）"
                    )
                c["channel"] = "stable"
                self._save(cards)
                return ModelCard.model_validate(c)
        raise KeyError(f"registry 无此模型：{name} v{version}")
