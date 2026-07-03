"""Model card（engineering §6）：模型产物的可追溯卡片，registry/OTA 的载体。

记录 backbone / 类数 / 输入 / **许可与数据来源** / 指标 / 量化 / 平台 / sha256 / 版本 / channel。
许可与 provenance 是商用硬约束（plan §C.1：换 head 不洗白上游 license）。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Task = Literal["detect", "classify", "cascade"]  # cascade=级联产品级评估卡（检测+分类组合）
Platform = Literal["v85x", "dev"]  # 边侧平台 Literal（engineering §6）
Channel = Literal["candidate", "stable"]
Precision = Literal["fp32", "int8"]


class Provenance(BaseModel):
    """数据/权重来源与许可（逐项可追溯，随产物披露）。"""

    datasets: list[str] = Field(default_factory=list)  # 如 ["coco-2017", "open-images-v7"]
    licenses: list[str] = Field(default_factory=list)  # 如 ["CC-BY-4.0", "unverified"]
    commercial_safe: bool = False  # 是否可进商用出货权重
    notes: str = ""


class ModelCard(BaseModel):
    """一个模型产物（.onnx/.nb/.ckpt）的卡片。"""

    name: str
    task: Task
    backbone: str
    num_classes: int
    input_size: int
    precision: Precision = "fp32"
    platform: Platform = "dev"
    artifact_path: str = ""
    sha256: str = ""
    version: str = "v0"
    channel: Channel = "candidate"
    provenance: Provenance = Field(default_factory=Provenance)
    metrics: dict[str, float] = Field(default_factory=dict)  # 如 {"top1": 0.9, "int8_drop": 0.04}
    gate_pass: bool = False  # gate 过了才可 promote 到 stable（engineering §3）

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ModelCard:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
