"""ModelCard：字段校验、provenance、存盘往返。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from edge_cam.contracts.schemas.model_card import ModelCard, Provenance


def test_minimal_card() -> None:
    card = ModelCard(
        name="m", task="classify", backbone="efficientnet_lite0", num_classes=525, input_size=224
    )
    assert card.precision == "fp32"
    assert card.channel == "candidate"
    assert card.gate_pass is False
    assert card.provenance.commercial_safe is False


def test_invalid_task_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelCard(name="m", task="segment", backbone="x", num_classes=1, input_size=224)  # type: ignore[arg-type]


def test_provenance_and_save_load(tmp_path: Path) -> None:
    card = ModelCard(
        name="birds525",
        task="classify",
        backbone="efficientnet_lite0",
        num_classes=525,
        input_size=224,
        precision="int8",
        platform="v85x",
        provenance=Provenance(
            datasets=["birds525"], licenses=["unverified"], commercial_safe=False
        ),
        metrics={"top1": 0.9, "int8_drop": 0.04},
    )
    out = tmp_path / "card.json"
    card.save(out)
    assert ModelCard.load(out) == card
