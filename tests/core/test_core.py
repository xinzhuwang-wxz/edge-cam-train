"""core：paths 标准目录、set_seed 可复现、logger 可用。"""

from __future__ import annotations

from edge_cam.core.logging import get_logger
from edge_cam.core.paths import CONFIGS_DIR, PROJECT_ROOT
from edge_cam.core.seed import set_seed


def test_project_root_has_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert CONFIGS_DIR.name == "configs"


def test_set_seed_reproducible() -> None:
    import random

    set_seed(42)
    a = [random.random() for _ in range(3)]
    set_seed(42)
    b = [random.random() for _ in range(3)]
    assert a == b


def test_get_logger() -> None:
    assert get_logger("test") is not None
