"""项目标准路径（engineering §6 core）。集中定义，避免各处硬编码相对路径。"""

from __future__ import annotations

from pathlib import Path

# src/edge_cam/core/paths.py → parents[3] = 仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure(path: Path) -> Path:
    """确保目录存在并返回它。"""
    path.mkdir(parents=True, exist_ok=True)
    return path
