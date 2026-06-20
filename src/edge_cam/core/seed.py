"""统一随机种子（engineering §6 core）：实验可复现。"""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 0, deterministic: bool = False) -> int:
    """设 random/numpy/torch 种子；deterministic=True 时尽量关非确定性算子。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return seed
