"""数据量 scaling 子集（round2 §7）：确定性抽 train split 的 frac，**val/test 固定不动**。

最佳实践（用户模型 + 修正）：
- **图存一份**（raw_root），manifest 只存引用 → scaling 子集 = records 子集，零图复制（ADR-0006）。
- **只抽 train、val/test 全保留** → 所有 fraction 在**同一 held-out test** 上评，曲线可比。
- **确定性 + 嵌套**（按 `path` sha256 排序取前 frac）→ 20% ⊂ 50% ⊂ 100%，"加数据"是真加。

用法（Hydra multirun 每 run 一个 frac）：`m2 = subsample_train(m, 0.2)` → 训练消费 m2（train 20%）。
"""

from __future__ import annotations

import hashlib
import math

from edge_cam.contracts.schemas.detection_manifest import DetectionManifest


def _hkey(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def subsample_train(manifest: DetectionManifest, frac: float) -> DetectionManifest:
    """抽 train split 的前 `frac`（确定性、嵌套），val/test 原样保留。返回新 manifest（图不复制）。

    frac∈(0,1]；按 `path` hash 排序取前 ceil(frac·n_train) 条 → 嵌套子集。frac≥1 原样返回。
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac 须 ∈ (0,1]，得到 {frac}")
    train = [r for r in manifest.records if r.split == "train"]
    other = [r for r in manifest.records if r.split != "train"]
    if frac >= 1.0 or not train:
        return manifest
    k = max(1, math.ceil(frac * len(train)))
    kept = sorted(train, key=lambda r: _hkey(r.path))[:k]
    return manifest.model_copy(update={"records": kept + other})
