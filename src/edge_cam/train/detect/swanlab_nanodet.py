"""NanoDet 训练日志 → SwanLab（非侵入，不 fork NanoDet §3）。

NanoDet 在子进程 + 自己的 Lightning 循环里训，我们**解析日志行**抽指标 → 实时 `swanlab.log`（train
loss 与 val 指标同一 run）。解析纯函数可测；runner 用 swanlab（box）。

NanoDet 日志样例：
  ...INFO:Train|Epoch1/24|Iter50(51/153)| lr:1.00e-04| loss_qfl:1.72| loss_bbox:1.18| loss_dfl:0.51|
  ...INFO:Val_metrics: {'mAP': 0.354, 'AP_50': 0.565, ...}
  | bird | 75.4 | 46.7 | squirrel | 43.4 | 22.2 |
"""

from __future__ import annotations

import ast
import re

_TRAIN_RE = re.compile(r"Train\|Epoch(\d+)/\d+\|Iter(\d+)\(")
_KV_RE = re.compile(r"(\w+):([\d.eE+-]+)")
_VAL_RE = re.compile(r"Val_metrics:\s*(\{.*\})")
# 逐类行：`| bird | 75.4 | 46.7 |`（AP50, mAP）；一行可含 2 个类（NanoDet 双列排版）
_CLS_RE = re.compile(
    r"\|\s*(bird|squirrel|cat|person|other_animal)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*"
)


def parse_train(line: str) -> dict | None:
    """训练行 → {step(global iter), epoch, lr, loss_*}。非训练行返回 None。"""
    m = _TRAIN_RE.search(line)
    if not m:
        return None
    epoch, it = int(m.group(1)), int(m.group(2))
    out: dict[str, float] = {"train/epoch": float(epoch), "_iter": float(it)}
    for k, v in _KV_RE.findall(
        line.split("Iter", 1)[1]
    ):  # Iter 之后才是 lr/loss（避开 Epoch 数字）
        if k in ("lr",) or k.startswith("loss") or k.startswith("aux_loss"):
            out[f"train/{k}"] = float(v)
    return out


def parse_val(line: str) -> dict | None:
    """Val_metrics 行 → {val/mAP, val/AP_50, ...}。非该行返回 None。"""
    m = _VAL_RE.search(line)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None
    return {f"val/{k}": float(v) for k, v in d.items() if _isnum(v)}


def parse_per_class(line: str) -> dict | None:
    """逐类行 → {val/bird_AP50, val/bird_mAP, ...}（含双列）。非该行返回 None。"""
    found = _CLS_RE.findall(line)
    if not found:
        return None
    out: dict[str, float] = {}
    for name, ap50, mAP in found:
        if ap50 != "nan":
            out[f"val/{name}_AP50"] = float(ap50)
        if mAP != "nan":
            out[f"val/{name}_mAP"] = float(mAP)
    return out or None


def _isnum(v) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and v != v)  # 排 nan
