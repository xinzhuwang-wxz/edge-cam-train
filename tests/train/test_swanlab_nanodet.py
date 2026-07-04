"""NanoDet 日志 → SwanLab 指标解析（train loss / val mAP·AP50 / 逐类）。"""

from __future__ import annotations

from edge_cam.train.detect.swanlab_nanodet import parse_per_class, parse_train, parse_val

_TRAIN = (
    "[NanoDet][07-04 10:05:14]INFO:Train|Epoch1/24|Iter50(51/153)| mem:18.7G| "
    "lr:1.00e-04| loss_qfl:1.7218| loss_bbox:1.1781| loss_dfl:0.5072| aux_loss_qfl:0.8265|"
)
_VAL = "[NanoDet][07-04 10:46:46]INFO:Val_metrics: {'mAP': 0.3539, 'AP_50': 0.5649, 'AP_75': 0.382}"
_CLS = "| bird         | 75.4   | 46.7  | squirrel | 43.4   | 22.2  |"


def test_parse_train():
    d = parse_train(_TRAIN)
    assert d is not None
    assert d["train/epoch"] == 1.0
    assert d["train/lr"] == 1.0e-4
    assert d["train/loss_qfl"] == 1.7218
    assert d["train/loss_bbox"] == 1.1781
    assert d["train/aux_loss_qfl"] == 0.8265


def test_parse_train_ignores_nontrain():
    assert parse_train(_VAL) is None
    assert parse_train("random line") is None


def test_parse_val():
    d = parse_val(_VAL)
    assert d == {"val/mAP": 0.3539, "val/AP_50": 0.5649, "val/AP_75": 0.382}


def test_parse_val_ignores_nonval():
    assert parse_val(_TRAIN) is None


def test_parse_per_class_double_column():
    """一行两个类（NanoDet 双列）都抽出。"""
    d = parse_per_class(_CLS)
    assert d["val/bird_AP50"] == 75.4
    assert d["val/bird_mAP"] == 46.7
    assert d["val/squirrel_AP50"] == 43.4
    assert d["val/squirrel_mAP"] == 22.2


def test_parse_per_class_nan_skipped():
    """person nan（无数据）跳过，不产出 nan 指标。"""
    d = parse_per_class("| person       | nan    | nan   | cat      | 24.4   | 17.0  |")
    assert "val/person_AP50" not in d
    assert d["val/cat_AP50"] == 24.4
