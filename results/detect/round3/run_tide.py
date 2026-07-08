"""TIDE 误差分解(round3 §6):把 AP gap 拆成 Cls/Loc/Both/Dupe/Bkg/Miss。
★Cls = "框对类错"的分类混淆(squirrel↔bird 等)量化成单值,逐 round 回归。
坑(verify G6):TIDE 的 COCO loader 要 `segmentation`,纯检测框没有 → 本脚本自动补 bbox 多边形 dummy。
用法: python run_tide.py --gt <GT coco json> --det <检测预测 coco json>
"""

import argparse
import json
import tempfile
import warnings

warnings.filterwarnings("ignore")


def _tide_compat_gt(gt_path: str) -> str:
    """给 GT 每框补 bbox-多边形 dummy segmentation(TIDE 需要),写临时文件返回路径。"""
    with open(gt_path) as fh:
        d = json.load(fh)
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        a.setdefault("segmentation", [[x, y, x + w, y, x + w, y + h, x, y + h]])
        a.setdefault("area", w * h)
        a.setdefault("iscrowd", 0)
    with tempfile.NamedTemporaryFile("w", suffix="_tide_gt.json", delete=False) as f:
        json.dump(d, f)
        name = f.name
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--det", required=True)
    a = ap.parse_args()

    from tidecv import TIDE, datasets

    gt = datasets.COCO(_tide_compat_gt(a.gt))
    res = datasets.COCOResult(a.det)
    tide = TIDE()
    tide.evaluate(gt, res, mode=TIDE.BOX)
    tide.summarize()  # 打印 Cls/Loc/Both/Dupe/Bkg/Miss 的 dAP 表
    print("\n★回归:Cls(混淆,round2=4.58) Bkg(误报,6.21) Miss(漏检,3.63) 逐 round 比。")


if __name__ == "__main__":
    main()
