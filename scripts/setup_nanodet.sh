#!/usr/bin/env bash
# NanoDet 检测训练环境搭建（在租的 GPU 机/AutoDL 上跑一次）。
#
# 为什么单独一个 env：NanoDet 依赖旧版 pytorch-lightning，与本仓 lightning 2.x 冲突
# （engineering §2）。本仓只「生成它的 config + 消费它导出的 ONNX」，不混环境。
#
# 用法：  bash scripts/setup_nanodet.sh
# 之后：  conda activate nanodet  再按下方「训练/评测/导出」三步跑。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NANODET_DIR="$REPO_ROOT/third_party/nanodet"
PINNED_COMMIT="be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0"  # = NANODET_PINNED_COMMIT

[ -d "$NANODET_DIR/nanodet" ] || { echo "✗ 缺 $NANODET_DIR（应随仓库带上）"; exit 1; }

echo "==> 锁版 NanoDet @ $PINNED_COMMIT"
git -C "$NANODET_DIR" rev-parse --git-dir >/dev/null 2>&1 \
  && git -C "$NANODET_DIR" checkout "$PINNED_COMMIT" 2>/dev/null \
  || echo "  (third_party/nanodet 非独立 git，跳过 checkout —— 确认随主仓锁定即可)"

echo "==> 建 conda env: nanodet (python 3.10)"
conda create -y -n nanodet python=3.10
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanodet

echo "==> 装 NanoDet 依赖（GPU 机上 torch 按 CUDA 版本，见 nanodet/requirements.txt）"
pip install -r "$NANODET_DIR/requirements.txt"
pip install -e "$NANODET_DIR"

echo ""
echo "==> 预训练权重"
echo "  基线权重已在仓内：weights/nanodet/nanodet-plus-m_416_checkpoint.ckpt（34M）"
echo "  如需其它档/全量，见 third_party/nanodet/README（Google Drive，手动下）。"
echo ""
echo "✓ nanodet env 就绪。接下来（在本仓根目录、edge-cam-train env 里生成 config，再切 nanodet env 跑）："
cat <<'STEPS'

# 1) 生成指向本仓检测数据的 NanoDet config（在 edge-cam-train env）
conda activate edge-cam-train
python - <<'PY'
from edge_cam.data.detection_classes import FEEDER_CAM_CLASSES
from edge_cam.train.detect.nanodet_config import generate_nanodet_config
names = [c.name for c in FEEDER_CAM_CLASSES]
generate_nanodet_config("third_party/nanodet", "data/processed/detection_feeder", names,
                        "outputs/detect/nanodet_feeder_416.yml",
                        base_config="config/nanodet-plus-m_416.yml", input_size=416, epochs=100)
print("config -> outputs/detect/nanodet_feeder_416.yml")
PY

# 2) 训练（在 nanodet env，从基线权重微调）
conda activate nanodet
python third_party/nanodet/tools/train.py outputs/detect/nanodet_feeder_416.yml

# 3) 评测 mAP / per-class（NanoDet 自带；结果手动汇总进消融表）
python third_party/nanodet/tools/test.py outputs/detect/nanodet_feeder_416.yml \
  --model outputs/detect/nanodet-plus-m/model_best/model_best.ckpt --task val

# 4) 导 FP32 ONNX（裸 backbone+head；decode/NMS 留 A7 CPU）。本仓会自动跑结构契约校验
python third_party/nanodet/tools/export_onnx.py --cfg_path outputs/detect/nanodet_feeder_416.yml \
  --model_path outputs/detect/nanodet-plus-m/model_best/model_best.ckpt \
  --out_path outputs/detect/nanodet_feeder_416.onnx --input_shape 416,416
STEPS
