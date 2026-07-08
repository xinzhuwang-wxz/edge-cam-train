#!/bin/bash
# P0 收口评测(训完跑,box nanodet env):固定test逐类AP + canary遗忘检查 + 产test预测(→混淆矩阵)。
# 机制=复用round2 eval_test.sh:复制P0 cfg、把 data.val.ann_path 换成 test/canary、跑 tools/test.py --task val。
# 用法: bash eval_p0.sh [model_best.ckpt路径(缺省用 p0/model_best)]
set -u
B=/root/autodl-tmp/nanodet_round3
ND=/root/autodl-tmp/ect/third_party/nanodet
PY=/root/miniconda3/envs/nanodet/bin/python
BASECFG=/root/autodl-tmp/nanodet_round3_p0.yml
TEST=/root/autodl-tmp/detect_round3/labels/test_test.json
CANARY=/root/autodl-tmp/detect_round3/canary_r2.json
CKPT=${1:-$B/p0/model_best/model_best.ckpt}
OUT=$B/p0_eval; mkdir -p "$OUT"

[ -f "$CKPT" ] || { echo "✗ 找不到 ckpt: $CKPT"; ls -R "$B/p0"/*/ 2>/dev/null | head; exit 1; }
echo "ckpt = $CKPT"

for tag in test canary; do
  ann=$TEST; [ "$tag" = canary ] && ann=$CANARY
  "$PY" - "$BASECFG" "$ann" "$OUT/$tag" "$OUT/cfg_$tag.yml" <<'PYEOF'
import sys, yaml
base, ann, save, out = sys.argv[1:5]
c = yaml.safe_load(open(base))
c["save_dir"] = save
c["data"]["val"]["ann_path"] = ann   # img_path 不变(detect_raw)
yaml.safe_dump(c, open(out, "w"), sort_keys=False, allow_unicode=True)
PYEOF
  echo ">>> eval $tag ($ann)"
  ( cd "$ND" && "$PY" tools/test.py --task val --config "$OUT/cfg_$tag.yml" --model "$CKPT" ) > "$OUT/$tag.log" 2>&1
  echo "--- $tag 逐类 AP50 ---"; tr '\r' '\n' < "$OUT/$tag.log" | grep -E '^\| bird |^\| cat ' | tail -2
  echo "--- $tag 整体 AP_50 ---"; tr '\r' '\n' < "$OUT/$tag.log" | grep -E 'IoU=0.50 .*area=   all' | tail -1
done

echo "=== test 预测(→混淆矩阵) ==="; find "$OUT/test" -name 'results0.json' 2>/dev/null
touch "$OUT/eval.done"
echo "DONE。下一步(本机):拉 test/results0.json + test_test.json → detect_confusion 出混淆矩阵 vs round2 基线。"
