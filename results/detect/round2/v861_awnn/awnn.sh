#!/usr/bin/env bash
# AWNN 工具链运行辅助（arm64 Mac 上以 amd64 模拟跑 awnn:1.0.2）
# 用法：
#   ./awnn.sh gen   <onnx_rel>            # generate_config_file
#   ./awnn.sh build <cfg_rel>             # 转换量化 → _ipu.param/.bin
#   ./awnn.sh profile <cfg_rel> <csv_rel> # 精度分析(INT8 vs FP32)
#   ./awnn.sh sim   <cfg_rel>             # 仿真 dump 输出
#   ./awnn.sh sh                          # 进容器交互 shell
# 所有 <*_rel> 均为相对本工作区(挂载到容器 /data)的路径。
set -euo pipefail
WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # = results/detect/round2/v861_awnn
IMG="awnn:1.0.2"
PLAT="linux/amd64"
CPUS="${AWNN_CPUS:-3}"   # 限核降温（amd64 模拟满载发热）；可用 AWNN_CPUS=N 覆盖
run() { docker run --rm --platform "$PLAT" --cpus="$CPUS" -v "$WORK":/data "$IMG" bash -lc "$*" 2>&1 | grep -vE "ttyname failed"; }

cmd="${1:-sh}"; shift || true
case "$cmd" in
  gen)     run "awnntools generate_config_file /data/configs/$(basename "$1" .onnx)_gen.yml /data/onnx/$1" ;;
  build)   run "cd /data && awnntools build /data/$1" ;;
  profile) run "cd /data && awnntools profile /data/$1 --log-file /data/${2:-build_out/profile.csv}" ;;
  sim)     run "cd /data && awnntools simulate /data/$1" ;;
  sh)      docker run --rm -it --platform "$PLAT" -v "$WORK":/data "$IMG" /bin/bash ;;
  raw)     run "$*" ;;
  *) echo "unknown: $cmd"; exit 1 ;;
esac
