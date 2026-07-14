#!/bin/bash
# V861 板端批量推理 —— 分块 + 熔断
#   /tmp 是 46MB RAM 盘: 每块 6 张 (输入 7.2MB + 输出 14.7MB + 模型 7.5MB ≈ 30MB)
#   profiler=0 (开着会把 NPU 内存从 18.02MB 顶到 22.71MB, 超过 20MB 保留池 → DMA 分配失败)
#   熔断: 任一块报 DMA/invalid → 立即停, 不反复砸驱动 (那会把内核搞挂)
set -u
B=/tmp/ppf
IN=/home/<用户>/ppf_batch/inputs
OUT=/home/<用户>/ppf_batch/out
CHUNK=6
mkdir -p "$OUT"

adb shell "mkdir -p $B/in $B/out" >/dev/null 2>&1
adb push /home/<用户>/tina-v861/out/v861/perf2/openwrt/build_dir/target/awnn/awnn_sdk/examples/awnn_verify/verify/awnn_batch $B/awnn_batch >/dev/null 2>&1
adb shell "chmod +x $B/awnn_batch" >/dev/null 2>&1

adb shell "cat > $B/batch_cfg.txt <<EOF
input_paths=in/x.bin
output_paths=o0.bin,o1.bin,o2.bin,o3.bin,o4.bin,o5.bin
input_blob_names=image
output_blob_names=conv2d_81.tmp_0,conv2d_84.tmp_0,conv2d_74.tmp_0,conv2d_77.tmp_0,conv2d_67.tmp_0,conv2d_70.tmp_0
inputs_w=640
inputs_h=640
inputs_c=3
input_data_type=DATA_TYPE_INT8
output_data_type=DATA_TYPE_FP32,DATA_TYPE_FP32,DATA_TYPE_FP32,DATA_TYPE_FP32,DATA_TYPE_FP32,DATA_TYPE_FP32
model_path=model/ppyoloe_s_640_ipu.bin
param_path=model/ppyoloe_s_640_ipu.param
use_static_mode=0
use_awnn_profiler=0
is_compare_result=0
dump_output_result=1
net_name=ppyoloe
loop_count=1
batch_list=in/list.txt
out_dir=out
EOF"

mapfile -t ALL < <(ls "$IN"/*.bin)
N=${#ALL[@]}; done_n=0; chunk_i=0
for ((s=0; s<N; s+=CHUNK)); do
  chunk_i=$((chunk_i+1))
  batch=("${ALL[@]:s:CHUNK}")
  adb shell "rm -rf $B/in $B/out && mkdir -p $B/in $B/out" >/dev/null 2>&1
  : > /tmp/list.txt
  for f in "${batch[@]}"; do
    adb push "$f" "$B/in/$(basename "$f")" >/dev/null 2>&1
    echo "in/$(basename "$f")" >> /tmp/list.txt
  done
  adb push /tmp/list.txt $B/in/list.txt >/dev/null 2>&1

  log=$(adb shell "cd $B && ./awnn_batch batch_cfg.txt 2>&1")
  if echo "$log" | grep -qiE "DMA_HEAP|invalid|inference error|precompiler error"; then
    echo "!!! 熔断: 块 $chunk_i 出错, 立即停止 (不重试, 避免搞挂驱动)"
    echo "$log" | grep -iE "DMA_HEAP|invalid|error|Memory Statistics" | head -5
    exit 1
  fi
  echo "$log" | grep -E "npu_ms|Memory Statistics|min =" 

  adb pull $B/out "$OUT/" >/dev/null 2>&1
  # adb pull 会建 out/out/ 子目录 → 摊平
  [ -d "$OUT/out" ] && mv "$OUT"/out/* "$OUT/" 2>/dev/null && rmdir "$OUT/out" 2>/dev/null
  adb shell "rm -rf $B/in $B/out && mkdir -p $B/in $B/out" >/dev/null 2>&1
  done_n=$((done_n + ${#batch[@]}))
  t=$(adb shell "cat /sys/class/thermal/thermal_zone0/temp" 2>/dev/null | tr -d '\r')
  echo "== 块 $chunk_i 完成: $done_n/$N 张, 温度 $((${t:-0}/1000))°C, 已落盘 $(ls $OUT | wc -l) 个 bin =="
done
echo "=== 全部完成: $(ls $OUT | wc -l) 个输出 bin (应为 $((N*6))) ==="
