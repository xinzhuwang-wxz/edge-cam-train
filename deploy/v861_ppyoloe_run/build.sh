#!/bin/bash
# 交叉编译 ppyoloe_run（在编译机上跑；工具链/库都来自固件构建产物）
# 用法: bash build.sh   （产物: ./ppyoloe_run, ELF32 RISC-V musl）
set -e
O=${TINA_OUT:-/home/yechen/tina-v861/out/v861/perf2/openwrt}
R=$O/build_dir/target/awnn/awnn_sdk/awnn_runtime
SRC=$(dirname "$0")/ppyoloe_run.c

$O/staging_dir/toolchain/bin/riscv32-unknown-linux-musl-gcc "$SRC" -o ppyoloe_run \
  -I$R/libawnn/include -I$R/libaw_simpleocv/include \
  -L$O/staging_dir/target/usr/lib -L$R/libaw_simpleocv/library/musl \
  -lawnn -lawipubsp -law_simpleocv -lm -O2
file ppyoloe_run
echo "OK. 部署: adb push ppyoloe_run /tmp/ppf/ && adb push $R/libaw_simpleocv/library/musl/libaw_simpleocv.so /tmp/ppf/lib/"
