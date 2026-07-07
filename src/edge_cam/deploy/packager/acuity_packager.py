"""ACUITY/pegasus 打包器（engineering §6 ★，**板子相关，当前为桩**）。

⚠️ **已弃用（[[ADR-0009]]）**：目标芯片 V85x→V861，工具链实为 **AWNN**（非 ACUITY/pegasus）、
板端格式为 `_ipu.param/.bin`（非 `.nb`）。**新代码用 `awnn_packager.AwnnPackager`（已实装可跑）**。
本桩仅为历史保留 / 万一回到 V85x 的备用。

链路（plan §7，历史）：ONNX → onnxsim 静态 → pegasus import → pegasus quantize(PTQ) → .nb。
真跑须 V85x Tina-SDK 内配套 pegasus（Ubuntu only），且 VIPLite .so 必须与 pegasus 对齐
（否则 VIP_ERROR_NETWORK_INCOMPATIBLE）。无板/无 SDK 时 pack() 抛 NotImplementedError，
但接口与 PackagerBackend 一致，有板子时填 subprocess 调 pegasus 即可（不改调用方）。"""

from __future__ import annotations

import subprocess
from pathlib import Path


class AcuityPackager:
    """实现 PackagerBackend。pegasus CLI 路径未配 → pack() 桩抛异常。"""

    def __init__(self, pegasus_bin: str | None = None) -> None:
        self.pegasus_bin = pegasus_bin  # 板子/SDK 上的 pegasus 可执行路径

    def pack(self, onnx_path: str | Path, out_path: str | Path, calib_dataset: str | Path) -> Path:
        # 板子相关，当前为桩。有板/SDK 后在此 subprocess 调 pegasus：
        #   import onnx → quantize(--dataset <calib_dataset>) → export ovxlib --pack-nbg-unify
        #   （plan §B.8；子命令/flag 随 ACUITY 版本，待 W1 实机对齐）。
        raise NotImplementedError(
            "AcuityPackager 桩：需 V85x Tina-SDK 内的 pegasus（配置 pegasus_bin），"
            f"输入 onnx={onnx_path} calib={calib_dataset} → out={out_path}（plan §7/§B.8）。"
        )

    def _run(self, cmd: list[str]) -> int:
        return subprocess.run(cmd, check=False).returncode
