"""VIPLite 端侧推理（engineering §6 ★，**板子相关，当前为桩**）。

借 frigate_npu_vivante 蓝本：ctypes 调 VIPLite 加载 .nb 跑 NPU。关键坑（engineering §8）：
**输出 buffer 按 CHW reshape**；检测后处理（decode/NMS）在 A7 用 OpenCV，不进 NPU 图。
VIPLite .so 必须与 pegasus 版本对齐（否则 VIP_ERROR_NETWORK_INCOMPATIBLE）。

无板时构造即抛 NotImplementedError；接口稳定，上板填 ctypes 调用即可。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VIPLiteRunner:
    """加载 .nb，单帧 NPU 推理。当前桩。"""

    def __init__(self, nb_path: str | Path, viplite_so: str | None = None) -> None:
        self.nb_path = Path(nb_path)
        self.viplite_so = viplite_so
        raise NotImplementedError(
            "VIPLiteRunner 桩：需 V85x 上的 VIPLite .so（与 pegasus 版本对齐）+ ctypes 绑定。"
            "上板实现：加载 .nb → set_input → run → get_output（**CHW reshape**）。"
        )

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """单帧 NCHW → NPU 卷积输出（后处理 decode/NMS 在 A7 CPU 另做）。"""
        raise NotImplementedError("待上板实现 ctypes VIPLite 调用")
