"""打包后端。**V861 用 `AwnnPackager`**（实装，[[ADR-0009]]）；`acuity_packager` 已弃用（V85x）。"""

from edge_cam.deploy.packager.awnn_packager import AwnnPackager, gen_config, write_config
from edge_cam.deploy.packager.base import PackagerBackend

__all__ = ["AwnnPackager", "PackagerBackend", "gen_config", "write_config"]
