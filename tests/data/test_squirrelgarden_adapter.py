"""round3 新增 roboflow_squirrelgarden adapter 注册/配置测试（无卡、无需数据）。"""

from __future__ import annotations

from edge_cam.data.adapters.detect import available_adapters, build_adapter


def test_squirrelgarden_registered_and_configured() -> None:
    assert "roboflow_squirrelgarden" in available_adapters()
    a = build_adapter("roboflow_squirrelgarden", "/tmp/raw")
    assert a.spec.name == "roboflow_squirrelgarden"
    assert a.spec.catch_all_label == "squirrel"  # Squirrel(根+类) → squirrel
    assert a.spec.commercial_safe is True
    # slug 对应 squirrel-iest3/squirrel-xgfti v1（README.dataset.txt 实核）
    assert a._rf == ("squirrel-iest3", "squirrel-xgfti", 1)
    # 盘位 = commercial/roboflow_squirrelgarden（round2 已下的目录）
    assert a._subpath == "commercial/roboflow_squirrelgarden"
