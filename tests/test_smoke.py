"""W0 冒烟测试：确保包可导入、pytest 链路在位（让 pre-commit 的 pytest 有东西可跑）。

随业务叶子落地，这里逐步替换为真实测试；安全面（状态机/硬规则/权限隔离）的改动
必须先在此目录写测试（先红后绿）。"""

import edge_cam


def test_package_importable() -> None:
    assert edge_cam is not None
