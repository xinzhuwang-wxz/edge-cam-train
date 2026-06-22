#!/usr/bin/env python
"""生成 clean markdown:去掉所有本地图片行(避免导入空占位)+ 去掉级联图片表(改文字),
图与画板全部走 lark-cli 后插。文字/数据表保留。"""

import re
from pathlib import Path

src = Path(__file__).resolve().parent / "开发笔记.md"
out = Path(__file__).resolve().parent / "开发笔记_clean.md"
lines = src.read_text(encoding="utf-8").splitlines()

res = []
skip_casc = False
for ln in lines:
    # 1) 跳过独立图片行 ![..](assets/..)
    if re.match(r"^!\[.*\]\(.*\)\s*$", ln):
        continue
    # 2) 级联图片表:从表头到最后一行 caption 整块删,换成一句话
    if ln.startswith("| 报种正确 ✓ | 报种正确 ✓ | 报种正确 ✓ |"):
        skip_casc = True
        res.append(  # noqa: E501
            "> 6 例逐步标注图见下(每张顶部黑条标 ①检测 ②裁框 ③细分种+top5;绿=检测框、黄=外扩输入框)。"  # noqa: E501
        )
        continue
    if skip_casc:
        if ln.startswith("| 普通鵟 |"):  # 级联表最后一行
            skip_casc = False
        continue
    res.append(ln)

out.write_text("\n".join(res) + "\n", encoding="utf-8")
print("clean md:", out, "| 行:", len(res), "(原", len(lines), ")")
print("残留 assets 引用:", sum("assets/" in x for x in res))
