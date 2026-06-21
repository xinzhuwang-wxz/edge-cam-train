#!/usr/bin/env python
"""把 检测实验报告.md 转成自包含单文件 HTML（图片 base64 内嵌），配色对齐 results/实验1/实验报告.html。"""
import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
MD = ROOT / "检测实验报告.md"
OUT = ROOT / "检测实验报告.html"

text = MD.read_text(encoding="utf-8")


def embed(m):
    alt, path = m.group(1), m.group(2)
    p = (ROOT / path).resolve()
    if not p.exists():
        return m.group(0)
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"![{alt}](data:image/png;base64,{b64})"


text = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", embed, text)

body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"])

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>边侧粗检测 实验报告 · NanoDet-Plus (feeder 5 类)</title>
<style>
  :root {{ --ink:#0a0e16; --panel:#121826; --panel2:#1a2233; --stroke:#27324a;
           --accent:#38bdf8; --accent2:#34d399; --warn:#fbbf24; --danger:#f87171;
           --text:#dfe6f2; --muted:#8a97b0; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--ink); color:var(--text);
          font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
          line-height:1.7; font-size:15px; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:48px 28px 96px; }}
  h1 {{ font-size:30px; font-weight:800; letter-spacing:-.02em; margin:.2em 0 .6em;
        background:linear-gradient(90deg,var(--accent),var(--accent2));
        -webkit-background-clip:text; background-clip:text; color:transparent; }}
  h2 {{ font-size:22px; font-weight:700; margin:2.2em 0 .8em; padding-bottom:.35em;
        border-bottom:2px solid var(--stroke); color:#fff; }}
  h3 {{ font-size:17px; font-weight:700; margin:1.6em 0 .6em; color:var(--accent); }}
  p,li {{ color:var(--text); }}
  a {{ color:var(--accent); }}
  strong {{ color:#fff; }}
  blockquote {{ margin:1em 0; padding:.6em 1.1em; background:var(--panel);
        border-left:3px solid var(--accent); border-radius:0 8px 8px 0; color:var(--muted); font-size:14px; }}
  blockquote strong {{ color:var(--text); }}
  table {{ border-collapse:collapse; width:100%; margin:1.2em 0; font-size:13.5px;
        background:var(--panel); border-radius:10px; overflow:hidden; }}
  th {{ background:var(--panel2); color:#fff; font-weight:700; text-align:left; }}
  th,td {{ padding:9px 12px; border-bottom:1px solid var(--stroke); }}
  tr:last-child td {{ border-bottom:none; }}
  tbody tr:hover {{ background:rgba(56,189,248,.06); }}
  img {{ max-width:100%; border:1px solid var(--stroke); border-radius:10px; margin:1em 0;
         background:#fff; }}
  code {{ background:var(--panel2); padding:.15em .45em; border-radius:5px; font-size:13px;
          color:var(--accent2); font-family:"SF Mono",Menlo,Consolas,monospace; }}
  pre {{ background:var(--panel); border:1px solid var(--stroke); border-radius:10px;
         padding:16px; overflow-x:auto; }}
  pre code {{ background:none; color:var(--muted); padding:0; }}
  hr {{ border:none; border-top:1px solid var(--stroke); margin:2.4em 0; }}
  em {{ color:var(--muted); }}
</style></head>
<body><div class="wrap">{body}</div></body></html>
"""
OUT.write_text(HTML, encoding="utf-8")
print("HTML 写出:", OUT, f"({OUT.stat().st_size // 1024} KB)")
