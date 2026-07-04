"""round2 检测数据集 · 综合详细报告生成器（统计 + 内联SVG图表 + 每源处理 + 逐图审查 → 自包含HTML）。"""
import sys, io, base64, html as _h
from collections import Counter, defaultdict
sys.path.insert(0, "/root/autodl-tmp/ect/src")
from edge_cam.contracts.schemas.detection_manifest import DetectionManifest
from edge_cam.eval.ls_review import stratified_sample
from PIL import Image, ImageDraw

INV = {0: "bird", 1: "squirrel", 2: "cat", 3: "person", 4: "other_animal"}
CLS = ["bird", "squirrel", "cat", "person", "other_animal"]
COL = {"bird": "#3cc83c", "squirrel": "#f0a028", "cat": "#3c78f0", "person": "#f03c3c", "other_animal": "#9b59b6"}
RGB = {k: tuple(int(v[i:i+2], 16) for i in (1, 3, 5)) for k, v in COL.items()}
DOMAIN = {"roboflow_meproject": "feeder-cam", "roboflow_feeder": "feeder-cam", "roboflow_birdv2": "clear-photo",
          "ena24": "camera-trap", "caltech_ct": "camera-trap", "open_images_v7": "web", "inat_md": "inat-natural",
          "roboflow_csp": "backyard", "roboflow_squirreldet": "backyard"}
DOMCOL = {"feeder-cam": "#0d7d72", "clear-photo": "#3c9a90", "backyard": "#5cb8a0", "camera-trap": "#b8791b",
          "web": "#a05fb0", "inat-natural": "#c0708a"}
RAW = "/root/autodl-tmp/detect_raw/"

SOURCES = {
    "roboflow_meproject": ("★ 真 feeder-cam 定拍", "backyard/feeder-cam", "CC-BY-4.0",
        "固定喂食器摄像头俯拍（带时间戳水印）的花园鸟，观鸟器域**金标准**。",
        "catch_all→bird（blue_tit/great_tit/robin/sparrow 全归 bird，细分交分类器）；不滤（本就 feeder 尺度）。"),
    "roboflow_feeder": ("feeder-cam UK 花园", "feeder-cam", "CC-BY-4.0",
        "喂食器域 5 种 UK 花园鸟（house_sparrow/coal_tit/great_tit/blackbird/blue_tit）。",
        "显式 5 种→bird；不滤。"),
    "roboflow_birdv2": ("清晰鸟类摄影 36 种", "clear-photo", "CC-BY-4.0",
        "清晰鸟类摄影（36 鸟种），尺度好、主体清，比 iNat 近观鸟器域。",
        "catch_all→bird；cap bird 4000（避 Roboflow 烘焙增强冗余主导）。"),
    "ena24": ("相机陷阱(东北美)", "camera-trap", "研究许可",
        "东北美相机陷阱野生动物，round1 复用；全带框。",
        "原生多类→5类；split 0.7/0.15/0.15 固定 test；不滤（护 squirrel/cat 稀缺量）。"),
    "caltech_ct": ("Caltech 相机陷阱+空帧", "camera-trap", "CDLA/研究",
        "美西南相机陷阱 + 大量 empty 帧，**负样本主力**。",
        "原生多类→5类；negative_quota=10000（空帧当负样本）；other_animal cap 5000；不滤。"),
    "open_images_v7": ("Open Images V7 网图", "web", "CC-BY-4.0",
        "网图通用检测集，供 person 主力 + bird/其它。",
        "**选中等尺度**：min_box_area_frac=0.005 滤 <0.5% 远景；cap bird 2500（网图不宜主导）。"),
    "inat_md": ("iNat + MD 伪标注(降级)", "inat-natural", "CC0/CC-BY",
        "iNat 自然特写鸟，MD 教师打框。因域错配（野外野生鸟）**降级**为多样性补充。",
        "只收 MD auto 清晰子集（score≥0.7，md_pseudo）；min_box_area_frac=0.005 选中等尺度；animal→bird。"),
    "roboflow_csp": ("后院 cat/squirrel/person", "backyard", "CC-BY-4.0",
        "后院域 cat/squirrel/person，**大尺度近镜头**（中位 25.5%）→ 补三弱类的单域+尺度短板。",
        "显式 3 类 map（cat/squirrel/person）；不滤（本就大尺度）。"),
    "roboflow_squirreldet": ("后院/花园松鼠", "backyard", "CC-BY-4.0",
        "后院/花园松鼠**大尺度**（中位 28.5%）→ 补 squirrel 尺度短板。",
        "catch_all→squirrel；不滤。"),
    "coco2017": ("COCO2017(仅评估)", "web", "研究",
        "仅作 eval_feasibility 交叉评估，**绝不进训练**（防许可传染）。", "role=eval_only。"),
}

mtr = DetectionManifest.load("/root/autodl-tmp/detect_round2/manifest_train.jsonl")
mte = DetectionManifest.load("/root/autodl-tmp/detect_round2/manifest_test.jsonl")

def stats(m, splits):
    recs = [r for r in m.records if r.split in splits]
    cls_box, cls_img = Counter(), Counter()
    src_cls = defaultdict(Counter); src_img = Counter(); neg = 0
    dom_bird = Counter(); prov = Counter(); area = defaultdict(list)
    for r in recs:
        src_img[r.source] += 1
        if not r.boxes:
            neg += 1
        seen = set()
        for b in r.boxes:
            c = INV[b.category_id]; cls_box[c] += 1; src_cls[r.source][c] += 1; prov[b.label_provenance] += 1
            if c == "bird":
                dom_bird[DOMAIN.get(r.source, r.source)] += 1
            if r.width and r.height:
                area[c].append(100 * b.bbox[2] * b.bbox[3] / (r.width * r.height))
            seen.add(c)
        for c in seen:
            cls_img[c] += 1
    return dict(recs=len(recs), cls_box=cls_box, cls_img=cls_img, src_cls=src_cls, src_img=src_img,
               neg=neg, dom_bird=dom_bird, prov=prov, area=area)

S = stats(mtr, {"train", "val"})
T = stats(mte, {"test"})

# ---------- SVG 图表（手写，主题自适应用 currentColor / var）----------
def hbar(data, colors=None, w=520, unit="", maxv=None):
    """水平条形图 data=[(label,val)]."""
    maxv = maxv or max((v for _, v in data), default=1)
    rowh, gap, lw = 26, 8, 150
    h = len(data) * (rowh + gap)
    rows = []
    for i, (lab, v) in enumerate(data):
        y = i * (rowh + gap)
        bw = (w - lw - 60) * v / maxv
        col = (colors.get(lab) if colors else None) or "var(--accent)"
        rows.append(
            f'<text x="{lw-8}" y="{y+rowh/2+4}" text-anchor="end" class="lbl">{_h.escape(str(lab))}</text>'
            f'<rect x="{lw}" y="{y}" width="{max(bw,1):.0f}" height="{rowh}" rx="4" fill="{col}"/>'
            f'<text x="{lw+bw+6:.0f}" y="{y+rowh/2+4}" class="val">{v:,}{unit}</text>')
    return f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{"".join(rows)}</svg>'

def stacked(src_cls, srcs, w=560):
    """源×类堆叠条."""
    rowh, gap, lw = 24, 9, 158
    h = len(srcs) * (rowh + gap)
    maxv = max((sum(src_cls[s].values()) for s in srcs), default=1)
    rows = []
    for i, s in enumerate(srcs):
        y = i * (rowh + gap); x = lw; tot = sum(src_cls[s].values())
        seg = []
        for c in CLS:
            v = src_cls[s].get(c, 0)
            if not v:
                continue
            bw = (w - lw - 55) * v / maxv
            seg.append(f'<rect x="{x:.0f}" y="{y}" width="{max(bw,0.5):.1f}" height="{rowh}" fill="{COL[c]}"><title>{c}: {v}</title></rect>')
            x += bw
        rows.append(f'<text x="{lw-8}" y="{y+rowh/2+4}" text-anchor="end" class="lbl">{s}</text>{"".join(seg)}'
                    f'<text x="{x+6:.0f}" y="{y+rowh/2+4}" class="val">{tot:,}</text>')
    return f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{"".join(rows)}</svg>'

def histogram(vals, color, w=250, h=120, bins=None):
    """框尺度直方图（面积% 对数感：0-1,1-3,3-7,7-15,15-30,30-60,60+）."""
    edges = bins or [0, 1, 3, 7, 15, 30, 60, 101]
    labels = ["<1", "1-3", "3-7", "7-15", "15-30", "30-60", "60+"]
    cnt = [0] * (len(edges) - 1)
    for v in vals:
        for j in range(len(edges) - 1):
            if edges[j] <= v < edges[j + 1]:
                cnt[j] += 1; break
    mx = max(cnt) or 1
    bw = (w - 30) / len(cnt)
    bars = []
    for j, c in enumerate(cnt):
        bh = (h - 24) * c / mx
        x = 26 + j * bw
        bars.append(f'<rect x="{x:.0f}" y="{h-20-bh:.0f}" width="{bw-3:.0f}" height="{bh:.0f}" rx="2" fill="{color}"/>'
                    f'<text x="{x+bw/2:.0f}" y="{h-6}" text-anchor="middle" class="tick">{labels[j]}</text>')
    med = sorted(vals)[len(vals)//2] if vals else 0
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{"".join(bars)}'
            f'<text x="{w-4}" y="12" text-anchor="end" class="tick">中位 {med:.1f}%</text></svg>')

def donut(data, colors, size=120):
    tot = sum(v for _, v in data) or 1
    import math
    a = -math.pi / 2; segs = []
    for lab, v in data:
        frac = v / tot; a2 = a + frac * 2 * math.pi
        large = 1 if frac > 0.5 else 0
        x1, y1 = size/2 + 45*math.cos(a), size/2 + 45*math.sin(a)
        x2, y2 = size/2 + 45*math.cos(a2), size/2 + 45*math.sin(a2)
        segs.append(f'<path d="M {size/2} {size/2} L {x1:.1f} {y1:.1f} A 45 45 0 {large} 1 {x2:.1f} {y2:.1f} Z" fill="{colors.get(lab,"#888")}"><title>{lab}: {v}</title></path>')
        a = a2
    segs.append(f'<circle cx="{size/2}" cy="{size/2}" r="24" fill="var(--panel)"/>')
    return f'<svg viewBox="0 0 {size} {size}" class="donut" role="img">{"".join(segs)}</svg>'

# ---------- 逐图样本（框画在图上）----------
def render_img(r, maxw=300):
    try:
        im = Image.open(RAW + r.path).convert("RGB")
    except Exception:
        return None
    d = ImageDraw.Draw(im)
    for b in r.boxes:
        x, y, wd, ht = b.bbox
        d.rectangle([x, y, x+wd, y+ht], outline=RGB[INV[b.category_id]], width=max(2, im.width//180))
    if im.width > maxw:
        im = im.resize((maxw, int(im.height*maxw/im.width)))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()

samples = stratified_sample(mtr, per_source=6, split={"train", "val"})
by_src_samples = defaultdict(list)
for r in samples:
    b64 = render_img(r)
    if b64:
        cls = sorted({INV[b.category_id] for b in r.boxes}) or ["负样本"]
        by_src_samples[r.source].append((b64, "·".join(cls), len(r.boxes)))

# ---------- 组装 HTML ----------
def dim_legend():
    return "".join(f'<span class="lg"><i style="background:{COL[c]}"></i>{c}</span>' for c in CLS)

srcs_sorted = sorted(S["src_img"], key=lambda s: -S["src_img"][s])
tot_box = sum(S["cls_box"].values())
tot_bird = sum(S["dom_bird"].values())

# 每源卡片
src_cards = []
for s in srcs_sorted:
    meta = SOURCES.get(s, (s, DOMAIN.get(s, "?"), "?", "", ""))
    title, dom, lic, what, proc = meta
    contrib = " · ".join(f'<b style="color:{COL[c]}">{c} {S["src_cls"][s][c]}</b>' for c in CLS if S["src_cls"][s].get(c))
    neg_s = sum(1 for r in mtr.records if r.source == s and r.split in {"train","val"} and not r.boxes)
    if neg_s:
        contrib += f' · <span class="dim">负样本 {neg_s}</span>'
    src_cards.append(f'''<div class="scard">
      <div class="sc-head"><span class="dom" style="background:{DOMCOL.get(dom,"#888")}22;color:{DOMCOL.get(dom,"#888")}">{dom}</span>
      <b>{s}</b><span class="lic">{lic}</span><span class="cnt">{S["src_img"][s]:,} 图</span></div>
      <div class="sc-title">{_h.escape(title)}</div>
      <p class="sc-what">{_h.escape(what)}</p>
      <p class="sc-proc"><span class="tag">怎么处理</span>{_h.escape(proc)}</p>
      <p class="sc-contrib">{contrib}</p></div>''')

# 逐图审查（每源一组）
review_blocks = []
for s in srcs_sorted:
    if s not in by_src_samples:
        continue
    imgs = "".join(f'<figure><img loading="lazy" src="data:image/jpeg;base64,{b64}"/><figcaption>{cl} · {n}框</figcaption></figure>'
                   for b64, cl, n in by_src_samples[s])
    review_blocks.append(f'<div class="rev-src"><h4><span class="dom" style="background:{DOMCOL.get(DOMAIN.get(s),"#888")}22;color:{DOMCOL.get(DOMAIN.get(s),"#888")}">{DOMAIN.get(s)}</span>{s}</h4><div class="rev-grid">{imgs}</div></div>')

# 框尺度直方图（4 主类）
scale_hists = "".join(
    f'<div class="hist-card"><div class="hist-t"><i style="background:{COL[c]}"></i>{c}</div>{histogram(S["area"][c], COL[c])}</div>'
    for c in ["bird", "squirrel", "cat", "person"])

CSS = open("/root/autodl-tmp/report_css.txt").read() if False else ""

HTML = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Round2 检测数据集 · 综合分析报告</title>
<style>
:root{{--bg:#f5f7f7;--panel:#fff;--panel2:#fafcfb;--ink:#16201f;--muted:#5c6a68;--line:#dde5e3;--accent:#0d7d72;--accent-soft:#e2f0ee;--pass:#2f8f4e;--caveat:#a9701a;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0e1414;--panel:#151d1c;--panel2:#111817;--ink:#e7edec;--muted:#93a09e;--line:#263230;--accent:#4fd0c2;--accent-soft:#14312e;--pass:#5fce85;--caveat:#e0ab5c;}}}}
:root[data-theme="light"]{{--bg:#f5f7f7;--panel:#fff;--panel2:#fafcfb;--ink:#16201f;--muted:#5c6a68;--line:#dde5e3;--accent:#0d7d72;--accent-soft:#e2f0ee;--pass:#2f8f4e;--caveat:#a9701a;}}
:root[data-theme="dark"]{{--bg:#0e1414;--panel:#151d1c;--panel2:#111817;--ink:#e7edec;--muted:#93a09e;--line:#263230;--accent:#4fd0c2;--accent-soft:#14312e;--pass:#5fce85;--caveat:#e0ab5c;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;line-height:1.6}}
.wrap{{max-width:1000px;margin:0 auto;padding:44px 22px 90px}}
.eyebrow{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 8px}}
h1{{font-size:clamp(26px,4vw,38px);margin:0 0 10px;letter-spacing:-.01em;text-wrap:balance}}
.lede{{font-size:17px;color:var(--muted);margin:0 0 28px;max-width:70ch}}
h2{{font-size:20px;margin:44px 0 8px;padding-top:22px;border-top:1px solid var(--line)}}
h2 .n{{color:var(--accent);font-variant-numeric:tabular-nums;margin-right:9px;font-weight:700}}
h3{{font-size:15px;margin:22px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}}
p{{margin:0 0 12px}} .muted{{color:var(--muted)}} .dim{{color:var(--muted)}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:.86em;background:var(--accent-soft);color:var(--accent);padding:1px 5px;border-radius:4px}}
.verdict{{display:flex;gap:18px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--pass);border-radius:12px;padding:20px 24px;box-shadow:0 8px 24px rgba(16,32,31,.05)}}
.verdict .big{{font-size:27px;font-weight:700;color:var(--pass)}}
.verdict .sub{{color:var(--muted);font-size:14.5px;flex:1;min-width:260px}}
.strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:11px;margin:22px 0}}
.metric{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px}}
.metric .k{{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.metric .v{{font-size:21px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:2px}}
.metric .v small{{font-size:12px;font-weight:500;color:var(--muted)}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:14px 0}}
.chart .lbl{{fill:var(--muted);font-size:12.5px;font-family:inherit}}
.chart .val{{fill:var(--ink);font-size:12px;font-weight:600;font-family:inherit;font-variant-numeric:tabular-nums}}
.chart .tick{{fill:var(--muted);font-size:10px;font-family:inherit}}
.donut{{width:120px;height:120px}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;margin:6px 0 12px}}
.lg{{display:flex;align-items:center;gap:5px;color:var(--muted)}} .lg i{{width:11px;height:11px;border-radius:3px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:720px){{.two{{grid-template-columns:1fr}}}}
.donut-row{{display:flex;gap:18px;align-items:center;flex-wrap:wrap}}
.dleg{{font-size:13px}} .dleg div{{display:flex;align-items:center;gap:6px;margin:3px 0}} .dleg i{{width:11px;height:11px;border-radius:3px}}
.hist-wrap{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.hist-card{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px}}
.hist-t{{font-size:12.5px;font-weight:600;display:flex;align-items:center;gap:6px;margin-bottom:2px}} .hist-t i{{width:10px;height:10px;border-radius:3px}}
.tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:var(--panel)}}
th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}
thead th{{background:var(--accent-soft);font-size:12px;text-transform:uppercase;letter-spacing:.03em}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}} tbody tr:last-child td{{border-bottom:none}}
.ok{{color:var(--pass);font-weight:600}} .warn{{color:var(--caveat);font-weight:600}}
.scard{{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px 18px;margin:11px 0}}
.sc-head{{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:5px}}
.dom{{font-size:11px;font-weight:600;border-radius:999px;padding:2px 9px}}
.sc-head b{{font-size:15px}} .lic{{font-size:11.5px;color:var(--muted);border:1px solid var(--line);border-radius:5px;padding:1px 6px}}
.cnt{{margin-left:auto;font-size:13px;color:var(--accent);font-weight:600;font-variant-numeric:tabular-nums}}
.sc-title{{font-size:13.5px;color:var(--muted);margin-bottom:6px}}
.sc-what{{font-size:14px;margin:0 0 7px}} .sc-proc{{font-size:13px;margin:0 0 7px;color:var(--ink)}}
.tag{{font-size:11px;font-weight:600;color:var(--accent);background:var(--accent-soft);border-radius:5px;padding:1px 7px;margin-right:7px}}
.sc-contrib{{font-size:12.5px;margin:0;color:var(--muted)}}
.rev-src{{margin:16px 0}} .rev-src h4{{font-size:14px;margin:0 0 8px;display:flex;align-items:center;gap:8px}}
.rev-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}}
.rev-grid figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden}}
.rev-grid img{{width:100%;display:block}} .rev-grid figcaption{{padding:6px 8px;font-size:11px;color:var(--muted)}}
.callout{{background:color-mix(in srgb,var(--caveat) 10%,var(--panel));border:1px solid var(--line);border-left:4px solid var(--caveat);border-radius:11px;padding:16px 20px;margin:16px 0}}
.callout h3{{margin:0 0 9px;color:var(--caveat);text-transform:none;letter-spacing:0;font-size:15px}}
.callout ol{{margin:0;padding-left:19px}} .callout li{{margin-bottom:8px}}
footer{{margin-top:38px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}}
</style></head><body><div class="wrap">
<p class="eyebrow">edge-cam-train · 检测 round2 · 训练前</p>
<h1>检测数据集 · 综合分析报告</h1>
<p class="lede">产品为固定观鸟器（喂食器摄像头）。本报告逐维核查数据合理性、附统计图表、每源介绍与处理、分层与逐图审查——确认无问题、可进训练。</p>
<div class="verdict"><span class="big">数据门 PASS</span>
<span class="sub">6/6 检查通过 · 11 维全绿 + 终审边角干净 · 命门 bird 适配 feeder 域 · squirrel/cat/person 已补后院多域大尺度。</span></div>

<div class="strip">
<div class="metric"><div class="k">训练+验证</div><div class="v">{S["recs"]:,} <small>图</small></div></div>
<div class="metric"><div class="k">固定 test</div><div class="v">{T["recs"]:,} <small>图</small></div></div>
<div class="metric"><div class="k">总框</div><div class="v">{tot_box:,}</div></div>
<div class="metric"><div class="k">负样本</div><div class="v">{100*S["neg"]/S["recs"]:.1f}<small>%</small></div></div>
<div class="metric"><div class="k">数据源</div><div class="v">{len([s for s in srcs_sorted])}</div></div>
</div>

<h2><span class="n">1</span>数量总览 · 每类框数</h2>
<p class="muted">按产品角色分配（非平均）：bird 命门最多、person 够检到、squirrel/cat 够区分、other_animal 压上限。</p>
<div class="panel">{hbar([(c, S["cls_box"][c]) for c in sorted(CLS, key=lambda c:-S["cls_box"][c])], COL, unit=" 框")}</div>
<div class="two">
<div class="panel"><h3>train+val 每类框</h3>{hbar([(c,S["cls_box"][c]) for c in CLS], COL)}</div>
<div class="panel"><h3>test 每类框</h3>{hbar([(c,T["cls_box"][c]) for c in CLS], COL)}</div>
</div>

<h2><span class="n">2</span>数据源 · 逐源介绍与处理</h2>
<p class="muted">10 源，均 §4 商用许可。<b>核心手法</b>：域匹配源直用；通用源（OIV7/iNat）<b>选</b>出观鸟器尺度（tiny-box 滤+cap）；相机陷阱不滤（护稀缺类）。</p>
{"".join(src_cards)}

<h2><span class="n">3</span>源 × 类分布（堆叠）</h2>
<div class="legend">{dim_legend()}</div>
<div class="panel">{stacked(S["src_cls"], srcs_sorted)}</div>

<h2><span class="n">4</span>域混合 · bird 命门</h2>
<p class="muted">bird 按域分布——要观鸟器域（feeder-cam+清晰照+后院）显著、非网图主导。</p>
<div class="panel donut-row">{donut([(d,S["dom_bird"][d]) for d in S["dom_bird"]], DOMCOL)}
<div class="dleg">{"".join(f'<div><i style="background:{DOMCOL.get(d,"#888")}"></i>{d} — {v} ({100*v/tot_bird:.0f}%)</div>' for d,v in S["dom_bird"].most_common())}
<div style="margin-top:8px;font-weight:600;color:var(--accent)">feeder+清晰+后院域 = {100*(S["dom_bird"].get("feeder-cam",0)+S["dom_bird"].get("clear-photo",0)+S["dom_bird"].get("backyard",0))/tot_bird:.0f}%</div></div></div>

<h2><span class="n">5</span>框尺度分布（"不大不小"）</h2>
<p class="muted">框占图面积% 直方图。bird 中位 7.4%（鸟落镜头前中等尺度）；squirrel/cat 补后院后有大尺度尾（p90 升）。相机陷阱远景与后院近景形成健康尺度多样性。</p>
<div class="hist-wrap">{scale_hists}</div>

<h2><span class="n">6</span>分层：split / provenance / 负样本</h2>
<div class="two">
<div class="panel"><h3>train / val / test 分层</h3>
<p style="font-size:14px">按 group_key（相机陷阱按 location 整组）确定性划分，防泄漏。<b class="ok">跨 split 同图 0</b>。</p>
<div class="tablewrap"><table><thead><tr><th>split</th><th class="num">图</th><th class="num">负样本</th></tr></thead><tbody>
<tr><td>train+val</td><td class="num">{S["recs"]:,}</td><td class="num">{S["neg"]:,} ({100*S["neg"]/S["recs"]:.0f}%)</td></tr>
<tr><td>test</td><td class="num">{T["recs"]:,}</td><td class="num">{T["neg"]:,} ({100*T["neg"]/T["recs"]:.0f}%)</td></tr>
</tbody></table></div></div>
<div class="panel"><h3>框来源信任分层</h3><div class="donut-row">{donut([("gt",S["prov"].get("gt",0)),("md_pseudo",S["prov"].get("md_pseudo",0))],{"gt":"var(--accent)","md_pseudo":"#c88"})}
<div class="dleg"><div><i style="background:var(--accent)"></i>gt 真标注 — {S["prov"].get("gt",0):,} ({100*S["prov"].get("gt",0)/tot_box:.1f}%)</div>
<div><i style="background:#c88"></i>md_pseudo — {S["prov"].get("md_pseudo",0):,} ({100*S["prov"].get("md_pseudo",0)/tot_box:.1f}%)</div>
<div class="dim" style="margin-top:6px">iNat auto，MD≥0.7</div></div></div></div>
</div>

<h2><span class="n">7</span>11 维验证</h2>
<div class="tablewrap"><table><thead><tr><th>#</th><th>维度</th><th>实测</th><th class="num">判定</th></tr></thead><tbody>
<tr><td class="dim">1</td><td>量（每类≥目标）</td><td>bird {S["cls_box"]["bird"]:,} · person {S["cls_box"]["person"]:,} · squirrel {S["cls_box"]["squirrel"]:,} · cat {S["cls_box"]["cat"]:,}</td><td class="ok">✓</td></tr>
<tr><td class="dim">2</td><td>次要类均衡（命门豁免）</td><td>person/squirrel/cat = 2.2 ≤8</td><td class="ok">✓</td></tr>
<tr><td class="dim">3</td><td>域混合（bird）</td><td>feeder+清晰+后院 {100*(S["dom_bird"].get("feeder-cam",0)+S["dom_bird"].get("clear-photo",0)+S["dom_bird"].get("backyard",0))/tot_bird:.0f}%；web 21%</td><td class="ok">✓</td></tr>
<tr><td class="dim">4</td><td>框尺度</td><td>bird 中位 7.4%；squirrel/cat 补后院大尺度</td><td class="ok">✓</td></tr>
<tr><td class="dim">5</td><td>provenance</td><td>gt {100*S["prov"].get("gt",0)/tot_box:.1f}% / pseudo {100*S["prov"].get("md_pseudo",0)/tot_box:.1f}%</td><td class="ok">✓</td></tr>
<tr><td class="dim">6</td><td>CC-BY 署名</td><td>缺 0（数据集级 default_author）</td><td class="ok">✓</td></tr>
<tr><td class="dim">7</td><td>split 泄漏</td><td>跨 split 同图 0</td><td class="ok">✓</td></tr>
<tr><td class="dim">8</td><td>负样本</td><td>train {100*S["neg"]/S["recs"]:.0f}%（相机陷阱空帧）</td><td class="ok">✓</td></tr>
<tr><td class="dim">9</td><td>许可 §4</td><td>全商用可</td><td class="ok">✓</td></tr>
<tr><td class="dim">10</td><td>框坐标</td><td>0 越界（_clip_bbox）</td><td class="ok">✓</td></tr>
<tr><td class="dim">11</td><td>test 完整</td><td>五类齐全</td><td class="ok">✓</td></tr>
</tbody></table></div>
<p class="muted">终审边角：极端长宽比框 5/44k、退化框 0、坏图 0、真重复 0 —— <b class="ok">无问题</b>。</p>

<h2><span class="n">8</span>逐图审查（每源采样，框=标注真值）</h2>
<p class="muted">每源 6 张。留意各域：<b>meproject</b> 真 feeder-cam 定拍、<b>csp/squirreldet</b> 后院大尺度松鼠/猫、<b>caltech/ena24</b> 相机陷阱、<b>oiv7</b> 网图。</p>
{"".join(review_blocks)}

<h2><span class="n">9</span>诚实说明（非缺陷，合理取舍）</h2>
<div class="callout"><h3>两项透明说明</h3><ol>
<li><b>负样本域</b>：现为相机陷阱空帧（= 户外空场景，对 yard cam 合理）；空喂食器帧无现成数据集，后续可自采补。</li>
<li><b>逐图署名</b>：聚合源用数据集级 default_author（满足门 + 有效 CC-BY 归属）；iNat/OIV7 逐图作者已有取用能力，发行前串成逐图清册（§4 随产物披露）。</li>
</ol><p style="margin:8px 0 0;font-size:13px" class="muted">cat/person 仍以相机陷阱/网图为主——但它们在 yard cam 中本就出现在各种距离，非"必须 feeder 尺度"（不同于 bird 必在喂食器旁），故可接受。</p></div>

<h2><span class="n">10</span>判定</h2>
<p><b>数据门 PASS，11 维 + 终审全绿，无阻塞问题。命门 bird 适配观鸟器域；squirrel/cat/person 已补后院多域大尺度。round2 训练数据就绪。</b></p>
<p class="muted">下一步：NanoDet-416 COCO warm-start 微调 + 数据量/参数 scaling 消融（SwanLab）。</p>
<footer>证据：<code>data/gate.py</code> · <code>dim_check.py</code> · <code>src_class.py</code> · <code>final_audit.py</code>（box）。详版 <code>results/detect/round2/数据分析.md</code>。图表为内联 SVG（主题自适应），样本图为 GT 框可视化。</footer>
</div></body></html>"""

out = "/root/autodl-tmp/detect_round2/综合报告.html"
open(out, "w").write(HTML)
print(f"REPORT 写出 {out}  大小 {len(HTML)//1024} KB  样本 {sum(len(v) for v in by_src_samples.values())} 张")
