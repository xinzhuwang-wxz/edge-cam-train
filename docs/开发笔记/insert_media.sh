#!/bin/bash
# 把本地图片逐张插入飞书文档对应锚点(失败的会打印,便于补)。
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"
cd "/Users/bamboo/Githubs/edge-cam-train/docs/开发笔记" || exit 1
DOC="https://www.feishu.cn/docx/BCt7dIUXzoN8c2x246gcxFfZnaf"

ins() { # 1=file 2=anchor 3=caption(可选)
  local out
  out=$(lark-cli docs +media-insert --as user --doc "$DOC" --file "assets/$1" \
        --selection-with-ellipsis "$2" ${3:+--caption "$3"} 2>&1 \
        | grep -oE '"ok": ?(true|false)|unsafe|no (block|match)|not found' | head -1)
  printf '%-28s %s\n' "$1" "${out:-?}"
  sleep 1
}

# 单图(唯一锚点)
ins diagram_product_chain.png "这就是本笔记的全部内容"
ins v_series_roadmap.jpg     "整条选型逻辑都被它的能力框住"
# (已插) ins diagram_pipeline.png     "图中虚线标出"
ins diagram_data_unify.png   "统一是单独一件工程"
ins det_320_vs_416.png       "换分辨率 + 更长训练"
ins det_recall.png           "才是产品成立依据"
ins det_quant.png            "方向性,非板子"
ins cls_bird_coverage.png    "很多是远景/多鸟/杂背景"
ins cls_v1_vs_v2.png         "消 domain gap"
ins cls_envelope.png         "量化前后(分类)+ 效果"
ins cls_regional.png         "in-region 子集比 mask on/off"

# 分类卡片(反序插于同锚点 → 最终 naturgucker/arter/inat)
ins card_cls_inat.png        "逐图过滤只留 CC0/CC-BY"
ins card_cls_arter.png       "逐图过滤只留 CC0/CC-BY"
ins card_cls_naturgucker.png "逐图过滤只留 CC0/CC-BY"

# 检测曲线对(反序 → overall/perclass)
ins det_perclass.png         "训练干净收敛"
ins det_overall_curve.png    "训练干净收敛"

# 级联 6 张(反序 → ok1..wrong)+ caption
ins casc_wrong.png    "分类器外扩输入框" "疣鼻天鹅误报绿头鸭(自信报错)"
ins casc_fallback.png "分类器外扩输入框" "苍鹭→置信不足回退 bird(安全回退)"
ins casc_ok4.png      "分类器外扩输入框" "普通鵟 buteo buteo ✓"
ins casc_ok3.png      "分类器外扩输入框" "红额金翅雀 carduelis carduelis ✓"
ins casc_ok2.png      "分类器外扩输入框" "大斑啄木鸟 dendrocopos major ✓"
ins casc_ok1.png      "分类器外扩输入框" "大鸬鹚 phalacrocorax carbo ✓"
echo "DONE"
