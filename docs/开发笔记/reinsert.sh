#!/bin/bash
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"
cd "/Users/bamboo/Githubs/edge-cam-train/docs/开发笔记" || exit 1
DOC="https://www.feishu.cn/docx/BCt7dIUXzoN8c2x246gcxFfZnaf"

ins() { # 1=file 2=anchor 3=caption
  local out
  out=$(lark-cli docs +media-insert --as user --doc "$DOC" --file "assets/$1" \
        --selection-with-ellipsis "$2" ${3:+--caption "$3"} 2>/dev/null \
        | grep -oE '"ok": ?(true|false)' | head -1)
  printf '  img %-26s %s\n' "$1" "$out"; sleep 1
}
wb() { # 1=mmd 2=anchor 3=name
  local tok ok
  tok=$(lark-cli docs +update --as user --doc "$DOC" \
        --markdown '<whiteboard type="blank"></whiteboard>' \
        --mode insert_after --selection-with-ellipsis "$2" 2>/dev/null \
        | grep -A2 board_tokens | grep -oE '[A-Za-z0-9]{24,}' | head -1)
  ok=$(lark-cli docs +whiteboard-update --as user --whiteboard-token "$tok" \
        --input_format mermaid --source @"$1" --overwrite --yes 2>/dev/null \
        | grep -oE '"ok": ?(true|false)' | head -1)
  printf '  wb  %-26s board=%s %s\n' "$3" "$tok" "$ok"; sleep 1
}

echo "[whiteboards]"
wb diagrams/product_chain.mmd "这就是本笔记的全部内容" product_chain
wb diagrams/pipeline.mmd      "图中虚线标出"          pipeline
wb diagrams/data_unify.mmd    "统一是单独一件工程"      data_unify

echo "[roadmap]"
ins v_series_roadmap.jpg "整条选型逻辑都被它的能力框住"

echo "[cls cards · 反序]"
ins card_cls_inat.png        "逐图过滤只留 CC0/CC-BY"
ins card_cls_arter.png       "逐图过滤只留 CC0/CC-BY"
ins card_cls_naturgucker.png "逐图过滤只留 CC0/CC-BY"

echo "[det 曲线对 · 反序]"
ins det_perclass.png      "训练干净收敛"
ins det_overall_curve.png "训练干净收敛"

echo "[det 单图]"
ins det_320_vs_416.png "换分辨率 + 更长训练"
ins det_recall.png     "才是产品成立依据"
ins det_quant.png      "方向性,非板子"

echo "[cls 图]"
ins cls_bird_coverage.png "很多是远景/多鸟/杂背景"
ins cls_v1_vs_v2.png      "消 domain gap"
ins cls_envelope.png      "量化前后(分类)+ 效果"
ins cls_regional.png      "in-region 子集比 mask on/off"

echo "[级联 6 张 · 反序 + caption]"
ins casc_wrong.png    "逐步标注图见下" "疣鼻天鹅误报绿头鸭(自信报错)"
ins casc_fallback.png "逐步标注图见下" "苍鹭→置信不足回退 bird(安全回退)"
ins casc_ok4.png      "逐步标注图见下" "普通鵟 buteo buteo ✓"
ins casc_ok3.png      "逐步标注图见下" "红额金翅雀 carduelis carduelis ✓"
ins casc_ok2.png      "逐步标注图见下" "大斑啄木鸟 dendrocopos major ✓"
ins casc_ok1.png      "逐步标注图见下" "大鸬鹚 phalacrocorax carbo ✓"
echo DONE
