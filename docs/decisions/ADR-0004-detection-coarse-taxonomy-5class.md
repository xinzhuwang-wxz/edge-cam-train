# ADR-0004：检测粗类体系 11 → 5（数据高效的分组）

- 状态：Accepted
- 日期：2026-06-20
- 相关：`docs/plan-v2.md` §5.1、[[ADR-0003]]、实验1（`results/实验1/`）、检测数据集方案 grill

## 背景

实验1 用 11 个 feeder-cam 细类（bird/squirrel/cat/dog/raccoon/rabbit/deer/fox/skunk/hedgehog/bear）训 NanoDet，
结果 **skunk AP=1.3、raccoon AP=42**——稀有类样本饿死、几乎不可用。检测器在级联里的职责是
**找 bird(→crop→种分类器) + 粗辨非鸟访客(非鸟直接出大类、不进种分类器)**，并不需要细分每种哺乳动物。
细类既无产品价值，又因数据稀少而废。用户初稿提"少类分组"，方向正确。

## 决策

采用 **5 类粗检测体系**：

| idx | 类 | 含义 / 数据来源映射 |
|---|---|---|
| 0 | `bird` | 目标；→crop→种分类器。COCO bird / OIV7 Bird / camera-trap bird |
| 1 | `squirrel` | 头号害兽/混淆源，单列（OIV7 Squirrel 充足）|
| 2 | `cat` | 捕食者（COCO cat / OIV7 Cat）|
| 3 | `person` | 补粮的人，检到→抑制告警（COCO person / OIV7 Person）|
| 4 | `other_animal` | 长尾哺乳动物合并：raccoon/rabbit/fox/dog/deer/opossum/rodent… 治样本饿死 |

- 高价值且数据足的类单列（bird/squirrel/cat/person）；数据稀的长尾合并（other_animal）。
- 各源 label → 5 类的映射各自维护（见数据集 adapter）。empty/无框图 = 负样本（非类别）。

## 理由

- **数据高效**：合并长尾直接解掉实验1 的稀有类饿死（skunk 1.3 / raccoon 42）。
- **产品够用**：消费侧只需"是不是鸟/松鼠/猫/人/其它动物"；细分种是分类器的事。
- **映射干净**：COCO/OIV7/Caltech CT/NACTI 的物种标签都能无歧义落进这 5 类。

## 影响 / 后果

- `data/detection_classes.py` 的 `FEEDER_CAM_CLASSES` 由 11 改 5（不可逆口径变更）；实验1 的 11 类检测产物作废，需按 5 类重训。
- 各源映射从"按源类型固定字段(coco=/oiv7=)"演进为"每数据集 adapter 自带映射"（[[ADR-0003]] 数据集 adapter 优化，配合本 ADR 落地）。
- person 为新类，需引入 COCO person（户外子集）数据。

## 备选与放弃理由

- **保留 11 类**：稀有类饿死（实验1 实证），细分无产品价值。
- **用户初稿 5 类(squirrel_rodent/medium_mammal)**：方向对，但丢了 dog、deer/bear 归属不清；本 ADR 的 5 类把长尾统一进 other_animal，更干净、无遗漏。
- **极简 3-4 类(bird/other)**：丢失 squirrel/cat/person 的高价值区分（害兽/捕食者/人各有产品动作）。
