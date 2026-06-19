# 租卡前准备 & 上机清单

> 目标：本地（mac，CPU）把**数据 + 权重 + 可移植 manifest** 全部备齐，租到 GPU 卡后
> 只做「上传 → 建环境 → 开训」三步，不再在卡上做数据准备。
> 关联：`CLAUDE.md §6 命令参考`、`engineering.md §7 W1 落地步骤`。

---

## 1. 本地已就绪的产物（截至准备完成）

| 产物 | 路径 | 体量 | 说明 |
|---|---|---|---|
| 分类 raw（BIRDS-525） | `~/Downloads/bird_species(DST1192)/` | 1.3G / 16796 图 | Kaggle DST1192，train/valid/test 子目录 |
| 分类 manifest（**可移植**） | `data/processed/birds525/manifest.json` | 6.1M | 525 类，相对路径 + 记录 root；换机用 `data_root` 覆盖 |
| 检测数据（COCO-json） | `data/processed/detection_feeder/{train,validation}/` | 4.8G（train 15000 图 / val 3335 图） | FiftyOne 导出，**图片自带**（`data/` + `labels.json`），便携；11 大类 |
| NanoDet 基线权重 | `weights/nanodet/nanodet-plus-m_416_checkpoint.ckpt` | 34M | 检测器固定基线 |
| timm 预训练权重 | `~/.cache/huggingface/hub/models--timm--*` | 各 ~5–20M | efficientnet_lite0 / mobilenetv3_large_100 / repvgg_a0（消融三骨干） |

> ⚠️ 许可：birds525 与 COCO/OIV7 均**可行性优先、不进商用权重**（plan §5.2 / §C.1 / ADR-0001）。
> OIV7 逐图 CC-BY 的署名清册仍是 TODO，进商用前必须补。
>
> ⚠️ 检测长尾极稀疏（train 标注数）：skunk 8 / hedgehog 46 / raccoon 63 / fox 152，
> 这几类实际不可训。建议上机时按 `detection_classes.CORE_CLASSES`（bird/squirrel/cat/dog）
> 或自定子集裁剪（plan §5.1 大类可裁剪）。bird 11154 / dog 7108 / cat 4513 充足。

---

## 2. 要上传到 GPU 机的东西

**谁随 git 走、谁要 scp**（`.gitignore` 挡了 `/data/processed/*` 和 `weights/`，
唯一例外是 birds525 manifest）：

| 产物 | 传输方式 | 目标 |
|---|---|---|
| `data/processed/birds525/manifest.json` (6.1M) | **随 git**（已开 gitignore 例外） | clone 自带 |
| birds525 raw 图 (1.3G) | **scp/tar** | `<BOX>/data/raw/birds525/` |
| `data/processed/detection_feeder/` (4.8G，自带图) | **scp/tar** | `<BOX>/data/processed/detection_feeder/` |
| `weights/nanodet/*.ckpt` (34M) | **scp**（weights/ 被 gitignore） | `<BOX>/weights/nanodet/` |
| timm 预训练权重 | **卡上联网重下**（AutoDL 可直连 HF）或拷 `~/.cache/huggingface/hub` | 同路径 |

打包示例：

```bash
# 分类 raw
cd ~/Downloads && tar czf birds525.tar.gz "bird_species(DST1192)"
# 检测数据（含图，4.8G）
cd ~/Githubs/edge-cam-train && tar czf detection_feeder.tar.gz -C data/processed detection_feeder
# scp 后在卡上解包到对应目标路径
```

> 备选：不传 raw 图也行——把 birds525 raw 上传后在卡上重跑 `python -m edge_cam.data.prep`
> 即可重建等价 manifest（split 用固定 seed，确定性）。但有了可移植 manifest + `data_root`
> 覆盖，**推荐直接复用 git 自带的 manifest**，省一步。

---

## 3. 卡上环境

```bash
git clone <repo> && cd edge-cam-train
conda env create -f environment.yml && conda activate edge-cam-train
# 校验护栏
pytest -q && ruff check src tests
```

---

## 4. 开训命令

### 分类器（关键：用 `data_root` 覆盖指向上传后的 raw 根）

```bash
# 真训（GPU，预训练，80 epoch）
python -m edge_cam.train.classify.train \
  data.manifest=data/processed/birds525/manifest.json \
  data.data_root=<BOX>/data/raw/birds525/'bird_species(DST1192)' \
  trainer.accelerator=gpu model.pretrained=true

# 消融（plan §B.3）：Hydra multirun 扫骨干
python -m edge_cam.train.classify.train -m \
  data.data_root=<BOX>/.../bird_species\(DST1192\) \
  model.name=efficientnet_lite0,mobilenetv3_large_100,repvgg_a0 \
  trainer.accelerator=gpu model.pretrained=true
```

> `data_root` 留空（null）时回退到 manifest 里记录的 mac 本地 root —— 卡上**必须覆盖**，
> 否则路径不存在。manifest 本身无需重生成。

### 检测器（NanoDet-Plus，B.2）—— 需先建独立环境

NanoDet 用旧版 pytorch-lightning，跑在**独立 conda env**（与本仓隔离）。一次性搭建：

```bash
bash scripts/setup_nanodet.sh     # 建 nanodet env + 装依赖 + 锁版；打印训练/eval/导出四步命令
```

之后：① 在 edge-cam-train env 生成 config（指向 detection_feeder）→ ② nanodet env 跑
`tools/train.py` → ③ `tools/test.py` 出 mAP（**手动汇总进消融表**，本仓不含检测 mAP 评测）
→ ④ `tools/export_onnx.py` 导 FP32 ONNX（本仓自动跑结构契约校验）。详见脚本输出。

### 地域过滤（B.5）—— eBird 映射 + 区域清单

`taxon_key` 已是 eBird code（`data/processed/birds525/ebird_map.csv`，377/525 自动匹配，随 git）。
要跑 B.5「地域过滤 on/off 对比」还需一份**区域在场物种清单**：

```bash
# 真实区域（需免费 eBird API key）→ 有意义的对比
curl -H "X-eBirdApiToken: <KEY>" "https://api.ebird.org/v2/product/spplist/US-CA" -o region_codes.json
python scripts/build_region_list.py --codes-file region_codes.json \
  --map data/processed/birds525/ebird_map.csv --out regions/us_ca.json

# 然后评估时带上 regional_json → 出 ④ regional 级
python -m edge_cam.eval.run_envelope manifest=data/processed/birds525/manifest.json \
  ckpt=<...> fp32_onnx=<...> regional_json=regions/us_ca.json
```

> 未填真实区域时可 `--demo` 生成占位清单**仅验证机制**（数字无地域意义）。
> 未匹配 eBird 的 148 类（BIRDS-525 笔误/泛称）`taxon_key=None`，不进区域 mask；
> 需要时编辑 `ebird_map.csv` 人工补别名后重跑 prep。

---

## 5. 训完（仍在卡上 / 回流）

- 两侧训完都导 **FP32 ONNX**（铁律，`export.enabled=true` 自动跑）。
- ONNX → 校准集 → ACUITY/pegasus PTQ → `.nb` 那段是**板子相关**，回到有板子/Tina-SDK 的环境做（W1 spike）。
- 校准集本地即可建：`build_calib_set(manifest, out, data_root=...)`（同样支持 `data_root`）。

---

## 6. 不在本清单内（明确边界）

- DVC 远端 / 实验追踪（aim）接入 —— 见审计工程债，可后续补。
- 检测框 MegaDetector bootstrap、OIV7 署名清册 —— TODO。
- 板上 latency/精度回灌 —— 需有板子（engineering §8.1）。
