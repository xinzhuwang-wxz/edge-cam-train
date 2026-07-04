# iNat 教师伪标注 + Label Studio 人审（操作手册）

> 数据管线 §3b「MD 高置信自动收 / 中置信 Label Studio 人审 / 低置信丢」的**上手步骤**。代码见
> `src/edge_cam/data/pseudolabel/`；设计见 [数据管线.md](数据管线.md) §3b、[[ADR-0006]] D7（信任分层）。

## 0. 一句话

iNat 拉鸟图（无框）→ MegaDetector 教师打框（带 score）→ 按置信度三分：**高**自动收、**低**丢、
**中**（拿不准）交 Label Studio 人审。人只看"中置信"那批（保规模又控质量）。

## 1. 跑伪标注（box/GPU，`md` 隔离 env）

```bash
conda activate md          # 隔离 env（pytorch-wildlife，AGPL 不进产物 §4）
cd <repo> && PYTHONPATH=src python -m edge_cam.data.pseudolabel.run \
  --raw-root /root/autodl-tmp/detect_raw --max-obs 8000 --per-taxon-cap 40 \
  --md-version MDV6-yolov9-c \
  --md-weights ~/.cache/torch/hub/checkpoints/MDV6-yolov9-c.pt \
  --conf-hi 0.7 --conf-lo 0.2
```

产物落 `raw_root/commercial/inat_md/`：

| 产物 | 内容 |
|---|---|
| `images/*.jpg` | iNat 拉的鸟图（CC0/CC-BY/research/geo） |
| `inat_md_coco.json` | **auto**（≥conf-hi，md_pseudo）→ 直接进 build |
| `inat_review_coco.json` | **中置信**（[lo,hi)）待人审 |
| `label_studio_tasks.json` | 中置信 → LS 导入任务（**MD 框作预标注**，人只微调） |
| `previews/*.jpg` | 教师打框可视化（绿=高置信 / 橙=中置信），肉眼直评 |
| `triage_stats.json` | 三层计数 |

## 2. Label Studio 人审（Apache-2.0，许可安全）

```bash
pip install label-studio          # 单独 env 即可；Apache-2.0 合规
# 允许本地图片服务（指向 images 目录）
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/root/autodl-tmp/detect_raw/commercial/inat_md
label-studio start                # 默认 http://localhost:8080（AutoDL 用端口转发）
```

1. **建 Project** → Labeling Setup 用这段配置（RectangleLabels，标签 bird）：
   ```xml
   <View>
     <Image name="image" value="$image"/>
     <RectangleLabels name="label" toName="image">
       <Label value="bird"/>
     </RectangleLabels>
   </View>
   ```
   > `name="label"` / `toName="image"` 必须与代码一致（`label_studio.py` 里 `_FROM/_TO`）。
2. **Import** → 上传 `label_studio_tasks.json`。每张图带 MD 预标注（`predictions`），**人只需
   确认/拉正/删除**，不用从零画。
3. **审核动作**：框对了→直接 Submit；框歪/太紧（如漏尾巴）→拖拽拉正；MD 框错东西→删掉；
   漏了鸟→补框。
4. **Export** → 选 **JSON**（LS 原生格式，带 `annotations[].result`）→ 存 `ls_export.json`。

## 3. 回收人审结果（→ 信任分层）

```bash
PYTHONPATH=src python -m edge_cam.data.pseudolabel.run \
  --raw-root /root/autodl-tmp/detect_raw \
  --import-ls /path/to/ls_export.json
```

产 `inat_verified_coco.json`（框 `label_provenance=md_human_verified`）。

## 4. 进 build（两行信任分层并列）

`configs/data/detect_round2.yaml`：
```yaml
  inat_md: {}            # auto 份（md_pseudo）读 inat_md_coco.json
  inat_md_verified: {}   # 人审份（md_human_verified）读 inat_verified_coco.json
```
两份同源同图、独立成行 → 训练可按信任分层加权（数据管线 §7）。

## 5. 阈值怎么定

`--conf-hi/--conf-lo` 默认 0.7 / 0.2。**先用 `eval/md_difficulty.py` 在已有 GT 带框 bird 图上量
MD 的召回 + 置信分布**，据此调阈：MD 对本域鸟越准，conf-hi 可越低（多自动收、少人审）。
conf-lo 对齐 MD 推理下限（低于此的弱检测多是噪声，丢）。
