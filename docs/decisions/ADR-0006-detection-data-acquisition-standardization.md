# ADR-0006：检测数据获取标准化 + 逐图溯源透明化（可插拔源 + JSONL 数据集）

- 状态：Accepted
- 日期：2026-07-04
- 相关：[[ADR-0003]]（DatasetAdapter seam）、[[ADR-0004]]（5 类）、[[ADR-0002]]（eBird 键）、`docs/detect/01-数据集.md`、`docs/detect/03-实操日志.md`、bird-tagger 采集器（外部参考）、CLAUDE.md §4（许可红线）
- 评审：经 critic 独立 verify（2026-07-04），本版并入其 3 项 MAJOR + 遗漏项修订

## 背景

检测数据链目前**两层不对称**：

1. **组装层（raw → 5 类 manifest）已标准化** —— `DatasetAdapter`（`load_raw()` + 基类映射/清洗/split/负样本）+ `build.py` + `detect_feeder5.yaml`，加源=配置加一行。这层干净。
2. **获取层（source → raw 落盘）拼凑** —— `build.py` 假设「raw 已下载好」。实际只有 1 个真脚本（`fetch_oiv7_direct.py`）；ENA24/Caltech-CT/COCO 全靠**手动 wget/unzip**，唯一记录是 `docs/detect/03` 里的散落 prose。换机/换人就得考古，**不可复现**。
3. **逐图溯源薄** —— `license_manifest.csv` 只有 `[path, source, license]`；`DatasetSpec.attribution` 只是个 **bool flag**，管线**从没捕获逐图作者/原图 URL** → CC-BY 商用署名（§4）**声明了却没兑现**。无 archive sha256 / 快照版本 / 下载命令 → 不可审计。

同时（本轮）要**吸取教训覆盖更多数据**：接入 iNat Open Data S3（补 bird 覆盖）、Roboflow bird-feeder（补当前为 0 的 feeder 真实域）、eBird 区域过滤，并让新源即插即用。

## 决策

把「获取 → 检测数据集」做成**标准化 / 独立化 / 可插拔**。

### D0. 完全替换，不新旧并存（no dual-track）

本 ADR 是检测**数据集管线的完全更新**：新管线上线即**替换**旧路径，**不保留旧机制并存**。
- 旧获取脚本（`fetch_oiv7_direct.py`）折进对应 adapter 的 `acquire()` 后**删除**，不留独立脚本。
- 旧手动下载（散在 `docs/detect/03` 的 prose 命令）由标准 `acquire()` **取代**，prose 降级为历史。
- 旧 11 类检测数据路径（`detection_classes` 的 11 类、`merge_map`、`detection_ingest` 的 FiftyOne 11 类拉取、旧 `detection_feeder.yaml`）**清除**（即 issue #18）；打标契约 `contracts/schemas/detection.py` **迁到 5 类**（`FEEDER5_CATEGORIES`），与新管线一致。
- 数据集序列化**只用 JSONL**（D5），不留 JSON 双轨。

原则：读到本管线任一处，看到的都应是**唯一的新做法**，不是"新的 + 兜底旧的"。

### D1. 获取元数据进 `DatasetSpec`（代码内自包含，不新增 config 文件）

给 `DatasetSpec` 加一个 `acquire: AcquireSpec` 子模型（**不是** top-level `SourceSpec` —— `edge_cam.data.prep.SourceSpec` 已占用该名，避冲突）：

```
AcquireSpec:
  method: "lila_http" | "s3_direct" | "inat_open_data" | "roboflow" | "local_mirror" | "manual"
  urls: list[str]            # 规范下载地址（archive / annotation / bucket 前缀）
  version: str               # 数据集快照标识（日期 / release tag）
  archive_sha256: dict[str,str]  # {url_basename: sha256}，下载后校验（压缩包完整性）
  # select 参数（已有的 caps/region/max 归此）：per-class caps、eBird region、max_samples
```

**adapter 的 `DatasetSpec`（在代码里声明）= 数据从哪来的单一事实源**，取代散落 prose 与手动命令。人读的"全源清单"由 `python -m edge_cam.data.adapters.detect.acquire --list` 从所有已注册 adapter 的 spec **导出**，无需另维护一份 yaml。`detect_feeder5.yaml` **职责不变**（选哪些源 + 组装期 overrides），不承载 URL/校验和。

### D2. adapter 加 `acquire()` seam（与 `load_raw()` 对称）

「一个源 = 一个自包含可插拔 adapter，既知道怎么**下**（`acquire`）也知道怎么**解析**（`load_raw`）」。

```
DetectionDatasetAdapter.acquire(self, raw_root) -> AcquireReceipt
```

- **基类**收公共逻辑：http/s3 下载、`archive_sha256` 校验、解压、断点续（跳过已存在且校验通过的文件）、幂等（重跑安全）、写 `_acquire.json` 收据。
- **子类**只声明源特定 URL/method（在 `DatasetSpec.acquire`）；特殊源（iNat 流式 TSV、Roboflow SDK）覆写少量下载逻辑。
- **收据** `<raw_root>/<layer>/<name>/_acquire.json`：`{source, method, urls, version, archive_sha256, downloaded_at, command, image_count, box_count, tool_versions}`（`downloaded_at` 由 caller 传入时间戳，纯函数化便于测试）→ 可复现 + 可审计。
- **`method: manual` 源**（暂无脚本可下）：`acquire()` **只校验 raw 是否就位 + archive_sha256**，缺失则**抛可执行错误**（错误信息直接给出该源的下载命令），**不静默放行**。

### D3. 集成 = 两阶段（acquire → build），build 保持离线纯净

- **独立 CLI**：`python -m edge_cam.data.adapters.detect.acquire --config <build_cfg> [--source X] [--dry-run]`。按 build config 里选中的源，逐源 `acquire()`。
- `build.py` **不自动 acquire**（保持 build 离线、快、可在无网机跑）；获取从"手动/散脚本"**替换**为唯一标准命令 `acquire` + 收据。
- **成本闸**：每源 `select.max` 有硬上限；iNat 类大源必须设 per-taxon 配额 + 总量硬顶；`--dry-run` 只算"会下多少"不真下。

### D4. 逐图溯源透明化（明确改哪些代码）

- `Provenanced` 扩 attribution 字段：`author / original_url / source_media_id / asset_sha256`（均可选，默认空 → **向后兼容**，`provenance_summary` 与 ModelCard 链不受影响）。`license` 收敛为 **SPDX 标识**（小映射表规范化：`CC-BY-4.0` / `CC0-1.0` / `CDLA-Permissive-1.0` …）。
- `RawSample` 加对应逐样本字段（当前它只有 path/box，**无逐样本溯源** —— 必补）。
- `DetectionDatasetAdapter.build_records()`（`base.py:118`）**必改**：从 `RawSample` 读逐样本 attribution 传入 `DetImageRecord`，缺省回退到 spec 级 `source/license`。
- `build.py:_write_license_manifest()`（`build.py:41`）**必改**：扩列 `[path, source, license_spdx, author, original_url, source_media_id, asset_sha256]` → **真正兑现 CC-BY 逐图署名**（§4）。

### D5. JSONL 标准化数据集（用户要求）

检测数据集记录序列化为 **JSONL**（一行一 `DetImageRecord`）+ `meta.json` sidecar（`name/version/categories/root/provenance` 汇总）—— 标准化、可流式、可 diff、下游逐行可插拔。**JSONL 是唯一格式**：`DetectionManifest.save/load` 迁到 JSONL+meta，**移除**旧的单文件 JSON 双轨（D0）。`to_coco()`/`write_nanodet_labels()`（NanoDet 仍吃 COCO）为派生视图，保留。现有消费方（build 写、训练/评估/promotion 读）一并迁到 JSONL。

### D6. 跨源一致性（去重 + split 防泄漏）

覆盖更多源后同一张图可能多源出现：
- **跨源去重**：assemble 前按 `asset_sha256` 全局精确去重（pHash 近似去重为后续增量）。
- **跨源 split 一致**：split key 改用**内容标识（asset_sha256 优先，回退 path）+ 全局固定 seed**（非现在的 per-adapter `name` seed）→ 同一图无论来自哪源都落同一 split，杜绝跨源泄漏。**此为对 `_split_of` seed 策略的变更，需迁移。**

### D7. 覆盖更多数据（首批新源，按可插拔契约接入）

- **iNat Open Data S3**（`method: inat_open_data`）：搬 bird-tagger 采集器思路（免鉴权流式 TSV → research-grade + geo + per-taxon 配额），**收紧 CC0/CC-BY**（去一切 NC）。iNat `taxon_id → eBird key` 走 crosswalk 填 `taxon_key`（检测层 bird 单一类，taxon_key 为次要、best-effort）。
- **MegaDetector 伪标注 = 独立 GPU 阶段**（**不在** `acquire()` 内）：iNat 无 bbox → 先跑 MD（`pytorch-wildlife`，隔离 env，MDV6-mit/apa）产 **COCO JSON**（自带 `_mdlabel.json` 收据）→ iNat 的 adapter 是个 `CocoJsonAdapter` 子类，`load_raw()` 吃这份 COCO。MD 权重不进产物、不发行、只出框坐标（框坐标非版权物，不传染）。伪标注是把 MD 知识迁进小 student 的**蒸馏**载体（比 soft-label KD 更实用）。
  - **先量 MD 可信度**：在**有框源**（ENA24/Caltech/OIV7/Roboflow）上跑 `eval/megadetector.py` 得 MD 的 AP/bird 召回 → 决定信任阈值。
  - **分层 QA（不全人审，保规模）**：MD 框按置信分三层 —— **高**→自动收；**中**→ Label Studio 人审（MMDetection 有官方半自动标注集成）；**低**→丢。
  - **信任分层入溯源**：每框记 `label_provenance ∈ {gt, md_pseudo, md_human_verified}`（见 D4）→ 训练可按信任加权/分阶段，且透明可审。
- **框级溯源**：`DetBox` 加 `label_provenance`（默认 `gt`）——与逐图 attribution 一并兑现"每个框哪来的"透明度。
- **Roboflow bird-feeder**（`method: roboflow`）：补 feeder 真实域（当前 0）。逐个核 license（要 CC-BY/Public）。API key 走 **`ROBOFLOW_API_KEY` 环境变量**（不入库）。
- **eBird 区域过滤**：`select.region`，把 bird 覆盖对齐下游区域 mask（[[ADR-0002]]）。

## 理由

- **透明**：一条命令复现全套 raw；每图可追作者/URL/license/来源/校验和 → §4「全链路可追溯」从声明变兑现。
- **独立/可插拔**：acquire + parse 同处自包含（不割裂"一个源"的知识，守 [[ADR-0003]]）；JSONL 解耦下游；加新源 = 一个 adapter + register，caller 零改。
- **覆盖**：iNat 补 bird 多样性、Roboflow 补 feeder 域（直击最大短板：训练分布≠部署分布，plan §8 的 99.5%→88% 教训）。

## 影响 / 后果

- `DatasetSpec` 加 `acquire: AcquireSpec` 子模型 + `select`；`DetectionDatasetAdapter` 加 `acquire()`（现有 4 源补下载声明，OIV7 收编 `fetch_oiv7_direct.py`，ENA24/CCT/COCO 声明 `method` + 校验和）。
- `Provenanced` / `RawSample` / `DetImageRecord` 加 attribution 字段（新字段默认空，旧记录仍可解析——非"旧路径并存"，是新 schema 的缺省）。
- `build_records()` + `_write_license_manifest()` 按 D4 改。
- `_split_of` seed 策略改全局内容 key（D6）—— 重建 split（口径变更，记入实操日志）。
- `DetectionManifest.save/load` 迁 JSONL+meta（移除单文件 JSON）；`license_manifest.csv` 扩列。
- 新依赖按 §4 隔离/lazy：iNat（stdlib urllib+gzip，无 SDK）、Roboflow（`roboflow`，可选 extra）、MD（`pytorch-wildlife`，隔离 env）。
- 新 CLI `data.adapters.detect.acquire`（+ `--list` / `--dry-run`）。
- **DVC**：acquired raw + `_acquire.json` 收据纳入 `data/` DVC 跟踪（已 DVC 化）。
- `docs/detect/01` 更新获取段；`configs/data/detect_feeder5.yaml` 结构不变（新源加行）。

**移除清单（D0 完全替换，不留旧路径）**：
- `scripts/fetch_oiv7_direct.py`（折进 `Oiv7Adapter.acquire()` 后删）。
- 旧 11 类检测路径（issue #18）：`data/detection_classes.py` 的 11 类常量/映射、`data/merge_map.py`、`data/detection_ingest.py` 的 FiftyOne 11 类拉取、`configs/data/detection_feeder.yaml` 及其关联测试。
- `contracts/schemas/detection.py` 打标契约由 11 类 **迁 5 类**（`FEEDER5_CATEGORIES`）。
- `DetectionManifest` 旧单文件 JSON `save/load`。
- `docs/detect/03` 里手动下载 prose 标注为历史（获取以 adapter `acquire()` + `--list` 为准）。

## 备选与放弃理由

- **维持现状（脚本 + 手动 + prose）**：不可复现、CC-BY 未兑现、加源靠考古。否。
- **独立 Fetcher 注册表 / 新 `sources.yaml`**：割裂"一个源"的下载 vs 解析知识（违 [[ADR-0003]]）、与 `detect_feeder5.yaml` 双文件同步、`SourceSpec` 命名冲突。否 —— acquire 收进 `DatasetSpec`+adapter，人读清单靠 `--list` 导出。
- **build 自动 acquire（一阶段）**：build 被网络/GPU 污染、无网机跑不了、iNat/MD 重活混进快构建。否 —— 两阶段。
- **MD 伪标注塞进 `acquire()`**：`acquire()` 变重（要 GPU + pytorch-wildlife），违"基类只收公共 HTTP/S3"。否 —— MD 为独立阶段，产 COCO 给 `load_raw`。
- **继续 JSON / JSON+JSONL 双轨**：覆盖更多数据后 JSON 全量载入笨重、不可流式、diff 噪声大；双轨违 D0「不新旧并存」。JSONL 为唯一格式。
- **iNat 直接进训练（不 MD 伪标）**：iNat 无 bbox，中心裁剪代理框差、伤 crop 完整率。否。
- **iNat 收 CC-BY-NC（如 bird-tagger）**：商用，NC 传染上游权重（§4）。否 —— 收紧 CC0/CC-BY。
