# vendored eBird/Clements taxonomy registry（ADR-0002 版本 pin）

分类段层级树（roll-up / genus-family aux loss / 类集 universe）的**身份地基**。
由姊妹 repo [`bird-tagger/taxonomy`](https://github.com/xinzhuwang-wxz) 产出，**vendor 进本仓
并 pin 版本**——不依赖姊妹 repo 活路径，保 provenance 干净、跨轮可复现。

| 文件 | 内容 |
|---|---|
| `species.jsonl` | 11,167 种：`ebird_code`(主键) · `sci_name` · `genus` · `family_code/sci` · `order` · `taxon_order` · `avibase_id` |
| `rollup.jsonl` | 4,120 非种级（issf/form/…）→ `reports_as_ebird_code`（上滚到种） |
| `_meta.json` | **版本 pin**：authority=eBird/Clements 2025.0，`raw_sha256`（权威身份） |

**只管「是谁」+ 层级，不含区域/分布**——区域是独立一层，从 eBird checklist / GBIF occurrence 来
（见 `data/region/`、`scripts/build_europe_species_list.py`、docs/classify/05 §模式1）。

消费入口：`src/edge_cam/data/ebird_registry.py`（`EbirdRegistry.load()` +
`Hierarchy.from_registry`）。重 vendor：从 `bird-tagger/taxonomy/data/` 复制同名文件即可
（换 eBird 版本时同步更新 `_meta.json` 的 `raw_sha256`）。
