# ADR-0002：eBird 规范键 taxonomy seam — 多源细分类合并的前置

- 状态：Accepted
- 日期：2026-06-19
- 相关：`docs/plan-v2.md` §5.2/§5.3/§5.4/§B.1、`docs/engineering.md` §6 data/taxonomy、[[ADR-0001]]、架构审查（taxonomy 假 seam）

## 背景

要「下载开源细粒度鸟类数据集 → 映射到统一物种键 → 与 BIRDS-525 合并」，并让**地域 mask**（plan §5.4 最大精度杠杆）与**层级回退**（属/科/bird）工作，都需要一个**跨数据集稳定的规范键**。

当前 `data/taxonomy.py` 只有占位 `IdentityTaxonomy`（小写化俗名当 `taxon_key`）。一个 adapter = **假 seam**：① 地域 mask 的 eBird 清单对不上小写俗名 → 静默失效；② 不同数据集词表无法合并；③ 无层级依据。方案 §5.2/§B.1 早已指定 **eBird/Clements 学名（带版本）为规范键**——本 ADR 记录如何落地，不是新决策。

## 决策

1. **采用 eBird/Clements 物种键为唯一规范 `taxon_key`**，版本化（如 `ebird-v2024`）记入 manifest 与 ModelCard provenance。
2. **保留 `Taxonomy` Protocol 作为 seam**；新增 `EbirdTaxonomy` adapter（由一张 `源标签 → eBird code` 映射表构造）。**每个数据源 = 一个用其映射表实例化的 adapter**（BIRDS-525 俗名表、开源集学名表各一份）。≥2 个 adapter → **真 seam**。
3. **`IdentityTaxonomy` 降级为「无 eBird 表时的 feasibility 默认」**，保留向后兼容；prep 经 config 选择 taxonomy。
4. **映射表是版本化数据产物**：`label → ebird_code` 表按源维护、随产物披露（许可可追溯）；eBird/Clements checklist 本身免费可商用（Cornell，CC0 性质），不触红线。
5. **未映射标签返回 None**，由调用方决定层级回退或丢弃——不静默编造键。

## 理由

- 规范键是 merge / 地域 mask / 层级回退三者的**共同前置**，集中在一个深模块（locality），换数据集只加一个 adapter（leverage）。
- 测试面 = 接口（`label → eBird key`），逐源纯函数单测，不碰图片/训练。
- eBird 是观鸟领域事实标准（Merlin/eBird 生态），学名带版本可跨集稳定合并、支持地域频率清单。

## 影响 / 后果

- `RegionalMask` 的「规范键」契约（架构审查 C 已加显式校验）落到实处：清单与 manifest 同为 eBird key。
- 引入 eBird/Clements taxonomy 数据依赖（版本化文件）；**数据获取（拉 checklist + 匹配 525 俗名）是独立后续步**，本 ADR 先确立 seam 架构。
- 分类多源合并（架构审查 B）解锁：依赖本规范键去重/合并。

## 备选与放弃理由

- **继续用占位小写俗名**：跨集合并/地域过滤做不了，假 seam 永久化。
- **自造一套内部物种 ID**：重复造轮子、丢掉 eBird 地域频率与生态对接，且无版本权威。
- **GBIF/Catalogue of Life 等其他权威**：可行，但 eBird 与地域频率清单（§5.4）和 Merlin 工业做法直接对齐，鸟类场景首选。
