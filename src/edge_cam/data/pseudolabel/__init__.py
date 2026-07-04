"""iNat 无框源 → MegaDetector 伪标注 → 置信分层 → Label Studio 人审（独立 box/GPU 流程）。

数据管线 §3b 的可跑实现（不进 `acquire()`，避免基类拖 GPU/pytorch-wildlife）：

    ① inat_fetch    iNat API 枚举 Aves(CC0/CC-BY/research/geo) → 并行 S3 下图
    ② md_label      MegaDetector 伪标注出框（隔离 env/GPU）→ COCO（保 score）
    ③ triage        按框置信度分三层：高→自动收(md_pseudo) / 中→人审 / 低→丢
    ④ label_studio  中置信 → LS 导入 JSON（MD 框作预标注）；人审导出 → md_human_verified

纯函数（解析/分层/LS 往返转换）可测；下图/MD 推理是薄 box 步骤。产物 `inat_md_coco.json`
（+ `inat_verified_coco.json`）就位后 `InatMdAdapter` 直接读进 5 类管线。
"""
