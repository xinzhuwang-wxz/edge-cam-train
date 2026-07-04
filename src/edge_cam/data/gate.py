"""检测数据门（round2）：训练前**数据质量 + 数据量**硬 gate——不达标不训。

用户要求"训练前数据质量和数据量都要过关"。查（全 manifest-derivable，纯函数可测）：
1. **量**：每类框数 ≥ 目标（`min_boxes`）——命门 bird 尤其。
2. **均衡**：最多/最少类框数比 ≤ `max_imbalance`（round1 曾 5:1 失衡）。
3. **框坐标合理**：框在图内、正面积、不超界——抓 [[CCT 坐标 bug]] 式错位（框比图大）。
4. **许可（§4）**：逐图 license 全在商用白名单；无 NC/unknown。
5. **CC-BY 署名**：CC-BY 图须有 author 或 original_url（§4 逐图署名兑现）。
6. **伪标注信任**：报 gt / md_pseudo / md_human_verified 框占比（md_pseudo 未审占比过高 → 提示）。

`gate()` 返回 `DataGateReport`（passed + 逐项）。CI/build 后调；不 pass 抛或拦训练。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from edge_cam.contracts.schemas.detection_manifest import DetectionManifest

# round2 每类**最少**框数目标（other_animal 是压上限、不设下限；docs/detect/round2-数据计划.md §3）
ROUND2_MIN_BOXES: dict[str, int] = {"bird": 8000, "person": 4000, "squirrel": 2000, "cat": 2000}

# §4 商用许可白名单（SPDX/常见写法）；不在此列（含 unknown / 任何 NC）= 红线拦。
COMMERCIAL_LICENSES: frozenset[str] = frozenset(
    {
        "CC0", "CC0-1.0", "CC-BY", "CC-BY-4.0", "CC-BY-2.0", "CC-BY-3.0",
        "CDLA-Permissive", "CDLA-Permissive-1.0", "CDLA-Permissive-2.0",
        "MIT", "Apache-2.0", "BSD-3-Clause", "Public", "Public Domain",
    }
)  # fmt: skip
_CC_BY_PREFIX = "CC-BY"  # 需逐图署名的许可前缀（不含 CC0）


@dataclass
class DataGateReport:
    """数据门结果：总 pass + 逐项 (名, 过, 详情) + 分布/信任统计。"""

    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    box_counts: dict[str, int] = field(default_factory=dict)
    provenance_mix: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        head = "PASS ✅" if self.passed else "FAIL ❌"
        lines = [f"检测数据门：{head}"]
        for name, ok, detail in self.checks:
            lines.append(f"  [{'✓' if ok else '✗'}] {name}：{detail}")
        return "\n".join(lines)


def _box_class_counts(m: DetectionManifest, split: str | None) -> dict[str, int]:
    inv = {v: k for k, v in m.categories.items()}
    c: Counter = Counter()
    for r in m.records:
        if split and r.split != split:
            continue
        for b in r.boxes:
            c[inv.get(b.category_id, str(b.category_id))] += 1
    return dict(c)


def gate(
    manifest: DetectionManifest,
    *,
    split: str | None = "train",
    min_boxes: dict[str, int] | None = None,
    max_imbalance: float = 6.0,
    bounds_tol: float = 1.02,
) -> DataGateReport:
    """数据门：量 + 均衡 + 框合理 + 许可 + 署名 + 信任分层。返回 DataGateReport。"""
    min_boxes = ROUND2_MIN_BOXES if min_boxes is None else min_boxes
    counts = _box_class_counts(manifest, split)
    checks: list[tuple[str, bool, str]] = []

    # 1. 量：每类框数 ≥ 目标
    short = {k: (counts.get(k, 0), v) for k, v in min_boxes.items() if counts.get(k, 0) < v}
    checks.append(
        (
            "数据量(每类≥目标)",
            not short,
            "全达标" if not short else f"未达标 {{类:(实,标)}}={short}",
        )
    )

    # 2. 均衡：max/min（只看有目标的类 + 有框的类）
    present = [counts.get(k, 0) for k in min_boxes] or [1]
    lo = min(x for x in present if x > 0) if any(present) else 0
    hi = max(present)
    ratio = (hi / lo) if lo else float("inf")
    checks.append(
        (f"类均衡(max/min≤{max_imbalance:.0f})", ratio <= max_imbalance, f"比={ratio:.1f}")
    )

    # 3. 框坐标合理（抓 CCT 式坐标错位：框超界/零负面积）
    bad = 0
    total = 0
    for r in manifest.records:
        if split and r.split != split:
            continue
        for b in r.boxes:
            total += 1
            x, y, w, h = b.bbox
            if (
                w <= 0
                or h <= 0
                or x < 0
                or y < 0
                or x + w > r.width * bounds_tol
                or y + h > r.height * bounds_tol
            ):
                bad += 1
    frac = bad / total if total else 0.0
    checks.append(("框坐标合理(无超界/零负面积)", bad == 0, f"{bad}/{total} 越界({frac:.1%})"))

    # 4. 许可（§4 商用白名单，无 NC/unknown）
    bad_lic = sorted({r.license for r in manifest.records if r.license not in COMMERCIAL_LICENSES})
    checks.append(
        (
            "许可(§4 商用白名单)",
            not bad_lic,
            "全商用可" if not bad_lic else f"红线 license={bad_lic}",
        )
    )

    # 5. CC-BY 署名（须 author 或 original_url）
    no_attr = sum(
        1
        for r in manifest.records
        if r.license.startswith(_CC_BY_PREFIX) and not (r.author or r.original_url)
    )
    checks.append(("CC-BY 逐图署名", no_attr == 0, f"{no_attr} 张 CC-BY 缺署名"))

    # 6. 伪标注信任分层（informational + md_pseudo 未审占比提示）
    prov: Counter = Counter()
    for r in manifest.records:
        if split and r.split != split:
            continue
        for b in r.boxes:
            prov[b.label_provenance] += 1
    tot = sum(prov.values()) or 1
    pseudo_frac = prov.get("md_pseudo", 0) / tot
    checks.append(
        (
            "伪标注信任(md_pseudo 未审占比)",
            pseudo_frac <= 0.6,  # 未审 MD 框占比过半 → 提示（非硬拦，看 LS 复核策略）
            f"gt/pseudo/verified={dict(prov)}，md_pseudo {pseudo_frac:.0%}",
        )
    )

    passed = all(ok for _, ok, _ in checks)
    return DataGateReport(
        passed=passed, checks=checks, box_counts=counts, provenance_mix=dict(prov)
    )
