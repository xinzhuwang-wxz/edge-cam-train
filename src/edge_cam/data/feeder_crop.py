"""喂食器域"中大化"：crop-to-feeder-framing（round3 §A6，Bird Buddy 洞察）。

喂食器/观鸟器场景里生物凑到镜头前吃东西 → 画面里都是**中大近景**。本模块把**已有已标注的动物框**
（相机陷阱的小/中松鼠、猫、宽幅鸟）做**动物为中心的 zoom-crop**：裁一块居中于目标框、夹在图内的
正方窗，使该框在裁窗里占中大比例 → 用现有数据合成"喂食器域中大"实例，对齐部署域（M1 显示次要类
recall 是短板，此手段直接补 squirrel/cat 的中大近景）。

纯几何 + 纯函数（无图像 I/O，可无卡测）：`compute_feeder_crop` 算裁窗，`remap_boxes_to_crop` 把该图
所有框映射进裁窗坐标并按可见度过滤。实际像素裁剪 + 存图 + 新 record 由 build/acquire 阶段的薄 I/O
包一层调用（裁窗原生分辨率存盘，NanoDet 训练时再 stretch 到 416）。

⚠️ 坑（§A6）：**不放大过小的源框**（`min_box_px` 门）——
20px 松鼠放大到 400px 会糊 + 假细节 → 返回 None。
"""

from __future__ import annotations

from edge_cam.contracts.schemas.detection_manifest import DetBox, DetImageRecord


def compute_feeder_crop(
    img_w: int,
    img_h: int,
    box: list[float],
    *,
    target_box_frac: float = 0.5,
    min_box_px: float = 48.0,
    margin: float = 1.15,
) -> tuple[int, int, int, int] | None:
    """算一个居中于 `box`、夹在图内的**正方裁窗**，使 box 长边 ≈ `target_box_frac` × 裁窗边长。

    box=[x,y,w,h]（原图像素）。box 长边 < `min_box_px` → 返回 None（太小，放大会糊）。
    返回 (x, y, w, h) 整数正方裁窗（原图坐标）。`margin` 保证 box 带边距地容进裁窗。
    caller 可对 `target_box_frac` 在 [0.4,0.7] 随机抽以增多样性（数据增强）。
    """
    if not 0.0 < target_box_frac <= 1.0:
        raise ValueError(f"target_box_frac 须 ∈ (0,1]，得到 {target_box_frac}")
    x, y, w, h = box
    box_max = max(w, h)
    if box_max < min_box_px:
        return None
    side = box_max / target_box_frac  # 让 box 长边占 target_box_frac
    side = max(side, box_max * margin)  # 至少容下 box + 边距
    side = min(side, float(min(img_w, img_h)))  # 不超过图（不能凭空放大超出画面）
    cx, cy = x + w / 2.0, y + h / 2.0  # 居中于 box 中心
    left = cx - side / 2.0
    top = cy - side / 2.0
    left = max(0.0, min(left, img_w - side))  # 夹在图内
    top = max(0.0, min(top, img_h - side))
    return int(round(left)), int(round(top)), int(round(side)), int(round(side))


def remap_boxes_to_crop(
    boxes: list[DetBox],
    crop_rect: tuple[int, int, int, int],
    *,
    min_visible: float = 0.4,
) -> list[DetBox]:
    """把原图 `boxes` 变换进裁窗坐标系；保留可见比例 ≥ `min_visible`（框面积占比）的框并裁剪到边界。

    返回新 DetBox 列表（bbox 在裁窗坐标、未 resize；
    category_id / label_provenance 原样带过）。
    可见度过滤挡掉裁窗边缘只露一角的框（否则成半截标注噪声）。
    """
    cx, cy, cw, ch = crop_rect
    out: list[DetBox] = []
    for b in boxes:
        x, y, w, h = b.bbox
        area = w * h
        if area <= 0:
            continue
        ix1, iy1 = max(x, cx), max(y, cy)
        ix2, iy2 = min(x + w, cx + cw), min(y + h, cy + ch)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter / area < min_visible:
            continue
        out.append(
            DetBox(
                bbox=[ix1 - cx, iy1 - cy, iw, ih],  # 平移进裁窗坐标 + 已裁剪到边界
                category_id=b.category_id,
                label_provenance=b.label_provenance,
            )
        )
    return out


def plan_feeder_crops(
    records: list[DetImageRecord],
    target_category_ids: set[int],
    *,
    min_box_px: float = 48.0,
    target_box_frac: float = 0.5,
    max_crops_per_image: int = 2,
) -> list[tuple[DetImageRecord, tuple[int, int, int, int], list[DetBox]]]:
    """对每条 record 的**目标类框**（squirrel/cat）规划一个"动物为中心"的中大近景裁窗。

    纯函数（只用 record 的 width/height/boxes，不读像素）→ 返回
    `[(源 record, 裁窗 rect, 裁窗内重映射的 boxes)]`，供 build 的 I/O 步裁图存盘 + 建新 record。
    每图最多 `max_crops_per_image` 个（防爆量）；框太小或裁后目标类没留住则跳。
    """
    plans: list[tuple[DetImageRecord, tuple[int, int, int, int], list[DetBox]]] = []
    for r in records:
        n = 0
        for b in r.boxes:
            if n >= max_crops_per_image:
                break
            if b.category_id not in target_category_ids:
                continue
            crop = compute_feeder_crop(
                r.width, r.height, b.bbox, min_box_px=min_box_px, target_box_frac=target_box_frac
            )
            if crop is None:
                continue
            new_boxes = remap_boxes_to_crop(r.boxes, crop)
            if not any(nb.category_id in target_category_ids for nb in new_boxes):
                continue  # 裁窗边界把目标类切没了 → 跳
            plans.append((r, crop, new_boxes))
            n += 1
    return plans
