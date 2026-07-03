"""Caltech-CT sm 图坐标缩放（round1 揭出的 bug）：注解原图分辨率、图下采样 → 按实际缩放框。"""

from __future__ import annotations

import json

from PIL import Image

from edge_cam.data.adapters.detect.caltech_ct import CaltechCtAdapter


def _seed_cct(raw, anno_wh, img_wh):
    """造 ECCV18 注解(anno_wh 分辨率 + 框)+ 实际 sm 图(img_wh)。"""
    anno_dir = raw / "commercial/caltech_ct/eccv18/eccv_18_annotation_files"
    anno_dir.mkdir(parents=True)
    coco = {
        "images": [
            {
                "id": "u1",
                "file_name": "u1.jpg",
                "width": anno_wh[0],
                "height": anno_wh[1],
                "location": "loc1",
            }
        ],
        "annotations": [
            {"id": 1, "image_id": "u1", "category_id": 1, "bbox": [200, 300, 400, 500]}
        ],
        "categories": [{"id": 1, "name": "squirrel"}],
    }
    (anno_dir / "train_annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    img_dir = raw / "commercial/caltech_ct/eccv_18_all_images_sm"
    img_dir.mkdir(parents=True)
    Image.new("RGB", img_wh).save(img_dir / "u1.jpg")


def test_cct_rescales_boxes_to_actual_sm_image(tmp_path):
    """注解 2048×1494、实际 sm 图 1024×747(0.5×) → 框缩放 0.5、尺寸更新为实际。"""
    _seed_cct(tmp_path, anno_wh=(2048, 1494), img_wh=(1024, 747))
    s = next(x for x in CaltechCtAdapter(str(tmp_path)).load_raw() if x.boxes)
    assert (s.width, s.height) == (1024, 747)  # 更新为实际图尺寸
    name, box = s.boxes[0]
    assert name == "squirrel"
    assert box == [100.0, 150.0, 200.0, 250.0]  # [200,300,400,500]×0.5


def test_cct_no_rescale_when_dims_match(tmp_path):
    """注解尺寸==实际图尺寸 → 框不动(无下采样的情形)。"""
    _seed_cct(tmp_path, anno_wh=(1024, 747), img_wh=(1024, 747))
    s = next(x for x in CaltechCtAdapter(str(tmp_path)).load_raw() if x.boxes)
    assert (s.width, s.height) == (1024, 747)
    assert s.boxes[0][1] == [200.0, 300.0, 400.0, 500.0]  # 原样


def test_cct_build_records_scaled_box_in_5class(tmp_path):
    """端到端:缩放后的框进 5 类 DetImageRecord(box 已缩放、类=squirrel)。"""
    _seed_cct(tmp_path, anno_wh=(2048, 1494), img_wh=(1024, 747))
    recs = [r for r in CaltechCtAdapter(str(tmp_path)).build_records() if r.boxes]
    assert len(recs) == 1
    assert recs[0].boxes[0].bbox == [100.0, 150.0, 200.0, 250.0]
    assert recs[0].width == 1024 and recs[0].height == 747
