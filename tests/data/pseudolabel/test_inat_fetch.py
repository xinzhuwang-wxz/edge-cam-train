"""iNat API 一页解析 + medium 直链归一（纯函数）。"""

from __future__ import annotations

from edge_cam.data.pseudolabel.inat_fetch import medium_url, parse_inat_api_page


def _page(**over):
    res = {
        "id": 42,
        "quality_grade": "research",
        "location": "37.5,-122.3",
        "taxon": {"id": 7000, "name": "Passer domesticus"},
        "user": {"login": "birder1"},
        "photos": [{"id": 999, "license_code": "cc-by", "url": "https://x/photos/999/square.jpg"}],
    }
    res.update(over)
    return {"results": [res]}


def test_parse_maps_photo_license_and_geo() -> None:
    obs = parse_inat_api_page(_page())
    assert len(obs) == 1
    o = obs[0]
    assert o.photo_id == "999" and o.taxon_id == "7000"
    assert o.license == "CC-BY"  # cc-by → 大写归一，对上白名单
    assert o.lat == 37.5 and o.lon == -122.3
    assert o.author == "birder1"
    assert o.photo_url == "https://x/photos/999/medium.jpg"  # square→medium


def test_parse_nc_license_uppercased_and_rejectable() -> None:
    """cc-by-nc → CC-BY-NC（大写后仍非白名单，select_inat 会拒）。"""
    obs = parse_inat_api_page(
        _page(photos=[{"id": 1, "license_code": "cc-by-nc", "url": "a/square.jpg"}])
    )
    assert obs[0].license == "CC-BY-NC"


def test_parse_missing_license_becomes_empty() -> None:
    """无 license_code（保留权利）→ ""，被白名单拒。"""
    obs = parse_inat_api_page(_page(photos=[{"id": 1, "url": "a/square.jpg"}]))
    assert obs[0].license == ""


def test_parse_skips_obs_without_photos() -> None:
    assert parse_inat_api_page(_page(photos=[])) == []


def test_parse_bad_location_none() -> None:
    o = parse_inat_api_page(_page(location=None))[0]
    assert o.lat is None and o.lon is None


def test_medium_url_variants() -> None:
    assert medium_url("h/photos/1/square.jpeg") == "h/photos/1/medium.jpeg"
    assert medium_url("h/photos/1/large.png") == "h/photos/1/medium.png"
    assert medium_url("h/photos/1/medium.jpg") == "h/photos/1/medium.jpg"  # 已是 medium 不变
