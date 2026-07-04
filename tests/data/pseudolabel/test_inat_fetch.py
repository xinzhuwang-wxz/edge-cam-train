"""iNat API 一页解析 + medium 直链归一 + 分页游标/防空转（注入 fake fetcher）。"""

from __future__ import annotations

from edge_cam.data.pseudolabel.inat_fetch import (
    fetch_inat_aves_obs,
    medium_url,
    parse_inat_api_page,
)


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


def _fake_photo(oid):
    return {
        "id": oid,
        "quality_grade": "research",
        "location": "1.0,2.0",
        "taxon": {"id": 3},
        "user": {"login": "u"},
        "photos": [{"id": oid, "license_code": "cc0", "url": f"h/photos/{oid}/square.jpg"}],
    }


def test_fetch_paginates_by_id_cursor_and_caps_max_obs() -> None:
    """id_above 游标翻页 + max_obs 截断（注入 fake fetcher，不联网）。"""
    calls = []

    def fake(url):
        calls.append(url)
        above = int(url.split("id_above=")[1])
        ids = [above + 1, above + 2]  # 每页 2 条，id 递增
        return {"results": [_fake_photo(i) for i in ids]}

    obs = fetch_inat_aves_obs(max_obs=3, per_page=2, sleep=0, fetch_json=fake)
    assert len(obs) == 3  # max_obs 截断
    assert "id_above=0" in calls[0] and "id_above=2" in calls[1]  # 游标前进


def test_fetch_stops_on_empty_results() -> None:
    obs = fetch_inat_aves_obs(max_obs=100, sleep=0, fetch_json=lambda _url: {"results": []})
    assert obs == []


def test_fetch_breaks_when_cursor_stalls() -> None:
    """results 非空但 id 不再前进（异常分页）→ 防空转 break（有界，非翻遍全库）。"""
    n = {"c": 0}

    def stuck(_url):
        n["c"] += 1
        return {"results": [_fake_photo(5)]}  # id 恒为 5，游标卡住

    obs = fetch_inat_aves_obs(max_obs=1000, per_page=2, sleep=0, fetch_json=stuck)
    # 第 1 次合法前进 0→5，第 2 次检测到 5→5 卡住即 break（有界 2 次，绝不无限翻页）
    assert n["c"] == 2 and len(obs) == 2
