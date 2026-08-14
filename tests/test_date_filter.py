from search.dates import date_where
from tests.factories import add_photo


def _ids(conn, where, params):
    return {r["id"] for r in conn.execute(
        "SELECT id FROM photos p WHERE owner_id = 1" + where, params
    )}


def test_date_where_bounds_shot_at(conn):
    a = add_photo(conn, content_hash="a" * 64, shot_at="2025-07-01T10:00:00")
    b = add_photo(conn, content_hash="b" * 64, shot_at="2025-09-01T10:00:00")
    where, params = date_where({"date_from": "2025-06-01", "date_to": "2025-08-31"})
    got = _ids(conn, where, params)
    assert a in got and b not in got


def test_date_where_empty_without_bounds(conn):
    assert date_where({}) == ("", [])
