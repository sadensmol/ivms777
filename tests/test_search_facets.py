import pytest

from search.facets import (
    FacetFilter,
    build_where,
    facet_counts,
    parse_filters,
)
from tests.factories import add_photo


def add_photo_with_facets(conn, photo_id, facets):
    add_photo(conn, photo_id=photo_id, content_hash=f"h{photo_id}")
    for key, text, num in facets:
        conn.execute(
            "INSERT INTO photo_facets(photo_id, key, value_text, value_num) VALUES (?,?,?,?)",
            (photo_id, key, text, num),
        )


@pytest.fixture
def library(conn):
    add_photo_with_facets(conn, 1, [("camera_model", "X-T5", None), ("iso", None, 200.0)])
    add_photo_with_facets(conn, 2, [("camera_model", "X-T5", None), ("iso", None, 6400.0)])
    add_photo_with_facets(conn, 3, [("camera_model", "iPhone", None), ("iso", None, 400.0)])
    return conn


def matching_ids(conn, filters):
    where, params = build_where(filters)
    sql = f"SELECT id FROM photos p WHERE 1=1 {where} ORDER BY id"
    return [row["id"] for row in conn.execute(sql, params)]


def test_no_filters_matches_everything(library):
    assert matching_ids(library, []) == [1, 2, 3]


def test_categorical_filter_matches_one_value(library):
    assert matching_ids(library, [FacetFilter("camera_model", values=["iPhone"])]) == [3]


def test_categorical_values_are_ored(library):
    filters = [FacetFilter("camera_model", values=["iPhone", "X-T5"])]
    assert matching_ids(library, filters) == [1, 2, 3]


def test_numeric_range_is_inclusive(library):
    assert matching_ids(library, [FacetFilter("iso", gte=200, lte=400)]) == [1, 3]


def test_open_ended_range_works(library):
    assert matching_ids(library, [FacetFilter("iso", gte=1000)]) == [2]


def test_filters_across_keys_are_anded(library):
    filters = [FacetFilter("camera_model", values=["X-T5"]), FacetFilter("iso", lte=1000)]
    assert matching_ids(library, filters) == [1]


def test_parse_filters_reads_categorical_and_numeric_params():
    filters = parse_filters({"f_camera_model": "X-T5,iPhone", "n_iso": "200:800"})
    by_key = {f.key: f for f in filters}

    assert by_key["camera_model"].values == ["X-T5", "iPhone"]
    assert (by_key["iso"].gte, by_key["iso"].lte) == (200.0, 800.0)


def test_parse_filters_accepts_open_ended_ranges():
    by_key = {f.key: f for f in parse_filters({"n_iso": "1000:"})}
    assert (by_key["iso"].gte, by_key["iso"].lte) == (1000.0, None)


def test_parse_filters_ignores_unknown_and_malformed_params():
    assert parse_filters({"f_not_a_facet": "x", "n_iso": "junk", "page": "2"}) == []


def test_facet_counts_are_ordered_by_frequency(library):
    counts = facet_counts(library, owner_id=1, key="camera_model")
    assert counts == [("X-T5", 2), ("iPhone", 1)]
