from ingest.exif import ExifFacts
from ingest.facets import backfill_place_facets, derive_facets, store_facets
from tests.factories import add_photo


def facet_map(facets):
    return {f.key: (f.value_text, f.value_num) for f in facets}


def test_gps_yields_place_name_facets():
    facts = ExifFacts(gps_lat=41.9028, gps_lon=12.4964)  # Rome
    m = facet_map(derive_facets(facts, None, None))
    assert m["place_city"] == ("Rome", None)
    assert m["place_country"] == ("Italy", None)


def test_no_place_facets_without_gps():
    m = facet_map(derive_facets(ExifFacts(shot_at="2025-07-12T20:30:00"), None, None))
    assert "place_city" not in m


def test_backfill_place_facets_names_existing_gps_photos(conn):
    add_photo(conn, photo_id=1, content_hash="a", thumb_key="1.jpg",
              gps_lat=41.9028, gps_lon=12.4964)  # Rome
    add_photo(conn, photo_id=2, content_hash="b", thumb_key="2.jpg")  # no GPS -> skipped
    assert backfill_place_facets(conn) == 1
    city = conn.execute(
        "SELECT value_text FROM photo_facets WHERE photo_id = 1 AND key = 'place_city'"
    ).fetchone()
    assert city["value_text"] == "Rome"
    assert backfill_place_facets(conn) == 0  # idempotent


def test_time_facets_come_from_shot_at():
    facts = ExifFacts(shot_at="2025-07-12T20:30:00")  # a Saturday evening
    m = facet_map(derive_facets(facts, width=None, height=None))

    assert m["year"] == (None, 2025.0)
    assert m["month"] == (None, 7.0)
    assert m["hour"] == (None, 20.0)
    assert m["weekday"] == ("Saturday", None)
    assert m["time_of_day"] == ("evening", None)
    assert m["is_weekend"] == ("yes", None)


def test_time_of_day_buckets():
    def bucket(hour: int) -> str:
        facts = ExifFacts(shot_at=f"2025-07-12T{hour:02d}:00:00")
        return facet_map(derive_facets(facts, None, None))["time_of_day"][0]

    assert bucket(2) == "night"
    assert bucket(6) == "dawn"
    assert bucket(10) == "morning"
    assert bucket(15) == "afternoon"
    assert bucket(20) == "evening"
    assert bucket(23) == "night"


def test_exposure_facets_are_numeric():
    facts = ExifFacts(
        raw={"FNumber": 1.8, "ExposureTime": 0.005, "FocalLength": 35.0, "ISOSpeedRatings": 3200}
    )
    m = facet_map(derive_facets(facts, None, None))

    assert m["aperture"] == (None, 1.8)
    assert m["shutter_speed"] == (None, 0.005)
    assert m["focal_length"] == (None, 35.0)
    assert m["iso"] == (None, 3200.0)


def test_categorical_exposure_settings_are_named_not_numbered():
    facts = ExifFacts(raw={"Flash": 1, "WhiteBalance": 0, "MeteringMode": 5})
    m = facet_map(derive_facets(facts, None, None))

    assert m["flash"] == ("fired", None)
    assert m["white_balance"] == ("auto", None)
    assert m["metering_mode"] == ("pattern", None)


def test_camera_facets():
    facts = ExifFacts(camera="X-T5", lens="XF33mmF1.4", raw={"Make": "FUJIFILM"})
    m = facet_map(derive_facets(facts, None, None))

    assert m["camera_make"] == ("FUJIFILM", None)
    assert m["camera_model"] == ("X-T5", None)
    assert m["lens"] == ("XF33mmF1.4", None)


def test_image_shape_facets():
    m = facet_map(derive_facets(ExifFacts(), width=4000, height=3000))
    assert m["aspect"] == ("landscape", None)
    assert m["megapixels"][1] == 12.0

    portrait = facet_map(derive_facets(ExifFacts(), width=3000, height=4000))
    assert portrait["aspect"] == ("portrait", None)

    square = facet_map(derive_facets(ExifFacts(), width=1000, height=1000))
    assert square["aspect"] == ("square", None)


def test_gps_presence_is_a_facet():
    with_gps = facet_map(derive_facets(ExifFacts(gps_lat=51.5, gps_lon=-0.1), None, None))
    assert with_gps["has_gps"] == ("yes", None)
    assert with_gps["gps_lat"] == (None, 51.5)

    without = facet_map(derive_facets(ExifFacts(), None, None))
    assert without["has_gps"] == ("no", None)
    assert "gps_lat" not in without


def test_missing_exif_yields_only_the_facets_that_are_knowable():
    m = facet_map(derive_facets(ExifFacts(), None, None))
    assert set(m) == {"has_gps"}


def test_store_facets_replaces_previous_values(conn):
    add_photo(conn, photo_id=1, content_hash="h")
    store_facets(conn, 1, derive_facets(ExifFacts(camera="A"), None, None))
    store_facets(conn, 1, derive_facets(ExifFacts(camera="B"), None, None))

    rows = conn.execute(
        "SELECT value_text FROM photo_facets WHERE photo_id = 1 AND key = 'camera_model'"
    ).fetchall()
    assert [row["value_text"] for row in rows] == ["B"]
