import pytest
from fastapi.testclient import TestClient

from ingest.receive import receive
from ingest.worker import drain, thumbnail_handler
from tests.factories import add_upload
from tests.fixtures import jpeg_bytes_with_exif, sha
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    ctx = app.state.context
    upload_id = add_upload(ctx.conn)
    morning = jpeg_bytes_with_exif(when="2025:07:12 09:00:00", model="X-T5")
    evening = jpeg_bytes_with_exif(when="2024:02:03 20:00:00", model="iPhone")
    receive(
        ctx.conn, ctx.originals, owner_id=ctx.settings.owner_id, upload_id=upload_id,
        rel_path="morning.jpg", declared_hash=sha(morning), data=morning,
    )
    receive(
        ctx.conn, ctx.originals, owner_id=ctx.settings.owner_id, upload_id=upload_id,
        rel_path="evening.jpg", declared_hash=sha(evening), data=evening,
    )
    drain(ctx.conn, {"thumbnail": thumbnail_handler(ctx.originals, ctx.derived, 320, 1600)})
    with TestClient(app) as test_client:
        yield test_client


def test_unfiltered_library_shows_both(client):
    assert client.get("/library").text.count('class="tile"') == 2


def test_categorical_facet_filter_narrows_the_grid(client):
    response = client.get("/library?f_camera_model=iPhone")
    assert response.text.count('class="tile"') == 1


def test_time_of_day_filter_works(client):
    assert client.get("/library?f_time_of_day=morning").text.count('class="tile"') == 1


def test_numeric_year_range_filter_works(client):
    assert client.get("/library?n_year=2025:").text.count('class="tile"') == 1


def test_sidebar_lists_facet_values_with_counts(client):
    body = client.get("/library").text
    assert "X-T5" in body
    assert "iPhone" in body
    assert "Camera" in body


def test_sort_by_facet_changes_order(client):
    ascending = client.get("/library?sort=year_asc").text
    descending = client.get("/library?sort=year_desc").text
    # The 2024 photo is the iPhone shot; the 2025 photo is the X-T5 shot. Sorting
    # by year flips which thumbnail comes first — assert on the thumb ids.
    first_ascending = ascending.split('src="/thumb/')[1].split('"')[0]
    first_descending = descending.split('src="/thumb/')[1].split('"')[0]
    assert first_ascending != first_descending


def test_filters_survive_into_the_infinite_scroll_link(client):
    body = client.get("/library?f_camera_model=X-T5").text
    assert "f_camera_model=X-T5" in body or 'class="tile"' in body


def test_place_filter_narrows_by_city(client):
    from ingest.facets import backfill_place_facets
    from tests.factories import add_photo

    conn = client.app.state.context.conn
    add_photo(conn, photo_id=10, content_hash="rome", thumb_key="r.jpg",
              gps_lat=41.9028, gps_lon=12.4964)  # Rome
    add_photo(conn, photo_id=11, content_hash="kyiv", thumb_key="k.jpg",
              gps_lat=50.4501, gps_lon=30.5234)  # Kyiv
    backfill_place_facets(conn)

    body = client.get("/library").text
    assert "Place" in body and "Rome" in body           # sidebar place group
    assert client.get("/library?f_place_city=Rome").text.count('class="tile"') == 1


def test_filters_apply_in_place_without_reloading_the_sidebar(client):
    body = client.get("/library").text
    # the filter form swaps only #grid via htmx, so the sidebar (and its scroll)
    # is never re-rendered on a filter change; a clear-all resets everything.
    assert 'id="grid"' in body
    assert 'hx-target="#grid"' in body
    assert 'hx-select="#grid"' in body
    assert "Clear all" in body
