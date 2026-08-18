import pytest
from fastapi.testclient import TestClient

from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from ingest.receive import receive
from ingest.worker import drain, thumbnail_handler
from tests.factories import add_photo, add_upload
from tests.fixtures import jpeg_bytes_with_exif, sha
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    ctx = app.state.context
    upload_id = add_upload(ctx.conn)
    data = jpeg_bytes_with_exif(model="X-T5")
    receive(
        ctx.conn, ctx.originals, owner_id=1, upload_id=upload_id,
        rel_path="Pictures/holiday/a.jpg", declared_hash=sha(data), data=data,
    )
    drain(ctx.conn, {"thumbnail": thumbnail_handler(ctx.originals, ctx.derived, 320, 1600)})
    with TestClient(app) as test_client:
        yield test_client


def _first_id(client):
    return client.app.state.context.conn.execute("SELECT id FROM photos LIMIT 1").fetchone()["id"]


def test_photo_page_shows_exif_and_source_path(client):
    body = client.get(f"/photo/{_first_id(client)}").text
    assert "X-T5" in body
    assert "Pictures/holiday/a.jpg" in body


def test_photo_page_shows_every_duplicate_path_and_wasted_space(client):
    ctx = client.app.state.context
    base = _first_id(client)
    upload_id = add_upload(ctx.conn)
    ctx.conn.execute("UPDATE photos SET bytes = 1000 WHERE id = ?", (base,))
    ctx.conn.execute(
        "INSERT INTO photo_sources(photo_id, upload_id, rel_path, filename)"
        " VALUES (?, ?, 'Backup/a.jpg', 'a.jpg')",
        (base, upload_id),
    )
    body = client.get(f"/photo/{base}").text
    assert "Pictures/holiday/a.jpg" in body
    assert "Backup/a.jpg" in body


def test_photo_page_serves_the_detail_image(client):
    photo_id = _first_id(client)
    assert client.get(f"/photo/{photo_id}").status_code == 200
    detail = client.get(f"/thumb/{photo_id}?size=detail")
    assert detail.status_code == 200
    assert detail.headers["content-type"] == "image/jpeg"


def test_unknown_photo_is_404(client):
    assert client.get("/photo/99999").status_code == 404


def _dated(conn, pid, shot_at):
    add_photo(conn, photo_id=pid, content_hash=f"n{pid}", thumb_key=f"{pid}.jpg", shot_at=shot_at)


def test_photo_page_links_to_newer_and_older_neighbors(client):
    conn = client.app.state.context.conn
    # Dates in 2030 keep these three contiguous, clear of the fixture photo.
    _dated(conn, 101, "2030-01-01T00:00:00")  # older
    _dated(conn, 102, "2030-06-01T00:00:00")  # this one
    _dated(conn, 103, "2030-12-01T00:00:00")  # newer
    body = client.get("/photo/102").text
    assert 'class="photo-nav prev" href="/photo/103"' in body  # ‹ newer
    assert 'class="photo-nav next" href="/photo/101"' in body  # › older


def test_photo_page_omits_the_newer_arrow_at_the_newest(client):
    conn = client.app.state.context.conn
    _dated(conn, 301, "2099-01-01T00:00:00")  # newest in the whole library
    body = client.get("/photo/301").text
    assert "photo-nav prev" not in body   # nothing newer
    assert "photo-nav next" in body       # something older exists


def test_close_and_paging_preserve_the_library_view(client):
    body = client.get(f"/photo/{_first_id(client)}").text
    assert "history.back()" in body    # close returns to the exact filtered/scrolled library
    assert "location.replace" in body  # paging replaces history so close still lands on the list


def test_arrow_key_navigation_script_carries_the_neighbor_ids(client):
    conn = client.app.state.context.conn
    _dated(conn, 401, "2025-03-01T00:00:00")
    _dated(conn, 402, "2025-04-01T00:00:00")
    _dated(conn, 403, "2025-05-01T00:00:00")
    body = client.get("/photo/402").text
    assert "ArrowLeft" in body and "ArrowRight" in body
    assert "var prev = 403, next = 401" in body  # ← newer, → older


def test_photo_page_shows_ai_title_description_and_caption(client):
    conn = client.app.state.context.conn
    base = _first_id(client)
    conn.execute(
        "UPDATE photos SET caption = ?, caption_model = ?, ai_title = ?, ai_description = ?"
        " WHERE id = ?",
        ("a dog on a beach", "fake-vlm", "Beach day", "A dog runs on the sand.", base),
    )
    body = client.get(f"/photo/{base}").text
    assert "Beach day" in body                # ai_title
    assert "A dog runs on the sand." in body  # ai_description
    assert "a dog on a beach" in body         # caption
    assert "fake-vlm" in body                 # caption model


def test_photo_page_shows_model_tags_grouped_by_dimension(client):
    conn = client.app.state.context.conn
    base = _first_id(client)
    # tags are seeded at app startup; attach one to this photo
    tag_id = conn.execute(
        "SELECT id FROM tags WHERE dimension = 'setting' AND label = 'beach'"
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, 0.87, 'siglip')",
        (base, tag_id),
    )
    body = client.get(f"/photo/{base}").text
    assert "beach" in body
    assert "setting" in body
    assert "siglip" in body  # the source badge


def test_photo_page_defers_the_similar_strip(client):
    # The similar-photos scan is expensive; the page must render immediately and
    # hand the strip to an async HTMX fetch instead of blocking on it.
    photo_id = _first_id(client)
    body = client.get(f"/photo/{photo_id}").text
    assert "similar-item" not in body
    assert f'hx-get="/photo/{photo_id}/similar"' in body
    assert 'hx-trigger="load"' in body


def test_similar_strip_lists_other_photos_with_reasons_and_ctx(client):
    # This strip now loads via the async fragment route, not the /photo body.
    ctx = client.app.state.context
    fake = FakeEmbedder()
    base = _first_id(client)
    write_vector(ctx.conn, base, fake.embed_texts(["a"])[0])
    other = add_photo(ctx.conn, content_hash="bb" * 32, thumb_key="bb.jpg")
    write_vector(ctx.conn, other, fake.embed_texts(["a"])[0])  # identical → nearest
    body = client.get(f"/photo/{base}/similar").text
    assert f"/thumb/{other}" in body
    assert "why-line" in body                                    # reasons overlaid on the photo
    assert f'href="/photo/{other}?ctx=similar:{base}"' in body   # drills into the similar layer


def test_similar_photo_closes_up_to_its_origin_photo_then_to_the_grid(client):
    ctx = client.app.state.context
    fake = FakeEmbedder()
    base = _first_id(client)
    write_vector(ctx.conn, base, fake.embed_texts(["a"])[0])
    other = add_photo(ctx.conn, content_hash="cc" * 32, thumb_key="cc.jpg", caption="x")
    write_vector(ctx.conn, other, fake.embed_texts(["a"])[0])

    body = client.get(f"/photo/{other}?ctx=similar:{base}").text

    assert "Similar to" in body  # origin shown as context
    # Close goes UP one level — to the origin photo, in its own library layer, so
    # ITS close then goes up to the grid (§13.1 rule 4): similar -> origin -> grid.
    assert f'class="photo-close" href="/photo/{base}?ctx=library"' in body
    assert "location.replace(ivmsOrigin)" in body  # replace, never a push (rule 3)
    assert "var ivmsCloseReplaces = true" in body

    # …and the origin photo itself closes to the grid with a plain history.back().
    origin_body = client.get(f"/photo/{base}?ctx=library").text
    assert 'class="photo-close" href="/library"' in origin_body
    assert "var ivmsCloseReplaces = false" in origin_body


def test_similar_layer_shows_both_photos_own_words_before_the_table(client):
    # What is being compared is read BEFORE how it scored: the base photo's and this
    # photo's own title/description/caption, in the table's Base|This order (§13).
    ctx = client.app.state.context
    fake = FakeEmbedder()
    base = _first_id(client)
    write_vector(ctx.conn, base, fake.embed_texts(["a"])[0])
    other = add_photo(ctx.conn, content_hash="dd" * 32, thumb_key="dd.jpg")
    write_vector(ctx.conn, other, fake.embed_texts(["a"])[0])
    for pid, title, desc, cap in (
        (base, "Beach day", "A dog runs on the sand.", "a dog on a beach"),
        (other, "Park day", "A dog runs on the grass.", "a dog in a park"),
    ):
        ctx.conn.execute(
            "UPDATE photos SET ai_title=?, ai_description=?, caption=? WHERE id=?",
            (title, desc, cap, pid),
        )

    body = client.get(f"/photo/{other}?ctx=similar:{base}").text

    head = body[: body.index("Why similar")]
    for text in ("Beach day", "A dog runs on the sand.", "a dog on a beach",
                 "Park day", "A dog runs on the grass.", "a dog in a park"):
        assert text in head
    assert head.index("Beach day") < head.index("Park day")  # Base, then This
    # …and the leaf's own block below the table does not repeat them.
    assert body.count("Park day") == 1


# --- "Show more" — the loose tail (§9) --------------------------------------


def _moment_photo(ctx, content_hash, *, shot_at, caption):
    """A photo that shares only a MOMENT with the base — no vector, no tags. Its
    only route into the strip is the loose pass."""
    return add_photo(
        ctx.conn, content_hash=content_hash, thumb_key=f"{content_hash[:4]}.jpg",
        caption=caption, shot_at=shot_at, gps_lat=42.0, gps_lon=42.0,
    )


def test_show_more_is_offered_only_when_the_strict_strip_is_short(client):
    ctx = client.app.state.context
    base = _first_id(client)
    write_vector(ctx.conn, base, FakeEmbedder().embed_texts(["a"])[0])
    body = client.get(f"/photo/{base}/similar").text
    assert "Show more" in body


def test_show_more_reveals_looser_matches_the_strict_pass_refused(client):
    ctx = client.app.state.context
    base = _first_id(client)
    ctx.conn.execute(
        "UPDATE photos SET shot_at = ?, gps_lat = 42.0, gps_lon = 42.0 WHERE id = ?",
        ("2023-12-01T14:00:00", base),
    )
    write_vector(ctx.conn, base, FakeEmbedder().embed_texts(["a"])[0])
    # Same minute, same spot, but nothing else in common — a moment-only pair.
    nearby = _moment_photo(ctx, "dd" * 32, shot_at="2023-12-01T14:02:00", caption="something else")

    strict = client.get(f"/photo/{base}/similar").text
    assert f"/thumb/{nearby}" not in strict          # too weak for the default strip

    loose = client.get(f"/photo/{base}/similar?loose=1").text
    assert f"/thumb/{nearby}" in loose               # …but worth offering
    assert "same time &amp; place" in loose
    assert "similar-loose" in loose                  # rendered as visibly weaker


def test_the_loose_pass_never_repeats_a_strict_result(client):
    ctx = client.app.state.context
    fake = FakeEmbedder()
    base = _first_id(client)
    write_vector(ctx.conn, base, fake.embed_texts(["a"])[0])
    twin = add_photo(ctx.conn, content_hash="ee" * 32, thumb_key="ee.jpg")
    write_vector(ctx.conn, twin, fake.embed_texts(["a"])[0])   # identical → strict hit

    assert f"/thumb/{twin}" in client.get(f"/photo/{base}/similar").text
    assert f"/thumb/{twin}" not in client.get(f"/photo/{base}/similar?loose=1").text


def test_a_loose_thumbnail_pages_within_the_expanded_set(client):
    """§13.1 rule 6: prev/next moves within the layer the user can SEE. A loose
    thumbnail carries `:more`, so its layer is strict + loose, not strict alone."""
    ctx = client.app.state.context
    base = _first_id(client)
    ctx.conn.execute(
        "UPDATE photos SET shot_at = ?, gps_lat = 42.0, gps_lon = 42.0 WHERE id = ?",
        ("2023-12-01T14:00:00", base),
    )
    write_vector(ctx.conn, base, FakeEmbedder().embed_texts(["a"])[0])
    nearby = _moment_photo(ctx, "ff" * 32, shot_at="2023-12-01T14:02:00", caption="something else")

    loose = client.get(f"/photo/{base}/similar?loose=1").text
    assert f'href="/photo/{nearby}?ctx=similar:{base}:more"' in loose

    # The expanded layer resolves — same origin identity, and close still goes UP
    # to the origin photo exactly like the plain similar layer. This is also the
    # regression guard for the separator: `+more` decoded to a SPACE in the query
    # string, so the origin lookup failed and the layer silently fell back to the
    # whole library — losing the user's place, which §13.1 forbids.
    body = client.get(f"/photo/{nearby}?ctx=similar:{base}:more").text
    assert "Similar to" in body
    assert f'class="photo-close" href="/photo/{base}?ctx=library"' in body
