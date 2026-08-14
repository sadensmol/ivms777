from ingest.jobs import claim_next, enqueue, stage_counts
from ingest.thumbs import thumb_key
from ingest.worker import drain, thumbnail_handler
from storage.keys import content_key
from storage.local import LocalStorage
from tests.factories import add_photo
from tests.fixtures import make_jpeg


def test_drain_runs_handler_and_marks_done(conn):
    add_photo(conn, photo_id=1, content_hash="h")
    enqueue(conn, 1, "thumbnail")
    seen: list[int] = []

    completed = drain(conn, {"thumbnail": lambda c, pid: seen.append(pid)})

    assert seen == [1]
    assert completed["thumbnail"] == 1
    assert stage_counts(conn, "thumbnail")["done"] == 1


def test_drain_records_failure_without_stopping_the_queue(conn):
    for index in (1, 2):
        add_photo(conn, photo_id=index, content_hash=f"h{index}")
        enqueue(conn, index, "thumbnail")

    def handler(c, photo_id):
        if photo_id == 1:
            raise OSError("cannot read")

    drain(conn, {"thumbnail": handler})

    counts = stage_counts(conn, "thumbnail")
    assert counts["done"] == 1
    assert counts["pending"] == 1  # photo 1 retries; it has attempts left
    assert claim_next(conn, "thumbnail") == 1


def test_drain_finishes_each_stage_before_starting_the_next(conn):
    for index in (1, 2):
        add_photo(conn, photo_id=index, content_hash=f"h{index}")
        enqueue(conn, index, "thumbnail")
        enqueue(conn, index, "embed")

    order: list[str] = []
    drain(
        conn,
        {
            "thumbnail": lambda c, pid: order.append(f"thumbnail:{pid}"),
            "embed": lambda c, pid: order.append(f"embed:{pid}"),
        },
    )

    assert order == ["thumbnail:1", "thumbnail:2", "embed:1", "embed:2"]


def test_thumbnail_handler_writes_files_and_sets_thumb_key(conn, tmp_path):
    originals = LocalStorage(tmp_path / "originals")
    derived = LocalStorage(tmp_path / "thumbs")
    digest = "ee" * 32
    key = content_key(digest, ".jpg")
    make_jpeg(originals.local_path(key), size=(800, 600))
    add_photo(conn, photo_id=1, content_hash=digest, suffix=".jpg")
    enqueue(conn, 1, "thumbnail")

    drain(conn, {"thumbnail": thumbnail_handler(originals, derived, 320, 1600)})

    row = conn.execute("SELECT content_hash, thumb_key FROM photos").fetchone()
    assert row["thumb_key"] == thumb_key(digest, 320)
    assert derived.exists(row["thumb_key"])
