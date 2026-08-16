from albums.memory_store import Memory, read_memories, replace_memories
from embedding.fakes import FakeEmbedder
from embedding.store import read_caption_vector, write_caption_vector, write_vector
from ingest.folders import (
    enqueue_folder_deletion,
    list_folders,
    process_folder_deletions,
)
from ingest.thumbs import thumb_key
from storage.keys import content_key
from storage.local import LocalStorage
from tests.factories import add_photo, add_upload


def _storages(tmp_path):
    return LocalStorage(tmp_path / "orig"), LocalStorage(tmp_path / "thumbs")


def test_delete_folder_removes_its_photos_and_all_metadata(conn, tmp_path):
    originals, derived = _storages(tmp_path)
    up = add_upload(conn, root_label="Trip")
    h = "aa" * 32
    pid = add_photo(conn, content_hash=h, sources=("Trip/a.jpg",), upload_id=up,
                    thumb_key=thumb_key(h, 320), caption="a dog")
    write_vector(conn, pid, FakeEmbedder().embed_texts(["x"])[0])
    write_caption_vector(conn, pid, [1.0, 0.0])
    conn.execute("INSERT INTO tags(dimension, label) VALUES ('subject', 'dog')")
    tid = conn.execute("SELECT id FROM tags WHERE label = 'dog'").fetchone()["id"]
    conn.execute("INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, 1.0, 'siglip')", (pid, tid))
    conn.execute("INSERT INTO photo_facets(photo_id, key, value_text) VALUES (?, 'year', '2024')", (pid,))
    conn.execute("INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (?, 'a dog', 'dog')", (pid,))
    conn.execute("INSERT INTO jobs(photo_id, stage, status, updated_at) VALUES (?, 'caption', 'done', 'n')", (pid,))
    originals.write(content_key(h, ".jpg"), b"orig")
    derived.write(thumb_key(h, 320), b"t")
    derived.write(thumb_key(h, 1600), b"t")

    enqueue_folder_deletion(conn, 1, "Trip")
    assert process_folder_deletions(conn, originals, derived, 1, 320, 1600) == 1

    gone = lambda sql, *p: conn.execute(sql, p).fetchone() is None  # noqa: E731
    assert gone("SELECT 1 FROM photos WHERE id = ?", pid)
    assert gone("SELECT 1 FROM photo_tags WHERE photo_id = ?", pid)
    assert gone("SELECT 1 FROM photo_facets WHERE photo_id = ?", pid)
    assert gone("SELECT 1 FROM jobs WHERE photo_id = ?", pid)
    assert gone("SELECT 1 FROM photo_vec WHERE rowid = ?", pid)
    assert gone("SELECT 1 FROM photo_fts WHERE rowid = ?", pid)
    assert gone("SELECT 1 FROM photo_sources WHERE photo_id = ?", pid)
    assert gone("SELECT 1 FROM uploads WHERE id = ?", up)
    assert gone("SELECT 1 FROM folder_deletions")            # outbox drained
    assert not originals.exists(content_key(h, ".jpg"))       # storage cleaned
    assert not derived.exists(thumb_key(h, 320))
    assert not derived.exists(thumb_key(h, 1600))
    assert read_caption_vector(conn, pid) is None


def test_delete_folder_keeps_a_photo_shared_with_another_folder(conn, tmp_path):
    originals, derived = _storages(tmp_path)
    up_a, up_b = add_upload(conn, root_label="A"), add_upload(conn, root_label="B")
    h = "bb" * 32
    pid = add_photo(conn, content_hash=h, thumb_key=thumb_key(h, 320))
    for up, path in ((up_a, "A/x.jpg"), (up_b, "B/x.jpg")):
        conn.execute(
            "INSERT INTO photo_sources(photo_id, upload_id, rel_path, filename)"
            " VALUES (?, ?, ?, 'x.jpg')", (pid, up, path))

    enqueue_folder_deletion(conn, 1, "A")
    process_folder_deletions(conn, originals, derived, 1, 320, 1600)

    assert conn.execute("SELECT 1 FROM photos WHERE id = ?", (pid,)).fetchone() is not None
    remaining = [r["rel_path"] for r in conn.execute(
        "SELECT rel_path FROM photo_sources WHERE photo_id = ?", (pid,))]
    assert remaining == ["B/x.jpg"]                          # only B's path kept
    assert conn.execute("SELECT 1 FROM uploads WHERE id = ?", (up_a,)).fetchone() is None
    assert conn.execute("SELECT 1 FROM uploads WHERE id = ?", (up_b,)).fetchone() is not None


def test_delete_folder_prunes_a_memory_left_empty_and_keeps_a_surviving_one(conn, tmp_path):
    originals, derived = _storages(tmp_path)
    up_trip = add_upload(conn, root_label="Trip")
    up_keep = add_upload(conn, root_label="Keep")
    trip = [add_photo(conn, content_hash=f"a{i}" * 32, sources=(f"Trip/{i}.jpg",),
                      upload_id=up_trip, thumb_key=thumb_key(f"a{i}" * 32, 320)) for i in range(2)]
    keep = add_photo(conn, content_hash="bb" * 32, sources=("Keep/k.jpg",),
                     upload_id=up_keep, thumb_key=thumb_key("bb" * 32, 320))
    replace_memories(conn, 1, [
        Memory("Trip only", "gone", trip, "sig"),          # every photo from Trip -> empties
        Memory("Mixed", "stays", [keep, *trip], "sig"),    # keeps one surviving photo
    ])

    enqueue_folder_deletion(conn, 1, "Trip")
    process_folder_deletions(conn, originals, derived, 1, 320, 1600)

    titles = {m.title: m for m in read_memories(conn, 1)}
    assert "Trip only" not in titles                       # empty memory pruned
    assert titles["Mixed"].photo_ids == [keep]             # survivor kept, dead ids dropped
    fts = conn.execute("SELECT name FROM memory_fts").fetchall()
    assert {r["name"] for r in fts} == {"Mixed"}           # fts rebuilt in lockstep


def test_list_folders_reports_counts_and_deleting_flag(conn):
    up_a, up_b = add_upload(conn, root_label="A"), add_upload(conn, root_label="B")
    add_photo(conn, content_hash="c" * 64, sources=("A/1.jpg",), upload_id=up_a)
    add_photo(conn, content_hash="d" * 64, sources=("B/1.jpg",), upload_id=up_b)
    enqueue_folder_deletion(conn, 1, "A")
    by_name = {f["name"]: f for f in list_folders(conn, 1)}
    assert by_name["A"]["photos"] == 1 and by_name["A"]["deleting"] is True
    assert by_name["B"]["photos"] == 1 and by_name["B"]["deleting"] is False
