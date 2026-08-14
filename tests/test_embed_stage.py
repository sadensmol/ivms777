from embedding.fakes import FakeEmbedder
from embedding.store import knn, read_vector, write_vector
from embedding.vectors import l2_normalize
from ingest.embed import embed_handler
from ingest.jobs import enqueue, stage_counts
from ingest.worker import drain
from storage.keys import content_key
from storage.local import LocalStorage
from tests.factories import add_photo
from tests.fixtures import make_jpeg


def test_write_and_read_round_trip(conn):
    add_photo(conn, photo_id=1, content_hash="h")
    vector = l2_normalize([0.5] * 1152)
    write_vector(conn, 1, vector)
    restored = read_vector(conn, 1)
    assert restored is not None
    assert all(abs(a - b) < 1e-6 for a, b in zip(vector, restored))


def test_knn_orders_by_distance_and_can_exclude_self(conn):
    for pid, seed in ((1, "a"), (2, "b"), (3, "c")):
        add_photo(conn, photo_id=pid, content_hash=seed)
        write_vector(conn, pid, FakeEmbedder().embed_texts([seed])[0])
    query = read_vector(conn, 1)
    hits = knn(conn, owner_id=1, vector=query, k=10)
    assert hits[0][0] == 1  # a photo is nearest to its own vector
    without_self = knn(conn, owner_id=1, vector=query, k=10, exclude_id=1)
    assert 1 not in [pid for pid, _ in without_self]


def test_knn_is_owner_scoped(conn):
    add_photo(conn, photo_id=1, owner_id=1, content_hash="a")
    add_photo(conn, photo_id=2, owner_id=2, content_hash="b")
    vec = FakeEmbedder().embed_texts(["a"])[0]
    write_vector(conn, 1, vec)
    write_vector(conn, 2, vec)
    hits = knn(conn, owner_id=1, vector=vec, k=10)
    assert [pid for pid, _ in hits] == [1]


def test_embed_handler_populates_the_vector_table(conn, tmp_path):
    originals = LocalStorage(tmp_path / "originals")
    key = content_key("ff" * 32, ".jpg")
    make_jpeg(originals.local_path(key))
    add_photo(conn, photo_id=1, content_hash="ff" * 32)
    enqueue(conn, 1, "embed")

    drain(conn, {"embed": embed_handler(originals, FakeEmbedder(), "fake-model")})

    assert read_vector(conn, 1) is not None
    assert stage_counts(conn, "embed")["done"] == 1
    assert conn.execute(
        "SELECT embedding_model FROM photos WHERE id = 1"
    ).fetchone()["embedding_model"] == "fake-model"
