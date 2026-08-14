import hashlib
import io
import threading

from PIL import Image

from config import Settings
from ingest.receive import receive
from web.deps import build_context


def _jpeg(i: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (i % 255, (i * 7) % 255, (i * 13) % 255)).save(buf, "JPEG")
    return buf.getvalue()


def test_concurrent_uploads_through_context_conn_do_not_corrupt(tmp_path):
    # FastAPI serves uploads from a threadpool, so several threads run receive()
    # at once. receive() hashes, opens the image, and writes a file — all release
    # the GIL, so the threads genuinely interleave on the DB. A single shared
    # sqlite3 connection corrupts under that ("bad parameter or other API
    # misuse", phantom FK/UNIQUE errors); each thread needs its own connection.
    settings = Settings(data_dir=tmp_path, use_fake_embedder=True, use_fake_inference=True)
    ctx = build_context(settings)
    ctx.conn.execute("INSERT INTO uploads(id, owner_id, root_label, started_at) VALUES (1,1,'x','now')")

    errors: list[str] = []

    def worker(i: int) -> None:
        try:
            data = _jpeg(i)
            receive(
                ctx.conn, ctx.originals, owner_id=1, upload_id=1,
                rel_path=f"f/img{i}.jpg", declared_hash=hashlib.sha256(data).hexdigest(),
                data=data,
            )
        except Exception as error:  # noqa: BLE001
            errors.append(repr(error))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors[:5]
    total = ctx.conn.execute("SELECT count(*) AS c FROM photos").fetchone()["c"]
    assert total == 40
