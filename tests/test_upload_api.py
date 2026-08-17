import pytest
from fastapi.testclient import TestClient

from tests.fixtures import jpeg_bytes, sha
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def conn(client):
    return client.app.state.context.conn


def start(client, root_label="Pictures") -> int:
    response = client.post("/api/upload/start", json={"root_label": root_label})
    assert response.status_code == 200
    return response.json()["upload_id"]


def send(client, upload_id, rel_path, data):
    return client.post(
        "/api/upload/file",
        data={"upload_id": upload_id, "rel_path": rel_path, "content_hash": sha(data)},
        files={"file": (rel_path.rsplit("/", 1)[-1], data, "image/jpeg")},
    )


def test_probe_asks_for_everything_when_the_library_is_empty(client):
    upload_id = start(client)
    data = jpeg_bytes()
    response = client.post(
        "/api/upload/probe",
        json={"upload_id": upload_id, "files": [{"hash": sha(data), "rel_path": "a.jpg"}]},
    )
    assert response.json()["needed"] == [sha(data)]


def test_probe_skips_known_bytes_but_still_records_the_new_path(client, conn):
    upload_id = start(client)
    data = jpeg_bytes()
    assert send(client, upload_id, "Pictures/a.jpg", data).status_code == 200

    response = client.post(
        "/api/upload/probe",
        json={
            "upload_id": upload_id,
            "files": [{"hash": sha(data), "rel_path": "Backup/a.jpg"}],
        },
    )
    assert response.json()["needed"] == []
    paths = {
        row["rel_path"] for row in conn.execute("SELECT rel_path FROM photo_sources")
    }
    assert paths == {"Pictures/a.jpg", "Backup/a.jpg"}


def test_uploading_a_file_stores_it_and_reports_the_photo(client, conn):
    upload_id = start(client)
    data = jpeg_bytes()
    response = send(client, upload_id, "Pictures/a.jpg", data)
    assert response.status_code == 200
    assert response.json()["status"] == "stored"
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 1


def test_sending_bytes_already_held_links_instead_of_storing(client):
    upload_id = start(client)
    data = jpeg_bytes()
    send(client, upload_id, "Pictures/a.jpg", data)
    response = send(client, upload_id, "Desktop/a.jpg", data)
    assert response.json()["status"] == "linked"


def test_a_corrupted_transfer_is_rejected_with_422(client, conn):
    upload_id = start(client)
    data = jpeg_bytes()
    response = client.post(
        "/api/upload/file",
        data={"upload_id": upload_id, "rel_path": "a.jpg", "content_hash": "00" * 32},
        files={"file": ("a.jpg", data, "image/jpeg")},
    )
    assert response.status_code == 422
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 0
    assert conn.execute(
        "SELECT files_failed FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()[0] == 1


def test_a_non_image_is_rejected_with_415(client):
    upload_id = start(client)
    data = b"not an image at all"
    response = client.post(
        "/api/upload/file",
        data={"upload_id": upload_id, "rel_path": "notes.jpg", "content_hash": sha(data)},
        files={"file": ("notes.jpg", data, "image/jpeg")},
    )
    assert response.status_code == 415


def test_counters_add_up_across_an_upload(client, conn):
    upload_id = start(client)
    first, second = jpeg_bytes(color="red"), jpeg_bytes(color="green")
    client.post(
        "/api/upload/probe",
        json={
            "upload_id": upload_id,
            "files": [
                {"hash": sha(first), "rel_path": "a.jpg"},
                {"hash": sha(second), "rel_path": "b.jpg"},
            ],
        },
    )
    send(client, upload_id, "a.jpg", first)
    send(client, upload_id, "b.jpg", second)
    summary = client.post("/api/upload/finish", json={"upload_id": upload_id}).json()
    assert summary == {"offered": 2, "sent": 2, "failed": 0}
    assert conn.execute(
        "SELECT finished_at FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()[0] is not None


def test_finish_returns_the_receipt_even_when_the_inline_drain_fails(client, conn, monkeypatch):
    # The upload RECEIPT is complete once the bytes are stored (§5, §8); the inline
    # drain is a best-effort convenience and the worker container does the real
    # processing. So a drain failure — e.g. the app container cannot init CUDA on
    # jetson (RuntimeError 801) — must NOT fail the upload. Regression for the
    # reported "upload failed: /api/upload/finish returned 500".
    from config import Settings
    upload_id = start(client)
    send(client, upload_id, "a.jpg", jpeg_bytes())

    def boom(self):
        raise RuntimeError("Unexpected error from cudaGetDeviceCount ... Error 801")

    monkeypatch.setattr(Settings, "build_embedder", boom)
    summary = client.post("/api/upload/finish", json={"upload_id": upload_id}).json()
    assert summary == {"offered": 0, "sent": 1, "failed": 0}  # the receipt still returns
    assert conn.execute(
        "SELECT finished_at FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()[0] is not None                                # the upload is marked finished


def test_inline_drain_off_leaves_the_work_to_the_worker(settings):
    # Wherever a `worker` container runs — every compose profile — the app must
    # NOT drain inline (§5: app serves reads, worker owns writes). The inline pass
    # covers the WHOLE queue, so it held the finish request open for the entire
    # library and put a second drainer on the same jobs; on the Jetson that pinned
    # `app` at 100 % CPU racing the worker for a GPU with `gpu_concurrency: 1`.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.inline_drain = False
    with TestClient(create_app(settings)) as client:
        conn = client.app.state.context.conn
        upload_id = start(client)
        send(client, upload_id, "Pictures/a.jpg", jpeg_bytes())
        summary = client.post(
            "/api/upload/finish", json={"upload_id": upload_id}
        ).json()

    assert summary == {"offered": 0, "sent": 1, "failed": 0}  # receipt still returns
    # The bytes landed and every stage is queued — but nothing ran any of them.
    assert conn.execute("SELECT thumb_key FROM photos").fetchone()["thumb_key"] is None
    assert [tuple(row) for row in conn.execute(
        "SELECT stage, status FROM jobs ORDER BY stage"
    )] == [
        ("caption", "pending"),
        ("embed", "pending"),
        ("taxonomy", "pending"),
        ("thumbnail", "pending"),
    ]


def test_an_unknown_upload_id_is_rejected(client):
    data = jpeg_bytes()
    response = send(client, 999, "a.jpg", data)
    assert response.status_code == 404


def test_uploaded_photos_appear_in_the_library_with_a_thumbnail(client, conn):
    upload_id = start(client)
    send(client, upload_id, "Pictures/a.jpg", jpeg_bytes())
    client.post("/api/upload/finish", json={"upload_id": upload_id})
    assert conn.execute(
        "SELECT thumb_key FROM photos LIMIT 1"
    ).fetchone()["thumb_key"] is not None
    assert client.get("/library").status_code == 200
