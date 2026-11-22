"""Naming face groups (design §6, §13, §13.1): "Faces to name" on the People grid, the group
leaf `/faces/{id}` and its actions, the person's faces on `/people/{id}`, and the face
crops. A person is a `people` row, never a saved voice. Every query is scoped to the owner."""

import io
import itertools
import re

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ingest.faces import assign_to_group
from ingest.thumbs import thumb_key
from tests.factories import add_face, add_face_group, add_person, add_photo
from web.app import create_app

OWNER = 1
OTHER_OWNER = 2
DETAIL_SIZE = (400, 300)
_hashes = itertools.count(1)


def _detail_jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", DETAIL_SIZE, "gray").save(buffer, format="JPEG")
    return buffer.getvalue()


def _app_client(settings) -> TestClient:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return TestClient(create_app(settings))


@pytest.fixture
def client(settings):
    with _app_client(settings) as test_client:
        yield test_client


@pytest.fixture
def jetson(settings):
    settings.profile = "jetson"
    with _app_client(settings) as test_client:
        yield test_client


def _conn(client):
    return client.app.state.context.conn


def _group(client, faces: int, *, owner_id: int = OWNER, person_id: int | None = None):
    """A face group of `faces` faces, each in its own photo with a detail thumbnail."""
    ctx = client.app.state.context
    group = add_face_group(ctx.conn, owner_id=owner_id, person_id=person_id)
    face_ids = []
    for _ in range(faces):
        content_hash = f"{next(_hashes):064x}"
        photo = add_photo(ctx.conn, owner_id=owner_id, content_hash=content_hash,
                          thumb_key=thumb_key(content_hash, 320))
        ctx.derived.write(thumb_key(content_hash, ctx.settings.thumb_detail_px), _detail_jpeg())
        face_ids.append(add_face(ctx.conn, photo_id=photo, group_id=group, vector=(1.0, 0.0)))
    return group, face_ids


def _person_of(client, group: int) -> int | None:
    return _conn(client).execute("SELECT person_id FROM face_groups WHERE id = ?",
                                 (group,)).fetchone()[0]


def _people(client) -> list[tuple[int, str]]:
    return [tuple(r) for r in _conn(client).execute(
        "SELECT owner_id, name FROM people ORDER BY id")]


def _voices(client) -> list[tuple[int, str]]:
    return [tuple(r) for r in _conn(client).execute(
        "SELECT owner_id, name FROM saved_voices ORDER BY id")]


def _group_link(body: str, group: int) -> str | None:
    found = re.search(rf'<a\b[^>]*href="/faces/{group}"[^>]*>', body)
    return found.group(0) if found else None


def test_faces_to_name_lists_an_unnamed_group_with_its_crops_count_and_link(client):
    group, faces = _group(client, 2)

    body = client.get("/people").text

    assert "Faces to name" in body
    assert _group_link(body, group) is not None
    assert all(f'src="/faces/crop/{face}"' in body for face in faces)
    assert "2 faces" in body


def test_a_group_in_faces_to_name_shows_at_most_four_crops(client):
    group, faces = _group(client, 6)

    body = client.get("/people").text

    assert _group_link(body, group) is not None
    assert sum(f'src="/faces/crop/{face}"' in body for face in faces) == 4
    assert "6 faces" in body


def test_the_minimum_face_count_of_faces_to_name_comes_from_the_settings(settings):
    settings.face_group_min_faces = 1
    with _app_client(settings) as client:
        group, _ = _group(client, 1)
        body = client.get("/people").text
    assert _group_link(body, group) is not None


def test_faces_to_name_shows_on_jetson_too(jetson):
    group, _ = _group(jetson, 2)
    assert _group_link(jetson.get("/people").text, group) is not None


def test_faces_to_name_never_shows_a_group_of_another_owner(client):
    mine, _ = _group(client, 2)
    theirs, _ = _group(client, 2, owner_id=OTHER_OWNER)

    body = client.get("/people").text

    assert _group_link(body, mine) is not None
    assert _group_link(body, theirs) is None


def test_the_group_leaf_shows_its_faces_and_the_owners_people(client):
    add_person(_conn(client), name="Bob")
    group, faces = _group(client, 2)

    response = client.get(f"/faces/{group}")

    assert response.status_code == 200
    assert all(f'src="/faces/crop/{face}"' in response.text for face in faces)
    assert "Bob" in response.text
    assert "Faces to name" in response.text


def test_attaching_a_group_to_a_person_takes_it_out_of_faces_to_name(client):
    bob = add_person(_conn(client), name="Bob")
    group, _ = _group(client, 2)

    response = client.post(f"/faces/{group}/assign", data={"person_id": str(bob)})

    assert response.status_code == 200
    assert _person_of(client, group) == bob
    assert _group_link(client.get("/people").text, group) is None


def test_a_new_person_created_for_a_group_owns_it(client):
    group, _ = _group(client, 2)

    response = client.post(f"/faces/{group}/person", data={"name": "Ann"})

    assert response.status_code == 200
    rows = _conn(client).execute("SELECT id FROM people WHERE owner_id = ? AND name = 'Ann'",
                                 (OWNER,)).fetchall()
    assert len(rows) == 1
    assert _person_of(client, group) == rows[0]["id"]
    assert _voices(client) == []


def test_a_new_person_can_be_created_on_jetson(jetson):
    group, _ = _group(jetson, 2)

    response = jetson.post(f"/faces/{group}/person", data={"name": "Ann"})

    assert response.status_code == 200
    assert _people(jetson) == [(OWNER, "Ann")]
    assert _person_of(jetson, group) is not None
    assert _voices(jetson) == []


def test_a_taken_name_is_refused_with_its_reason_and_nothing_is_written(client):
    add_person(_conn(client), name="Ann")
    group, _ = _group(client, 2)

    response = client.post(f"/faces/{group}/person", data={"name": "ann"})

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]
    assert _people(client) == [(OWNER, "Ann")]
    assert _voices(client) == []
    assert _person_of(client, group) is None


def test_an_empty_name_is_refused_with_its_reason_and_nothing_is_written(client):
    group, _ = _group(client, 2)

    response = client.post(f"/faces/{group}/person", data={"name": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Give the person a name."
    assert _people(client) == [] and _voices(client) == []
    assert _person_of(client, group) is None


def test_not_this_person_moves_the_face_into_a_pinned_group_of_its_own(client):
    group, faces = _group(client, 3)

    response = client.post(f"/faces/face/{faces[0]}/release")

    assert response.status_code == 200
    row = _conn(client).execute("SELECT group_id, pinned FROM faces WHERE id = ?",
                                (faces[0],)).fetchone()
    assert row["pinned"] == 1 and row["group_id"] not in (None, group)
    members = [r[0] for r in _conn(client).execute(
        "SELECT id FROM faces WHERE group_id = ? ORDER BY id", (row["group_id"],))]
    assert members == [faces[0]]
    owner = _conn(client).execute("SELECT owner_id FROM face_groups WHERE id = ?",
                                  (row["group_id"],)).fetchone()[0]
    assert owner == OWNER
    left = [r[0] for r in _conn(client).execute(
        "SELECT id FROM faces WHERE group_id = ? ORDER BY id", (group,))]
    assert left == faces[1:]


def test_a_released_face_never_rejoins_its_old_group_by_itself(client):
    group, faces = _group(client, 3)
    client.post(f"/faces/face/{faces[0]}/release")

    assign_to_group(_conn(client), faces[0], 0.5)

    now = _conn(client).execute("SELECT group_id FROM faces WHERE id = ?",
                                (faces[0],)).fetchone()[0]
    assert now not in (None, group)


def test_the_faces_tab_of_a_person_lists_their_groups_in_place(client):
    ann = add_person(_conn(client), name="Ann")
    group, faces = _group(client, 2, person_id=ann)

    body = client.get(f"/people/{ann}").text

    assert all(f'src="/faces/crop/{face}"' in body for face in faces)
    assert "Remove from this person" in body
    assert _group_link(body, group) is None


def test_the_faces_tab_of_a_person_with_no_group_says_so(client):
    ann = add_person(_conn(client), name="Ann")
    assert "No faces yet." in client.get(f"/people/{ann}").text


def test_removing_a_group_from_a_person_returns_it_to_faces_to_name(client):
    ann = add_person(_conn(client), name="Ann")
    group, _ = _group(client, 2, person_id=ann)

    response = client.post(f"/faces/{group}/unassign")

    assert response.status_code == 200
    assert _person_of(client, group) is None
    assert _group_link(client.get("/people").text, group) is not None


def test_the_grid_opens_a_group_with_a_plain_link(client):
    group, _ = _group(client, 2)
    link = _group_link(client.get("/people").text, group)
    assert link is not None and "hx-" not in link


def test_the_group_leaf_closes_back_to_the_grid_and_never_pushes_history(client):
    group, _ = _group(client, 2)

    page = client.get(f"/faces/{group}").text
    script = client.get("/static/faces.js")

    assert "/static/faces.js" in page
    assert script.status_code == 200
    assert "history.back()" in script.text
    assert re.search(r"""location\.replace\(\s*["'`]/people["'`]\s*\)""", script.text)
    assert "data-from-grid" in page + script.text
    assert "pushState" not in page + script.text
    assert "location.assign" not in page + script.text


def test_the_crop_of_a_face_is_a_jpeg_cut_from_the_detail_thumbnail(client):
    _, faces = _group(client, 2)

    response = client.get(f"/faces/crop/{faces[0]}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    width, height = Image.open(io.BytesIO(response.content)).size
    assert 0 < width < DETAIL_SIZE[0] and 0 < height < DETAIL_SIZE[1]


def test_the_crop_of_another_owners_face_is_a_404(client):
    _, mine = _group(client, 2)
    _, theirs = _group(client, 2, owner_id=OTHER_OWNER)

    assert client.get(f"/faces/crop/{mine[0]}").status_code == 200
    assert client.get(f"/faces/crop/{theirs[0]}").status_code == 404
