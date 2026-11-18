"""The `/faces/*` routes refuse what the owner does not hold (design §6, §13): another
owner's group or face, a missing person, a person being deleted, a crop with no readable
detail thumbnail. Each refusal leaves the rows as they were."""

import io
import itertools

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from feed.saved_voices import create_voice, request_voice_deletion
from ingest.thumbs import thumb_key
from tests.factories import add_face, add_face_group, add_photo
from web.app import create_app

OWNER = 1
OTHER_OWNER = 2
MISSING_ID = 999_999
_hashes = itertools.count(10_000)


def _detail_jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (400, 300), "gray").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _conn(client):
    return client.app.state.context.conn


def _group(
    client,
    faces: int,
    *,
    owner_id: int = OWNER,
    person_id: int | None = None,
    detail: bytes | None = b"",
) -> tuple[int, list[int]]:
    """A group of `faces` faces, one per photo. `detail` is the detail thumbnail written for
    each photo: empty bytes = a real JPEG, None = no file at all."""
    ctx = client.app.state.context
    group = add_face_group(ctx.conn, owner_id=owner_id, person_id=person_id)
    face_ids = []
    for _ in range(faces):
        content_hash = f"{next(_hashes):064x}"
        photo = add_photo(
            ctx.conn,
            owner_id=owner_id,
            content_hash=content_hash,
            thumb_key=thumb_key(content_hash, 320),
        )
        if detail is not None:
            ctx.derived.write(
                thumb_key(content_hash, ctx.settings.thumb_detail_px), detail or _detail_jpeg()
            )
        face_ids.append(add_face(ctx.conn, photo_id=photo, group_id=group, vector=(1.0, 0.0)))
    return group, face_ids


def _person_of(client, group: int) -> int | None:
    return (
        _conn(client)
        .execute("SELECT person_id FROM face_groups WHERE id = ?", (group,))
        .fetchone()[0]
    )


def _face_row(client, face: int) -> tuple:
    return tuple(
        _conn(client).execute("SELECT group_id, pinned FROM faces WHERE id = ?", (face,)).fetchone()
    )


def _voice_names(client) -> list[str]:
    return [r[0] for r in _conn(client).execute("SELECT name FROM saved_voices ORDER BY id")]


def test_the_leaf_of_another_owners_group_is_a_404(client):
    theirs, _ = _group(client, 2, owner_id=OTHER_OWNER)

    assert client.get(f"/faces/{theirs}").status_code == 404
    assert client.get(f"/faces/{MISSING_ID}").status_code == 404


def test_another_owners_group_is_never_attached_to_my_person(client):
    person = create_voice(_conn(client), OWNER, "Ann")
    theirs, _ = _group(client, 2, owner_id=OTHER_OWNER)

    response = client.post(f"/faces/{theirs}/assign", data={"person_id": person.id})

    assert response.status_code == 404
    assert _person_of(client, theirs) is None


def test_my_group_is_never_attached_to_another_owners_person(client):
    theirs = create_voice(_conn(client), OTHER_OWNER, "Bob")
    mine, _ = _group(client, 2)

    response = client.post(f"/faces/{mine}/assign", data={"person_id": theirs.id})

    assert response.status_code == 404
    assert _person_of(client, mine) is None


def test_attaching_to_a_person_being_deleted_is_a_409(client):
    person = create_voice(_conn(client), OWNER, "Ann")
    assert request_voice_deletion(_conn(client), OWNER, person.id)
    group, _ = _group(client, 2)

    response = client.post(f"/faces/{group}/assign", data={"person_id": person.id})

    assert response.status_code == 409
    assert _person_of(client, group) is None


def test_a_new_person_for_another_owners_group_is_a_404_and_creates_no_person(client):
    theirs, _ = _group(client, 2, owner_id=OTHER_OWNER)

    response = client.post(f"/faces/{theirs}/person", data={"name": "Ann"})

    assert response.status_code == 404
    assert _voice_names(client) == []
    assert _person_of(client, theirs) is None


def test_releasing_another_owners_face_is_a_404_and_moves_nothing(client):
    theirs, faces = _group(client, 3, owner_id=OTHER_OWNER)

    response = client.post(f"/faces/face/{faces[0]}/release")

    assert response.status_code == 404
    assert _face_row(client, faces[0]) == (theirs, 0)


def test_removing_another_owners_group_from_its_person_is_a_404(client):
    their_person = create_voice(_conn(client), OTHER_OWNER, "Bob")
    theirs, _ = _group(client, 2, owner_id=OTHER_OWNER, person_id=their_person.id)

    response = client.post(f"/faces/{theirs}/unassign")

    assert response.status_code == 404
    assert _person_of(client, theirs) == their_person.id


def test_releasing_the_only_face_of_a_group_answers_its_new_group_and_drops_the_old(client):
    group, faces = _group(client, 1)

    response = client.post(f"/faces/face/{faces[0]}/release")

    assert response.status_code == 200
    new_group, pinned = _face_row(client, faces[0])
    assert pinned == 1 and new_group != group
    assert (
        _conn(client)
        .execute("SELECT COUNT(*) FROM face_groups WHERE id = ?", (group,))
        .fetchone()[0]
        == 0
    )
    assert f'src="/faces/crop/{faces[0]}"' in response.text


@pytest.mark.parametrize(
    ("detail", "case"),
    [
        pytest.param(None, "missing", id="no-detail-thumbnail"),
        pytest.param(b"not a jpeg", "unreadable", id="unreadable-detail-thumbnail"),
    ],
)
def test_a_crop_with_no_readable_detail_thumbnail_is_a_404(client, detail, case):
    _, faces = _group(client, 1, detail=detail)

    response = client.get(f"/faces/crop/{faces[0]}")

    assert response.status_code == 404, case
