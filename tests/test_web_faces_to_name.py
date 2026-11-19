"""Who shows in "Faces to name" (design §6, §13, AC6): at the default minimum of two faces a
1-face group is hidden, and a face the owner released ("Not this person") is still reached
from there and can be named, whatever the size of its new group."""

import io
import itertools
import re

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from feed.saved_voices import create_voice
from ingest.thumbs import thumb_key
from tests.factories import add_face, add_face_group, add_photo
from web.app import create_app

OWNER = 1
_hashes = itertools.count(1)


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _conn(client):
    return client.app.state.context.conn


def _group(client, faces: int, *, person_id: int | None = None) -> tuple[int, list[int]]:
    ctx = client.app.state.context
    group = add_face_group(ctx.conn, owner_id=OWNER, person_id=person_id)
    face_ids = []
    for _ in range(faces):
        content_hash = f"{next(_hashes):064x}"
        photo = add_photo(
            ctx.conn,
            owner_id=OWNER,
            content_hash=content_hash,
            thumb_key=thumb_key(content_hash, 320),
        )
        buffer = io.BytesIO()
        Image.new("RGB", (400, 300), "gray").save(buffer, format="JPEG")
        ctx.derived.write(thumb_key(content_hash, ctx.settings.thumb_detail_px), buffer.getvalue())
        face_ids.append(add_face(ctx.conn, photo_id=photo, group_id=group, vector=(1.0, 0.0)))
    return group, face_ids


def _links_to(body: str, group: int) -> bool:
    return re.search(rf'<a\b[^>]*href="/faces/{group}"', body) is not None


def _group_of(client, face: int) -> int:
    return _conn(client).execute("SELECT group_id FROM faces WHERE id = ?", (face,)).fetchone()[0]


def _person_of(client, group: int) -> int | None:
    return (
        _conn(client)
        .execute("SELECT person_id FROM face_groups WHERE id = ?", (group,))
        .fetchone()[0]
    )


def test_at_the_default_minimum_a_one_face_group_is_hidden_and_a_two_face_group_shown(client):
    assert client.app.state.context.settings.face_group_min_faces == 2
    single, _ = _group(client, 1)
    pair, _ = _group(client, 2)

    body = client.get("/voices").text

    assert not _links_to(body, single)
    assert _links_to(body, pair)


def test_a_released_face_is_listed_in_faces_to_name_and_can_be_named(client):
    ann = create_voice(_conn(client), OWNER, "Ann").id
    bob = create_voice(_conn(client), OWNER, "Bob").id
    ann_group, faces = _group(client, 2, person_id=ann)

    released = client.post(f"/faces/face/{faces[0]}/release")
    new_group = _group_of(client, faces[0])
    listed = client.get("/voices").text
    named = client.post(f"/faces/{new_group}/assign", data={"person_id": str(bob)})
    after = client.get("/voices").text

    assert released.status_code == 200
    assert new_group != ann_group
    assert _links_to(listed, new_group)
    assert named.status_code == 200
    assert _person_of(client, new_group) == bob
    assert _person_of(client, ann_group) == ann
    assert not _links_to(after, new_group)
