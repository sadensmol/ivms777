"""The person leaf and the voice form against the people rules (fix round 8, design §13,
D8): the leaf lists photos newest first with a photo of no shot time last, and a voice
re-created with the name of a deleted voice needs its kept person."""

import re

import pytest
from fastapi.testclient import TestClient

from feed.saved_voices import create_voice, process_voice_deletions, request_voice_deletion
from storage.local import LocalStorage
from tests.factories import add_face, add_face_group, add_person, add_photo
from web.app import create_app

OWNER = 1


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _conn(client):
    return client.app.state.context.conn


def _photo_of(client, person: int, shot_at: str | None) -> int:
    conn = _conn(client)
    photo = add_photo(conn, content_hash=f"{shot_at or 'none'}-{person}", shot_at=shot_at)
    add_face(conn, photo_id=photo, group_id=add_face_group(conn, person_id=person),
             vector=(1.0, 0.0))
    return photo


def _thumbs(body: str) -> list[int]:
    section = body[body.index('class="person-photos"'):]
    return [int(found) for found in re.findall(r'<img src="/thumb/(\d+)"', section)]


def _delete_voice(client, voice_id: int, tmp_path) -> None:
    conn = _conn(client)
    request_voice_deletion(conn, OWNER, voice_id)
    process_voice_deletions(conn, LocalStorage(tmp_path / "voices"), OWNER)


def _voices(client) -> list[tuple[str, int]]:
    return [tuple(r) for r in _conn(client).execute(
        "SELECT name, person_id FROM saved_voices ORDER BY id")]


def test_the_person_leaf_lists_photos_newest_first_and_no_shot_time_last(client):
    ann = add_person(_conn(client), name="Ann")
    unknown = _photo_of(client, ann, None)
    older = _photo_of(client, ann, "2019-03-01T00:00:00")
    newer = _photo_of(client, ann, "2021-06-01T00:00:00")

    response = client.get(f"/people/{ann}")

    assert response.status_code == 200
    assert _thumbs(response.text) == [newer, older, unknown]


def test_re_creating_a_deleted_voice_with_a_new_person_is_refused_with_its_reason(
    client, tmp_path
):
    voice = create_voice(_conn(client), OWNER, "Grandma")
    _delete_voice(client, voice.id, tmp_path)

    response = client.post("/voices", data={"name": "Grandma"})

    assert response.status_code == 409
    assert "Grandma" in response.json()["detail"]
    assert _voices(client) == []


def test_re_creating_a_deleted_voice_for_its_kept_person_works(client, tmp_path):
    voice = create_voice(_conn(client), OWNER, "Grandma")
    _delete_voice(client, voice.id, tmp_path)

    response = client.post("/voices", data={"name": "Grandma", "person_id": str(voice.person_id)})

    assert response.status_code == 200
    assert _voices(client) == [("Grandma", voice.person_id)]
