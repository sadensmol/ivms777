"""The photos of the person leaf `/people/{id}` (design §13, docs/ui.md): each photo that
holds a face of the person, once, newest shot first; no photo of anyone else."""

import re

import pytest
from fastapi.testclient import TestClient

from tests.factories import add_face, add_face_group, add_person, add_photo
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _conn(client):
    return client.app.state.context.conn


def _thumbs(body: str) -> list[int]:
    section = body[body.index('class="person-photos"'):]
    return [int(found) for found in re.findall(r'<img src="/thumb/(\d+)"', section)]


def test_a_photo_with_two_faces_of_the_person_is_listed_once(client):
    conn = _conn(client)
    ann = add_person(conn, name="Ann")
    both = add_photo(conn, content_hash="both", shot_at="2020-01-01T00:00:00")
    single = add_photo(conn, content_hash="single", shot_at="2018-01-01T00:00:00")
    first_group = add_face_group(conn, person_id=ann)
    second_group = add_face_group(conn, person_id=ann)
    add_face(conn, photo_id=both, group_id=first_group, vector=(1.0, 0.0))
    add_face(conn, photo_id=both, group_id=second_group, vector=(0.0, 1.0))
    add_face(conn, photo_id=single, group_id=first_group, vector=(1.0, 0.0))

    response = client.get(f"/people/{ann}")

    assert response.status_code == 200
    assert _thumbs(response.text) == [both, single]


def test_a_photo_of_another_person_is_not_on_the_leaf(client):
    conn = _conn(client)
    ann = add_person(conn, name="Ann")
    bob = add_person(conn, name="Bob")
    anns = add_photo(conn, content_hash="ann", shot_at="2018-01-01T00:00:00")
    bobs = add_photo(conn, content_hash="bob", shot_at="2022-01-01T00:00:00")
    add_face(conn, photo_id=anns, group_id=add_face_group(conn, person_id=ann),
             vector=(1.0, 0.0))
    add_face(conn, photo_id=bobs, group_id=add_face_group(conn, person_id=bob),
             vector=(0.0, 1.0))

    response = client.get(f"/people/{ann}")

    assert response.status_code == 200
    assert _thumbs(response.text) == [anns]
