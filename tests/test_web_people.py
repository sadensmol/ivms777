"""People (fix round 8, design §6, §13, §13.1): a person is its own record. The People tab
`/people` with "New person" and "Faces to name", the person leaf `/people/{id}`, a face
group attached to a person, a voice that belongs to a person, and deleting a person.
Every lookup is scoped to the owner."""

import io
import itertools
import re

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from feed.saved_voices import create_voice, request_voice_deletion
from ingest.thumbs import thumb_key
from tests.factories import add_face, add_face_group, add_person, add_photo
from web.app import create_app

OWNER = 1
OTHER_OWNER = 2
MISSING_ID = 999_999
NAME_MAX = 80
_hashes = itertools.count(50_000)


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _conn(client):
    return client.app.state.context.conn


def _require_people(client) -> None:
    found = _conn(client).execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'people'").fetchone()
    assert found is not None, "the schema has no people table"


def _people(client) -> list[tuple[int, str]]:
    _require_people(client)
    return [tuple(r) for r in _conn(client).execute(
        "SELECT owner_id, name FROM people ORDER BY id")]


def _voices(client) -> list[tuple[str, str | None]]:
    """Each saved voice with the name of its person."""
    _require_people(client)
    return [tuple(r) for r in _conn(client).execute(
        "SELECT v.name, pe.name FROM saved_voices v LEFT JOIN people pe ON pe.id = v.person_id"
        " ORDER BY v.id")]


def _person_of_voice(client, voice_id: int) -> int | None:
    _require_people(client)
    return _conn(client).execute(
        "SELECT person_id FROM saved_voices WHERE id = ?", (voice_id,)).fetchone()[0]


def _person_of_group(client, group: int) -> int | None:
    return _conn(client).execute(
        "SELECT person_id FROM face_groups WHERE id = ?", (group,)).fetchone()[0]


def _group(client, faces: int, *, owner_id: int = OWNER, person_id: int | None = None):
    """A face group of `faces` faces, each in its own photo with a detail thumbnail."""
    ctx = client.app.state.context
    group = add_face_group(ctx.conn, owner_id=owner_id, person_id=person_id)
    photos, face_ids = [], []
    for _ in range(faces):
        content_hash = f"{next(_hashes):064x}"
        photo = add_photo(ctx.conn, owner_id=owner_id, content_hash=content_hash,
                          thumb_key=thumb_key(content_hash, 320))
        buffer = io.BytesIO()
        Image.new("RGB", (400, 300), "gray").save(buffer, format="JPEG")
        ctx.derived.write(thumb_key(content_hash, ctx.settings.thumb_detail_px),
                          buffer.getvalue())
        photos.append(photo)
        face_ids.append(add_face(ctx.conn, photo_id=photo, group_id=group, vector=(1.0, 0.0)))
    return group, face_ids, photos


def _links_to(body: str, href: str) -> bool:
    return re.search(rf'<a\b[^>]*href="{re.escape(href)}"', body) is not None


def _count(client, table: str) -> int:
    return _conn(client).execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# AC18 — the People tab, its grid, "New person" and "Faces to name".

def test_the_nav_has_people_before_voices(client):
    body = client.get("/library").text
    nav = body[body.index('<nav class="nav"'):body.index("</nav>")]
    links = re.findall(r'<a href="(/[a-z]+)">', nav)

    assert "/people" in links
    assert links.index("/people") == links.index("/voices") - 1


def test_the_people_grid_lists_the_owners_people_each_a_plain_link(client):
    ann = add_person(_conn(client), name="Ann")
    theirs = add_person(_conn(client), owner_id=OTHER_OWNER, name="Zebediah")

    response = client.get("/people")

    assert response.status_code == 200
    assert "Ann" in response.text and _links_to(response.text, f"/people/{ann}")
    assert "Zebediah" not in response.text and not _links_to(response.text, f"/people/{theirs}")


def test_new_person_creates_a_person_and_answers_the_cards(client):
    response = client.post("/people", data={"name": "  Ann  "})

    assert response.status_code == 200
    assert "Ann" in response.text
    assert _people(client) == [(OWNER, "Ann")]
    assert _voices(client) == []


@pytest.mark.parametrize(("name", "status"), [
    pytest.param("   ", 400, id="empty"),
    pytest.param("x" * (NAME_MAX + 1), 400, id="too-long"),
    pytest.param("aNN", 409, id="taken-case-blind"),
])
def test_new_person_refuses_a_bad_name_with_its_reason_and_writes_nothing(client, name, status):
    first = client.post("/people", data={"name": "Ann"})

    response = client.post("/people", data={"name": name})

    assert first.status_code == 200
    assert response.status_code == status
    assert response.json()["detail"]
    assert _people(client) == [(OWNER, "Ann")]


def test_a_name_of_the_longest_length_is_accepted(client):
    response = client.post("/people", data={"name": "x" * NAME_MAX})

    assert response.status_code == 200
    assert _people(client) == [(OWNER, "x" * NAME_MAX)]


def test_faces_to_name_is_on_people_and_no_longer_on_voices(client):
    group, _, _ = _group(client, 2)

    people = client.get("/people").text
    voices = client.get("/voices").text

    assert "Faces to name" in people and _links_to(people, f"/faces/{group}")
    assert "Faces to name" not in voices and not _links_to(voices, f"/faces/{group}")


# AC19 — the person leaf.

def test_the_person_leaf_shows_the_name_voices_faces_and_photos_in_place(client):
    ann = add_person(_conn(client), name="Ann")
    voice = create_voice(_conn(client), OWNER, "Grandma's voice", person_id=ann)
    group, faces, photos = _group(client, 2, person_id=ann)

    response = client.get(f"/people/{ann}")
    body = response.text

    assert response.status_code == 200
    assert "Ann" in body and "Grandma&#39;s voice" in body
    assert all(f'src="/faces/crop/{face}"' in body for face in faces)
    assert "2 faces" in body and "Remove from this person" in body
    assert all(f'src="/thumb/{photo}"' in body for photo in photos)
    assert not _links_to(body, f"/faces/{group}")
    assert not _links_to(body, f"/voices/{voice.id}")
    assert all(not _links_to(body, f"/photo/{photo}") for photo in photos)


def test_a_person_with_no_voice_says_so(client):
    ann = add_person(_conn(client), name="Ann")

    body = client.get(f"/people/{ann}").text

    assert "No voices yet." in body and "No faces yet." in body


def test_rename_changes_the_name_and_answers_the_leaf(client):
    ann = add_person(_conn(client), name="Ann")

    response = client.post(f"/people/{ann}/rename", data={"name": "Anna"})

    assert response.status_code == 200
    assert "Anna" in response.text
    assert _people(client) == [(OWNER, "Anna")]


@pytest.mark.parametrize(("name", "status"), [
    pytest.param("  ", 400, id="empty"),
    pytest.param("x" * (NAME_MAX + 1), 400, id="too-long"),
    pytest.param("bob", 409, id="taken-case-blind"),
])
def test_rename_refuses_a_bad_name_and_keeps_the_old_one(client, name, status):
    ann = add_person(_conn(client), name="Ann")
    add_person(_conn(client), name="Bob")

    response = client.post(f"/people/{ann}/rename", data={"name": name})

    assert response.status_code == status
    assert response.json()["detail"]
    assert _people(client) == [(OWNER, "Ann"), (OWNER, "Bob")]


def test_another_owners_person_or_a_missing_one_is_a_404(client):
    theirs = add_person(_conn(client), owner_id=OTHER_OWNER, name="Zebediah")

    assert client.get(f"/people/{theirs}").status_code == 404
    assert client.get(f"/people/{MISSING_ID}").status_code == 404
    assert client.post(f"/people/{theirs}/rename", data={"name": "Mine"}).status_code == 404
    assert client.post(f"/people/{theirs}/delete").status_code == 404
    assert _people(client) == [(OTHER_OWNER, "Zebediah")]


# AC20 — the group leaf belongs to People.

def test_the_group_leaf_is_under_people_and_its_select_lists_people(client):
    bob = add_person(_conn(client), name="Bob")
    add_person(_conn(client), owner_id=OTHER_OWNER, name="Zebediah")
    group, _, _ = _group(client, 2)

    body = client.get(f"/faces/{group}").text

    assert _links_to(body, "/people")
    assert re.search(rf'<option value="{bob}">Bob</option>', body)
    assert "Zebediah" not in body
    assert _voices(client) == []


# AC21 — a voice belongs to a person.

def test_a_new_voice_by_default_makes_a_person_with_its_name(client):
    response = client.post("/voices", data={"name": "Grandma"})

    assert response.status_code == 200
    assert "Voice of Grandma" in response.text
    assert _voices(client) == [("Grandma", "Grandma")]
    assert _people(client) == [(OWNER, "Grandma")]


def test_a_new_voice_can_belong_to_an_existing_person(client):
    ann = add_person(_conn(client), name="Ann")

    response = client.post("/voices", data={"name": "Nana", "person_id": str(ann)})

    assert response.status_code == 200
    assert _voices(client) == [("Nana", "Ann")]
    assert _people(client) == [(OWNER, "Ann")]


def test_a_new_voice_whose_name_a_person_has_is_refused_and_writes_nothing(client):
    add_person(_conn(client), name="Grandma")

    response = client.post("/voices", data={"name": "grandma"})

    assert response.status_code == 409
    assert "Grandma" in response.json()["detail"]
    assert _voices(client) == []
    assert _people(client) == [(OWNER, "Grandma")]


@pytest.mark.parametrize("person", ["foreign", "missing"])
def test_a_new_voice_for_a_person_the_owner_has_not_is_refused(client, person):
    theirs = add_person(_conn(client), owner_id=OTHER_OWNER, name="Zebediah")
    person_id = theirs if person == "foreign" else MISSING_ID

    response = client.post("/voices", data={"name": "Nana", "person_id": str(person_id)})

    assert response.status_code == 400
    assert _voices(client) == []
    assert _people(client) == [(OTHER_OWNER, "Zebediah")]


def test_the_voice_leaf_names_its_person_offers_a_move_and_has_no_faces_tab(client):
    ann = add_person(_conn(client), name="Ann")
    bob = add_person(_conn(client), name="Bob")
    voice = create_voice(_conn(client), OWNER, "Nana", person_id=ann)

    body = client.get(f"/voices/{voice.id}").text
    tabs = re.findall(r'class="voice-tab" data-tab="(\w+)"', body)

    assert "Voice of" in body and "Ann" in body
    assert f'action="/voices/{voice.id}/person"' in body
    assert re.search(rf'<option value="{bob}">Bob</option>', body)
    assert tabs == ["recordings", "test"]


def test_moving_a_voice_to_another_person_of_the_owner(client):
    ann = add_person(_conn(client), name="Ann")
    bob = add_person(_conn(client), name="Bob")
    voice = create_voice(_conn(client), OWNER, "Nana", person_id=ann)

    response = client.post(f"/voices/{voice.id}/person", data={"person_id": str(bob)})

    assert response.status_code == 200
    assert "Bob" in response.text
    assert _person_of_voice(client, voice.id) == bob


def test_moving_a_voice_to_another_owners_person_is_refused(client):
    ann = add_person(_conn(client), name="Ann")
    theirs = add_person(_conn(client), owner_id=OTHER_OWNER, name="Zebediah")
    voice = create_voice(_conn(client), OWNER, "Nana", person_id=ann)

    response = client.post(f"/voices/{voice.id}/person", data={"person_id": str(theirs)})

    assert response.status_code == 400
    assert _person_of_voice(client, voice.id) == ann


# AC22 — deleting a person.

def test_deleting_a_person_who_owns_a_voice_is_refused_with_its_reason(client):
    ann = add_person(_conn(client), name="Ann")
    create_voice(_conn(client), OWNER, "Nana", person_id=ann)

    response = client.post(f"/people/{ann}/delete")

    assert response.status_code == 409
    assert "Nana" in response.json()["detail"]
    assert _people(client) == [(OWNER, "Ann")]


def test_deleting_a_person_whose_voice_is_being_deleted_is_refused(client):
    ann = add_person(_conn(client), name="Ann")
    voice = create_voice(_conn(client), OWNER, "Nana", person_id=ann)
    assert request_voice_deletion(_conn(client), OWNER, voice.id)

    response = client.post(f"/people/{ann}/delete")

    assert response.status_code == 409
    assert _people(client) == [(OWNER, "Ann")]


def test_deleting_a_person_returns_their_groups_and_deletes_no_photo_or_face(client):
    ann = add_person(_conn(client), name="Ann")
    group, faces, photos = _group(client, 2, person_id=ann)
    voice = create_voice(_conn(client), OWNER, "Someone else's voice")

    response = client.post(f"/people/{ann}/delete")

    assert response.status_code in (200, 204)
    assert (OWNER, "Ann") not in _people(client)
    assert _person_of_group(client, group) is None
    assert _links_to(client.get("/people").text, f"/faces/{group}")
    assert _count(client, "faces") == len(faces) and _count(client, "photos") == len(photos)
    assert [v for v, _ in _voices(client)] == [voice.name]
