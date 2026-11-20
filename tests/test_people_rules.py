"""People library rules (fix round 8, design §6, D7, D8): create, rename and delete a
person, the voices and photos of a person, moving a voice, and a voice re-created with
the name of a deleted voice. Names are unique per owner, compared case-blind, Cyrillic too."""

import pytest

from feed.saved_voices import (
    VoiceDeleting,
    create_voice,
    get_voice,
    move_voice,
    process_voice_deletions,
    request_voice_deletion,
)
from ingest import people
from storage.local import LocalStorage
from tests.factories import add_face, add_face_group, add_person, add_photo

OWNER = 1
OTHER_OWNER = 2
MISSING_ID = 999_999
NAME_MAX = 80
ANNA = "Анна"
ANNA_UPPER = "АННА"


@pytest.fixture
def store(tmp_path):
    return LocalStorage(tmp_path / "voices")


def _people(conn) -> list[tuple[int, str]]:
    return [tuple(r) for r in conn.execute("SELECT owner_id, name FROM people ORDER BY id")]


def _voices(conn) -> list[tuple[str, int | None]]:
    return [tuple(r) for r in conn.execute("SELECT name, person_id FROM saved_voices ORDER BY id")]


def _group_person(conn, group: int) -> int | None:
    return conn.execute("SELECT person_id FROM face_groups WHERE id = ?", (group,)).fetchone()[0]


def _photo_of(conn, person: int, shot_at: str | None, *, owner_id: int = OWNER) -> int:
    photo = add_photo(conn, owner_id=owner_id, content_hash=f"{shot_at or 'none'}-{person}-{owner_id}",
                      shot_at=shot_at)
    group = add_face_group(conn, owner_id=owner_id, person_id=person)
    add_face(conn, photo_id=photo, group_id=group, vector=(1.0, 0.0))
    return photo


# create_person

def test_create_person_stores_the_trimmed_name(conn):
    person = people.create_person(conn, OWNER, "  Ann  ")

    assert person.name == "Ann"
    assert _people(conn) == [(OWNER, "Ann")]
    assert people.get_person(conn, OWNER, person.id).name == "Ann"


@pytest.mark.parametrize(
    "name",
    [pytest.param("   ", id="empty"), pytest.param("x" * (NAME_MAX + 1), id="too-long")],
)
def test_create_person_refuses_a_bad_name_and_writes_nothing(conn, name):
    with pytest.raises(people.PersonError):
        people.create_person(conn, OWNER, name)

    assert _people(conn) == []


def test_create_person_takes_the_longest_name(conn):
    people.create_person(conn, OWNER, "x" * NAME_MAX)

    assert _people(conn) == [(OWNER, "x" * NAME_MAX)]


def test_create_person_refuses_a_cyrillic_name_taken_in_another_case(conn):
    people.create_person(conn, OWNER, ANNA)

    with pytest.raises(people.PersonNameTaken):
        people.create_person(conn, OWNER, ANNA_UPPER)

    assert _people(conn) == [(OWNER, ANNA)]


def test_two_owners_may_have_a_person_of_the_same_name(conn):
    people.create_person(conn, OTHER_OWNER, ANNA)

    people.create_person(conn, OWNER, ANNA)

    assert _people(conn) == [(OTHER_OWNER, ANNA), (OWNER, ANNA)]


def test_people_are_listed_by_name_case_blind_with_their_voices(conn):
    boris = people.create_person(conn, OWNER, "борис")
    anna = people.create_person(conn, OWNER, ANNA)
    create_voice(conn, OWNER, "Nana", person_id=anna.id)
    create_voice(conn, OWNER, "anna reading", person_id=anna.id)
    people.create_person(conn, OTHER_OWNER, "Aaron")

    listed = people.list_people(conn, OWNER)

    assert [(p.id, p.voice_names) for p in listed] == [
        (anna.id, ("anna reading", "Nana")),
        (boris.id, ()),
    ]


# rename_person

def test_rename_person_may_change_only_the_case_of_its_own_name(conn):
    anna = people.create_person(conn, OWNER, "анна")

    renamed = people.rename_person(conn, OWNER, anna.id, ANNA)

    assert renamed.name == ANNA
    assert _people(conn) == [(OWNER, ANNA)]


def test_rename_person_refuses_a_cyrillic_name_of_another_person(conn):
    people.create_person(conn, OWNER, ANNA)
    boris = people.create_person(conn, OWNER, "Борис")

    with pytest.raises(people.PersonNameTaken):
        people.rename_person(conn, OWNER, boris.id, ANNA_UPPER)

    assert _people(conn) == [(OWNER, ANNA), (OWNER, "Борис")]


@pytest.mark.parametrize("owner", ["other-owner", "missing"])
def test_rename_person_of_another_owner_or_missing_is_not_found(conn, owner):
    theirs = add_person(conn, owner_id=OTHER_OWNER, name="Zebediah")
    person_id = theirs if owner == "other-owner" else MISSING_ID

    with pytest.raises(people.PersonNotFound):
        people.rename_person(conn, OWNER, person_id, "Ann")

    assert _people(conn) == [(OTHER_OWNER, "Zebediah")]


def test_renaming_a_person_renames_no_voice(conn):
    ann = people.create_person(conn, OWNER, "Ann")
    voice = create_voice(conn, OWNER, "Nana", person_id=ann.id)

    people.rename_person(conn, OWNER, ann.id, "Grandma")

    assert get_voice(conn, OWNER, voice.id).name == "Nana"
    assert get_voice(conn, OWNER, voice.id).person_name == "Grandma"


# delete_person

def test_delete_person_with_a_voice_is_refused_and_keeps_everything(conn):
    ann = people.create_person(conn, OWNER, "Ann")
    create_voice(conn, OWNER, "Nana", person_id=ann.id)
    group = add_face_group(conn, owner_id=OWNER, person_id=ann.id)

    with pytest.raises(people.PersonHasVoices, match="Nana"):
        people.delete_person(conn, OWNER, ann.id)

    assert _people(conn) == [(OWNER, "Ann")]
    assert _voices(conn) == [("Nana", ann.id)]
    assert _group_person(conn, group) == ann.id


def test_delete_person_whose_voice_is_being_deleted_is_refused(conn):
    ann = people.create_person(conn, OWNER, "Ann")
    voice = create_voice(conn, OWNER, "Nana", person_id=ann.id)
    request_voice_deletion(conn, OWNER, voice.id)

    with pytest.raises(people.PersonHasVoices):
        people.delete_person(conn, OWNER, ann.id)

    assert _people(conn) == [(OWNER, "Ann")]


def test_delete_person_returns_their_groups_and_keeps_faces_and_photos(conn):
    ann = people.create_person(conn, OWNER, "Ann")
    photo = _photo_of(conn, ann.id, "2020-01-01T00:00:00")
    group = conn.execute("SELECT group_id FROM faces WHERE photo_id = ?", (photo,)).fetchone()[0]

    people.delete_person(conn, OWNER, ann.id)

    assert _people(conn) == []
    assert _group_person(conn, group) is None
    assert conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 1


def test_delete_person_after_their_voice_is_gone_succeeds(conn, store):
    ann = people.create_person(conn, OWNER, "Ann")
    voice = create_voice(conn, OWNER, "Nana", person_id=ann.id)
    request_voice_deletion(conn, OWNER, voice.id)
    process_voice_deletions(conn, store, OWNER)

    people.delete_person(conn, OWNER, ann.id)

    assert _people(conn) == []


@pytest.mark.parametrize("owner", ["other-owner", "missing"])
def test_delete_person_of_another_owner_or_missing_is_not_found(conn, owner):
    theirs = add_person(conn, owner_id=OTHER_OWNER, name="Zebediah")
    person_id = theirs if owner == "other-owner" else MISSING_ID

    with pytest.raises(people.PersonNotFound):
        people.delete_person(conn, OWNER, person_id)

    assert _people(conn) == [(OTHER_OWNER, "Zebediah")]


# person_voices

def test_person_voices_lists_only_their_voices_by_name_with_a_deleting_one(conn):
    ann = people.create_person(conn, OWNER, "Ann")
    bob = people.create_person(conn, OWNER, "Bob")
    nana = create_voice(conn, OWNER, "nana", person_id=ann.id)
    ann_reading = create_voice(conn, OWNER, "Ann reading", person_id=ann.id)
    create_voice(conn, OWNER, "Bob", person_id=bob.id)
    request_voice_deletion(conn, OWNER, nana.id)

    voices = people.person_voices(conn, OWNER, ann.id)

    assert [(v.id, v.deleting) for v in voices] == [(ann_reading.id, False), (nana.id, True)]


def test_person_voices_of_another_owners_person_is_empty(conn):
    theirs = add_person(conn, owner_id=OTHER_OWNER, name="Zebediah")
    create_voice(conn, OTHER_OWNER, "Zeb", person_id=theirs)

    assert people.person_voices(conn, OWNER, theirs) == []


# person_photos

def test_person_photos_are_oldest_first_and_no_shot_time_last(conn):
    ann = people.create_person(conn, OWNER, "Ann")
    unknown = _photo_of(conn, ann.id, None)
    newer = _photo_of(conn, ann.id, "2021-06-01T00:00:00")
    older = _photo_of(conn, ann.id, "2019-03-01T00:00:00")

    assert people.person_photos(conn, OWNER, ann.id) == [older, newer, unknown]


def test_person_photos_lists_a_photo_with_two_of_their_faces_once(conn):
    ann = people.create_person(conn, OWNER, "Ann")
    photo = add_photo(conn, content_hash="ab" * 32, shot_at="2020-01-01T00:00:00")
    group = add_face_group(conn, owner_id=OWNER, person_id=ann.id)
    add_face(conn, photo_id=photo, group_id=group, vector=(1.0, 0.0))
    add_face(conn, photo_id=photo, group_id=group, vector=(0.0, 1.0))

    assert people.person_photos(conn, OWNER, ann.id) == [photo]


def test_person_photos_skip_other_people_and_unnamed_groups(conn):
    ann = people.create_person(conn, OWNER, "Ann")
    bob = people.create_person(conn, OWNER, "Bob")
    own = _photo_of(conn, ann.id, "2020-01-01T00:00:00")
    _photo_of(conn, bob.id, "2020-01-02T00:00:00")
    unnamed = add_photo(conn, content_hash="cd" * 32)
    add_face(conn, photo_id=unnamed, group_id=add_face_group(conn, owner_id=OWNER),
             vector=(1.0, 0.0))

    assert people.person_photos(conn, OWNER, ann.id) == [own]


def test_person_photos_of_another_owners_person_is_empty(conn):
    theirs = add_person(conn, owner_id=OTHER_OWNER, name="Zebediah")
    _photo_of(conn, theirs, "2020-01-01T00:00:00", owner_id=OTHER_OWNER)

    assert people.person_photos(conn, OWNER, theirs) == []


# move_voice

def test_move_voice_gives_the_voice_to_another_person(conn):
    ann = people.create_person(conn, OWNER, "Ann")
    bob = people.create_person(conn, OWNER, "Bob")
    voice = create_voice(conn, OWNER, "Nana", person_id=ann.id)

    moved = move_voice(conn, OWNER, voice.id, bob.id)

    assert (moved.person_id, moved.person_name) == (bob.id, "Bob")
    assert people.person_voices(conn, OWNER, ann.id) == []
    people.delete_person(conn, OWNER, ann.id)
    assert _people(conn) == [(OWNER, "Bob")]


@pytest.mark.parametrize("person", ["other-owner", "missing"])
def test_move_voice_to_a_person_the_owner_has_not_is_refused(conn, person):
    ann = people.create_person(conn, OWNER, "Ann")
    voice = create_voice(conn, OWNER, "Nana", person_id=ann.id)
    theirs = add_person(conn, owner_id=OTHER_OWNER, name="Zebediah")
    target = theirs if person == "other-owner" else MISSING_ID

    with pytest.raises(people.PersonNotFound):
        move_voice(conn, OWNER, voice.id, target)

    assert _voices(conn) == [("Nana", ann.id)]


def test_move_voice_being_deleted_is_refused(conn):
    ann = people.create_person(conn, OWNER, "Ann")
    bob = people.create_person(conn, OWNER, "Bob")
    voice = create_voice(conn, OWNER, "Nana", person_id=ann.id)
    request_voice_deletion(conn, OWNER, voice.id)

    with pytest.raises(VoiceDeleting):
        move_voice(conn, OWNER, voice.id, bob.id)

    assert _voices(conn) == [("Nana", ann.id)]


@pytest.mark.parametrize("voice", ["other-owner", "missing"])
def test_move_voice_of_another_owner_or_missing_is_refused(conn, voice):
    theirs = add_person(conn, owner_id=OTHER_OWNER, name="Zebediah")
    their_voice = create_voice(conn, OTHER_OWNER, "Zeb", person_id=theirs)
    ann = people.create_person(conn, OWNER, "Ann")
    voice_id = their_voice.id if voice == "other-owner" else MISSING_ID

    with pytest.raises(VoiceDeleting):
        move_voice(conn, OWNER, voice_id, ann.id)

    assert _voices(conn) == [("Zeb", theirs)]


# create_voice and the person (D8)

def test_a_new_voice_named_like_a_cyrillic_person_in_another_case_is_refused(conn):
    people.create_person(conn, OWNER, ANNA)

    with pytest.raises(people.PersonNameTaken):
        create_voice(conn, OWNER, ANNA_UPPER)

    assert _voices(conn) == []
    assert _people(conn) == [(OWNER, ANNA)]


def test_re_creating_a_deleted_voice_with_a_new_person_is_refused_by_the_kept_person(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    request_voice_deletion(conn, OWNER, voice.id)
    process_voice_deletions(conn, store, OWNER)

    with pytest.raises(people.PersonNameTaken, match="Grandma"):
        create_voice(conn, OWNER, "Grandma")

    assert _voices(conn) == []
    assert _people(conn) == [(OWNER, "Grandma")]


def test_re_creating_a_deleted_voice_for_its_kept_person_works(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    person = voice.person_id
    request_voice_deletion(conn, OWNER, voice.id)
    process_voice_deletions(conn, store, OWNER)

    again = create_voice(conn, OWNER, "Grandma", person_id=person)

    assert (again.person_id, again.person_name) == (person, "Grandma")
    assert _voices(conn) == [("Grandma", person)]
    assert _people(conn) == [(OWNER, "Grandma")]
