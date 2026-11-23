"""The `?person=` value of the Voices grid that names no person (design §13 `/voices`; plan 29).

Only ASCII digits that fit a SQLite INTEGER are looked up; spaces around them are dropped.
Anything else answers 200 with the default New voice form, never a 500 and never another
person."""

import pytest
from fastapi.testclient import TestClient

from tests.factories import add_person
from tests.test_web_person_record_voice import NEW_PERSON_OPTION, _conn, _voices_form
from web.app import create_app

FIFTH_PERSON_ID = 5
FULLWIDTH_FIVE = "５"
SUPERSCRIPT_TWO = "²"
LARGEST_ID_DIGITS = 18


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _five_people(client) -> int:
    ids = [add_person(_conn(client), name=f"Person {n}") for n in range(1, FIFTH_PERSON_ID + 1)]
    assert ids[-1] == FIFTH_PERSON_ID
    return ids[-1]


@pytest.mark.parametrize(
    "person",
    [
        "9" * (LARGEST_ID_DIGITS + 1),
        "9" * 40,
        "9" * LARGEST_ID_DIGITS,
        FULLWIDTH_FIVE,
        SUPERSCRIPT_TWO,
        "-1",
        "",
        "   ",
        "1e3",
        "+5",
        "5.0",
    ],
    ids=[
        "19 digits",
        "40 digits",
        "18 digits",
        "fullwidth five",
        "superscript two",
        "negative",
        "empty",
        "spaces only",
        "exponent",
        "plus sign",
        "decimal",
    ],
)
def test_a_value_that_names_no_person_gives_the_default_form(client, person):
    _five_people(client)

    form = _voices_form(client, person)

    assert form.selected() == [NEW_PERSON_OPTION]
    assert form.name_value == ""


def test_spaces_around_the_id_still_select_the_person(client):
    fifth = _five_people(client)

    form = _voices_form(client, f"  {fifth}  ")

    assert form.selected() == [f"Person {FIFTH_PERSON_ID}"]
    assert form.name_value == f"Person {FIFTH_PERSON_ID}"
