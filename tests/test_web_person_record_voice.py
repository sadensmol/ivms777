"""Record a voice from a person (design §13 `/people/{id}` and `/voices`, §13.1; plan 29).

The person leaf links to the Voices grid with `?person=<id>`; the grid's New voice form
then has that person selected and, when the person owns no voice yet, their name in the
name field. Any other `person` value gives the default form. A profile with no voice copy
shows the reason on the person page instead of the link."""

from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from feed.saved_voices import create_voice
from tests.factories import add_person
from tests.test_web_people import _group
from web.app import create_app

OWNER = 1
OTHER_OWNER = 2
MISSING_ID = 999_999
NO_VOICE_COPY = "Voice copy is not available on this device."
RECORD_LINK_TEXT = "Record a voice"
NEW_PERSON_OPTION = "New person with this name"


class _Page(HTMLParser):
    """The links of a page, and the New voice form of the Voices grid."""

    def __init__(self, body: str) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.name_value: str | None = None
        self.options: list[tuple[str, bool, str]] = []
        self._href: str | None = None
        self._text: list[str] = []
        self._option: tuple[str, bool] | None = None
        self._in_person_select = False
        self.feed(body)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        found = dict(attrs)
        if tag == "a" and found.get("href") is not None:
            self._href, self._text = found["href"], []
        elif tag == "input" and found.get("id") == "voice-new-name":
            self.name_value = found.get("value") or ""
        elif tag == "select" and found.get("id") == "voice-new-person":
            self._in_person_select = True
        elif tag == "option" and self._in_person_select:
            self._option = (found.get("value") or "", "selected" in found)
            self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join("".join(self._text).split())))
            self._href = None
        elif tag == "option" and self._option is not None:
            self.options.append((*self._option, "".join(self._text).strip()))
            self._option = None
        elif tag == "select":
            self._in_person_select = False

    def handle_data(self, data: str) -> None:
        self._text.append(data)

    def hrefs(self) -> set[str]:
        return {href for href, _ in self.links}

    def selected(self) -> list[str]:
        return [label for _, is_selected, label in self.options if is_selected]


def _client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return TestClient(create_app(settings))


@pytest.fixture
def client(settings):
    with _client(settings) as test_client:
        yield test_client


@pytest.fixture
def jetson(settings):
    settings.profile = "jetson"
    with _client(settings) as test_client:
        yield test_client


def _conn(client):
    return client.app.state.context.conn


def _person_page(client, person_id: int) -> tuple[str, _Page]:
    response = client.get(f"/people/{person_id}")
    assert response.status_code == 200
    return response.text, _Page(response.text)


def _voices_form(client, person: str) -> _Page:
    response = client.get("/voices", params={"person": person})
    assert response.status_code == 200
    return _Page(response.text)


def test_the_person_page_links_to_record_a_voice_when_the_person_has_none(client):
    ann = add_person(_conn(client), name="Ann")

    _, page = _person_page(client, ann)

    assert (f"/voices?person={ann}", RECORD_LINK_TEXT) in page.links


def test_the_person_page_links_to_record_a_voice_when_the_person_has_one(client):
    ann = add_person(_conn(client), name="Ann")
    create_voice(_conn(client), OWNER, "Nana", person_id=ann)

    body, page = _person_page(client, ann)

    assert "Nana" in body
    assert (f"/voices?person={ann}", RECORD_LINK_TEXT) in page.links


def test_no_voice_copy_shows_the_reason_and_no_record_link(jetson):
    ann = add_person(_conn(jetson), name="Ann")

    body, page = _person_page(jetson, ann)

    assert NO_VOICE_COPY in body
    assert not any(href.startswith("/voices?person=") for href in page.hrefs())


def test_no_voice_copy_still_lists_the_persons_voices_beside_the_reason(jetson):
    ann = add_person(_conn(jetson), name="Ann")
    create_voice(_conn(jetson), OWNER, "Nana", person_id=ann)

    body, page = _person_page(jetson, ann)

    assert "Nana" in body and NO_VOICE_COPY in body
    assert not any(href.startswith("/voices?person=") for href in page.hrefs())


def test_the_voices_grid_selects_the_person_of_the_link(client):
    add_person(_conn(client), name="Bob")
    ann = add_person(_conn(client), name="Ann")

    form = _voices_form(client, str(ann))

    assert form.selected() == ["Ann"]


def test_the_voices_grid_selects_a_person_who_already_owns_a_voice(client):
    add_person(_conn(client), name="Ann")
    bob = add_person(_conn(client), name="Bob")
    create_voice(_conn(client), OWNER, "Grandpa", person_id=bob)

    form = _voices_form(client, str(bob))

    assert form.selected() == ["Bob"]


def test_the_name_is_the_persons_name_when_they_own_no_voice(client):
    ann = add_person(_conn(client), name="Ann")

    form = _voices_form(client, str(ann))

    assert form.name_value == "Ann"


def test_the_prefilled_name_keeps_quotes_and_ampersands(client):
    person = add_person(_conn(client), name="O'Brien & \"Co\"")

    form = _voices_form(client, str(person))

    assert form.name_value == "O'Brien & \"Co\""


def test_the_name_is_empty_when_the_person_owns_a_voice(client):
    ann = add_person(_conn(client), name="Ann")
    create_voice(_conn(client), OWNER, "Nana", person_id=ann)

    form = _voices_form(client, str(ann))

    assert form.selected() == ["Ann"]
    assert form.name_value == ""


@pytest.mark.parametrize("value", ["missing", "foreign", "not-a-number"])
def test_a_person_the_owner_has_not_gives_the_default_form(client, value):
    ann = add_person(_conn(client), name="Ann")
    theirs = add_person(_conn(client), owner_id=OTHER_OWNER, name="Zebediah")
    person = {"missing": str(MISSING_ID), "foreign": str(theirs), "not-a-number": "ann"}[value]

    linked = _voices_form(client, str(ann))
    form = _voices_form(client, person)

    assert linked.selected() == ["Ann"]
    assert form.selected() == [NEW_PERSON_OPTION]
    assert form.name_value == ""
    assert "Zebediah" not in [label for _, _, label in form.options]


def _create_from_the_form(client, person_id: int, typed_name: str) -> None:
    _, person_page = _person_page(client, person_id)
    record = [href for href, text in person_page.links if text == RECORD_LINK_TEXT]
    assert record == [f"/voices?person={person_id}"]
    response = client.get(record[0])
    assert response.status_code == 200
    form = _Page(response.text)
    chosen = [value for value, is_selected, _ in form.options if is_selected]
    assert chosen == [str(person_id)]

    created = client.post(
        "/voices", data={"name": typed_name or form.name_value, "person_id": chosen[0]})

    assert created.status_code == 200


def test_a_voice_created_from_the_linked_form_is_listed_on_the_person_page(client):
    ann = add_person(_conn(client), name="Ann")

    _create_from_the_form(client, ann, typed_name="")

    body, _ = _person_page(client, ann)
    assert "No voices yet." not in body
    assert [name for (name,) in _conn(client).execute(
        "SELECT name FROM saved_voices WHERE person_id = ?", (ann,))] == ["Ann"]


def test_a_second_voice_from_the_linked_form_joins_the_first(client):
    ann = add_person(_conn(client), name="Ann")
    create_voice(_conn(client), OWNER, "Nana", person_id=ann)

    _create_from_the_form(client, ann, typed_name="Nana singing")

    body, _ = _person_page(client, ann)
    assert "Nana" in body and "Nana singing" in body
    assert _conn(client).execute(
        "SELECT COUNT(*) FROM saved_voices WHERE person_id = ?", (ann,)).fetchone()[0] == 2


def test_the_person_page_links_only_to_people_and_the_voices_grid(client):
    ann = add_person(_conn(client), name="Ann")
    voice = create_voice(_conn(client), OWNER, "Nana", person_id=ann)
    _group(client, 2, person_id=ann)

    _, page = _person_page(client, ann)
    hrefs = page.hrefs()

    assert {href for href in hrefs if href.startswith(("/voices/", "/voices?"))} == {f"/voices?person={ann}"}
    assert f"/voices/{voice.id}" not in hrefs
    assert not any(href.startswith("/faces/") for href in hrefs), hrefs
    assert not any(href.startswith("/photo") for href in hrefs), hrefs
