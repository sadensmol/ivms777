"""The ⚙ settings popup: `/settings/models` (design §13, §4.1).

The app is built against a fake `ModelsClient` that speaks the real service API
(over `FakeBackend`), so the routes are exercised end to end without a model.
"""

import re

import pytest
from fastapi.testclient import TestClient

from config import Settings
from db.settings import get_setting
from inference.models_client import ModelsClient
from modelsvc.app import create_models_app
from modelsvc.backends.fake import FakeBackend
from tests.factories import add_photo
from web.app import create_app


def _service_client(profile: str) -> ModelsClient:
    transport = TestClient(create_models_app(FakeBackend(profile=profile)))._transport
    return ModelsClient("http://modelsvc", transport=transport)


@pytest.fixture
def client(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    service = _service_client(settings.profile)
    monkeypatch.setattr(Settings, "build_models_client", lambda self: service)
    app = create_app(settings)
    with TestClient(app) as tc:
        tc.app_state = app.state
        tc.service = service
        yield tc


@pytest.fixture
def cloud_client(tmp_path, monkeypatch):
    settings = Settings(
        data_dir=tmp_path, profile="cloud", use_fake_embedder=True, use_fake_inference=True
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    service = _service_client("cloud")
    monkeypatch.setattr(Settings, "build_models_client", lambda self: service)
    with TestClient(create_app(settings)) as tc:
        yield tc


def _checked(body: str) -> set[str]:
    return set(re.findall(r'value="([^"]+)"[^>]*checked', body)) | set(
        re.findall(r'checked[^>]*value="([^"]+)"', body)
    )


# --- rendering -------------------------------------------------------------


def test_the_fragment_shows_every_slot(client):
    body = client.get("/settings/models").text
    for label in ("Image embeddings", "Caption text", "Captions", "Planner"):
        assert label in body


def test_the_active_model_is_the_checked_one(client):
    body = client.get("/settings/models").text
    assert "gemma4-E2B" in _checked(body)
    assert "siglip2-so400m-384" in _checked(body)


def test_a_stored_choice_wins_over_the_default(client):
    client.post("/settings/models", data={"slot": "planner", "key": "qwen3-4b-2507"})
    assert "qwen3-4b-2507" in _checked(client.get("/settings/models").text)


def test_a_partly_downloaded_entry_says_what_is_left_to_fetch(client, monkeypatch):
    # `caption`/`planner` share one GGUF and differ only by the vision projector,
    # so the SAME displayed model reads "on disk" in one slot and offers a full
    # 5.8 GB download in the other. The popup must say what is actually missing.
    real = client.service.catalog
    entry = next(e for e in real()["entries"] if e["key"] == "qwen3-vl-8b" and e["slot"] == "caption")
    have = 4795 * 1024 * 1024

    def partly(**kw):
        payload = real(**kw)
        for e in payload["entries"]:
            if (e["slot"], e["key"]) == ("caption", "qwen3-vl-8b"):
                e["download"] = {"state": "absent", "bytes": have, "total": 0, "error": None}
        return payload

    monkeypatch.setattr(client.service, "catalog", partly)
    body = client.get("/settings/models").text
    assert "4.7 GB already on disk" in body
    assert "1.1 GB still to fetch" in body
    assert entry["size_mb"] == 5900  # weights 4795 + the F16 projector 1105


def test_entries_show_size_cost_and_whether_the_cost_is_measured(client):
    body = client.get("/settings/models").text
    assert "3400 MB" in body  # the measured SigLIP cost
    assert "estimated" in body  # at least one unverified entry says so


def test_a_downloaded_model_says_so_and_an_absent_one_offers_download(client):
    body = client.get("/settings/models").text
    assert "on disk" in body
    assert "Download" in body


# --- the consequence line --------------------------------------------------


def test_selecting_another_model_states_what_it_re_runs(client):
    add_photo(client.app_state.context.conn, content_hash="a" * 8)
    add_photo(client.app_state.context.conn, content_hash="b" * 8)
    body = client.get("/settings/models?select=caption:qwen3-vl-4b").text
    assert "caption" in body and "caption_embed" in body
    assert "2 photos" in body


def test_the_planner_slot_has_no_re_run_cost(client):
    add_photo(client.app_state.context.conn, content_hash="a" * 8)
    body = client.get("/settings/models?select=planner:qwen3-4b-2507").text
    assert "photos" not in body.split("qwen3-4b-2507", 1)[1][:400]


def test_switch_is_disabled_until_the_model_is_on_disk(client):
    body = client.get("/settings/models?select=caption:qwen3-vl-4b").text
    assert re.search(r"<button[^>]*disabled[^>]*>\s*Switch", body)
    client.post("/settings/models/download", data={"slot": "caption", "key": "qwen3-vl-4b"})
    body = client.get("/settings/models?select=caption:qwen3-vl-4b").text
    assert not re.search(r"<button[^>]*disabled[^>]*>\s*Switch", body)


# --- switching -------------------------------------------------------------


def test_switching_stores_the_choice_and_pushes_it_to_the_service(client):
    resp = client.post("/settings/models", data={"slot": "caption", "key": "qwen3-vl-4b"})
    assert resp.status_code == 200
    conn = client.app_state.context.conn
    assert get_setting(conn, 1, "model_slot.caption") == "qwen3-vl-4b"
    assert client.service.models_state()["slots"]["caption"] == "qwen3-vl-4b"
    assert "qwen3-vl-4b" in _checked(resp.text)


def test_switching_requeues_the_invalidated_stages(client):
    conn = client.app_state.context.conn
    pid = add_photo(conn, content_hash="a" * 8)
    client.post("/settings/models", data={"slot": "caption", "key": "qwen3-vl-4b"})
    stages = {
        r["stage"] for r in conn.execute("SELECT stage FROM jobs WHERE photo_id = ?", (pid,))
    }
    assert stages == {"caption", "caption_embed"}


def test_an_unknown_model_is_a_400_and_changes_nothing(client):
    resp = client.post("/settings/models", data={"slot": "caption", "key": "nope"})
    assert resp.status_code == 400
    assert get_setting(client.app_state.context.conn, 1, "model_slot.caption") is None


def test_cloud_slots_render_read_only_and_refuse_a_switch(cloud_client):
    body = cloud_client.get("/settings/models").text
    assert "not switchable" in body.lower()
    assert not re.search(r"<input[^>]*type=\"radio\"", body)
    resp = cloud_client.post("/settings/models", data={"slot": "caption", "key": "qwen2.5vl-7b"})
    assert resp.status_code == 400


# --- downloads -------------------------------------------------------------


def test_download_starts_and_the_fragment_shows_the_new_state(client):
    resp = client.post(
        "/settings/models/download", data={"slot": "caption", "key": "qwen3-vl-4b"}
    )
    assert resp.status_code == 200
    # the entry's own block, up to the next model in the list
    section = resp.text.split("qwen3-vl-4b", 1)[1].split("</li>", 1)[0]
    assert "on disk" in section or "%" in section


def test_download_of_an_unknown_model_is_a_400(client):
    assert (
        client.post("/settings/models/download", data={"slot": "caption", "key": "x"}).status_code
        == 400
    )


# --- the generation re-push ------------------------------------------------


def test_the_resources_poll_re_pushes_slots_the_service_lost(client):
    # The service holds no DB (design §4.1): after a restart it is back on the
    # profile defaults, and the next poll must put the user's choice back.
    client.post("/settings/models", data={"slot": "caption", "key": "qwen3-vl-4b"})
    client.service.set_slots({"caption": "gemma4-E2B"})  # simulate a restart
    client.get("/api/resources")
    assert client.service.models_state()["slots"]["caption"] == "qwen3-vl-4b"


def test_the_resources_poll_does_not_push_when_they_already_agree(client):
    pushed = []
    original = client.service.set_slots
    client.service.set_slots = lambda slots, **kw: (pushed.append(slots), original(slots, **kw))[1]
    client.get("/api/resources")
    assert pushed == []


def test_the_bar_names_the_models_the_slots_actually_hold(client):
    client.post("/settings/models", data={"slot": "planner", "key": "qwen3-4b-2507"})
    client.get("/api/resources")  # re-push, so the service agrees
    from models.resources import display_names

    assert display_names(
        ["llm"],
        planner_model="qwen3-4b-2507",
        caption_model="gemma4-E2B",
        embed_model="siglip2-so400m-384",
        text_embed_model="nomic-1.5",
    ) == ["qwen3-4b-2507"]


# --- the popup shell (design §13) -----------------------------------------


def test_the_nav_has_a_settings_button_that_is_not_a_link(client):
    body = client.get("/library").text
    button = re.search(r'<button[^>]*id="settings-open"[^>]*>', body)
    assert button, "the ⚙ must be in the nav on every page"
    # A <button>, never an <a href>: an overlay pushes no history entry and
    # changes no URL, so §13.1's [grid, leaf] invariant is untouched.
    assert 'href' not in button.group(0)
    assert '<dialog id="settings"' in body


def test_the_settings_button_does_not_inherit_the_navs_hx_select(client):
    # The ⚙ sits inside <nav hx-select="main">, and hx-select is INHERITED by
    # htmx. Without an explicit unset the popup fragment (which has no <main>)
    # is filtered down to nothing and the dialog opens empty.
    body = client.get("/library").text
    button = re.search(r'<button[^>]*id="settings-open"[^>]*>', body).group(0)
    assert 'hx-select="unset"' in button


def test_the_fragment_does_not_poll_when_nothing_is_downloading(client):
    assert "every 1s" not in client.get("/settings/models").text


def test_the_fragment_polls_while_a_download_is_in_flight(client, monkeypatch):
    real = client.service.catalog

    def downloading(**kw):
        payload = real(**kw)
        for entry in payload["entries"]:
            if entry["key"] == "qwen3-vl-4b":
                entry["download"] = {
                    "state": "downloading", "bytes": 512, "total": 1024, "error": None
                }
        return payload

    monkeypatch.setattr(client.service, "catalog", downloading)
    body = client.get("/settings/models").text
    assert 'hx-trigger="every 1s"' in body
    assert "50%" in body


def test_a_dead_models_service_still_lists_the_catalog(client, monkeypatch):
    def dead(**kw):
        raise ConnectionError("models service is down")

    monkeypatch.setattr(client.service, "catalog", dead)
    body = client.get("/settings/models").text
    assert "unreachable" in body
    assert "gemma4-E2B" in body  # the catalog is local data, so it still renders
