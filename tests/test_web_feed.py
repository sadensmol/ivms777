"""The fullscreen memory feed (design §11.1, §13, §13.1)."""

import pathlib
import re
import shutil
import threading
from html.parser import HTMLParser

import numpy as np
import pytest
from fastapi.testclient import TestClient

from albums.memory_store import Memory, replace_memories
from config import Settings
from feed import saved_voices as sv
from models.catalog import cloning_entry_for
from storage.wav import encode_wav
from tests.factories import add_photo
from tests.test_feed_build import FAKE_ASR, WORDS
from tests.test_web_voices import clip
from web.app import create_app


class FakeModels:
    """The three calls the prepare job makes, returning real WAV bytes."""

    def __init__(self):
        self.story_prompts: list[str] = []

    def text_complete(self, model, messages, *, json_schema=None, **kw):
        import json

        self.story_prompts.append(messages[-1]["content"])

        # Narration is mandatory (§11.1), so the fake planner has to answer the
        # translation pass too — an English script for a Russian film now FAILS the
        # build rather than producing a mute one.
        prompt = messages[-1]["content"]
        # The story pass is two stages (§11.1): the STORY first, then the call that
        # gives each of its sentences a photograph.
        if prompt.startswith("Translate into"):
            return json.dumps({"story":
                               "Тёплый день у воды, и совершенно некуда спешить. "
                               "Мы купались до самого заката и не хотели уходить. "
                               "Потом мы пошли домой самой длинной дорогой через парк."})
        if "WHAT THE DAY WAS" in prompt:
            # Sentences long enough to be their own BEAT (`story._MIN_SENTENCE_CHARS`).
            return json.dumps({"story":
                               "A warm afternoon by the water, with nowhere to be. "
                               "We swam until the light went off the surface. "
                               "Then we walked all the way home the long way round."})
        if "ЧТО ЭТО ЗА ДЕНЬ" in prompt:
            return json.dumps({"story":
                               "Тёплый день у воды, и совершенно некуда спешить. "
                               "Мы купались до самого заката и не хотели уходить. "
                               "Потом мы пошли домой самой длинной дорогой через парк."})
        if "more sentences" in prompt or "Допиши" in prompt:
            return json.dumps({"story": ""})
        if "SENTENCE:" in prompt or "ПРЕДЛОЖЕНИЕ" in prompt:
            return json.dumps({"text": ""})
        return json.dumps({
            "script": [{"sentence": 1, "shot": 1, "voice": ""},
                       {"sentence": 2, "shot": 2, "voice": ""},
                       {"sentence": 3, "shot": 3, "voice": ""}],
            "narration": "",
            "narrator": "",
            "cards": [{"sentence": 1, "text": "The water"}],
            "music_prompt": "warm ambient drone",
        })

    def speak(self, text, language="en", voice=None, segments=None,
              voices=None):
        return {"audio": encode_wav(np.zeros(24000, dtype=np.float32), 24000),
                "sample_rate": 24000, "model": "fake-voice"}

    def music(self, prompt="", seconds=30.0, seed=None, *, sections=None,
              temperature=None, guidance=None):
        return {"audio": encode_wav(np.zeros((2, 44100), dtype=np.float32), 44100),
                "sample_rate": 44100, "model": "fake-music", "segments": []}


@pytest.fixture
def client(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Settings, "build_models_client", lambda self: FakeModels())
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  width=4000, height=3000, ai_title=f"caption {pid}",
                  shot_at=f"2025-07-12T1{pid}:00:00")
    replace_memories(conn, settings.owner_id,
                     [Memory("Beach day", "Warm afternoon.", [1, 2, 3], "sig")])
    with TestClient(app) as test_client:
        yield test_client


def _prepare(client) -> None:
    client.post("/feed/memory-0/prepare", follow_redirects=False)
    client.app.state.await_feed_build()


def test_an_unprepared_memory_offers_to_prepare(client):
    body = client.get("/feed/memory-0").text
    assert "Prepare" in body
    assert "Beach day" in body


def test_an_unknown_key_is_404(client):
    assert client.get("/feed/memory-9").status_code == 404


def test_prepare_redirects_back_to_the_player(client):
    response = client.post("/feed/memory-0/prepare", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/feed/memory-0?lang=en"
    client.app.state.await_feed_build()


def test_prepare_then_play(client):
    _prepare(client)
    body = client.get("/feed/memory-0").text
    assert "feed-gl.js" in body
    assert "<canvas" in body
    assert "feed.js?" not in body        # the CSS-layer player is gone
    assert "Beach day" in body


def test_the_timeline_endpoint_serves_the_shots(client):
    _prepare(client)
    timeline = client.get("/feed/memory-0/timeline.json").json()
    assert [s["photo_id"] for s in timeline["shots"]] == [1, 2, 3]
    assert timeline["grade"] in timeline["grades"]


def test_the_timeline_is_404_before_preparation(client):
    assert client.get("/feed/memory-0/timeline.json").status_code == 404


def test_the_audio_endpoints_serve_wav(client):
    _prepare(client)
    for kind in ("music", "narration"):
        response = client.get(f"/feed/memory-0/audio/{kind}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content[:4] == b"RIFF"


def test_an_unknown_audio_kind_is_404(client):
    _prepare(client)
    assert client.get("/feed/memory-0/audio/podcast").status_code == 404


def test_progress_refreshes_the_page_when_the_build_is_done(client):
    client.app.state.feed_building = None
    response = client.get("/feed/memory-0/prepare/progress")
    assert response.headers.get("HX-Refresh") == "true"


def test_progress_reports_the_stage_and_a_percentage(client):
    # A static sentence is indistinguishable from a hung process, and the music stage
    # alone runs for minutes (§11.1).
    client.app.state.feed_building = ("memory-0", "en")
    client.app.state.feed_progress = {"stage": "composing the music", "fraction": 0.97}
    body = client.get("/feed/memory-0/prepare/progress").text
    assert "composing the music" in body
    assert "97%" in body
    assert "width: 97%" in body          # a bar the CSS can size
    assert "hx-get" in body              # still polling


def test_progress_offers_a_cancel_button(client):
    client.app.state.feed_building = ("memory-0", "en")
    body = client.get("/feed/memory-0/prepare/progress").text
    assert "Cancel" in body
    assert 'action="/feed/memory-0/cancel"' in body


def test_cancel_frees_the_ui_immediately(client):
    # The worker cannot be interrupted inside a multi-minute model call, so cancelling
    # must release the USER at once and discard whatever the thread produces.
    client.app.state.feed_building = ("memory-0", "en")
    client.app.state.feed_progress = {"stage": "composing the music", "fraction": 0.97}
    before = client.app.state.feed_run
    response = client.post("/feed/memory-0/cancel", data={"lang": "en"},
                           follow_redirects=False)
    assert response.status_code == 303
    assert client.app.state.feed_run > before      # the running build is now stale
    assert client.app.state.feed_building is None  # UI is free at once
    assert client.app.state.feed_progress == {}
    # And the page goes straight back to offering Prepare.
    assert "HX-Refresh" in client.get("/feed/memory-0/prepare/progress").headers


def test_cancel_leaves_nothing_queued(client):
    client.app.state.feed_building = ("memory-0", "en")
    client.app.state.feed_queued = ("memory-0", "ru")
    client.post("/feed/memory-0/cancel", data={"lang": "en"}, follow_redirects=False)
    assert client.app.state.feed_queued is None


def test_progress_for_one_language_does_not_block_the_other(client):
    # Preparing English must not make Russian look busy — each language is its own
    # film (§11.1).
    client.app.state.feed_building = ("memory-0", "en")
    response = client.get("/feed/memory-0/prepare/progress?lang=ru")
    assert response.headers.get("HX-Refresh") == "true"


def test_the_organize_memory_card_links_to_the_feed(client):
    assert "/feed/memory-0" in client.get("/organize?by=memories").text


def test_only_memory_cards_get_a_play_control(client):
    # `date`/`camera`/`place` albums are live views with nothing to play (§11.1).
    assert "/feed/" not in client.get("/organize?by=date").text


GRID = "/organize?by=memories"


def _memory_group(client) -> int:
    conn = client.app.state.context.conn
    return conn.execute("SELECT id FROM groups WHERE kind='memory'").fetchone()["id"]


def _film(client, language="en", *, status="ready", narration_key="n.wav", decisions=None,
          error=None) -> None:
    """A stored film row, written as the build writes it."""
    from feed.store import Presentation, save_presentation

    save_presentation(client.app.state.context.conn, Presentation(
        group_id=_memory_group(client), language=language, signature="sig", timeline={},
        music_key=None, narration_key=narration_key, status=status, error=error,
        decisions=decisions if decisions is not None else {}))


def _cast(language="en", **cast) -> dict:
    return {"language": language, "cast": cast}


def _row(grid: str, language: str) -> str:
    """One film row of the card: '<label> · <state>'."""
    found = re.search(rf'<a class="album-film" href="/feed/memory-0\?lang={language}">([^<]*)</a>'
                      r' · <span class="album-film-state">([^<]*)</span>', grid)
    assert found, f"no {language} film row"
    return f"{found.group(1)} · {found.group(2)}"


def _block(grid: str) -> str:
    start = grid.index('<div class="album-film-block"')
    return grid[start:grid.index("</div>", start)]


def test_a_card_offers_to_prepare_each_language_before_any_film(client):
    grid = client.get(GRID).text
    assert _row(grid, "en") == "Prepare English film · not made yet"
    assert _row(grid, "ru") == "Prepare Russian film · not made yet"
    assert ">▶ Play</a>" not in grid


def test_a_ready_automatic_film_plays_from_the_card(client):
    from feed.store import read_presentation

    _prepare(client)
    speaker = read_presentation(client.app.state.context.conn, _memory_group(client),
                                "en").decisions["cast"]["narrator_used"]
    grid = client.get(GRID).text
    assert _row(grid, "en") == f"▶ Play English film · ready · {speaker} (auto)"
    assert _row(grid, "ru") == "Prepare Russian film · not made yet"


def test_a_film_with_a_chosen_preset_reads_your_choice(client):
    client.post("/feed/memory-0/prepare", data={"lang": "en", "narrator": "preset:af_bella"},
                follow_redirects=False)
    client.app.state.await_feed_build()
    assert _row(client.get(GRID).text, "en") == "▶ Play English film · ready · af_bella (your choice)"


def test_the_card_reads_the_films_own_cast_not_the_current_choice(client):
    _film(client, decisions=_cast(narrator_used="af_bella", narrator_chosen_by="user"))
    assert sv.read_narrator(client.app.state.context.conn, _memory_group(client), "en") is None
    assert _row(client.get(GRID).text, "en") == "▶ Play English film · ready · af_bella (your choice)"


def test_a_film_being_made_or_waiting_opens_from_the_card(client):
    _film(client, "ru", decisions=_cast("ru", narrator_used="aidar"))
    client.app.state.feed_building = ("memory-0", "ru")
    assert _row(client.get(GRID).text, "ru") == "Open Russian film · being made…"
    client.app.state.feed_building = None
    client.app.state.feed_queued = ("memory-0", "ru")
    assert _row(client.get(GRID).text, "ru") == "Open Russian film · waiting to be made…"
    client.app.state.feed_queued = None


@pytest.mark.parametrize(("status", "narration_key", "decisions", "expected"), [
    ("failed", None, {}, "Prepare English film · the last try failed"),
    ("running", None, {}, "Prepare English film · interrupted"),
    ("ready", None, {}, "Prepare English film · no voice"),
    ("ready", "n.wav", {}, "▶ Play English film · ready · voice not recorded"),
])
def test_a_stored_film_row_reads_its_state(client, status, narration_key, decisions, expected):
    _film(client, status=status, narration_key=narration_key, decisions=decisions,
          error="boom" if status == "failed" else None)
    assert _row(client.get(GRID).text, "en") == expected


def test_retell_says_what_it_deletes(client):
    grid = client.get(GRID).text
    assert "↻ Retell memory" in grid
    assert ("Rewrites the title, description and photo order. Makes no film. "
            "No film to delete.") in grid
    assert "Rebuild memories" in grid
    _prepare(client)
    assert "Makes no film. Deletes the films in: English." in client.get(GRID).text


def test_the_chat_card_has_no_film_rows():
    # The chat renders the same partial with only `album` and `ctx` (web/app.py, the
    # chat memory card), so it gets no film rows and no Retell.
    from types import SimpleNamespace

    from jinja2 import Environment, FileSystemLoader

    from web.app import TEMPLATE_DIR

    album = SimpleNamespace(key="memory-0", photo_ids=[1, 2, 3], cover_id=1,
                            title="Beach day", description="Warm afternoon.", size=3)
    card = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True).get_template(
        "_album_card.html").render(album=album, ctx="ctx=chat-memory:memory-0")
    assert "Beach day" in card
    assert "album-films" not in card and "album-retell" not in card


def test_each_card_has_one_film_block_holding_rows_and_retell_but_not_notes(client):
    client.post("/feed/memory-0/note", data={"at": 1, "text": "a note"})
    grid = client.get(GRID).text
    assert grid.count('<div class="album-film-block" id="films-memory-0"') == 1
    block = _block(grid)
    assert "album-films" in block and "album-retell" in block and "album-retell-note" in block
    assert "album-notes" not in block
    assert "hx-get" not in block and "hx-trigger" not in block
    assert grid.count("window.__filmBlocksWired") == 2      # the guard and the flag, once


@pytest.mark.parametrize("state", ["feed_building", "feed_queued"])
def test_the_film_block_polls_only_while_its_memory_has_work(client, state):
    setattr(client.app.state, state, ("memory-0", "en"))
    block = _block(client.get(GRID).text)
    assert 'hx-get="/organize?by=memories"' in block
    assert 'hx-select="#films-memory-0"' in block
    assert 'hx-trigger="every 1500ms"' in block and 'hx-swap="outerHTML"' in block
    setattr(client.app.state, state, ("memory-9", "en"))
    assert "hx-trigger" not in _block(client.get(GRID).text)
    setattr(client.app.state, state, None)


def test_the_player_closes_back_to_the_memory_grid(client):
    # §13.1 rules 4/5: close goes UP exactly one level, with a deep-link fallback.
    _prepare(client)
    body = client.get("/feed/memory-0").text
    assert "history.back()" in body
    assert "/organize?by=memories" in body


def test_the_player_is_not_hx_boosted(client):
    # Boosting a leaf would put htmx's history cache in front of bfcache and break
    # the [grid, leaf] invariant (§13.1).
    _prepare(client)
    body = client.get("/feed/memory-0").text
    assert "hx-boost" not in body


def test_the_player_offers_both_languages(client):
    body = client.get("/feed/memory-0").text
    assert "English" in body and "Русский" in body


def test_preparing_in_russian_stores_a_russian_film(client):
    client.post("/feed/memory-0/prepare", data={"lang": "ru"}, follow_redirects=False)
    client.app.state.await_feed_build()
    assert "feed-gl.js" in client.get("/feed/memory-0?lang=ru").text


def test_each_language_is_prepared_separately(client):
    _prepare(client)                                   # en only
    assert "Prepare" in client.get("/feed/memory-0?lang=ru").text


def test_an_unknown_language_falls_back_to_the_default(client):
    assert client.get("/feed/memory-0?lang=klingon").status_code == 200


def test_the_edit_carries_camera_moves_and_transitions(client):
    _prepare(client)
    edit = client.get("/feed/memory-0/timeline.json").json()
    assert set(edit) == {"shots", "grade", "grades", "seconds", "narration", "effects",
                         "passages", "script"}
    move = edit["shots"][0]["move"]
    assert move["scale_from"] != move["scale_to"]
    assert edit["shots"][1]["transition"] in edit["effects"]["transition"]
    # The player looks a shot up by the soundtrack's `currentTime` (§11.1), so every
    # shot needs an absolute start and a named effect with its four numbers.
    for shot in edit["shots"]:
        assert shot["effect"] in edit["effects"]["shot"]
        assert len(shot["params"]) == 4
    assert edit["shots"][0]["at"] == 0.0


def test_the_story_is_a_narration_with_sparse_title_cards(client):
    _prepare(client)
    edit = client.get("/feed/memory-0/timeline.json").json()
    assert edit["narration"].startswith("A warm afternoon by the water")
    # One card, not a caption per shot.
    assert [s["line"] for s in edit["shots"]] == ["The water", "", ""]


def test_the_caption_is_hidden_behind_the_start_card(client):
    # Before the first play the title card is up; a caption underneath it competes
    # with the memory's own title (§11.1).
    _prepare(client)
    css = client.get("/static/feed.css").text
    assert ".feed-body:not(.is-playing) .feed-text.is-in { opacity: 0; }" in css


def test_a_prepared_film_can_be_regenerated(client):
    # The story pass is sampled, so a second run genuinely differs — and a run that
    # lost its music or voice must be retryable without deleting rows by hand.
    _prepare(client)
    body = client.get("/feed/memory-0").text
    assert "Regenerate" in body
    assert 'action="/feed/memory-0/prepare"' in body


def test_regenerate_carries_the_current_language(client):
    client.post("/feed/memory-0/prepare", data={"lang": "ru"}, follow_redirects=False)
    client.app.state.await_feed_build()
    body = client.get("/feed/memory-0?lang=ru").text
    assert 'name="lang" value="ru"' in body


def test_an_unprepared_film_offers_prepare_not_regenerate(client):
    body = client.get("/feed/memory-0").text
    assert "Prepare this memory" in body
    assert "feed-regen" not in body          # the button, not the word in a comment


def test_a_cancelled_build_leaves_no_presentation(client):
    from feed.build import Cancelled, build_presentation

    conn = client.app.state.context.conn
    gid = conn.execute("SELECT id FROM groups WHERE kind='memory'").fetchone()["id"]

    with pytest.raises(Cancelled):
        build_presentation(conn, FakeModels(), client.app.state.context.audio, 1, gid,
                           cancelled=lambda: True)
    from feed.store import read_presentation

    assert read_presentation(conn, gid, "en") is None


def test_progress_advances_through_every_stage(client):
    from feed.build import build_presentation

    conn = client.app.state.context.conn
    gid = conn.execute("SELECT id FROM groups WHERE kind='memory'").fetchone()["id"]
    seen = []
    build_presentation(conn, FakeModels(), client.app.state.context.audio, 1, gid,
                       progress=lambda stage, fraction: seen.append((stage, fraction)))
    stages = [stage for stage, _ in seen]
    # Each stage NAMES what it is making, so a run that takes minutes says where it is
    # rather than just "working" (§11.1).
    assert stages[0] == "reading your photos"
    assert stages[-1] == "saving the film"
    assert "recording the voice" in stages
    assert "composing the music" in stages
    assert "matching effects to the music" in stages
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0


def test_the_progress_fragment_escapes_what_it_interpolates(client):
    # The fragment is hand-built rather than rendered through Jinja, so nothing that
    # reaches it may carry markup — even though `key` is validated elsewhere.
    client.app.state.feed_building = ("memory-0", "en")
    client.app.state.feed_progress = {"stage": "<img src=x onerror=alert(1)>",
                                      "fraction": 0.5}
    body = client.get("/feed/memory-0/prepare/progress").text
    assert "<img src=x" not in body
    assert "&lt;img src=x" in body


def test_the_progress_percentage_is_clamped(client):
    client.app.state.feed_building = ("memory-0", "en")
    client.app.state.feed_progress = {"stage": "x", "fraction": 9.9}
    assert "width: 100%" in client.get("/feed/memory-0/prepare/progress").text


def test_the_progress_label_is_visible_not_the_canvas_class(client):
    # `.feed-stage` is the CANVAS (position: fixed; inset: 0). Reusing that class for
    # the progress label stretched it fullscreen and made the text invisible, which
    # left a bare line with no percentage on screen.
    client.app.state.feed_building = ("memory-0", "en")
    client.app.state.feed_progress = {"stage": "composing the music", "fraction": 0.97}
    body = client.get("/feed/memory-0/prepare/progress").text
    assert 'class="feed-step"' in body
    assert 'class="feed-stage"' not in body
    assert "composing the music · 97%" in body


def test_regenerate_while_busy_cancels_and_queues(client):
    # A button that says "go cancel it somewhere else" is a dead end. One click must
    # stop the running film and start this one.
    client.app.state.feed_building = ("memory-9", "ru")
    response = client.post("/feed/memory-0/prepare", data={"lang": "en"},
                           follow_redirects=False)
    assert response.status_code == 303
    assert "busy=memory-9:ru" in response.headers["location"]
    assert client.app.state.feed_run > 0                         # stop the other one
    assert client.app.state.feed_queued == ("memory-0", "en")    # then run mine
    body = client.get("/feed/memory-0?busy=memory-9:ru").text
    assert "Stopping" in body and "memory-9:ru" in body
    client.app.state.feed_building = None
    client.app.state.feed_queued = None


def test_a_second_regenerate_replaces_the_queued_one(client):
    # At most one film waits; pressing Regenerate again swaps what is waiting rather
    # than growing a backlog.
    client.app.state.feed_building = ("memory-9", "ru")
    client.post("/feed/memory-0/prepare", data={"lang": "en"}, follow_redirects=False)
    assert client.app.state.feed_queued == ("memory-0", "en")
    client.post("/feed/memory-0/prepare", data={"lang": "ru"}, follow_redirects=False)
    assert client.app.state.feed_queued == ("memory-0", "ru")
    client.app.state.feed_building = None
    client.app.state.feed_queued = None


def test_no_busy_notice_when_nothing_is_building(client):
    assert "Stopping" not in client.get("/feed/memory-0").text


def test_regenerating_immediately_hides_the_old_film(client):
    # "I clicked the button and nothing changed" — the page must switch to progress at
    # once, even though the previous presentation is still in the database.
    _prepare(client)
    assert "feed-gl.js" in client.get("/feed/memory-0").text     # plays before

    client.app.state.feed_building = ("memory-0", "en")
    body = client.get("/feed/memory-0").text
    assert "feed-gl.js" not in body          # the old film is gone from the page
    assert "feed-bar" in body                # progress instead
    client.app.state.feed_building = None


def test_a_queued_regenerate_also_shows_progress_not_the_old_film(client):
    _prepare(client)
    client.app.state.feed_building = ("memory-9", "ru")
    client.app.state.feed_queued = ("memory-0", "en")
    body = client.get("/feed/memory-0").text
    assert "feed-gl.js" not in body
    assert "feed-bar" in body
    fragment = client.get("/feed/memory-0/prepare/progress").text
    assert "waiting for memory-9:ru to stop" in fragment
    assert "hx-get" in fragment              # still polling, not page-reloading
    client.app.state.feed_building = None
    client.app.state.feed_queued = None


def test_in_film_navigation_replaces_instead_of_pushing(client):
    # §13.1 rule 3: history stays [grid, film]. Switching language or pressing
    # Regenerate must REPLACE the entry — otherwise watching a film three times means
    # pressing X three times to get back to the grid.
    _prepare(client)
    body = client.get("/feed/memory-0").text
    assert "location.replace" in body
    assert "preventDefault" in body
    # And the close button still goes UP one level.
    assert "history.back()" in body


def test_a_film_missing_its_voice_says_so_on_the_page(client):
    # A silent film with no explanation is what made "no Russian voice" take hours to
    # work out. The page must state what is missing and why.
    from dataclasses import replace as _replace

    from feed.store import read_presentation, save_presentation

    _prepare(client)
    conn = client.app.state.context.conn
    gid = conn.execute("SELECT id FROM groups WHERE kind='memory'").fetchone()["id"]
    stored = read_presentation(conn, gid, "en")
    save_presentation(conn, _replace(
        stored, error="no voice — the models service refused to speak ru"))
    body = client.get("/feed/memory-0").text
    assert "no voice" in body
    assert "refused to speak ru" in body


def test_a_complete_film_shows_no_missing_notice(client):
    _prepare(client)
    assert "feed-missing" not in client.get("/feed/memory-0").text


def test_the_waiting_screen_scatters_the_memorys_own_photographs(client):
    # Preparing a film is minutes, most of it one MusicGen call. A bar on a black page
    # is indistinguishable from a hung process, so the memory's own pictures drift
    # behind it (§11.1).
    body = client.get("/feed/memory-0").text
    assert "feed-scatter" in body
    assert body.count("feed-scatter-shot") >= 1
    # Positions are computed server-side and are therefore STABLE: the progress bar
    # polls every second, and a layout that re-randomised would flicker.
    assert body == client.get("/feed/memory-0").text


def test_the_waiting_screen_is_gone_once_the_film_is_ready(client):
    _prepare(client)
    assert "feed-scatter-shot" not in client.get("/feed/memory-0").text


def test_a_ready_film_offers_a_download(client):
    # The export records the canvas and the audio bus the player already has, so it is
    # only offered once the film exists (§11.2).
    _prepare(client)
    body = client.get("/feed/memory-0").text
    assert 'id="download"' in body
    # The wait is real time and the user has to know before starting a 50 s recording.
    assert "real" in body and "time" in body


def test_an_unprepared_film_offers_no_download(client):
    assert 'id="download"' not in client.get("/feed/memory-0").text


def test_a_stored_film_with_no_voice_is_not_played(client):
    # Rows written before narration became mandatory (§11.1) are silent, and some carry
    # an English script under a Russian film. Playing one is worse than saying it needs
    # preparing again.
    from feed.store import Presentation, read_presentation, save_presentation

    _prepare(client)
    conn = client.app.state.context.conn
    gid = conn.execute("SELECT id FROM groups WHERE kind='memory'").fetchone()["id"]
    stored = read_presentation(conn, gid, "en")
    save_presentation(conn, Presentation(
        group_id=gid, language="en", signature=stored.signature,
        timeline=stored.timeline, music_key=stored.music_key,
        narration_key=None, status="ready", error=None,
    ))
    body = client.get("/feed/memory-0").text
    assert "feed-gl.js" not in body           # no player
    assert 'id="download"' not in body        # and nothing to download
    assert "no voice" in body                 # it says why


def _notes_payload(html: str) -> list:
    """The notes the page hands the browser, parsed exactly as the browser parses it."""
    import json
    import re

    match = re.search(r'<script id="notes" type="application/json">(.*?)</script>',
                      html, re.DOTALL)
    assert match, "the page carries no notes payload at all"
    return json.loads(match.group(1))


def test_a_note_is_saved_against_the_moment_it_was_written(client):
    _prepare(client)
    saved = client.post("/feed/memory-0/note", data={"at": 12.5, "text": "это мой брат"})
    assert saved.status_code == 200
    body = saved.json()
    assert body["saved"] and body["stamp"] == "0:12" and body["text"] == "это мой брат"
    # And it comes back with the page as PARSEABLE JSON, so the panel can list it and
    # the player can float it. Asserting only that the text appears is what let an
    # autoescaped payload through: every quote became &#34;, `JSON.parse` threw, and
    # the panel silently listed nothing.
    page = client.get("/feed/memory-0").text
    assert "это мой брат" in page
    assert [n["text"] for n in _notes_payload(page)] == ["это мой брат"]


def test_an_empty_note_is_refused_rather_than_stored(client):
    _prepare(client)
    assert client.post("/feed/memory-0/note", data={"at": 1, "text": "  "}).status_code == 400


def test_a_note_can_be_rewritten_and_removed(client):
    _prepare(client)
    note = client.post("/feed/memory-0/note", data={"at": 3, "text": "first"}).json()
    edited = client.post(f"/feed/memory-0/note/{note['id']}",
                         data={"at": 3, "text": "second"}).json()
    assert edited["text"] == "second"
    # Editing to nothing deletes it — a scratchpad, not a form.
    gone = client.post(f"/feed/memory-0/note/{edited['id']}",
                       data={"at": 3, "text": ""}).json()
    assert gone["deleted"] and not gone["saved"]
    assert "second" not in client.get("/feed/memory-0").text


def test_notes_reach_the_next_generation(client):
    # The whole point: what the viewer said becomes part of the narration next time
    # (§11.3). The story pass must be SHOWN it.
    _prepare(client)
    client.post("/feed/memory-0/note", data={"at": 4, "text": "we got completely lost"})
    conn = client.app.state.context.conn
    gid = conn.execute("SELECT id FROM groups WHERE kind='memory'").fetchone()["id"]
    models = FakeModels()
    from feed.build import build_presentation

    build_presentation(conn, models, client.app.state.context.audio, 1, gid)
    prompt = models.story_prompts[0]
    assert "we got completely lost" in prompt
    assert "0:04" in prompt


def test_notes_show_on_the_memory_card_in_organize(client):
    # They were only visible inside the player, which means they could only be found
    # by playing the film again (§11.3).
    _prepare(client)
    client.post("/feed/memory-0/note", data={"at": 8, "text": "это мой брат Костя"})
    body = client.get("/organize?by=memories").text
    assert "это мой брат Костя" in body
    assert "0:08" in body


def test_notes_show_under_the_memory_title_on_a_photo(client):
    # A photo opened INSIDE a memory shows what the viewer said about that memory,
    # under the memory's own title and above the photograph's words.
    _prepare(client)
    client.post("/feed/memory-0/note", data={"at": 65, "text": "we got completely lost"})
    conn = client.app.state.context.conn
    pid = conn.execute("SELECT photo_id FROM group_photos LIMIT 1").fetchone()["photo_id"]
    body = client.get(f"/photo/{pid}?ctx=album:memories::memory-0").text
    assert "we got completely lost" in body
    assert "1:05" in body


def test_a_photo_outside_a_memory_shows_no_notes(client):
    _prepare(client)
    client.post("/feed/memory-0/note", data={"at": 3, "text": "only for the memory"})
    conn = client.app.state.context.conn
    pid = conn.execute("SELECT photo_id FROM group_photos LIMIT 1").fetchone()["photo_id"]
    assert "only for the memory" not in client.get(f"/photo/{pid}").text


def test_notes_are_offered_before_the_film_exists(client):
    # The Prepare screen is where notes are most worth writing — there is no film yet,
    # so they are the only thing that can steer the first cut (§11.3). While the panel
    # lived inside the player it simply was not there.
    body = client.get("/feed/memory-0").text          # never prepared
    assert 'id="notes-panel"' in body
    assert "feed-notes.js" in body
    assert "feed-gl.js" not in body                   # and no player to go with it


def test_notes_written_before_the_first_cut_survive_to_it(client):
    client.post("/feed/memory-0/note", data={"at": 0, "text": "start with the sea"})
    body = client.get("/feed/memory-0").text
    assert "start with the sea" in body
    _prepare(client)
    assert "start with the sea" in client.get("/feed/memory-0").text


def test_the_panel_starts_open_without_a_player_and_hidden_with_one(client):
    import re

    unprepared = client.get("/feed/memory-0").text
    assert re.search(r'id="notes-panel"\s*\n?\s*>', unprepared), "should not be hidden"
    _prepare(client)
    assert 'hidden' in client.get("/feed/memory-0").text


def test_every_note_reaches_the_panel_as_usable_json(client):
    # Several notes, in any language, with quotes and brackets in them — all of which
    # an HTML-escaped payload would break.
    for at, text in ((1.5, '[страшным голосом] это был вечер перед Хэллоуином'),
                     (7.2, 'выли кошки и собаки'),
                     (11.0, 'Юля готовилась очень тщательно'),
                     (14.8, 'she said "look at this" and laughed')):
        assert client.post("/feed/memory-0/note",
                           data={"at": at, "text": text}).status_code == 200
    payload = _notes_payload(client.get("/feed/memory-0").text)
    assert len(payload) == 4
    assert [n["stamp"] for n in payload] == ["0:01", "0:07", "0:11", "0:14"]
    assert 'she said "look at this" and laughed' in [n["text"] for n in payload]


def test_the_payload_cannot_close_its_own_script_tag(client):
    client.post("/feed/memory-0/note", data={"at": 1, "text": "</script><b>hi</b>"})
    html = client.get("/feed/memory-0").text
    payload = _notes_payload(html)
    assert payload[0]["text"] == "</script><b>hi</b>"
    assert "<b>hi</b>" not in html            # never as live markup


def test_only_close_and_cancel_are_offered_while_a_film_is_being_made(client):
    # A run has already read the notes and cannot act on an edit, and the language
    # pills would switch to a film nobody is building (§11.1).
    client.post("/feed/memory-0/note", data={"at": 2, "text": "before the build"})
    client.post("/feed/memory-0/prepare", data={"lang": "en"}, follow_redirects=False)
    body = client.get("/feed/memory-0").text
    assert "is-building" in body                 # the class the styling hangs off
    assert "feed-close" in body                  # close stays
    client.app.state.await_feed_build()


def test_the_controls_come_back_once_the_film_is_ready(client):
    _prepare(client)
    body = client.get("/feed/memory-0").text
    assert "is-building" not in body
    assert 'id="download"' in body and "feed-regen" in body


def test_the_notes_sheet_can_be_folded_out_of_the_way(client):
    # It sits over the memory's own description on the Prepare screen, and covering the
    # thing you are writing notes ABOUT is not a trade worth making (§11.3).
    body = client.get("/feed/memory-0").text
    assert 'id="notes-toggle"' in body
    assert 'id="notes-body"' in body
    assert 'aria-controls="notes-body"' in body


def test_the_sheet_starts_folded(client):
    # The first thing anyone wants on opening a film is to look at it, not to be handed
    # a form. The unfolded state is only remembered once the viewer asks for it.
    body = client.get("/feed/memory-0").text
    assert "stored !== '0'" not in body          # the rule lives in JS, not the page
    assert 'aria-expanded="true"' in body        # markup default; JS folds on load


def test_the_editor_is_hidden_only_while_the_film_plays(client):
    # Hidden during PLAYBACK; shown whenever the film is not running, title card
    # included — that is where you read what you already said before deciding to watch
    # or to prepare again (§11.3).
    _prepare(client)
    body = client.get("/feed/memory-0").text
    assert 'id="notes-panel"' in body
    player = pathlib.Path("web/static/feed-gl.js").read_text()
    assert "showPanel(false)" in player          # play() closes it
    assert player.count("showPanel(true)") == 2  # the title card, and a pause


def test_the_waiting_screen_scatters_the_notes_with_the_photographs(client):
    # The run is READING them (§11.3); a waiting screen showing only pictures gives no
    # sign of that.
    client.post("/feed/memory-0/note", data={"at": 3, "text": "это мой брат Костя"})
    body = client.get("/feed/memory-0").text
    assert "feed-scatter-note" in body
    assert "это мой брат Костя" in body


def test_the_scatter_appears_for_notes_even_with_no_thumbnails(client):
    conn = client.app.state.context.conn
    conn.execute("UPDATE photos SET thumb_key = NULL")
    client.post("/feed/memory-0/note", data={"at": 1, "text": "only words"})
    body = client.get("/feed/memory-0").text
    assert 'class="feed-scatter"' in body and "feed-scatter-note" in body


def test_the_scatter_is_gone_once_the_film_is_ready(client):
    client.post("/feed/memory-0/note", data={"at": 1, "text": "still here"})
    _prepare(client)
    body = client.get("/feed/memory-0").text
    assert "feed-scatter-note" not in body


def test_a_memory_can_be_retold_on_its_own(client):
    # Rebuilding EVERY memory re-clusters the library, so one you liked can come back
    # split or merged. This retells one and leaves the rest alone (§11).
    # The button lives on the MEMORY, not on the film screen: it changes the memory,
    # and every film made of it follows.
    grid = client.get("/organize?by=memories").text
    assert "/rebuild" in grid and "↻ Retell memory" in grid
    assert "/rebuild" not in client.get("/feed/memory-0").text

    conn = client.app.state.context.conn
    before = conn.execute("SELECT COUNT(*) c FROM groups WHERE kind='memory'").fetchone()["c"]
    client.post("/feed/memory-0/rebuild", data={"lang": "en"}, follow_redirects=False)
    thread = client.app.state.memories_build_thread
    if thread is not None:
        thread.join(timeout=10)
    after = conn.execute("SELECT COUNT(*) c FROM groups WHERE kind='memory'").fetchone()["c"]
    assert after == before


def test_retelling_a_memory_keeps_the_notes_written_about_it(client):
    # The viewer's own words survive the memory being retold — they are the input to
    # the next telling (§11.3). `replace_memories` would cascade them away.
    _prepare(client)
    client.post("/feed/memory-0/note", data={"at": 4, "text": "this is my brother"})
    client.post("/feed/memory-0/rebuild", data={"lang": "en"}, follow_redirects=False)
    thread = client.app.state.memories_build_thread
    if thread is not None:
        thread.join(timeout=10)
    assert "this is my brother" in client.get("/feed/memory-0").text


def test_retelling_a_memory_drops_its_film(client):
    # The film narrated a memory that no longer exists in that shape (§11.1).
    _prepare(client)
    conn = client.app.state.context.conn
    gid = conn.execute("SELECT id FROM groups WHERE kind='memory'").fetchone()["id"]
    from albums.memory_store import Memory, rewrite_memory

    rewrite_memory(conn, gid, Memory("Retold", "A new telling.", [2, 1], "sig"))
    left = conn.execute("SELECT COUNT(*) c FROM memory_presentations"
                        " WHERE group_id = ?", (gid,)).fetchone()["c"]
    assert left == 0


VOXCPM = cloning_entry_for("mac", "en")


class CloningModels(FakeModels):
    """FakeModels whose `speak` also takes saved-voice references, as the models service
    does (§5.1): it answers as the voice-copy model and reports, per reference, the
    conditioning it built from the take (D86)."""

    def __init__(self):
        super().__init__()
        self.references: list[dict | None] = []

    def speak(self, text, language="en", voice=None, segments=None, voices=None, *,
              references=None, **kw):
        self.references.append(references)
        reply = super().speak(text, language, voice, segments, voices)
        if references is None:
            return reply
        return reply | {
            "model": VOXCPM.key, "revision": VOXCPM.source.revision,
            "conditioning": {token: {"state": "built", "model": VOXCPM.key,
                                     "revision": VOXCPM.source.revision, "reason": None, "blob": None}
                             for token in references},
            "adapters": {token: {"state": "none", "sha256": None} for token in references},
        }


def _voiced_app(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Settings, "build_models_client", lambda self: CloningModels())
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  width=4000, height=3000, ai_title=f"caption {pid}",
                  shot_at=f"2025-07-12T1{pid}:00:00")
    replace_memories(conn, settings.owner_id,
                     [Memory("Beach day", "Warm afternoon.", [1, 2, 3], "sig")])
    return app


@pytest.fixture
def voiced(settings, monkeypatch):
    with TestClient(_voiced_app(settings, monkeypatch)) as test_client:
        yield test_client


@pytest.fixture
def voiced_jetson(settings, monkeypatch):
    settings.profile = "jetson"
    with TestClient(_voiced_app(settings, monkeypatch)) as test_client:
        yield test_client


def _ctx(client):
    return client.app.state.context


def _group(client) -> int:
    from albums.memory_store import group_id_for_album_key

    ctx = _ctx(client)
    return group_id_for_album_key(ctx.conn, ctx.settings.owner_id, "memory-0")


def _voice(client, name, *languages, freq=220.0, words: str | None = WORDS):
    """A saved voice with one take per language; `words` is what ASR heard in each (None =
    not heard yet), so it is ready for a language when they are usable (§11.1)."""
    ctx = _ctx(client)
    voice = sv.create_voice(ctx.conn, ctx.settings.owner_id, name)
    for i, language in enumerate(languages):
        added = sv.add_example(ctx.conn, ctx.voices, ctx.settings.owner_id, voice.id,
                               language=language, source="uploaded",
                               data=clip(6.0, freq=freq + 10 * i))
        if words is not None:
            sv.set_heard_words(ctx.conn, ctx.settings.owner_id, voice.id, added.id, text=words,
                               model=FAKE_ASR)
    return sv.get_voice(ctx.conn, ctx.settings.owner_id, voice.id)


def _select(body: str) -> str | None:
    found = re.search(r'<select name="narrator">.*?</select>', body, re.DOTALL)
    return found.group(0) if found else None


AUTOMATIC = "Automatic — the app picks a voice for this memory"
SAVED = "Saved voices"
PRESETS = "Preset voices"


def _disabled(body: str) -> list[tuple[str, str]]:
    """(value, label) of each disabled saved-voice option: a voice that cannot narrate."""
    return re.findall(r'<option value="(\d+)" disabled>([^<]*)</option>', _select(body) or "")


def _options(body: str, group: str | None = None) -> list[tuple[str, str, bool]]:
    """The options of the narrator select; with `group`, Automatic plus that group only."""
    select = _select(body) or ""
    if group is not None:
        automatic = re.search(r'<option value=""[^>]*>[^<]*</option>', select).group(0)
        found = re.search(rf'<optgroup label="{group}">(.*?)</optgroup>', select, re.DOTALL)
        select = automatic + (found.group(1) if found else "")
    return [(value, label.strip(), bool(selected))
            for value, selected, label in re.findall(
                r'<option value="([^"]*)" ?(selected)?\s*>([^<]*)</option>', select)]


def _no_build_started(client) -> bool:
    return (client.app.state.feed_build_thread is None
            and client.app.state.feed_building is None
            and client.app.state.feed_queued is None)


def test_the_picker_offers_the_voices_ready_for_the_language(voiced):
    english = _voice(voiced, "Grandma", "en")
    russian = _voice(voiced, "Babushka", "ru", freq=300.0)
    pending = _voice(voiced, "Leaving", "en", freq=400.0)
    sv.request_voice_deletion(_ctx(voiced).conn, 1, pending.id)
    options = _options(voiced.get("/feed/memory-0?lang=en").text, SAVED)
    assert options == [("", AUTOMATIC, True), (str(english.id), "Grandma", False)]
    options = _options(voiced.get("/feed/memory-0?lang=ru").text, SAVED)
    assert options == [("", AUTOMATIC, True), (str(russian.id), "Babushka", False)]


def test_every_other_saved_voice_is_a_disabled_option_with_its_reason(voiced):
    waiting = _voice(voiced, "Anna", "en", words=None)
    silent = _voice(voiced, "Boris", "en", freq=300.0, words="♪")
    russian = _voice(voiced, "Clara", "ru", freq=400.0)
    nothing = _voice(voiced, "Dora")
    pending = _voice(voiced, "Eva", "en", freq=500.0, words=None)
    sv.request_voice_deletion(_ctx(voiced).conn, 1, pending.id)
    body = voiced.get("/feed/memory-0?lang=en").text
    assert _disabled(body) == [
        (str(waiting.id), "«Anna» — words not heard yet (open it under Voices)"),
        (str(silent.id), "«Boris» — no English take with usable words"),
        (str(russian.id), "«Clara» — no English take"),
        (str(nothing.id), "«Dora» — no English take"),
    ]
    assert _options(body, SAVED) == [("", AUTOMATIC, True)]   # none can be chosen
    assert "needs a English take" not in body


def test_a_waiting_voice_reads_hearing_its_words_only_while_the_hearing_job_hears_it(
        voiced, monkeypatch):
    heard_now = _voice(voiced, "Anna", "en", words=None)
    left = _voice(voiced, "Boris", "en", freq=300.0, words=None)
    monkeypatch.setattr(voiced.app.state.voice_hearings, "hears",
                        lambda voice_id: voice_id == heard_now.id)
    assert _disabled(voiced.get("/feed/memory-0?lang=en").text) == [
        (str(heard_now.id), "«Anna» — hearing its words"),
        (str(left.id), "«Boris» — words not heard yet (open it under Voices)"),
    ]


def test_the_picker_shows_presets_and_no_saved_group_with_no_saved_voice(voiced):
    from feed.voice import offered

    body = voiced.get("/feed/memory-0?lang=en").text
    assert _options(body)[0] == ("", AUTOMATIC, True)
    assert _options(body, PRESETS)[1:] == [
        (f"preset:{s.name}", f"{s.name} — {s.gender}, {s.register}: {s.note}", False)
        for s in offered("en")]
    assert f'<optgroup label="{SAVED}">' not in body
    assert 'Add a voice under <a href="/voices">Voices</a>.' in body
    assert 'href="/organize/voices"' not in body
    russian = voiced.get("/feed/memory-0?lang=ru").text
    assert [value for value, _, _ in _options(russian, PRESETS)[1:]] == [
        f"preset:{s.name}" for s in offered("ru")]


def test_the_picker_on_jetson_offers_presets_and_says_voice_copy_is_not_available(voiced_jetson):
    from feed.voice import offered

    _voice(voiced_jetson, "Grandma", "en")
    body = voiced_jetson.get("/feed/memory-0?lang=en").text
    form = body[body.index('<form method="post" action="/feed/memory-0/prepare">'):]
    assert form.index('name="narrator"') < form.index("</form>")
    assert ("", AUTOMATIC, True) in _options(body)
    assert _options(body, PRESETS)[1:] == [
        (f"preset:{s.name}", f"{s.name} — {s.gender}, {s.register}: {s.note}", False)
        for s in offered("en")]
    assert _options(body, SAVED) == [("", AUTOMATIC, True)]
    assert "<option disabled>Voice copy not available on this device</option>" in body
    assert "Add a voice" not in body


def test_prepare_stores_the_choice_and_regenerate_preselects_it(voiced):
    voice = _voice(voiced, "Grandma", "en")
    response = voiced.post("/feed/memory-0/prepare",
                           data={"lang": "en", "narrator": str(voice.id)},
                           follow_redirects=False)
    assert response.status_code == 303
    voiced.app.state.await_feed_build()
    assert sv.read_narrator(_ctx(voiced).conn, _group(voiced), "en") == sv.NarratorChoice(voice.id)
    body = voiced.get("/feed/memory-0?lang=en").text
    regen = body[body.index('class="feed-regen"'):]
    assert _options(regen, SAVED) == [("", AUTOMATIC, False), (str(voice.id), "Grandma", True)]


def test_an_empty_narrator_clears_the_choice(voiced):
    voice = _voice(voiced, "Grandma", "en")
    sv.set_narrator(_ctx(voiced).conn, _group(voiced), "en", voice.id)
    voiced.post("/feed/memory-0/prepare", data={"lang": "en", "narrator": ""},
                follow_redirects=False)
    voiced.app.state.await_feed_build()
    assert sv.read_narrator(_ctx(voiced).conn, _group(voiced), "en") is None


def test_a_post_without_the_field_keeps_the_stored_choice(voiced):
    voice = _voice(voiced, "Grandma", "en")
    sv.set_narrator(_ctx(voiced).conn, _group(voiced), "en", voice.id)
    voiced.post("/feed/memory-0/prepare", data={"lang": "en"}, follow_redirects=False)
    voiced.app.state.await_feed_build()
    assert sv.read_narrator(_ctx(voiced).conn, _group(voiced), "en") == sv.NarratorChoice(voice.id)


@pytest.mark.parametrize("narrator", ["999", "abc", "-1", "²"])
def test_an_unknown_narrator_is_404_and_starts_no_build(voiced, narrator):
    response = voiced.post("/feed/memory-0/prepare",
                           data={"lang": "en", "narrator": narrator}, follow_redirects=False)
    assert response.status_code == 404
    assert _no_build_started(voiced)


def test_a_foreign_narrator_is_404(voiced):
    ctx = _ctx(voiced)
    other = sv.create_voice(ctx.conn, 2, "Theirs")
    sv.add_example(ctx.conn, ctx.voices, 2, other.id, language="en", source="uploaded",
                   data=clip())
    response = voiced.post("/feed/memory-0/prepare",
                           data={"lang": "en", "narrator": str(other.id)},
                           follow_redirects=False)
    assert response.status_code == 404
    assert _no_build_started(voiced)


@pytest.mark.parametrize(("words", "languages", "detail"), [
    (None, ("en",), ("The words of «Babushka»'s English takes are not heard yet. Open «Babushka» "
                     "under Voices and let it hear them, or choose another narrator.")),
    ("♪", ("en",), ("No English take of «Babushka» has usable words. Record a take where you "
                    "speak, or choose another narrator.")),
    (WORDS, ("ru",), ("«Babushka» has no English take yet. Record one under Voices, or choose "
                      "another narrator.")),
])
def test_a_narrator_that_is_not_ready_for_the_language_is_409(voiced, words, languages, detail):
    voice = _voice(voiced, "Babushka", *languages, words=words)
    response = voiced.post("/feed/memory-0/prepare",
                           data={"lang": "en", "narrator": str(voice.id)},
                           follow_redirects=False)
    assert response.status_code == 409
    assert response.json()["detail"] == detail
    assert _no_build_started(voiced)
    assert sv.read_narrator(_ctx(voiced).conn, _group(voiced), "en") is None


def test_a_narrator_on_jetson_is_409(voiced_jetson):
    voice = _voice(voiced_jetson, "Grandma", "en")
    response = voiced_jetson.post("/feed/memory-0/prepare",
                                  data={"lang": "en", "narrator": str(voice.id)},
                                  follow_redirects=False)
    assert response.status_code == 409
    assert "voice copy is not available on this device" in response.json()["detail"]
    assert _no_build_started(voiced_jetson)


def test_a_stored_narrator_being_deleted_stays_selected_and_is_refused(voiced):
    voice = _voice(voiced, "Grandma", "en")
    sv.set_narrator(_ctx(voiced).conn, _group(voiced), "en", voice.id)
    sv.request_voice_deletion(_ctx(voiced).conn, 1, voice.id)
    options = _options(voiced.get("/feed/memory-0?lang=en").text, SAVED)
    assert options == [("", AUTOMATIC, False), (str(voice.id), "«Grandma» — being deleted", True)]
    response = voiced.post("/feed/memory-0/prepare",
                           data={"lang": "en", "narrator": str(voice.id)},
                           follow_redirects=False)
    assert response.status_code == 409
    assert response.json()["detail"] == (
        "The saved voice «Grandma» is being deleted. Choose another narrator.")
    assert _no_build_started(voiced)


def test_a_stored_narrator_that_was_deleted_stays_selected_and_is_refused(voiced):
    ctx = _ctx(voiced)
    voice = _voice(voiced, "Grandma", "en")
    sv.set_narrator(ctx.conn, _group(voiced), "en", voice.id)
    sv.request_voice_deletion(ctx.conn, 1, voice.id)
    sv.process_voice_deletions(ctx.conn, ctx.voices, 1)
    assert sv.read_narrator(ctx.conn, _group(voiced), "en") == sv.NarratorChoice(None)
    options = _options(voiced.get("/feed/memory-0?lang=en").text, SAVED)
    assert options == [("", AUTOMATIC, False), ("deleted", "A deleted saved voice", True)]
    response = voiced.post("/feed/memory-0/prepare",
                           data={"lang": "en", "narrator": "deleted"}, follow_redirects=False)
    assert response.status_code == 409
    assert response.json()["detail"] == (
        "The saved voice chosen for this film was deleted. Choose another narrator.")
    assert _no_build_started(voiced)


@pytest.mark.parametrize(("languages", "words", "label"), [
    (("ru",), None, "«Grandma» — words not heard yet (open it under Voices)"),
    (("ru",), "♪", "«Grandma» — no Russian take with usable words"),
    (("en",), WORDS, "«Grandma» — no Russian take"),
])
def test_a_stored_narrator_that_cannot_narrate_stays_selected_and_links_to_its_voice(
        voiced, languages, words, label):
    voice = _voice(voiced, "Grandma", *languages, words=words)
    sv.set_narrator(_ctx(voiced).conn, _group(voiced), "ru", voice.id)
    body = voiced.get("/feed/memory-0?lang=ru").text
    assert _options(body, SAVED) == [("", AUTOMATIC, False), (str(voice.id), label, True)]
    assert _disabled(body) == []                     # once, as the selected option
    assert (f'<p class="feed-note">«Grandma» needs a Russian take with heard words for Russian films: '
            f'<a href="/voices/{voice.id}">open «Grandma» in Voices</a>.</p>') in body
    response = voiced.post("/feed/memory-0/prepare",
                           data={"lang": "ru", "narrator": str(voice.id)},
                           follow_redirects=False)
    assert response.status_code == 409
    assert _no_build_started(voiced)


def test_a_stored_narrator_on_jetson_shows_not_available(voiced_jetson):
    voice = _voice(voiced_jetson, "Grandma", "en")
    sv.set_narrator(_ctx(voiced_jetson).conn, _group(voiced_jetson), "en", voice.id)
    options = _options(voiced_jetson.get("/feed/memory-0?lang=en").text, SAVED)
    assert options == [
        ("", AUTOMATIC, False),
        (str(voice.id), "«Grandma» — voice copy is not available on this device", True),
    ]


def test_the_picker_sits_in_both_forms(voiced):
    _voice(voiced, "Grandma", "en")
    prepare = voiced.get("/feed/memory-0?lang=en").text
    form = prepare[prepare.index('<form method="post" action="/feed/memory-0/prepare">'):]
    assert form.index('name="narrator"') < form.index("</form>")
    voiced.post("/feed/memory-0/prepare", data={"lang": "en"}, follow_redirects=False)
    voiced.app.state.await_feed_build()
    ready = voiced.get("/feed/memory-0?lang=en").text
    regen = ready[ready.index('class="feed-regen"'):]
    assert regen.index('name="narrator"') < regen.index("</form>")


def test_a_refused_prepare_shows_its_reason_and_stays(voiced):
    # The submit script: a 4xx reply shows its `detail` under the form; others reload.
    body = voiced.get("/feed/memory-0").text
    script = body[body.index("form[action$=\"/prepare\"]"):]
    assert "response.status >= 400 && response.status < 500" in script
    assert "body.detail" in script and "form.after(line)" in script


def test_a_finished_saved_voice_film_plays_after_its_voice_is_deleted(voiced):
    voice = _voice(voiced, "Grandma", "en")
    voiced.post("/feed/memory-0/prepare", data={"lang": "en", "narrator": str(voice.id)},
                follow_redirects=False)
    voiced.app.state.await_feed_build()
    assert voiced.post(f"/voices/{voice.id}/delete").status_code == 200
    voiced.app.state.await_voice_drain()
    assert sv.get_voice(_ctx(voiced).conn, 1, voice.id) is None
    response = voiced.get("/feed/memory-0/audio/narration?lang=en")
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"


def test_a_saved_voice_film_stores_no_conditioning(voiced):
    # A film sends the stored conditionings of its take and stores no new one (§6), so the
    # build leaves no row and no file for the drain.
    ctx = _ctx(voiced)
    voice = _voice(voiced, "Grandma", "en")
    sv.set_narrator(ctx.conn, _group(voiced), "en", voice.id)
    voiced.post("/feed/memory-0/prepare", data={"lang": "en"}, follow_redirects=False)
    voiced.app.state.await_feed_build()
    voiced.app.state.await_voice_drain()
    from feed.store import read_presentation

    assert read_presentation(ctx.conn, _group(voiced), "en").status == "ready"
    assert ctx.conn.execute("SELECT COUNT(*) FROM voice_conditionings").fetchone()[0] == 0
    assert ctx.conn.execute("SELECT COUNT(*) FROM voice_file_deletions").fetchone()[0] == 0


def test_a_film_build_wakes_the_voice_drain(voiced):
    # `_start_film`'s finally: a key left in the outbox drains after the build, with no
    # delete and no restart.
    ctx = _ctx(voiced)
    ctx.voices.write("orphan.cond", b"left behind")
    ctx.conn.execute("INSERT INTO voice_file_deletions(storage_key, deletion_id, requested_at)"
                     " VALUES ('orphan.cond', NULL, 'now')")
    voiced.post("/feed/memory-0/prepare", data={"lang": "en"}, follow_redirects=False)
    voiced.app.state.await_feed_build()
    voiced.app.state.await_voice_drain()
    assert not ctx.voices.exists("orphan.cond")


def test_the_language_selector_marks_each_ready_film(client):
    body = client.get("/feed/memory-0").text
    assert "Film language" in body
    assert "English ✓" not in body and "Русский ✓" not in body
    _prepare(client)
    body = client.get("/feed/memory-0").text
    assert "English ✓" in body
    assert "Русский ✓" not in body and "Русский" in body
    client.app.state.feed_building = ("memory-0", "en")
    assert "English ✓" not in client.get("/feed/memory-0?lang=ru").text
    client.app.state.feed_building = None


@pytest.mark.parametrize(("cast", "expected"), [
    ({"narrator_used": "af_bella", "narrator_chosen_by": "user"},
     "This film: English · narrated by af_bella (female, bright) — your choice"),
    ({"narrator_used": "af_bella", "narrator_chosen_by": "story", "narrator_asked_was_real": True},
     "This film: English · narrated by af_bella (female, bright) — picked automatically by the story"),
    ({"narrator_used": "af_bella", "narrator_chosen_by": "mood"},
     ("This film: English · narrated by af_bella (female, bright) — "
      "picked automatically using the memory's mood")),
    ({"narrator_used": "af_bella"},
     "This film: English · narrated by af_bella (female, bright) — picked automatically"),
])
def test_this_film_says_who_narrates_it_and_who_chose(client, cast, expected):
    _film(client, decisions=_cast(**cast))
    body = client.get("/feed/memory-0").text
    assert f'<p class="feed-this-film">{expected}</p>' in body


def test_a_film_without_a_recorded_voice_says_so(client):
    _film(client, decisions={})
    assert "This film: English · Voice not recorded" in client.get("/feed/memory-0").text


def test_other_voices_lists_each_passage_spoken_by_another_voice(client):
    _film(client, decisions=_cast(narrator_used="af_bella", passages=[
        {"passage": 0, "asked": "", "used": "af_bella"},
        {"passage": 2, "asked": "low man's voice", "used": "am_adam"},
        {"passage": 4, "asked": "bm_george", "used": "bm_george"},
    ]))
    body = client.get("/feed/memory-0").text
    assert ("Other voices: passage 3 → am_adam (male, neutral); "
            "passage 5 → bm_george (male, deep)</p>") in body


def test_a_rejected_passage_cast_is_not_listed(client):
    _film(client, decisions=_cast(narrator_used="af_bella", passages=[
        {"passage": 1, "asked": "count_dracula", "used": "af_bella"}]))
    body = client.get("/feed/memory-0").text
    assert "Other voices" not in body
    assert "count_dracula" not in body


def test_after_an_automatic_film_automatic_stays_selected(client):
    body = client.get("/feed/memory-0").text
    assert "narrated by" not in body                     # nothing is predicted
    _prepare(client)
    body = client.get("/feed/memory-0").text
    regen = body[body.index('class="feed-regen"'):]
    assert _options(regen)[0] == ("", AUTOMATIC, True)
    assert "narrated by" in regen


AUTOMATIC_BRACKET = ("A [voice] note can request another voice for one passage: another preset "
                     "when a preset narrates, another saved voice when a saved voice narrates.")


def test_the_picker_hints_how_the_voice_is_chosen(client):
    body = client.get("/feed/memory-0").text
    assert ("The story picks the voice: a preset voice or one of your saved voices ready for "
            "English. Otherwise the memory's mood picks a preset voice.") in body
    assert AUTOMATIC_BRACKET in body
    assert "never picked" not in body


def test_on_jetson_the_hint_says_the_story_picks_a_preset_voice(voiced_jetson):
    body = voiced_jetson.get("/feed/memory-0?lang=en").text
    assert ("The story picks a preset voice; otherwise the memory's mood guides "
            "the choice.") in body
    assert "one of your saved voices" not in body
    assert AUTOMATIC_BRACKET in body
    assert "never picked" not in body


def test_the_bracket_line_follows_the_stored_narrator(voiced):
    conn = _ctx(voiced).conn
    sv.set_narrator(conn, _group(voiced), "en", preset_speaker="af_bella")
    body = voiced.get("/feed/memory-0?lang=en").text
    assert "The chosen narrator reads the whole film." in body
    assert AUTOMATIC_BRACKET not in body
    voice = _voice(voiced, "Grandma", "en")
    sv.set_narrator(conn, _group(voiced), "en", voice.id)
    body = voiced.get("/feed/memory-0?lang=en").text
    assert "The chosen narrator reads the whole film." in body
    assert AUTOMATIC_BRACKET not in body


def test_prepare_with_a_preset_stores_it_and_regenerate_preselects_it(voiced):
    response = voiced.post("/feed/memory-0/prepare",
                           data={"lang": "en", "narrator": "preset:af_bella"},
                           follow_redirects=False)
    assert response.status_code == 303
    voiced.app.state.await_feed_build()
    assert sv.read_narrator(_ctx(voiced).conn, _group(voiced), "en") == sv.NarratorChoice(None, "af_bella")
    body = voiced.get("/feed/memory-0?lang=en").text
    regen = body[body.index('class="feed-regen"'):]
    assert ("preset:af_bella", "af_bella — female, bright: " + _note("en", "af_bella"), True) in _options(regen)


def test_a_preset_name_is_stored_as_the_catalog_names_it(voiced):
    voiced.post("/feed/memory-0/prepare", data={"lang": "en", "narrator": "preset:AF_BELLA"},
                follow_redirects=False)
    voiced.app.state.await_feed_build()
    assert sv.read_narrator(_ctx(voiced).conn, _group(voiced), "en") == sv.NarratorChoice(None, "af_bella")


def test_a_preset_of_another_language_is_409_and_starts_no_build(voiced):
    response = voiced.post("/feed/memory-0/prepare",
                           data={"lang": "en", "narrator": "preset:xenia"}, follow_redirects=False)
    assert response.status_code == 409
    assert response.json()["detail"] == "«xenia» cannot narrate in English. Choose another narrator."
    assert _no_build_started(voiced)
    assert sv.read_narrator(_ctx(voiced).conn, _group(voiced), "en") is None


@pytest.mark.parametrize("narrator", ["preset:nobody", "preset:", "preset:  "])
def test_an_unknown_preset_is_404_and_starts_no_build(voiced, narrator):
    response = voiced.post("/feed/memory-0/prepare",
                           data={"lang": "en", "narrator": narrator}, follow_redirects=False)
    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown preset voice."
    assert _no_build_started(voiced)


def test_a_preset_replaces_a_saved_choice_and_back(voiced):
    conn = _ctx(voiced).conn
    voice = _voice(voiced, "Grandma", "en")
    sv.set_narrator(conn, _group(voiced), "en", voice.id)
    voiced.post("/feed/memory-0/prepare", data={"lang": "en", "narrator": "preset:af_bella"},
                follow_redirects=False)
    voiced.app.state.await_feed_build()
    assert sv.read_narrator(conn, _group(voiced), "en") == sv.NarratorChoice(None, "af_bella")
    voiced.post("/feed/memory-0/prepare", data={"lang": "en", "narrator": str(voice.id)},
                follow_redirects=False)
    voiced.app.state.await_feed_build()
    assert sv.read_narrator(conn, _group(voiced), "en") == sv.NarratorChoice(voice.id, None)


def test_a_stored_preset_the_catalog_rejects_stays_selected(voiced):
    sv.set_narrator(_ctx(voiced).conn, _group(voiced), "en", preset_speaker="xenia")
    options = _options(voiced.get("/feed/memory-0?lang=en").text, PRESETS)
    assert options[0] == ("", AUTOMATIC, False)
    assert options[-1] == ("preset:xenia", "«xenia» — not offered for English films", True)
    assert [selected for _, _, selected in options].count(True) == 1


def _note(language: str, name: str) -> str:
    from feed.voice import offered

    return next(s.note for s in offered(language) if s.name == name)


def _between(body: str, start: str, end: str) -> str:
    i = body.index(start)
    return body[i:body.index(end, i)]


def test_the_ready_screen_puts_every_control_in_one_header(client):
    _prepare(client)
    body = client.get("/feed/memory-0").text
    assert body.count('<header class="feed-head">') == 1
    head = _between(body, '<header class="feed-head">', "</header>")
    actions = _between(head, '<div class="feed-actions">', '<div class="feed-settings">')
    assert (actions.index('class="feed-lang"') < actions.index('id="download"')
            < actions.index('class="feed-close"'))
    settings = head[head.index('<div class="feed-settings">'):]
    regen = _between(settings, 'class="feed-regen"', "</form>")
    assert regen.index("This film:") < regen.index('<select name="narrator">')
    assert "The story picks the voice" in regen and "A [voice] note" in regen
    assert "<details" not in settings
    for kept in ('id="download"', 'id="stage"', 'id="start"', 'id="notes-panel"', "location.replace"):
        assert kept in body


@pytest.mark.parametrize("building", [False, True])
def test_the_prepare_and_building_screens_have_the_header_without_settings(client, building):
    if building:
        client.app.state.feed_building = ("memory-0", "en")
    body = client.get("/feed/memory-0").text
    head = _between(body, '<header class="feed-head">', "</header>")
    assert 'class="feed-lang"' in head and 'class="feed-close"' in head
    assert "feed-settings" not in body
    client.app.state.feed_building = None


def test_no_control_is_fixed_apart_from_the_header():
    css = pathlib.Path("web/static/feed.css").read_text()
    for control in (".feed-lang", ".feed-regen", ".feed-download", ".feed-close"):
        rule = re.search(rf"^{re.escape(control)} \{{(.*?)\}}", css, re.DOTALL | re.MULTILINE)
        assert rule, control
        assert "position" not in rule.group(1), control
    head = re.search(r"^\.feed-head \{(.*?)\}", css, re.DOTALL | re.MULTILINE).group(1)
    assert "position: fixed" in head and "pointer-events: none" in head


def test_the_notes_body_is_the_panels_one_scroller():
    css = pathlib.Path("web/static/feed.css").read_text()

    def rule(selector):
        return re.search(rf"^{re.escape(selector)} \{{(.*?)\}}", css,
                         re.DOTALL | re.MULTILINE).group(1)

    panel, body = rule(".feed-notes"), rule(".feed-notes-body")
    assert all(line in panel for line in ("display: flex;", "flex-direction: column;",
                                          "overflow: hidden;"))
    assert "flex: none;" in rule(".feed-notes-title")
    assert all(line in body for line in ("flex: 1 1 auto;", "min-height: 0;",
                                         "overflow-y: auto;", "max-height: 60vh;",
                                         "scrollbar-width: thin;"))
    assert "overflow: hidden" not in body
    assert ".feed-notes-body::-webkit-scrollbar-thumb" in css
    folded = rule(".feed-notes.is-collapsed .feed-notes-body")
    assert "overflow: hidden;" in folded and "max-height: 0;" in folded


def test_a_voice_whose_newest_take_is_deleted_narrates_from_the_one_before(voiced):
    ctx = _ctx(voiced)
    voice = _voice(voiced, "Grandma", "en", "en")
    sv.request_example_deletion(ctx.conn, 1, voice.id, voice.examples[1].id)
    assert _options(voiced.get("/feed/memory-0?lang=en").text, SAVED) == [
        ("", AUTOMATIC, True), (str(voice.id), "Grandma", False)]
    sv.process_voice_deletions(ctx.conn, ctx.voices, 1)
    response = voiced.post("/feed/memory-0/prepare",
                           data={"lang": "en", "narrator": str(voice.id)},
                           follow_redirects=False)
    assert response.status_code == 303


@pytest.mark.parametrize(("cast", "decisions", "tail"), [
    ({"narrator_name": "me", "narrator_label": "me (saved voice)", "narrator_model": "spoken from take #12"},
     {"saved": True, "takes": {"saved:1": {"example_id": 12}}}, " · spoken from take #12"),
    ({"narrator_name": "me", "narrator_label": "me (saved voice)", "narrator_model": "voice model v2"},
     {"saved": True, "models": {"saved:1": {"version": 2}}}, " · voice model v2"),
    ({"narrator_name": "me", "narrator_label": "me (saved voice)"}, {"saved": True},
     " · made from one take, before voice models"),
    ({}, {}, ""),
])
def test_this_film_names_the_take_or_the_voice_model_that_narrated_it(client, cast, decisions, tail):
    used = {"narrator_used": "saved:1" if cast else "af_bella", "narrator_chosen_by": "user"}
    _film(client, decisions=_cast(**used, **cast) | {"voice": decisions})
    body = client.get("/feed/memory-0").text
    label = "me (saved voice)" if cast else "af_bella (female, bright)"
    assert (f'<p class="feed-this-film">This film: English · narrated by {label} — your choice'
            f"{tail}</p>") in body


def test_a_film_waits_for_the_voice_model_and_then_finishes(client):
    import time

    lock = client.app.state.voice_calls
    lock.acquire()
    try:
        client.post("/feed/memory-0/prepare", follow_redirects=False)
        deadline = time.monotonic() + 30
        body = ""
        while "waiting for the voice model" not in body and time.monotonic() < deadline:
            time.sleep(0.1)
            body = client.get("/feed/memory-0/prepare/progress").text
        assert "waiting for the voice model" in body
    finally:
        lock.release()
    client.app.state.await_feed_build()
    from feed.store import read_presentation

    conn = client.app.state.context.conn
    gid = conn.execute("SELECT id FROM groups WHERE kind='memory'").fetchone()["id"]
    assert read_presentation(conn, gid, "en").status == "ready"
    assert not lock.locked()


@pytest.mark.parametrize("fixture", ["voiced", "voiced_jetson"])
def test_a_prepare_passes_the_app_profile_to_the_build(fixture, request, monkeypatch):
    test_client = request.getfixturevalue(fixture)
    seen: list[dict] = []
    monkeypatch.setattr("web.app.build_presentation",
                        lambda *args, **kwargs: seen.append(kwargs))
    test_client.post("/feed/memory-0/prepare", data={"lang": "en"}, follow_redirects=False)
    test_client.app.state.await_feed_build()
    assert len(seen) == 1
    assert seen[0]["profile"] == _ctx(test_client).settings.profile
    assert seen[0]["profile"] == ("jetson" if fixture == "voiced_jetson" else "mac")


FILM_TOKEN = "a" * 32
PASSAGES = ("A warm afternoon by the water.",
            "We swam until the light went off the surface.",
            "Then we walked home the long way round.")
STALE = "Not saved: this narration changed in another window, or a new film replaced it."
DRAFT_TITLE = "Speak your edited narration, and score this memory again"
FILM_LINE = ("The film's narration is in this list with your notes. Change its words, and "
             "↻ Regenerate speaks them as written. With no change, ↻ Regenerate writes a new "
             "story.")
CHANGED_LINE = "Your edited narration. ↻ Regenerate speaks it as written."
DISCLOSURE = ("Notes cannot rewrite these words or provide a new voice or music interpretation. "
              "Existing voice choices still pass current note gating and validation. Effect "
              "instructions reach the conductor; matched speech fixes shot timing.")
LEGACY_LINE = ("This older film has no timed narration text. ↻ Regenerate makes a new film "
               "whose narration you can edit here.")
PICKER_TWIN = ("With your edited narration, a [voice] note cannot ask for a new voice: it keeps "
               "the voice the story already gave the passage it points at.")
LEGEND_TWIN = ("With your edited narration, a note's words are not narrated and cannot change "
               "its words, voices or music. Effect brackets still reach the conductor; matched "
               "speech fixes shot timing.")
NOT_STARTED = "Not started: an edit to the narration is not saved. See Tell the story better."


def _narrated(client, language="en", *, texts=PASSAGES, token=FILM_TOKEN) -> None:  # pragma: allowlist secret
    """A ready film whose timeline carries its passages and script, as a build writes them."""
    from feed.store import Presentation, save_presentation

    passages = [{"n": i, "text": text, "start": 4.0 * i, "end": 4.0 * i + 3.5, "photo": i,
                 "photo_id": i + 1, "voice_asked": "", "voice_used": "af_heart"}
                for i, text in enumerate(texts)]
    script = {"token": token, "narrator_asked": "", "narrator_used": "af_heart", "cards": [],
              "score_plan": [], "music_prompt": "warm ambient drone"}
    save_presentation(client.app.state.context.conn, Presentation(
        group_id=_memory_group(client), language=language, signature="sig",
        timeline={"shots": [], "passages": passages, "script": script}, music_key=None,
        narration_key="n.wav", status="ready", decisions={}))


def _narration_payload(html: str) -> dict:
    import json

    match = re.search(r'<script id="narration-data" type="application/json">(.*?)</script>',
                      html, re.DOTALL)
    assert match, "the page carries no narration payload"
    return json.loads(match.group(1))


def _shown_states(html: str) -> list[str]:
    """The state lines the page shows before any script runs."""
    return [state for state, hidden in re.findall(
        r'<p class="feed-narration-state" data-state="([\w-]+)"\s*(hidden)?\s*>', html)
        if not hidden]


def _shown_twins(html: str) -> list[str]:
    return [kind for kind, hidden in re.findall(
        r'data-narration="(draft|no-draft)"\s*(hidden)?\s*>', html) if not hidden]


def _change(client, n, text, rows: dict, *, lang="en", route=""):
    return client.post(f"/feed/memory-0/narration/{n}{route}", data={
        "lang": lang, "source": rows["source"] or "", "draft": rows["draft"] or "",
        "revision": rows["revision"], "text": text})


def _film_rows(client, lang="en") -> dict:
    return client.get(f"/feed/memory-0/narration?lang={lang}").json()


def test_a_prepared_film_lists_its_narration_as_rows(client):
    from feed.store import read_presentation

    _prepare(client)
    passages = read_presentation(client.app.state.context.conn, _memory_group(client),
                                 "en").timeline["passages"]
    page = client.get("/feed/memory-0").text
    payload = _narration_payload(page)
    assert payload["state"] == "film"
    assert [row["text"] for row in payload["rows"]] == [p["text"] for p in passages]
    assert [row["stamp"] for row in payload["rows"]] == [
        "–:––" if p["start"] is None else f"{int(p['start']) // 60}:{int(p['start']) % 60:02d}"
        for p in passages]
    assert _shown_states(page) == ["film"]
    assert FILM_LINE in page


def test_the_ready_page_keeps_every_id_unique(client):
    from collections import Counter

    _narrated(client)
    page = client.get("/feed/memory-0").text
    ids = Counter(re.findall(r'\sid="([^"]+)"', page))
    assert ids["narration"] == 1 and ids["narration-data"] == 1
    assert '<audio id="narration"' in page
    assert [name for name, seen in ids.items() if seen > 1] == []


def test_a_film_without_passages_reads_as_legacy(client):
    _film(client)
    page = client.get("/feed/memory-0").text
    assert _narration_payload(page)["state"] == "legacy"
    assert _shown_states(page) == ["legacy"]
    assert LEGACY_LINE in page


def test_no_film_shows_no_state_line(client):
    page = client.get("/feed/memory-0").text
    assert _narration_payload(page) == {"state": "none", "source": None, "draft": None,
                                        "revision": 0, "seekable": False, "changed": False,
                                        "rows": []}
    assert _shown_states(page) == []


def test_a_first_change_seeds_a_draft_and_the_page_says_so(client):
    _narrated(client)
    reply = _change(client, 1, "We swam  until\nthe sun went down.", _film_rows(client))
    assert reply.status_code == 200
    body = reply.json()
    assert body["saved"] and body["state"] == "draft" and body["revision"] == 1
    assert re.fullmatch(r"[0-9a-f]{32}", body["draft"])
    assert body["changed"] and body["source"] == FILM_TOKEN
    assert [row["text"] for row in body["rows"]] == [
        PASSAGES[0], "We swam until the sun went down.", PASSAGES[2]]

    page = client.get("/feed/memory-0").text
    assert _shown_states(page) == ["draft-changed", "draft"]
    assert CHANGED_LINE in page and DISCLOSURE in page and ">Write a new story</button>" in page
    assert len(re.findall(rf'\stitle="{DRAFT_TITLE}"', page)) == 1      # ↻ Regenerate
    assert _shown_twins(page) == ["draft", "draft"]                      # picker, legend
    assert PICKER_TWIN in page and LEGEND_TWIN in page

    _film(client, status="failed", narration_key=None, error="boom")    # the Prepare screen
    prepare = client.get("/feed/memory-0").text
    assert len(re.findall(rf'\stitle="{DRAFT_TITLE}"', prepare)) == 1   # Prepare this memory


def test_without_a_draft_the_page_shows_the_plain_texts(client):
    _narrated(client)
    page = client.get("/feed/memory-0").text
    assert _shown_twins(page) == ["no-draft", "no-draft"]
    assert re.findall(rf'\stitle="{DRAFT_TITLE}"', page) == []


@pytest.mark.parametrize("case", ["old revision", "second tab", "old draft"])
def test_a_save_on_rows_that_are_gone_is_409(client, case):
    _narrated(client)
    first = _change(client, 0, "One.", _film_rows(client)).json()
    if case == "old revision":
        _change(client, 0, "Two.", first)
        posted = first
    elif case == "second tab":
        posted = {"source": "b" * 32, "draft": "", "revision": 0}
    else:
        client.post("/feed/memory-0/narration/discard", data={
            "lang": "en", "source": first["source"], "draft": first["draft"],
            "revision": first["revision"]})
        second = _change(client, 0, "Again.", _film_rows(client)).json()
        assert second["draft"] != first["draft"] and second["revision"] == first["revision"]
        posted = first
    reply = _change(client, 1, "Late words.", posted)
    assert reply.status_code == 409
    assert reply.json() == {"saved": False, "error": "stale", "message": STALE}


def test_each_refusal_answers_its_status_and_message(client):
    _narrated(client)
    rows = _film_rows(client)
    bracket = _change(client, 0, "[glitch] here", rows)
    assert (bracket.status_code, bracket.json()["message"]) == (
        400, "The narration is read aloud word for word. Put a [bracket] instruction in a note.")
    long = "x" * 10_001
    too_long = _change(client, 0, long, rows)
    total = len(long) + len(PASSAGES[1]) + len(PASSAGES[2])
    assert (too_long.status_code, too_long.json()["message"]) == (
        400, f"The narration is longer than this app allows: {total} characters, at most 10000.")
    missing = _change(client, 7, "Words.", rows)
    assert (missing.status_code, missing.json()["message"]) == (
        404, "This passage is no longer in the narration.")
    assert client.post("/feed/memory-9/narration/0", data={"text": "x"}).status_code == 404
    assert client.get("/feed/memory-9/narration").status_code == 404


def test_the_delete_route_removes_a_passage_but_never_the_last(client):
    _narrated(client)
    rows = _change(client, 0, "", _film_rows(client), route="/delete").json()
    assert [row["n"] for row in rows["rows"]] == [1, 2]
    rows = _change(client, 1, "", rows, route="/delete").json()
    last = _change(client, 2, "", rows, route="/delete")
    assert (last.status_code, last.json()) == (400, {
        "saved": False, "error": "empty",
        "message": "A film needs narration. Keep at least one passage."})
    assert [row["text"] for row in _film_rows(client)["rows"]] == [PASSAGES[2]]


def test_write_a_new_story_brings_the_film_rows_back(client):
    _narrated(client)
    draft = _change(client, 0, "Other words.", _film_rows(client)).json()
    reply = client.post("/feed/memory-0/narration/discard", data={
        "lang": "en", "source": draft["source"], "draft": draft["draft"],
        "revision": draft["revision"]})
    assert reply.status_code == 200
    body = reply.json()
    assert body["saved"] and body["state"] == "film" and body["draft"] is None
    assert [row["text"] for row in body["rows"]] == list(PASSAGES)
    assert _film_rows(client) == {key: value for key, value in body.items() if key != "saved"}


def test_the_narration_route_serves_the_page_payload(client):
    _narrated(client)
    _change(client, 2, "Home at last.", _film_rows(client))
    assert _film_rows(client) == _narration_payload(client.get("/feed/memory-0").text)


def test_a_russian_save_leaves_the_english_rows(client):
    _narrated(client)
    _narrated(client, "ru", texts=("Тёплый день.", "Мы купались.", "Домой."), token="c" * 32)
    reply = _change(client, 0, "Жаркий день.", _film_rows(client, "ru"), lang="ru")
    assert reply.json()["state"] == "draft"
    english = _film_rows(client, "en")
    assert english["state"] == "film"
    assert [row["text"] for row in english["rows"]] == list(PASSAGES)


def test_a_narration_save_is_never_a_note(client):
    _narrated(client)
    _change(client, 0, "Other words.", _film_rows(client))
    conn = client.app.state.context.conn
    assert conn.execute("SELECT COUNT(*) FROM memory_notes").fetchone()[0] == 0
    assert _notes_payload(client.get("/feed/memory-0").text) == []


def test_retell_names_the_edited_narration_it_deletes(client):
    _narrated(client)
    assert "Deletes your edited narration" not in client.get(GRID).text
    _change(client, 0, "Other words.", _film_rows(client))
    assert ("Makes no film. Deletes the films in: English. "
            "Deletes your edited narration in: English.") in client.get(GRID).text


class _Tree(HTMLParser):
    """The served page as a JSON tree, for the fake DOM the scripts run against."""

    VOID = frozenset({"input", "br", "meta", "link", "img", "source", "hr"})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = {"tag": "#document", "attrs": {}, "children": []}
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = {"tag": tag, "attrs": {k: v if v is not None else "" for k, v in attrs},
                "children": []}
        self.stack[-1]["children"].append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1]["children"].append(
            {"tag": tag, "attrs": {k: v if v is not None else "" for k, v in attrs},
             "children": []})

    def handle_endtag(self, tag):
        for depth in range(len(self.stack) - 1, 0, -1):
            if self.stack[depth]["tag"] == tag:
                del self.stack[depth:]
                return

    def handle_data(self, data):
        self.stack[-1]["children"].append(data)


PANEL_HARNESS = r"""
const flush = async () => { for (let i = 0; i < 40; i += 1) await new Promise((r) => setImmediate(r)); };
const kebab = (k) => k.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);
const descendants = (root) => root.children.flatMap((child) => [child, ...descendants(child)]);
function compound(text) {
  const parts = [...text.matchAll(/^([a-zA-Z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:([$^]?=)"([^"]*)")?\]/g)];
  return (el) => parts.every((m) => {
    if (m[1]) return el.localName === m[1].toLowerCase();
    if (m[2]) return el.id === m[2];
    if (m[3]) return el.classList.contains(m[3]);
    const value = el.getAttribute(m[4]);
    if (value === null) return false;
    if (!m[5]) return true;
    if (m[5] === '=') return value === m[6];
    return m[5] === '$=' ? value.endsWith(m[6]) : value.startsWith(m[6]);
  });
}
function matchesChain(el, chain) {
  if (!chain[chain.length - 1](el)) return false;
  let i = chain.length - 2;
  for (let node = el.parentNode; i >= 0 && node; node = node.parentNode) {
    if (chain[i](node)) i -= 1;
  }
  return i < 0;
}
let doc = null;
class El {
  constructor(tag, attrs = {}) {
    this.localName = tag;
    this.attrs = { ...attrs };
    this.childNodes = [];
    this.parentNode = null;
    this.listeners = {};
    this.style = {};
    this.scrollHeight = 0;
    const self = this;
    this.dataset = new Proxy({}, {
      get: (_, k) => (typeof k === 'string' ? self.getAttribute(`data-${kebab(k)}`) ?? undefined : undefined),
      set: (_, k, v) => { self.setAttribute(`data-${kebab(k)}`, v); return true; },
    });
    const names = () => (self.attrs.class || '').split(/\s+/).filter(Boolean);
    this.classList = {
      contains: (c) => names().includes(c),
      add: (c) => { if (!names().includes(c)) self.attrs.class = [...names(), c].join(' '); },
      remove: (c) => { self.attrs.class = names().filter((x) => x !== c).join(' '); },
      toggle: (c, on) => {
        const want = on === undefined ? !names().includes(c) : on;
        if (want) { self.classList.add(c); } else { self.classList.remove(c); }
        return want;
      },
    };
  }
  getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  removeAttribute(name) { delete this.attrs[name]; }
  hasAttribute(name) { return name in this.attrs; }
  flag(name, on) { if (on) { this.attrs[name] = ''; } else { delete this.attrs[name]; } }
  get hidden() { return this.hasAttribute('hidden'); }
  set hidden(on) { this.flag('hidden', on); }
  get disabled() { return this.hasAttribute('disabled'); }
  set disabled(on) { this.flag('disabled', on); }
  get readOnly() { return this.hasAttribute('readonly'); }
  set readOnly(on) { this.flag('readonly', on); }
  get id() { return this.getAttribute('id') || ''; }
  get className() { return this.getAttribute('class') || ''; }
  set className(v) { this.setAttribute('class', v); }
  get title() { return this.getAttribute('title') || ''; }
  set title(v) { this.setAttribute('title', v); }
  get type() { return this.getAttribute('type') || ''; }
  set type(v) { this.setAttribute('type', v); }
  set rows(v) { this.setAttribute('rows', v); }
  get children() { return this.childNodes.filter((node) => node instanceof El); }
  get nextElementSibling() {
    if (!this.parentNode) return null;
    const siblings = this.parentNode.children;
    return siblings[siblings.indexOf(this) + 1] || null;
  }
  get textContent() {
    return this.childNodes.map((node) => (node instanceof El ? node.textContent : node.data)).join('');
  }
  set textContent(v) { this.replaceChildren(String(v)); }
  get value() {
    if (this.current !== undefined) return this.current;
    if (this.localName === 'textarea') return this.textContent;
    if (this.localName === 'select') {
      const options = descendants(this).filter((el) => el.localName === 'option');
      const chosen = options.find((el) => el.hasAttribute('selected')) || options[0];
      return chosen ? (chosen.getAttribute('value') ?? chosen.textContent) : '';
    }
    return this.getAttribute('value') ?? '';
  }
  set value(v) { this.current = String(v); }
  detach(node) {
    if (node.parentNode) {
      const siblings = node.parentNode.childNodes;
      siblings.splice(siblings.indexOf(node), 1);
      node.parentNode = null;
    }
  }
  insertBefore(node, ref) {
    if (node === ref) return node;
    this.detach(node);
    const at = ref ? this.childNodes.indexOf(ref) : this.childNodes.length;
    this.childNodes.splice(at, 0, node);
    node.parentNode = this;
    return node;
  }
  appendChild(node) { return this.insertBefore(node, null); }
  append(...nodes) {
    for (const node of nodes) { this.appendChild(typeof node === 'string' ? { data: node } : node); }
  }
  replaceChildren(...nodes) {
    for (const node of [...this.childNodes]) { this.detach(node); }
    this.append(...nodes);
  }
  remove() { if (this.parentNode) { this.parentNode.detach(this); } }
  after(node) {
    const siblings = this.parentNode.childNodes;
    this.parentNode.insertBefore(node, siblings[siblings.indexOf(this) + 1] || null);
  }
  contains(node) {
    for (let at = node; at; at = at.parentNode) { if (at === this) return true; }
    return false;
  }
  querySelectorAll(selector) {
    const chain = selector.trim().split(/\s+/).map(compound);
    return descendants(this).filter((el) => matchesChain(el, chain));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  fire(type, extra = {}) {
    const event = { type, target: this, preventDefault() {}, ...extra };
    for (const listener of this.listeners[type] || []) { listener(event); }
  }
  focus() {
    const before = doc.activeElement;
    if (before === this) return;
    doc.activeElement = this;
    if (before && before !== doc.body) { before.fire('blur'); }
  }
  blur() {
    if (doc.activeElement !== this) return;
    doc.activeElement = doc.body;
    this.fire('blur');
  }
  requestSubmit() { this.fire('submit'); }
  scrollIntoView() { this.scrolled = (this.scrolled || 0) + 1; }
}
const build = (node) => {
  if (typeof node === 'string') return { data: node };
  const el = new El(node.tag, node.attrs);
  for (const child of node.children) { el.appendChild(build(child)); }
  return el;
};
doc = build(TREE);
doc.body = doc.querySelector('body');
doc.activeElement = doc.body;
doc.createElement = (tag) => new El(tag);
doc.getElementById = (id) => descendants(doc).find((el) => el.id === id) || null;

const h = { fetches: [], confirms: [], confirmAnswer: true, replaced: [] };
globalThis.document = doc;
globalThis.window = globalThis;
globalThis.FormData = class {
  constructor(form) {
    this.map = new Map();
    for (const el of form ? descendants(form) : []) {
      const name = el.getAttribute('name');
      if (name && ['input', 'select', 'textarea'].includes(el.localName)) { this.map.set(name, el.value); }
    }
  }
  set(name, value) { this.map.set(name, String(value)); }
  get(name) { return this.map.has(name) ? this.map.get(name) : null; }
};
globalThis.fetch = (url, options = {}) => new Promise((resolve, reject) => {
  h.fetches.push({ url, method: options.method || 'GET',
                   body: options.body ? Object.fromEntries(options.body.map) : null, resolve, reject });
});
globalThis.confirm = (text) => { h.confirms.push(text); return h.confirmAnswer; };
globalThis.location = { pathname: '/feed/memory-0', replace: (url) => h.replaced.push(url) };
globalThis.history = { length: 1, back() {} };
Object.defineProperty(globalThis, 'localStorage', {
  value: { getItem: () => null, setItem() {} }, configurable: true,
});
h.session = new Map(Object.entries(SESSION));
Object.defineProperty(globalThis, 'sessionStorage', {
  value: { getItem: (k) => h.session.get(k) ?? null, setItem: (k, v) => h.session.set(k, v) },
  configurable: true,
});
h.timers = [];
globalThis.setTimeout = (run) => { h.timers.push(run); return h.timers.length; };
h.tick = async () => {
  for (const run of h.timers.splice(0)) { run(); }
  await flush();
};

for (const script of doc.querySelectorAll('script')) {
  if (!script.hasAttribute('src') && !script.hasAttribute('type')) { (0, eval)(script.textContent); }
}
(0, eval)(NOTES_JS);

const initial = JSON.parse(doc.getElementById('narration-data').textContent);
h.answer = async (i, status, body) => {
  h.fetches[i].resolve({ ok: status >= 200 && status < 300, status, json: async () => body });
  await flush();
};
h.garbled = async (i) => {
  h.fetches[i].resolve({ ok: true, status: 200, json: async () => { throw new SyntaxError('bad'); } });
  await flush();
};
h.down = async (i) => { h.fetches[i].reject(new TypeError('Failed to fetch')); await flush(); };
h.passages = () => doc.getElementById('notes-list').children
  .filter((row) => row.classList.contains('is-narration'));
h.field = (i) => h.passages()[i].querySelector('textarea');
h.type = (i, text) => { const field = h.field(i); field.focus(); field.value = text; return field; };
h.texts = () => h.passages().map((row) => row.querySelector('textarea').value);
h.readOnly = () => h.passages().map((row) => row.querySelector('textarea').readOnly);
h.posted = () => h.fetches.map((call) => call.url);
h.reply = (changes) => ({ saved: true, ...initial, ...changes });
h.draftRows = (i, text) => initial.rows.map((row) => (row.n === i ? { ...row, text } : row));
h.refused = (form) => {
  const line = form.nextElementSibling;
  return line && line.classList.contains('feed-refused') ? line.textContent : null;
};
const byId = (id) => doc.getElementById(id);
h.base = { source: initial.source, draft: initial.draft || '', revision: initial.revision };
h.ask = async (text) => {
  byId('rephrase-text').focus();
  byId('rephrase-text').value = text;
  byId('rephrase-form').requestSubmit();
  await flush();
};
h.started = async (i, job = 'j1') => h.answer(i, 202, { job });
h.running = (stage, percent) => ({ job: 'j1', state: 'running', stage, percent, error: null,
                                   result: null });
h.done = (changes, extra = {}) => ({
  job: 'j1', state: 'done', stage: '', percent: 0, error: null, outdated: false,
  result: { base: h.base, reply: 'Done.', total: 3, changes,
            summary: `Changed ${changes.length} of 3 passages.` }, ...extra,
});
h.change = (n, next) => ({ n, stamp: '0:00', old: initial.rows[n].text, new: next });
h.chat = () => ({
  shown: !byId('rephrase').hidden,
  log: byId('rephrase-log').children.map((row) => row.textContent),
  summary: byId('rephrase-summary').textContent,
  proposal: !byId('rephrase-proposal').hidden,
  changes: byId('rephrase-changes').children.map((row) => row.textContent),
  use: !byId('rephrase-use').hidden,
  useDisabled: byId('rephrase-use').disabled,
  sendDisabled: byId('rephrase-send').disabled,
  textDisabled: byId('rephrase-text').disabled,
  outdated: !byId('rephrase-outdated').hidden,
  progress: byId('rephrase-progress').hidden ? null : byId('rephrase-stage').textContent,
  refused: byId('rephrase-refused').hidden ? null : byId('rephrase-refused').textContent,
  text: byId('rephrase-text').value,
});
h.error = () => {
  const line = doc.querySelector('.feed-narration-error');
  return {
    shown: !line.hidden,
    message: line.querySelector('.feed-narration-message').textContent,
    tail: !line.querySelector('[data-error-tail]').hidden,
    reload: !line.querySelector('[data-error-reload]').hidden,
  };
};
"""


def _panel(client, scenario: str, session: dict | None = None) -> dict:
    """Run the served page's scripts in a fake DOM under node; `scenario` prints JSON.
    `session` is what `sessionStorage` already holds."""
    import json
    import subprocess

    tree = _Tree()
    tree.feed(client.get("/feed/memory-0").text)
    notes_js = (pathlib.Path(__file__).parent.parent / "web" / "static"
                / "feed-notes.js").read_text()
    script = (f"const TREE = {json.dumps(tree.root)};\nconst NOTES_JS = {json.dumps(notes_js)};\n"
              f"const SESSION = {json.dumps(session or {})};\n"
              f"{PANEL_HARNESS}\n(async () => {{\n{scenario}\n}})();")
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30,
                          check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@needs_node
def test_all_returns_the_notes_only(client):
    _narrated(client)
    client.post("/feed/memory-0/note", data={"at": 2, "text": "This is Julia"})
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      const rows = document.getElementById('notes-list').children;
      console.log(JSON.stringify({
        all: window.feedNotes.all().map((note) => note.text),
        rows: rows.length,
        tags: rows.map((row) => (row.querySelector('.feed-note-tag') || { textContent: '' }).textContent),
        count: document.getElementById('notes-count').textContent,
      }));
    """)
    assert out["all"] == ["This is Julia"]
    # Sorted by time: 0:00, the note at 0:02, 0:04, 0:08.
    assert out["tags"] == ["narration", "", "narration", "narration"]
    assert out["count"] == "4"


@needs_node
@pytest.mark.parametrize(("case", "lit"), [("film", False), ("changed", True),
                                           ("unchanged", False)])
def test_regenerate_is_lit_at_load_only_for_a_changed_draft(client, case, lit):
    _narrated(client)
    if case != "film":
        draft = _change(client, 0, "Other words.", _film_rows(client)).json()
        if case == "unchanged":
            assert not _change(client, 0, PASSAGES[0], draft).json()["changed"]
    out = _panel(client, """
      console.log(JSON.stringify({
        lit: document.querySelector('.feed-regen button').classList.contains('is-stale'),
      }));
    """)
    assert out["lit"] is lit


@needs_node
def test_a_second_quick_save_posts_the_first_replys_draft_and_revision(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      h.type(0, 'First.').blur();
      h.type(1, 'Second.').blur();
      await flush();
      const waiting = h.posted();
      await h.answer(0, 200, h.reply({ state: 'draft', draft: 'd1', revision: 1, changed: true,
                                      rows: h.draftRows(0, 'First.') }));
      console.log(JSON.stringify({ waiting, first: h.fetches[0].body, second: h.fetches[1].body,
        lit: document.querySelector('.feed-regen button').classList.contains('is-stale') }));
    """)
    assert out["waiting"] == ["/feed/memory-0/narration/0"]
    assert out["first"] == {"lang": "en", "source": FILM_TOKEN, "draft": "", "revision": "0",
                            "text": "First."}
    assert out["second"] == {"lang": "en", "source": FILM_TOKEN, "draft": "d1", "revision": "1",
                             "text": "Second."}
    assert out["lit"] is True


@needs_node
@pytest.mark.parametrize("answer", ["saved", "refused"])
def test_regenerate_waits_for_the_save_in_flight(client, answer):
    _narrated(client)
    out = _panel(client, f"""
      window.feedNotes.open(true, 5);
      h.type(0, 'Typed and not blurred.');
      const form = document.querySelector('form.feed-regen');
      form.fire('submit');
      await flush();
      const before = h.posted();
      if ('{answer}' === 'saved') {{
        await h.answer(0, 200, h.reply({{ state: 'draft', draft: 'd1', revision: 1, changed: true,
                                         rows: h.draftRows(0, 'Typed and not blurred.') }}));
      }} else {{
        await h.answer(0, 400, {{ saved: false, error: 'bracket', message: 'refused' }});
      }}
      console.log(JSON.stringify({{ before, after: h.posted(), refused: h.refused(form),
                                    error: h.error() }}));
    """)
    assert out["before"] == ["/feed/memory-0/narration/0"]
    if answer == "saved":
        assert out["after"] == ["/feed/memory-0/narration/0", "/feed/memory-0/prepare"]
        assert out["refused"] is None
    else:
        assert out["after"] == ["/feed/memory-0/narration/0"]
        assert out["refused"] == NOT_STARTED
        assert out["error"] == {"shown": True, "message": "refused", "tail": False,
                                "reload": False}


@needs_node
def test_a_reply_keeps_the_row_being_typed_in(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      h.type(0, 'A saved.').blur();
      const typing = h.type(1, 'B half typed');
      const row = h.passages()[1];
      await flush();
      await h.answer(0, 200, h.reply({ state: 'draft', draft: 'd1', revision: 1, changed: true,
                                      rows: h.draftRows(0, 'A saved.') }));
      console.log(JSON.stringify({
        sameRow: h.passages()[1] === row, sameField: h.field(1) === typing,
        focused: document.activeElement === typing, texts: h.texts(),
      }));
    """)
    assert out["sameRow"] and out["sameField"] and out["focused"]
    assert out["texts"] == ["A saved.", "B half typed", PASSAGES[2]]


@needs_node
def test_a_conflict_freezes_the_panel_and_drops_the_queue(client):
    _narrated(client)
    out = _panel(client, f"""
      window.feedNotes.open(true, 5);
      h.type(0, 'A2').blur();
      h.type(1, 'B2').blur();
      h.type(2, 'C2').blur();
      await flush();
      await h.answer(0, 409, {{ saved: false, error: 'stale', message: {STALE!r} }});
      const frozen = {{ posted: h.posted(), readOnly: h.readOnly(), texts: h.texts(),
                        error: h.error() }};
      window.feedNotes.open(true, 6);                 // a pause re-renders
      const afterRender = h.texts();
      h.type(1, 'B3').blur();
      await flush();
      const postedWhileFrozen = h.posted().length;
      document.querySelector('[data-error-reload]').fire('click');
      await flush();
      const load = h.fetches[1];
      await h.answer(1, 200, {{ ...h.reply({{}}), saved: undefined }});
      console.log(JSON.stringify({{ frozen, afterRender, postedWhileFrozen,
        load: [load.url, load.method], texts: h.texts(), readOnly: h.readOnly(),
        error: h.error() }}));
    """)
    assert out["frozen"]["posted"] == ["/feed/memory-0/narration/0"]
    assert out["frozen"]["readOnly"] == [True, True, True]
    assert out["frozen"]["texts"] == ["A2", "B2", "C2"]
    assert out["frozen"]["error"] == {"shown": True, "message": STALE, "tail": True,
                                      "reload": True}
    assert out["afterRender"] == ["A2", "B2", "C2"]
    assert out["postedWhileFrozen"] == 1
    assert out["load"] == ["/feed/memory-0/narration?lang=en", "GET"]
    assert out["texts"] == list(PASSAGES)
    assert out["readOnly"] == [False, False, False]
    assert out["error"]["shown"] is False


@needs_node
def test_a_network_failure_keeps_the_text_and_blocks_regenerate(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      h.type(0, 'A2').blur();
      h.type(1, 'B2').blur();
      await flush();
      await h.down(0);
      const settled = await window.feedNotes.settled();
      const form = document.querySelector('form.feed-regen');
      form.fire('submit');
      await flush();
      console.log(JSON.stringify({ posted: h.posted(), texts: h.texts(), settled,
                                   refused: h.refused(form), error: h.error() }));
    """)
    assert out["posted"] == ["/feed/memory-0/narration/0"]
    assert out["texts"] == ["A2", "B2", PASSAGES[2]]
    assert out["settled"] is False
    assert out["refused"] == NOT_STARTED
    assert out["error"] == {"shown": True, "tail": False, "reload": False,
                            "message": "Not saved: the app did not answer. Your text stays in its row."}


@needs_node
@pytest.mark.parametrize("answer", ["saved", "refused"])
def test_write_a_new_story_waits_for_the_pending_save(client, answer):
    _narrated(client)
    draft = _change(client, 0, "Other words.", _film_rows(client)).json()
    out = _panel(client, f"""
      window.feedNotes.open(true, 5);
      h.type(1, 'B2').blur();
      document.querySelector('.feed-narration-undo').fire('click');
      await flush();
      const before = h.posted().length;
      if ('{answer}' === 'saved') {{
        await h.answer(0, 200, h.reply({{ revision: initial.revision + 1,
                                         rows: h.draftRows(1, 'B2') }}));
      }} else {{
        await h.answer(0, 400, {{ saved: false, error: 'bracket', message: 'refused' }});
      }}
      console.log(JSON.stringify({{ before, posted: h.posted(), confirms: h.confirms,
                                    discard: h.fetches[1] ? h.fetches[1].body : null }}));
    """)
    assert out["before"] == 1
    if answer == "saved":
        assert out["posted"] == ["/feed/memory-0/narration/1", "/feed/memory-0/narration/discard"]
        assert out["confirms"] == [("Stop using your edited narration? The next Prepare or "
                                    "Regenerate writes a new story.")]
        assert out["discard"] == {"lang": "en", "source": FILM_TOKEN, "draft": draft["draft"],
                                  "revision": str(draft["revision"] + 1)}
    else:
        assert out["posted"] == ["/feed/memory-0/narration/1"]
        assert out["confirms"] == []


@needs_node
def test_a_note_edited_back_to_its_first_text_is_saved_again(client):
    note = client.post("/feed/memory-0/note", data={"at": 2, "text": "A"}).json()
    out = _panel(client, f"""
      const field = () => document.getElementById('notes-list').children
        .find((row) => !row.classList.contains('is-narration')).querySelector('textarea');
      field().focus(); field().value = 'B'; field().blur();
      await flush();
      await h.answer(0, 200, {{ saved: true, id: {note["id"]}, at: 2, stamp: '0:02', text: 'B' }});
      field().focus(); field().value = 'A'; field().blur();
      await flush();
      console.log(JSON.stringify({{ posted: h.fetches.map((call) => [call.url, call.body.text]) }}));
    """)
    assert out["posted"] == [[f"/feed/memory-0/note/{note['id']}", "B"],
                             [f"/feed/memory-0/note/{note['id']}", "A"]]


@needs_node
def test_an_unreadable_reload_keeps_the_panel_frozen(client):
    _narrated(client)
    out = _panel(client, f"""
      window.feedNotes.open(true, 5);
      h.type(0, 'A2').blur();
      await flush();
      await h.answer(0, 409, {{ saved: false, error: 'stale', message: {STALE!r} }});
      document.querySelector('[data-error-reload]').fire('click');
      await flush();
      await h.garbled(1);
      console.log(JSON.stringify({{ texts: h.texts(), readOnly: h.readOnly(), error: h.error() }}));
    """)
    assert out["texts"] == ["A2", PASSAGES[1], PASSAGES[2]]
    assert out["readOnly"] == [True, True, True]
    assert out["error"] == {"shown": True, "tail": True, "reload": True,
                            "message": "Not saved: the app did not answer. Your text stays in its row."}


OFFLINE = "Not saved: the app did not answer. Your text stays in its row."
KEEP_ONE = "A film needs narration. Keep at least one passage."
GONE = "This passage is no longer in the narration."
SCREENS = ["regenerate", "prepare"]
DELETE_1 = "/feed/memory-0/narration/1/delete"


def _screen(client, screen: str) -> None:
    """The ready film's page (↻ Regenerate), or the Prepare screen with a draft's rows."""
    _narrated(client)
    if screen == "prepare":
        _change(client, 0, "Other words.", _film_rows(client))
        _film(client, status="failed", narration_key=None, error="boom")


# Runs after a scenario: the barrier, the build form, "Write a new story", and what posted.
BARRIER = """
  const settled = await window.feedNotes.settled();
  const form = document.querySelector('form[action$="/prepare"]');
  form.fire('submit');
  await flush();
  document.querySelector('.feed-narration-undo').fire('click');
  await flush();
  const barrier = { settled, refused: h.refused(form), confirms: h.confirms,
                    built: h.posted().some((url) => url.endsWith('/prepare')
                                                    || url.endsWith('/discard')) };
"""


@needs_node
@pytest.mark.parametrize("screen", SCREENS)
@pytest.mark.parametrize(("answer", "message"), [
    (f"h.answer(0, 400, {{ saved: false, error: 'empty', message: {KEEP_ONE!r} }})", KEEP_ONE),
    (f"h.answer(0, 404, {{ saved: false, error: 'gone', message: {GONE!r} }})", GONE),
    ("h.down(0)", OFFLINE),
    ("h.garbled(0)", OFFLINE),
], ids=["400", "404", "network", "garbled"])
def test_a_delete_with_no_200_keeps_its_row_and_blocks_the_build(client, screen, answer,
                                                                 message):
    _screen(client, screen)
    out = _panel(client, f"""
      window.feedNotes.open(true, 5);
      const before = h.texts();
      h.passages()[1].querySelector('.feed-note-remove').fire('click');
      await flush();
      await {answer};
      {BARRIER}
      console.log(JSON.stringify({{ before, texts: h.texts(), readOnly: h.readOnly(),
                                    posted: h.posted(), error: h.error(), ...barrier }}));
    """)
    assert out["texts"] == out["before"]
    assert out["readOnly"] == [False, False, False]
    assert out["posted"] == [DELETE_1]
    assert out["settled"] is False
    assert out["refused"] == NOT_STARTED
    assert out["built"] is False and out["confirms"] == []
    assert out["error"] == {"shown": True, "message": message, "tail": False, "reload": True}


@needs_node
@pytest.mark.parametrize("screen", SCREENS)
def test_a_retried_delete_with_its_own_200_unblocks(client, screen):
    _screen(client, screen)
    out = _panel(client, f"""
      window.feedNotes.open(true, 5);
      const remove = () => h.passages()[1].querySelector('.feed-note-remove').fire('click');
      remove();
      await flush();
      await h.answer(0, 400, {{ saved: false, error: 'empty', message: {KEEP_ONE!r} }});
      remove();
      await flush();
      await h.answer(1, 200, h.reply({{ revision: initial.revision + 1,
                                       rows: initial.rows.filter((row) => row.n !== 1) }}));
      const error = h.error();
      const settled = await window.feedNotes.settled();
      document.querySelector('form[action$="/prepare"]').fire('submit');
      await flush();
      console.log(JSON.stringify({{ settled, error, rows: h.passages().length,
                                    posted: h.posted() }}));
    """)
    assert out["settled"] is True
    assert out["error"]["shown"] is False
    assert out["rows"] == 2
    assert out["posted"] == [DELETE_1, DELETE_1, "/feed/memory-0/prepare"]


@needs_node
@pytest.mark.parametrize("screen", SCREENS)
def test_a_200_on_another_row_keeps_the_block_and_the_button(client, screen):
    _screen(client, screen)
    out = _panel(client, f"""
      window.feedNotes.open(true, 5);
      h.passages()[1].querySelector('.feed-note-remove').fire('click');
      await flush();
      await h.answer(0, 400, {{ saved: false, error: 'empty', message: {KEEP_ONE!r} }});
      h.type(0, 'A2').blur();
      await flush();
      await h.answer(1, 200, h.reply({{ revision: initial.revision + 1,
                                       rows: h.draftRows(0, 'A2') }}));
      const error = h.error();
      {BARRIER}
      console.log(JSON.stringify({{ error, texts: h.texts(), posted: h.posted(), ...barrier }}));
    """)
    assert out["error"] == {"shown": True, "message": KEEP_ONE, "tail": False, "reload": True}
    assert out["texts"][1] == PASSAGES[1]
    assert out["settled"] is False and out["refused"] == NOT_STARTED
    assert out["built"] is False
    assert out["posted"] == [DELETE_1, "/feed/memory-0/narration/0"]


@needs_node
@pytest.mark.parametrize("screen", SCREENS)
def test_a_delete_dropped_behind_a_failure_blocks(client, screen):
    _screen(client, screen)
    out = _panel(client, f"""
      window.feedNotes.open(true, 5);
      const remove = (i) => h.passages()[i].querySelector('.feed-note-remove').fire('click');
      remove(1);
      remove(2);
      await flush();
      await h.answer(0, 400, {{ saved: false, error: 'empty', message: {KEEP_ONE!r} }});
      remove(1);
      await flush();
      await h.answer(1, 200, h.reply({{ revision: initial.revision + 1,
                                       rows: initial.rows.filter((row) => row.n !== 1) }}));
      const error = h.error();
      {BARRIER}
      console.log(JSON.stringify({{ error, texts: h.texts(), posted: h.posted(), ...barrier }}));
    """)
    assert out["posted"] == [DELETE_1, DELETE_1]          # the delete of row 2 never posted
    assert out["texts"][-1] == PASSAGES[2]
    assert out["error"] == {"shown": True, "message": KEEP_ONE, "tail": False, "reload": True}
    assert out["settled"] is False and out["refused"] == NOT_STARTED
    assert out["built"] is False


@needs_node
@pytest.mark.parametrize("screen", SCREENS)
@pytest.mark.parametrize("load", ["loaded", "network", "garbled"])
def test_only_a_loaded_narration_clears_a_blocked_delete(client, screen, load):
    _screen(client, screen)
    out = _panel(client, f"""
      window.feedNotes.open(true, 5);
      h.passages()[1].querySelector('.feed-note-remove').fire('click');
      await flush();
      await h.down(0);
      document.querySelector('[data-error-reload]').fire('click');
      await flush();
      const get = [h.fetches[1].url, h.fetches[1].method];
      if ('{load}' === 'loaded') {{
        await h.answer(1, 200, {{ ...h.reply({{}}), saved: undefined }});
      }} else if ('{load}' === 'network') {{
        await h.down(1);
      }} else {{
        await h.garbled(1);
      }}
      const error = h.error();
      const settled = await window.feedNotes.settled();
      document.querySelector('form[action$="/prepare"]').fire('submit');
      await flush();
      console.log(JSON.stringify({{ get, error, settled, posted: h.posted() }}));
    """)
    assert out["get"] == ["/feed/memory-0/narration?lang=en", "GET"]
    if load == "loaded":
        assert out["settled"] is True
        assert out["error"]["shown"] is False
        assert out["posted"][-1] == "/feed/memory-0/prepare"
    else:
        assert out["settled"] is False
        assert out["error"] == {"shown": True, "message": OFFLINE, "tail": False,
                                "reload": True}
        assert not out["posted"][-1].endswith("/prepare")


@needs_node
@pytest.mark.parametrize("screen", SCREENS)
def test_the_reload_waits_for_the_save_in_flight_and_posts_nothing_while_it_loads(client,
                                                                                 screen):
    _screen(client, screen)
    out = _panel(client, f"""
      window.feedNotes.open(true, 5);
      h.passages()[1].querySelector('.feed-note-remove').fire('click');
      await flush();
      await h.answer(0, 400, {{ saved: false, error: 'empty', message: {KEEP_ONE!r} }});
      h.type(0, 'A2').blur();
      await flush();
      document.querySelector('[data-error-reload]').fire('click');
      await flush();
      const inFlight = h.posted();
      h.type(2, 'C2').blur();
      await flush();
      const whileWaiting = h.posted();
      await h.answer(1, 200, h.reply({{ revision: initial.revision + 1,
                                       rows: h.draftRows(0, 'A2') }}));
      const afterSave = h.fetches.map((call) => [call.url, call.method]);
      h.type(2, 'C3').blur();
      await flush();
      const whileLoading = h.posted();
      await h.answer(2, 200, {{ ...h.reply({{ revision: initial.revision + 1,
                                             rows: h.draftRows(0, 'A2') }}), saved: undefined }});
      const settled = await window.feedNotes.settled();
      console.log(JSON.stringify({{ inFlight, whileWaiting, afterSave, whileLoading, settled,
                                    texts: h.texts() }}));
    """)
    save_0 = "/feed/memory-0/narration/0"
    load = "/feed/memory-0/narration?lang=en"
    assert out["inFlight"] == [DELETE_1, save_0]
    assert out["whileWaiting"] == [DELETE_1, save_0]
    assert out["afterSave"] == [[DELETE_1, "POST"], [save_0, "POST"], [load, "GET"]]
    assert out["whileLoading"] == [DELETE_1, save_0, load]
    assert out["settled"] is True
    assert out["texts"][0] == "A2" and out["texts"][2] == PASSAGES[2]


REPHRASE = "/feed/memory-0/narration/rephrase"
LOST = "This did not finish. Ask again."
UNSAVED = "Not sent: an edit to the narration is not saved."


class RephraseModels(FakeModels):
    """Answers each rephrase call in turn. `gate` holds a call open, as a slow model does."""

    def __init__(self, answers=()):
        super().__init__()
        self.answers = list(answers)
        self.gate = None
        self.entered = threading.Event()
        self.calls = 0

    def text_complete(self, model, messages, *, json_schema=None, **kw):
        self.calls += 1
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def models(monkeypatch):
    fake = RephraseModels()
    monkeypatch.setattr(Settings, "build_models_client", lambda self: fake)
    return fake


def _answer(*changes, reply="I changed it.") -> str:
    import json

    return json.dumps({"reply": reply, "changes": [{"n": n, "text": text} for n, text in changes]})


def _rephrase_form(rows: dict, *, request="Make it warmer.", proposal=None, lang="en") -> dict:
    import json

    return {"lang": lang, "source": rows["source"] or "", "draft": rows["draft"] or "",
            "revision": rows["revision"], "request": request,
            "proposal": json.dumps(proposal or [])}


def _start(client, rows: dict, **options):
    return client.post(REPHRASE, data=_rephrase_form(rows, **options))


def _finished(client, response) -> dict:
    assert response.status_code == 202, response.text
    client.app.state.rephrases.join()
    return client.get(f"{REPHRASE}/{response.json()['job']}").json()


def _use(client, rows: dict, changes: list[dict], *, lang="en"):
    import json

    return client.post("/feed/memory-0/narration/rewrite", data={
        "lang": lang, "source": rows["source"] or "", "draft": rows["draft"] or "",
        "revision": rows["revision"], "changes": json.dumps(changes)})


def test_a_request_starts_a_job_and_writes_nothing(client, models):
    _narrated(client)
    models.answers = [_answer((1, "New words."))]
    rows = _film_rows(client)

    state = _finished(client, _start(client, rows))

    assert state["state"] == "done"
    assert state["outdated"] is False
    assert state["result"]["summary"] == "Changed 1 of 3 passages."
    assert state["result"]["changes"] == [{"n": 1, "stamp": "0:04", "old": PASSAGES[1],
                                           "new": "New words."}]
    assert state["result"]["base"] == {"source": FILM_TOKEN, "draft": "", "revision": 0}
    assert _film_rows(client) == rows


def test_the_two_turns_of_a_conversation_are_used_together(client, models):
    _narrated(client)
    models.answers = [_answer((0, "First new.")), _answer((2, "Third new."))]
    rows = _film_rows(client)

    first = _finished(client, _start(client, rows))
    on_screen = [{"n": c["n"], "text": c["new"]} for c in first["result"]["changes"]]
    second = _finished(client, _start(client, rows, proposal=on_screen))
    patch = [{"n": c["n"], "text": c["new"]} for c in second["result"]["changes"]]

    assert [c["n"] for c in second["result"]["changes"]] == [0, 2]
    saved = _use(client, rows, patch).json()
    assert saved["saved"] is True
    assert [row["text"] for row in saved["rows"]] == ["First new.", PASSAGES[1], "Third new."]
    assert _film_rows(client)["state"] == "draft"


def test_a_bad_answer_fails_the_job_with_the_reason(client, models):
    _narrated(client)
    models.answers = [_answer((0, "Fine."), (1, "With [a] bracket."))]
    state = _finished(client, _start(client, _film_rows(client)))
    assert state["state"] == "failed"
    assert state["error"] == "Not used: passage 1 has a [bracket]."
    assert state["result"] is None


def test_a_model_failure_fails_the_job(client, models):
    _narrated(client)
    models.answers = [RuntimeError("the model is gone")]
    state = _finished(client, _start(client, _film_rows(client)))
    assert state["state"] == "failed"
    assert state["error"] == "The AI did not answer. Nothing changed."


@pytest.mark.parametrize("field", ["feed_building", "feed_queued"])
def test_a_film_being_made_refuses_a_request(client, models, field):
    _narrated(client)
    setattr(client.app.state, field, ("memory-0", "en"))
    try:
        response = _start(client, _film_rows(client))
    finally:
        setattr(client.app.state, field, None)
    assert response.status_code == 409
    assert response.json()["error"] == "building"
    assert models.calls == 0


def test_a_film_of_another_language_being_made_refuses_nothing(client, models):
    _narrated(client)
    models.answers = [_answer(reply="Fine.")]
    client.app.state.feed_building = ("memory-0", "ru")
    try:
        response = _start(client, _film_rows(client))
    finally:
        client.app.state.feed_building = None
    assert _finished(client, response)["state"] == "done"


def test_rows_that_changed_refuse_a_request(client, models):
    _narrated(client)
    rows = _film_rows(client)
    _change(client, 0, "Other words.", rows)
    response = _start(client, rows)
    assert response.status_code == 409
    assert response.json() == {"error": "stale", "message": STALE}
    assert models.calls == 0


def test_no_film_and_a_legacy_film_refuse_a_request(client, models):
    from feed.store import Presentation, save_presentation

    empty = {"source": "", "draft": "", "revision": 0}
    assert _start(client, empty).status_code == 409
    save_presentation(client.app.state.context.conn, Presentation(
        group_id=_memory_group(client), language="en", signature="sig", timeline={"shots": []},
        music_key=None, narration_key="n.wav", status="ready", decisions={}))
    assert _start(client, empty).status_code == 409
    assert models.calls == 0


@pytest.mark.parametrize("request_text", ["   ", "x" * 1001])
def test_a_blank_or_long_request_is_refused(client, models, request_text):
    _narrated(client)
    response = _start(client, _film_rows(client), request=request_text)
    assert response.status_code == 400
    assert models.calls == 0


def test_an_unreadable_history_is_refused(client, models):
    _narrated(client)
    form = _rephrase_form(_film_rows(client)) | {"history": "not json"}
    assert client.post(REPHRASE, data=form).status_code == 400


def test_cancel_drops_the_job_and_never_waits_for_the_model(client, models):
    _narrated(client)
    models.gate = threading.Event()
    models.answers = [_answer((1, "Too late."))]
    job = _start(client, _film_rows(client)).json()["job"]
    assert models.entered.wait(5)

    cancelled = client.post(f"{REPHRASE}/{job}/cancel")

    assert cancelled.status_code == 200
    assert not models.gate.is_set()
    assert client.get(f"{REPHRASE}/{job}").status_code == 404
    models.gate.set()
    client.app.state.rephrases.join()
    assert client.get(f"{REPHRASE}/{job}").status_code == 404
    assert client.post(f"{REPHRASE}/{job}/cancel").status_code == 404


def test_a_new_request_replaces_the_running_one(client, models):
    _narrated(client)
    models.gate = threading.Event()
    models.answers = [_answer((0, "Older.")), _answer((1, "Newer."))]
    rows = _film_rows(client)
    older = _start(client, rows).json()["job"]
    assert models.entered.wait(5)

    newer = _start(client, rows)

    assert client.get(f"{REPHRASE}/{older}").status_code == 404
    models.gate.set()
    state = _finished(client, newer)
    assert state["state"] == "done"
    assert client.get(f"{REPHRASE}/{older}").status_code == 404


def test_a_running_job_reports_a_stage_and_a_percent_below_100(client, models):
    _narrated(client)
    models.gate = threading.Event()
    models.answers = [_answer(reply="Fine.")]
    job = _start(client, _film_rows(client)).json()["job"]
    assert models.entered.wait(5)
    jobs = client.app.state.rephrases
    jobs.state = jobs.state | {"stage": "writing new words", "fraction": 1.0}

    state = client.get(f"{REPHRASE}/{job}").json()

    assert state["state"] == "running"
    assert state["stage"] == "writing new words"
    assert state["percent"] == 99
    models.gate.set()
    jobs.join()


def test_a_saved_edit_makes_a_finished_result_outdated(client, models):
    _narrated(client)
    models.answers = [_answer((1, "New words."))]
    rows = _film_rows(client)
    job = _start(client, rows).json()["job"]
    client.app.state.rephrases.join()
    assert client.get(f"{REPHRASE}/{job}").json()["outdated"] is False

    _change(client, 0, "Other words.", rows)

    assert client.get(f"{REPHRASE}/{job}").json()["outdated"] is True


def test_an_edit_during_the_model_call_makes_the_result_outdated(client, models):
    _narrated(client)
    models.gate = threading.Event()
    models.answers = [_answer((1, "New words."))]
    rows = _film_rows(client)
    job = _start(client, rows).json()["job"]
    assert models.entered.wait(5)

    _change(client, 0, "Other words.", rows)
    models.gate.set()
    client.app.state.rephrases.join()

    assert client.get(f"{REPHRASE}/{job}").json()["outdated"] is True


def test_a_job_belongs_to_its_memory(client, models):
    _narrated(client)
    models.answers = [_answer(reply="Fine.")]
    job = _start(client, _film_rows(client)).json()["job"]
    client.app.state.rephrases.join()
    assert client.get(f"{REPHRASE}/unknown").status_code == 404
    assert client.get(f"/feed/memory-9/narration/rephrase/{job}").status_code == 404


def test_use_this_on_rows_that_changed_is_refused_and_writes_nothing(client, models):
    _narrated(client)
    rows = _film_rows(client)
    current = _change(client, 0, "Other words.", rows).json()

    response = _use(client, rows, [{"n": 1, "text": "New words."}])

    assert response.status_code == 409
    assert response.json()["message"] == STALE
    assert [row["text"] for row in _film_rows(client)["rows"]] == [
        row["text"] for row in current["rows"]]


@pytest.mark.parametrize(("changes", "status", "code"), [
    ([{"n": 0, "text": "Good."}, {"n": 1, "text": ""}], 400, "empty_text"),
    ([{"n": 0, "text": "Good."}, {"n": 1, "text": "A [b]."}], 400, "bracket"),
    ([{"n": 0, "text": "Good."}, {"n": 9, "text": "Gone."}], 404, "missing"),
    ([{"n": 0, "text": "Good."}, {"n": 0, "text": "Twice."}], 400, "unreadable"),
])
def test_one_bad_change_refuses_use_this_whole(client, models, changes, status, code):
    _narrated(client)
    rows = _film_rows(client)
    response = _use(client, rows, changes)
    assert response.status_code == status
    assert response.json()["error"] == code
    assert _film_rows(client) == rows


def test_the_page_offers_the_chat_only_with_rows(client):
    assert 'id="rephrase" hidden' in " ".join(client.get("/feed/memory-0").text.split())
    _narrated(client)
    page = client.get("/feed/memory-0").text
    assert 'id="rephrase" hidden' not in " ".join(page.split())
    assert page.index('id="notes-list"') < page.index('id="rephrase-log"')
    assert page.index('id="rephrase-log"') < page.index('id="rephrase-form"')
    assert page.index('id="rephrase-form"') < page.index('class="feed-legend"')


def test_model_text_is_only_ever_set_as_text():
    assert "innerHTML" not in pathlib.Path("web/static/feed-notes.js").read_text()


@needs_node
def test_send_waits_for_the_save_and_posts_the_rows_it_saved(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      h.type(1, 'B2');
      await h.ask('Warmer.');
      const during = h.posted();
      const revision = initial.revision + 1;
      await h.answer(0, 200, h.reply({ state: 'draft', draft: 'd', revision,
                                       rows: h.draftRows(1, 'B2') }));
      console.log(JSON.stringify({ during, posted: h.posted(), body: h.fetches[1].body }));
    """)
    assert out["during"] == ["/feed/memory-0/narration/1"]
    assert out["posted"] == ["/feed/memory-0/narration/1", REPHRASE]
    assert out["body"]["request"] == "Warmer."
    assert (out["body"]["draft"], out["body"]["revision"]) == ("d", "1")
    assert out["body"]["proposal"] == "[]"


@needs_node
def test_a_refused_save_sends_nothing_and_keeps_the_request(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      h.type(1, 'B2');
      await h.ask('Warmer.');
      await h.answer(0, 400, { saved: false, error: 'bracket', message: 'refused' });
      console.log(JSON.stringify({ posted: h.posted(), chat: h.chat() }));
    """)
    assert out["posted"] == ["/feed/memory-0/narration/1"]
    assert out["chat"]["refused"] == UNSAVED
    assert out["chat"]["text"] == "Warmer."
    assert out["chat"]["log"] == []


@needs_node
def test_a_running_job_shows_its_stage_and_the_result_opens_the_proposal(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      const first = h.chat();
      await h.answer(1, 200, h.running('writing new words — about 40s left', 40));
      const running = h.chat();
      await h.tick();
      await h.answer(2, 200, h.done([h.change(1, 'New words.')]));
      console.log(JSON.stringify({ first, running, done: h.chat(), posted: h.posted() }));
    """)
    assert out["first"]["text"] == ""
    assert out["first"]["log"] == ["you:Warmer."]
    assert out["first"]["sendDisabled"] is True
    assert out["running"]["progress"] == "writing new words — about 40s left · 40% (estimate)"
    done = out["done"]
    assert done["progress"] is None and done["sendDisabled"] is False
    assert done["log"] == ["you:Warmer.", "AI:Done."]
    assert done["summary"] == "Changed 1 of 3 passages."
    assert done["changes"] == [f"0:00{PASSAGES[1]}→ New words."]
    assert done["proposal"] is True and done["use"] is True
    assert out["posted"] == [REPHRASE, f"{REPHRASE}/j1", f"{REPHRASE}/j1"]


@needs_node
def test_a_reply_with_no_change_offers_no_use_this(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Is it warm?');
      await h.started(0);
      const answer = h.done([]);
      answer.result.reply = 'It is warm.';
      await h.answer(1, 200, answer);
      console.log(JSON.stringify(h.chat()));
    """)
    assert out["log"] == ["you:Is it warm?", "AI:It is warm."]
    assert out["proposal"] is False


@needs_node
def test_a_rejected_proposal_says_why_and_keeps_the_one_on_screen(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      await h.answer(1, 200, h.done([h.change(0, 'First new.')]));
      await h.ask('Shorter.');
      await h.started(2, 'j2');
      await h.answer(3, 200, { state: 'failed', error: 'Not used: passage 1 has a [bracket].' });
      console.log(JSON.stringify(h.chat()));
    """)
    assert out["log"][-1] == "AI:Not used: passage 1 has a [bracket]."
    assert out["proposal"] is True and out["use"] is True


@needs_node
def test_a_follow_up_posts_the_proposal_and_use_this_posts_the_whole_patch(client):
    import json

    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      await h.answer(1, 200, h.done([h.change(0, 'First new.')]));
      await h.ask('Now the last one.');
      const follow = h.fetches[2].body;
      await h.started(2, 'j2');
      await h.answer(3, 200, h.done([h.change(0, 'First new.'), h.change(2, 'Third new.')]));
      const shown = h.chat();
      document.getElementById('rephrase-use').fire('click');
      await flush();
      const use = h.fetches[4];
      await h.answer(4, 200, h.reply({ state: 'draft', draft: 'd', revision: 1,
        rows: initial.rows.map((row, i) => ({ ...row,
          text: ['First new.', row.text, 'Third new.'][i] })) }));
      console.log(JSON.stringify({ follow, shown, url: use.url, body: use.body,
                                   texts: h.texts(), chat: h.chat() }));
    """)
    assert json.loads(out["follow"]["proposal"]) == [{"n": 0, "text": "First new."}]
    assert json.loads(out["follow"]["history"]) == [
        {"role": "user", "text": "Warmer."}, {"role": "ai", "text": "Done."}]
    assert out["shown"]["summary"] == "Changed 2 of 3 passages."
    assert out["url"] == "/feed/memory-0/narration/rewrite"
    assert json.loads(out["body"]["changes"]) == [{"n": 0, "text": "First new."},
                                                  {"n": 2, "text": "Third new."}]
    assert (out["body"]["source"], out["body"]["draft"], out["body"]["revision"]) == (
        FILM_TOKEN, "", "0")
    assert out["texts"] == ["First new.", PASSAGES[1], "Third new."]
    assert out["chat"]["proposal"] is False


@needs_node
def test_the_conversation_history_is_the_last_six_messages(client):
    import json

    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      for (let i = 0; i < 4; i += 1) {
        await h.ask(`ask ${i}`);
        await h.started(h.fetches.length - 1, `j${i}`);
        await h.answer(h.fetches.length - 1, 200, h.done([]));
      }
      await h.ask('ask 4');
      console.log(JSON.stringify(h.fetches[h.fetches.length - 1].body));
    """)
    history = json.loads(out["history"])
    assert len(history) == 6
    assert history[0]["text"] == "ask 1"


USE_THIS_START = """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      await h.answer(1, 200, h.done([h.change(1, 'New words.')]));
      document.getElementById('rephrase-use').fire('click');
      await flush();
      const locked = { readOnly: h.readOnly(),
                       remove: h.passages().map((row) => row.querySelector('button.feed-note-remove').disabled) };
      const saved = h.reply({ state: 'draft', draft: 'd', revision: 1, rows: h.draftRows(1, 'New words.') });
      const open = () => ({ readOnly: h.readOnly(),
                            remove: h.passages().map((row) => row.querySelector('button.feed-note-remove').disabled) });
"""


@needs_node
def test_every_passage_is_read_only_during_use_this_and_comes_back_after(client):
    _narrated(client)
    out = _panel(client, USE_THIS_START + """
      await h.answer(2, 200, saved);
      console.log(JSON.stringify({ locked, after: open(), texts: h.texts(), chat: h.chat(),
                                   posted: h.posted() }));
    """)
    assert out["locked"] == {"readOnly": [True] * 3, "remove": [True] * 3}
    assert out["after"] == {"readOnly": [False] * 3, "remove": [False] * 3}
    assert out["texts"] == [PASSAGES[0], "New words.", PASSAGES[2]]
    assert out["chat"]["proposal"] is False
    assert len(out["posted"]) == 3


@needs_node
def test_text_typed_during_use_this_is_never_replaced(client):
    _narrated(client)
    out = _panel(client, USE_THIS_START + """
      h.field(2).value = 'Typed during the request.';
      await h.answer(2, 200, saved);
      console.log(JSON.stringify({ texts: h.texts(), readOnly: h.readOnly() }));
    """)
    assert out["texts"] == [PASSAGES[0], "New words.", "Typed during the request."]
    assert out["readOnly"] == [False] * 3


@needs_node
def test_a_row_focused_during_use_this_posts_no_old_words_back(client):
    _narrated(client)
    out = _panel(client, USE_THIS_START + """
      const field = h.field(1);
      field.focus();
      await h.answer(2, 200, saved);
      field.blur();
      await flush();
      console.log(JSON.stringify({ posted: h.posted(), texts: h.texts() }));
    """)
    assert len(out["posted"]) == 3
    assert out["texts"][1] == "New words."


@needs_node
@pytest.mark.parametrize("answer", ["refused", "down"])
def test_a_failed_use_this_gives_the_fields_back_and_keeps_the_proposal(client, answer):
    _narrated(client)
    out = _panel(client, USE_THIS_START + f"""
      if ('{answer}' === 'refused') {{
        await h.answer(2, 400, {{ saved: false, error: 'bracket', message: 'refused' }});
      }} else {{
        await h.down(2);
      }}
      console.log(JSON.stringify({{ after: open(), texts: h.texts(), chat: h.chat(),
                                    error: h.error() }}));
    """)
    assert out["after"] == {"readOnly": [False] * 3, "remove": [False] * 3}
    assert out["texts"] == list(PASSAGES)
    assert out["chat"]["proposal"] is True and out["chat"]["use"] is True
    assert out["chat"]["useDisabled"] is False
    assert out["error"]["shown"] is True


@needs_node
def test_a_conflict_on_use_this_freezes_and_keeps_the_proposal(client):
    _narrated(client)
    out = _panel(client, USE_THIS_START + f"""
      await h.answer(2, 409, {{ saved: false, error: 'stale', message: {STALE!r} }});
      await h.ask('Again.');
      console.log(JSON.stringify({{ after: open(), texts: h.texts(), chat: h.chat(),
                                    error: h.error(), posted: h.posted() }}));
    """)
    assert out["after"] == {"readOnly": [True] * 3, "remove": [True] * 3}
    assert out["texts"] == list(PASSAGES)
    assert out["chat"]["proposal"] is True
    assert out["chat"]["useDisabled"] is True and out["chat"]["sendDisabled"] is True
    assert out["chat"]["textDisabled"] is True
    assert out["error"]["reload"] is True
    assert len(out["posted"]) == 3


@needs_node
def test_use_this_with_an_unsaved_row_posts_nothing(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      await h.answer(1, 200, h.done([h.change(1, 'New words.')]));
      h.type(0, 'A2').blur();
      await flush();
      await h.answer(2, 400, { saved: false, error: 'bracket', message: 'refused' });
      document.getElementById('rephrase-use').fire('click');
      await flush();
      console.log(JSON.stringify({ posted: h.posted(), chat: h.chat(), readOnly: h.readOnly() }));
    """)
    assert out["posted"][-1] == "/feed/memory-0/narration/0"
    assert out["chat"]["refused"] == "Not used: an edit to the narration is not saved."
    assert out["chat"]["proposal"] is True
    assert out["readOnly"] == [False] * 3


@needs_node
def test_a_saved_edit_makes_the_proposal_outdated_and_the_next_request_starts_clean(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      await h.answer(1, 200, h.done([h.change(1, 'New words.')]));
      h.type(0, 'A2').blur();
      await flush();
      await h.answer(2, 200, h.reply({ state: 'draft', draft: 'd', revision: 1,
                                       rows: h.draftRows(0, 'A2') }));
      const chat = h.chat();
      document.getElementById('rephrase-use').fire('click');
      await flush();
      const clicks = h.posted().length;
      await h.ask('Again.');
      console.log(JSON.stringify({ chat, clicks, body: h.fetches[3].body }));
    """)
    assert out["chat"]["outdated"] is True and out["chat"]["use"] is False
    assert out["clicks"] == 3
    assert out["body"]["proposal"] == "[]"
    assert (out["body"]["draft"], out["body"]["revision"]) == ("d", "1")


@needs_node
def test_a_result_the_server_calls_outdated_never_looks_current(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      await h.answer(1, 200, h.done([h.change(1, 'New words.')], { outdated: true }));
      console.log(JSON.stringify(h.chat()));
    """)
    assert out["proposal"] is True
    assert out["outdated"] is True and out["use"] is False


@needs_node
def test_cancel_stops_the_polling_and_a_late_answer_is_dropped(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      document.getElementById('rephrase-cancel').fire('click');
      await flush();
      const cancelled = h.chat();
      await h.answer(1, 200, h.done([h.change(1, 'Too late.')]));
      await h.tick();
      console.log(JSON.stringify({ cancelled, chat: h.chat(), posted: h.posted(),
                                   method: h.fetches[2].method }));
    """)
    assert out["posted"] == [REPHRASE, f"{REPHRASE}/j1", f"{REPHRASE}/j1/cancel"]
    assert out["method"] == "POST"
    assert out["cancelled"]["log"] == ["you:Warmer.", "AI:Cancelled."]
    assert out["chat"]["proposal"] is False and out["chat"]["sendDisabled"] is False


@needs_node
def test_esc_cancels_a_running_job_before_the_film_hears_it(client):
    _narrated(client)
    out = _panel(client, """
      const film = [];
      window.feedFilm = { escape: () => { film.push('film'); return true; } };
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      document.fire('keydown', { key: 'Escape' });
      await flush();
      const first = { film: [...film], chat: h.chat() };
      document.fire('keydown', { key: 'Escape' });
      await flush();
      console.log(JSON.stringify({ first, film, posted: h.posted() }));
    """)
    assert out["first"]["film"] == []
    assert out["first"]["chat"]["log"] == ["you:Warmer.", "AI:Cancelled."]
    assert out["film"] == ["film"]
    assert out["posted"][-1] == f"{REPHRASE}/j1/cancel"


@needs_node
def test_esc_during_the_start_request_cancels_the_job_it_returns(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      const escaped = window.feedNotes.escape();
      await h.started(0);
      console.log(JSON.stringify({ escaped, posted: h.posted(), chat: h.chat() }));
    """)
    assert out["escaped"] is True
    assert out["posted"] == [REPHRASE, f"{REPHRASE}/j1/cancel"]
    assert out["chat"]["log"] == []
    assert out["chat"]["text"] == "Warmer."


@needs_node
def test_a_job_the_app_no_longer_knows_says_ask_again(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      await h.answer(1, 404, { detail: 'Not Found' });
      console.log(JSON.stringify(h.chat()));
    """)
    assert out["log"] == ["you:Warmer.", f"AI:{LOST}"]
    assert out["sendDisabled"] is False


@needs_node
def test_a_409_that_is_not_a_conflict_does_not_freeze(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.answer(0, 409, { error: 'building', message: 'being made' });
      console.log(JSON.stringify({ chat: h.chat(), readOnly: h.readOnly() }));
    """)
    assert out["chat"]["refused"] == "being made"
    assert out["chat"]["sendDisabled"] is False
    assert out["chat"]["text"] == "Warmer."
    assert out["readOnly"] == [False] * 3


@needs_node
def test_a_conflict_on_send_freezes_the_panel(client):
    _narrated(client)
    out = _panel(client, f"""
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.answer(0, 409, {{ error: 'stale', message: {STALE!r} }});
      console.log(JSON.stringify({{ chat: h.chat(), readOnly: h.readOnly(), error: h.error() }}));
    """)
    assert out["chat"]["sendDisabled"] is True and out["chat"]["textDisabled"] is True
    assert out["chat"]["text"] == "Warmer."
    assert out["readOnly"] == [True] * 3
    assert out["error"]["reload"] is True


@needs_node
def test_dismiss_closes_the_proposal_and_keeps_the_transcript(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      await h.answer(1, 200, h.done([h.change(1, 'New words.')]));
      document.getElementById('rephrase-dismiss').fire('click');
      await h.ask('Again.');
      console.log(JSON.stringify({ chat: h.chat(), body: h.fetches[2].body }));
    """)
    assert out["chat"]["proposal"] is False
    assert out["chat"]["log"] == ["you:Warmer.", "AI:Done."]
    assert out["body"]["proposal"] == "[]"


def _session(job=None, proposal=None, turns=None) -> dict:
    import json

    return {"ivms777.rephrase.memory-0.en": json.dumps({
        "turns": turns if turns is not None else [
            {"request": "Warmer.", "reply": None, "error": None}],
        "proposal": proposal, "job": job})}


@needs_node
def test_a_return_shows_the_conversation_and_polls_its_job(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await flush();
      const first = { chat: h.chat(), posted: h.posted() };
      await h.answer(0, 200, h.done([h.change(1, 'New words.')]));
      console.log(JSON.stringify({ first, chat: h.chat() }));
    """, _session(job="j1"))
    assert out["first"]["posted"] == [f"{REPHRASE}/j1"]
    assert out["first"]["chat"]["log"] == ["you:Warmer."]
    assert out["first"]["chat"]["progress"] is not None
    assert out["chat"]["proposal"] is True


@needs_node
def test_a_return_to_a_job_the_app_lost_says_ask_again(client):
    _narrated(client)
    out = _panel(client, """
      await flush();
      await h.answer(0, 404, {});
      console.log(JSON.stringify(h.chat()));
    """, _session(job="j1"))
    assert out["log"] == ["you:Warmer.", f"AI:{LOST}"]


@needs_node
def test_a_return_to_a_proposal_for_older_rows_shows_it_outdated(client):
    _narrated(client)
    proposal = {"base": {"source": FILM_TOKEN, "draft": "", "revision": 7}, "reply": "Done.",
                "summary": "Changed 1 of 3 passages.", "total": 3, "outdated": False,
                "changes": [{"n": 1, "stamp": "0:04", "old": "Old.", "new": "New."}]}
    out = _panel(client, "console.log(JSON.stringify(h.chat()));",
                 _session(proposal=proposal, turns=[]))
    assert out["proposal"] is True and out["outdated"] is True and out["use"] is False


@needs_node
def test_the_conversation_is_kept_per_memory_and_language(client):
    import json

    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      await h.ask('Warmer.');
      await h.started(0);
      console.log(JSON.stringify({ keys: [...h.session.keys()],
                                   stored: JSON.parse([...h.session.values()][0]) }));
    """)
    assert out["keys"] == ["ivms777.rephrase.memory-0.en"]
    assert out["stored"]["job"] == "j1"
    assert json.dumps(out["stored"]["turns"][0]["request"]) == '"Warmer."'


@needs_node
def test_another_languages_conversation_is_not_read(client):
    _narrated(client)
    other = {"ivms777.rephrase.memory-0.ru": _session(job="j9")["ivms777.rephrase.memory-0.en"]}
    out = _panel(client, "await flush(); console.log(JSON.stringify({ posted: h.posted(), chat: h.chat() }));",
                 other)
    assert out["posted"] == []
    assert out["chat"]["log"] == []


@needs_node
def test_the_chat_hides_when_no_rows_are_left(client):
    _narrated(client)
    draft = _change(client, 0, "Other words.", _film_rows(client)).json()
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      document.querySelector('.feed-narration-undo').fire('click');
      await flush();
      await h.answer(0, 200, { saved: true, state: 'none', source: null, draft: null, revision: 0,
                               seekable: false, changed: false, rows: [] });
      console.log(JSON.stringify(h.chat()));
    """)
    assert draft["state"] == "draft"
    assert out["shown"] is False


@needs_node
def test_the_form_comes_into_view_after_each_turn(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      const form = document.getElementById('rephrase-form');
      const before = form.scrolled || 0;
      await h.ask('Warmer.');
      await h.started(0);
      await h.answer(1, 200, h.running('writing new words', 40));
      const running = form.scrolled;
      await h.tick();
      await h.answer(2, 200, h.done([h.change(1, 'New words.')]));
      console.log(JSON.stringify({ before, running, after: form.scrolled }));
    """)
    assert out["before"] == 0
    assert out["running"] == 1
    assert out["after"] == 2


@needs_node
def test_an_answer_never_moves_a_viewer_who_is_reading_the_rows(client):
    _narrated(client)
    out = _panel(client, """
      window.feedNotes.open(true, 5);
      const form = document.getElementById('rephrase-form');
      await h.ask('Warmer.');
      await h.started(0);
      const sent = form.scrolled;
      h.field(0).focus();
      await h.answer(1, 200, h.done([h.change(1, 'New words.')]));
      console.log(JSON.stringify({ sent, after: form.scrolled, chat: h.chat() }));
    """)
    assert out["after"] == out["sent"]
    assert out["chat"]["proposal"] is True
