"""The fullscreen memory feed (design §11.1, §13, §13.1)."""

import pathlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from albums.memory_store import Memory, replace_memories
from config import Settings
from storage.wav import encode_wav
from tests.factories import add_photo
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
            "cards": [{"shot": 1, "text": "The water"}],
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
    assert set(edit) == {"shots", "grade", "grades", "seconds", "narration", "effects"}
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
    assert "/rebuild" in grid and "Rebuild" in grid
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
