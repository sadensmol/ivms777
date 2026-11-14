"""The film prepare job (design §11.1) — offline generation, fully faked."""

import json
import threading
from itertools import pairwise

import httpx
import numpy as np
import pytest

from albums.memory_store import Memory, replace_memories
from feed.build import Cancelled, NarrationFailed, build_presentation, voice_turn
from feed.store import read_presentation
from storage.local import LocalStorage
from storage.wav import encode_wav
from tests.factories import add_photo


class FakeModels:
    """The calls the job makes, with recorders for what it asked for."""

    SECONDS_PER_CHUNK = 2.0

    # Each sentence is long enough to be its own BEAT (`story._MIN_SENTENCE_CHARS`):
    # a passage holds a photograph for as long as it is spoken.
    PROSE_EN = ("A warm afternoon by the water with nowhere at all to be. "
                "The light came off the surface of it in long sheets. "
                "We stayed there until the whole day went cold.")
    PROSE_RU = ("Тёплый день у воды, и совершенно некуда было спешить. "
                "Свет ложился на неё длинными светлыми полосами. "
                "Мы сидели там, пока не стало совсем холодно.")

    def __init__(self, *, voice_off: bool = False, music_off: bool = False) -> None:
        self.voice_off = voice_off
        self.music_off = music_off
        self.spoken: list[tuple[str, str]] = []
        self.spoken_voice: str | None = None
        self.spoken_voices: list = []
        self.segments: list[str] = []
        self.prompted: list[tuple[str, float]] = []
        self.scores: list[dict] = []
        self.story_prompts: list[str] = []

    def _off(self) -> httpx.HTTPStatusError:
        request = httpx.Request("POST", "http://models/x")
        return httpx.HTTPStatusError(
            "slot off", request=request, response=httpx.Response(503, request=request)
        )

    def text_complete(self, model, messages, *, json_schema=None, **kw):
        prompt = messages[-1]["content"]
        self.story_prompts.append(prompt)
        # STAGE 1 — the story itself, prose only (§11.1).
        if "WHAT THE DAY WAS" in prompt:
            return json.dumps({"story": self.PROSE_EN})
        if "ЧТО ЭТО ЗА ДЕНЬ" in prompt:
            return json.dumps({"story": self.PROSE_RU})
        if "more sentences" in prompt or "Допиши" in prompt:
            return json.dumps({"story": ""})
        if "SENTENCE:" in prompt or "ПРЕДЛОЖЕНИЕ" in prompt:
            return json.dumps({"text": ""})
        if prompt.startswith("Translate into"):
            return json.dumps({"story": self.PROSE_RU})
        if "THE PHOTOGRAPHS available to you" in prompt:
            # The conductor pass. Reverses the order on purpose: matching pictures to
            # spoken words is allowed to reorder the film (§11.1), and the test suite
            # must exercise that rather than the identity arrangement.
            self.arranged = True
            count = prompt.count("\n") and len(self.shot_briefs(prompt))
            return json.dumps({"shots": [
                {"shot": count - i, "at": round(i * 4.0, 2), "seconds": 4.0,
                 "effect": "bloom", "params": {"strength": 0.4},
                 "transition": "dissolve" if i else "none"}
                for i in range(count)
            ]})
        # The story pass now returns the FILM: passages of story, each with the
        # photograph that illustrates it. Photo 1 is used twice and photo 3 not at all,
        # because the story leads and the pictures serve it (§11.1).
        # STAGE 2 — which photograph carries each SENTENCE of the story above. Photo 1
        # is used twice and photo 3 not at all: the story leads, the pictures serve it.
        return json.dumps({
            "script": [
                {"sentence": 1, "shot": 1, "voice": ""},
                {"sentence": 2, "shot": 2, "voice": ""},
                {"sentence": 3, "shot": 1, "voice": ""},
            ],
            "narration": "",
            "narrator": "",
            "cards": [{"sentence": 1, "text": "The water"}],
            "music_prompt": "warm ambient drone, no vocals",
        })

    @staticmethod
    def shot_briefs(prompt: str) -> list[str]:
        """The numbered photograph lines the conductor brief lists."""
        block = prompt.split("THE PHOTOGRAPHS available to you", 1)[1]
        block = block.split("WHAT THE VOICE SAYS", 1)[0]
        return [line for line in block.splitlines()
                if line[:1].isdigit() and ". " in line]

    def speak(self, text, language="en", voice=None, segments=None,
              voices=None):
        if self.voice_off:
            raise self._off()
        self.spoken.append((text, language))
        self.spoken_voice = voice
        self.spoken_voices = list(voices or [])
        self.segments = list(segments or [])
        # Real voices report where each requested chunk landed; the timing of the film
        # is derived from exactly this, so the fake has to provide it too.
        chunks = self.segments or [text]
        each, timings, cursor = self.SECONDS_PER_CHUNK, [], 0.0
        for chunk in chunks:
            timings.append({"text": chunk, "start": round(cursor, 3),
                            "end": round(cursor + each, 3)})
            cursor += each
        return {"audio": encode_wav(np.zeros(int(24000 * cursor), dtype=np.float32), 24000),
                "sample_rate": 24000, "model": "fake-voice", "segments": timings}

    def music(self, prompt="", seconds=30.0, seed=None, *, sections=None,
              temperature=None, guidance=None):
        if self.music_off:
            raise self._off()
        self.scores.append({"sections": sections or [], "seed": seed,
                            "temperature": temperature, "guidance": guidance})
        self.prompted.append((" ".join(s["brief"] for s in (sections or [])), seconds))
        total = sum(s["seconds"] for s in sections) if sections else seconds
        frames = int(44100 * max(1.0, total))
        return {"audio": encode_wav(np.zeros((2, frames), dtype=np.float32), 44100),
                "sample_rate": 44100, "model": "fake-music", "segments": []}


def dressed(models) -> str:
    """The CUT-AND-DRESS prompt — the one that carries the photographs, the notes' rules
    and the speaker table. `story_prompts[0]` is now the story call, which deliberately
    carries none of that (§11.1)."""
    return next(p for p in models.story_prompts if "EACH PHOTOGRAPH" in p)


def told(models) -> str:
    """The STORY prompt — the first stage, prose only."""
    return next(p for p in models.story_prompts
                if "WHAT THE DAY WAS" in p or "ЧТО ЭТО ЗА ДЕНЬ" in p)


def bracket_the_second_passage(conn, group_id, audio, language="en"):
    """A previous cut, and a `[voice]` note half-way through it: the gate marks the
    second of its three passages, the one every casting fake here casts. Without a
    bracket the model's cast is dropped (§11.3)."""
    from feed.notes import add_note

    build_presentation(conn, FakeModels(), audio, 1, group_id, language=language)
    seconds = read_presentation(conn, group_id, language).timeline["seconds"]
    add_note(conn, group_id, seconds / 2, "[another voice] the light on the water")


@pytest.fixture
def memory_group(conn):
    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  width=4000, height=3000, ai_title=f"caption {pid}",
                  shot_at=f"2025-07-12T1{pid}:00:00")
    replace_memories(conn, 1, [Memory("Beach day", "Warm afternoon.", [1, 2, 3], "sig")])
    return conn.execute("SELECT id FROM groups WHERE kind = 'memory'").fetchone()["id"]


@pytest.fixture
def audio(tmp_path):
    return LocalStorage(tmp_path / "audio")


def test_stores_both_clips_and_a_ready_presentation(conn, memory_group, audio):
    result = build_presentation(conn, FakeModels(), audio, 1, memory_group)
    assert result.status == "ready"
    assert audio.exists(result.music_key) and audio.exists(result.narration_key)
    assert read_presentation(conn, memory_group, "en").status == "ready"


def test_the_edit_replaces_the_old_timeline_shape(conn, memory_group, audio):
    edit = build_presentation(conn, FakeModels(), audio, 1, memory_group).timeline
    assert set(edit) == {"shots", "grade", "grades", "seconds", "narration", "effects",
                         "passages", "script"}
    move = edit["shots"][0]["move"]
    assert move["scale_from"] != move["scale_to"]
    assert edit["shots"][0]["transition"] == "none"


def test_the_film_records_that_it_kept_the_memorys_photo_order(conn, memory_group, audio):
    # The film records whether the photographs follow the memory's order (§11.1). This
    # fake's match is in order.
    result = build_presentation(conn, FakeModels(), audio, 1, memory_group)
    shown = [s["photo_id"] for s in result.timeline["shots"]]
    first_seen = [pid for i, pid in enumerate(shown) if pid not in shown[:i]]
    assert first_seen == sorted(first_seen), shown
    assert result.decisions["story"]["memory_order"] == "kept"


def test_a_photograph_the_story_never_uses_is_left_out_and_recorded(conn, memory_group, audio):
    # The fake's script shows photo 1 twice and never photo 3 (§11.1).
    result = build_presentation(conn, FakeModels(), audio, 1, memory_group)
    assert [s["photo_id"] for s in result.timeline["shots"]] == [1, 2, 1]
    assert result.decisions["story"]["photos"] == [0, 1, 0]
    assert result.decisions["story"]["votes"] == [None, None, None]
    assert result.decisions["story"]["matched"] is True


def test_every_shot_starts_when_its_own_words_are_spoken(conn, memory_group, audio):
    # The sync, and it is arithmetic rather than judgement: passage `i` was handed to
    # the voice as its own chunk, so segment `i` IS the words for shot `i`.
    models = FakeModels()
    edit = build_presentation(conn, models, audio, 1, memory_group).timeline
    assert [s["at"] for s in edit["shots"]] == [0.0, 2.0, 4.0]   # the fake speaks 2s each
    assert [s["photo_id"] for s in edit["shots"]] == [1, 2, 1]
    assert all(s["seconds"] > 0 for s in edit["shots"])


def test_the_passages_are_what_the_voice_is_asked_to_speak(conn, memory_group, audio):
    # Handing the voice our own chunk boundaries is the ONLY way a shot can begin
    # exactly on its sentence — a boundary we did not choose is one we cannot sync to.
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    assert models.segments == [
        FakeModels.PROSE_EN.split(". ")[0] + ".",
        FakeModels.PROSE_EN.split(". ")[1] + ".",
        FakeModels.PROSE_EN.split(". ")[2],
    ]


def test_the_narration_is_the_script_read_end_to_end(conn, memory_group, audio):
    edit = build_presentation(conn, FakeModels(), audio, 1, memory_group).timeline
    assert edit["narration"] == FakeModels.PROSE_EN


def test_title_cards_stay_sparse(conn, memory_group, audio):
    # A card per shot is a caption track, not a film (§11.1).
    edit = build_presentation(conn, FakeModels(), audio, 1, memory_group).timeline
    assert sum(1 for s in edit["shots"] if s["line"]) <= 1


def test_title_cards_are_sparse_and_the_rest_is_wordless(conn, memory_group, audio):
    # A card per shot is a caption track, not a film (§11.1).
    edit = build_presentation(conn, FakeModels(), audio, 1, memory_group).timeline
    assert sorted(s["line"] for s in edit["shots"]) == ["", "", "The water"]


def test_the_narration_travels_with_the_edit(conn, memory_group, audio):
    # The story must be visible in the JSON, not locked inside the WAV.
    edit = build_presentation(conn, FakeModels(), audio, 1, memory_group).timeline
    assert edit["narration"].startswith(FakeModels.PROSE_EN.split(". ")[0] + ".")


def test_the_model_is_shown_the_rich_descriptions(conn, memory_group, audio):
    # Not the four-word title: a story cannot be written from "People by Flower Arch".
    conn.execute("UPDATE photos SET ai_description = 'a long vivid sentence' WHERE id = 1")
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    assert "a long vivid sentence" in dressed(models)
    assert "EACH PHOTOGRAPH" in dressed(models)


def test_the_story_pass_sees_every_signal_not_just_the_description(conn, memory_group,
                                                                   audio):
    # It is the pass that WRITES the memory, and it used to be blind to the labels that
    # turn a description of pictures into a memory of an evening (§11.1).
    for dimension, label in (("occasion", "holiday"), ("season_weather", "snow")):
        conn.execute("INSERT INTO tags(dimension, label) VALUES (?, ?)", (dimension, label))
        tag_id = conn.execute(
            "SELECT id FROM tags WHERE dimension = ? AND label = ?", (dimension, label)
        ).fetchone()["id"]
        conn.execute("INSERT INTO photo_tags(photo_id, tag_id, score, source)"
                     " VALUES (1, ?, 0.9, 'siglip')", (tag_id,))
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    prompt = dressed(models)
    assert "occasion=holiday" in prompt and "season_weather=snow" in prompt
    # And the memory as a WHOLE, which no per-photo line can carry.
    assert "WHAT THIS MEMORY IS" in prompt
    assert "photographs." in prompt


def test_the_story_is_written_from_the_photographs_not_the_labels(conn, memory_group,
                                                                    audio):
    conn.execute("UPDATE photos SET ai_description = 'friends around a campfire' "
                 "WHERE id = 2")
    conn.execute("INSERT INTO tags(dimension, label) VALUES ('season_weather', 'rain')")
    tag_id = conn.execute("SELECT id FROM tags WHERE label = 'rain'").fetchone()["id"]
    conn.execute("INSERT INTO photo_tags(photo_id, tag_id, score, source)"
                 " VALUES (1, ?, 0.3, 'siglip')", (tag_id,))
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    prompt = told(models)
    assert "1. caption 1" in prompt and "2. friends around a campfire" in prompt
    assert "rain" not in prompt


def test_the_narration_is_the_story_script_not_the_captions(conn, memory_group, audio):
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    assert models.spoken[0][0].startswith(FakeModels.PROSE_EN.split(". ")[0] + ".")


def test_the_music_brief_leads_with_the_memorys_own_mood(conn, memory_group, audio):
    # A small model reaches for "warm ambient pads, gentle" whatever it is shown, and
    # that is funeral music over a birthday (§11.1). So the mood-derived clause goes
    # FIRST and is not optional; the story pass's sentence is kept after it as colour.
    for pid, (dimension, label) in ((1, ("occasion", "birthday")),
                                    (1, ("vibe", "festive"))):
        conn.execute("INSERT INTO tags(dimension, label) VALUES (?, ?)", (dimension, label))
        tag_id = conn.execute("SELECT id FROM tags WHERE dimension = ? AND label = ?",
                              (dimension, label)).fetchone()["id"]
        conn.execute("INSERT INTO photo_tags(photo_id, tag_id, score, source)"
                     " VALUES (?, ?, 0.9, 'siglip')", (pid, tag_id))
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    briefs = [section["brief"] for section in models.scores[0]["sections"]]
    assert briefs[0].startswith("Music for a birthday party.")
    assert all("BPM" in brief for brief in briefs)   # a tempo, not a vibe word
    assert "warm ambient drone, no vocals" in " ".join(briefs)  # the model's sentence


class LongFilm(FakeModels):
    """A film long enough to carry a score. The default fixture's three passages make a
    nine-second film, which is one section by design (`feed/score.py`)."""

    # Twenty beats, each long enough to hold a photograph on its own.
    STORY = " ".join(f"Passage number {i}, long enough to be a beat of its own."
                     for i in range(20))

    def text_complete(self, model, messages, *, json_schema=None, **kw):
        prompt = messages[-1]["content"]
        if "WHAT THE DAY WAS" in prompt or "ЧТО ЭТО ЗА ДЕНЬ" in prompt:
            self.story_prompts.append(prompt)
            return json.dumps({"story": self.STORY})
        if "EACH PHOTOGRAPH" in prompt:
            self.story_prompts.append(prompt)
            return json.dumps({
                "script": [{"sentence": i + 1, "shot": (i % 3) + 1, "voice": ""}
                           for i in range(20)],
                "narration": "", "narrator": "", "cards": [],
                "music_prompt": "warm ambient drone, no vocals",
                "score": [{"from": 1, "brief": "solo piano"},
                          {"from": 11, "brief": "drums and brass"}],
            })
        return super().text_complete(model, messages, json_schema=json_schema, **kw)


def test_the_score_is_sectioned_and_covers_the_film(conn, memory_group, audio):
    # One clip looped under everything is what made every film sound the same (§11.1).
    models = LongFilm()
    result = build_presentation(conn, models, audio, 1, memory_group)
    sections = models.scores[0]["sections"]
    assert 2 <= len(sections) <= 4
    assert "solo piano" in sections[0]["brief"]
    assert any("drums and brass" in s["brief"] for s in sections[1:])
    assert len({s["brief"] for s in sections}) == len(sections)   # they differ
    assert sum(s["seconds"] for s in sections) > 0
    assert result.music_key is not None


def test_the_score_is_sampled_from_the_memorys_energy(conn, memory_group, audio):
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    from feed.mood import CALM, SAMPLING

    assert models.scores[0]["guidance"] == SAMPLING[CALM]["guidance"]
    assert models.scores[0]["temperature"] == SAMPLING[CALM]["temperature"]


def test_the_same_film_is_always_scored_with_the_same_seed(conn, memory_group, audio):
    first, second = FakeModels(), FakeModels()
    build_presentation(conn, first, audio, 1, memory_group)
    build_presentation(conn, second, audio, 1, memory_group)
    assert first.scores[0]["seed"] == second.scores[0]["seed"]
    assert first.scores[0]["seed"] is not None


def test_the_language_is_recorded_and_passed_to_the_voice(conn, memory_group, audio):
    models = FakeModels()
    result = build_presentation(conn, models, audio, 1, memory_group, language="ru")
    assert result.language == "ru"
    assert models.spoken[0][1] == "ru"
    assert "in Russian" in dressed(models)
    # What is SPOKEN must be the Russian text, never the English first take.
    assert models.spoken[0][0] == FakeModels.PROSE_RU
    assert result.timeline["narration"] == FakeModels.PROSE_RU


def test_both_languages_coexist(conn, memory_group, audio):
    build_presentation(conn, FakeModels(), audio, 1, memory_group, language="en")
    build_presentation(conn, FakeModels(), audio, 1, memory_group, language="ru")
    assert read_presentation(conn, memory_group, "en") is not None
    assert read_presentation(conn, memory_group, "ru") is not None


def test_a_voice_that_cannot_speak_fails_the_whole_film(conn, memory_group, audio):
    # A film with no voice is a slideshow, not a film (§11.1). It must FAIL, loudly,
    # rather than being handed over mute.
    with pytest.raises(NarrationFailed):
        build_presentation(conn, FakeModels(voice_off=True), audio, 1, memory_group)


def test_a_music_slot_that_is_off_still_produces_a_playable_film(conn, memory_group, audio):
    result = build_presentation(conn, FakeModels(music_off=True), audio, 1, memory_group)
    assert result.status == "ready" and result.music_key is None
    assert result.narration_key is not None


def test_a_failed_film_is_recorded_with_its_reason(conn, memory_group, audio):
    # Recorded BEFORE raising, so reloading the page states why instead of offering a
    # bare Prepare button that looks like nothing ever ran.
    with pytest.raises(NarrationFailed):
        build_presentation(conn, FakeModels(voice_off=True), audio, 1, memory_group)
    stored = read_presentation(conn, memory_group, "en")
    assert stored.status == "failed"
    assert "voice" in stored.error and "slot off" in stored.error
    assert stored.narration_key is None and stored.music_key is None


def test_a_failed_film_never_reports_ready(conn, memory_group, audio):
    build_presentation(conn, FakeModels(), audio, 1, memory_group)
    assert read_presentation(conn, memory_group, "en").status == "ready"
    with pytest.raises(NarrationFailed):
        build_presentation(conn, FakeModels(voice_off=True), audio, 1, memory_group)
    assert read_presentation(conn, memory_group, "en").status == "failed"


def test_a_failed_story_pass_fails_the_film(conn, memory_group, audio):
    class NoStory(FakeModels):
        def text_complete(self, model, messages, **kw):
            raise RuntimeError("planner is down")

    # No script means no narration means no film (§11.1) — the edit alone is a
    # slideshow, and the row says so instead of shipping one.
    with pytest.raises(NarrationFailed):
        build_presentation(conn, NoStory(), audio, 1, memory_group)
    stored = read_presentation(conn, memory_group, "en")
    assert stored.status == "failed" and "narration" in stored.error
    # A failed story pass leaves a WORDLESS edit, not 40 machine labels on screen.
    # Showing "Two people outdoors" over every shot is what made the first version
    # read as a slideshow with captions (§11.1).
    assert all(s["line"] == "" for s in stored.timeline["shots"])


def test_a_model_that_raises_something_else_does_not_lose_the_film(conn, memory_group, audio):
    class Exploding(FakeModels):
        def music(self, prompt="", seconds=30.0, seed=None, **kw):
            raise RuntimeError("cuda is on fire")

    result = build_presentation(conn, Exploding(), audio, 1, memory_group)
    assert result.status == "ready" and result.music_key is None


def test_identical_audio_is_stored_once(conn, memory_group, audio):
    build_presentation(conn, FakeModels(), audio, 1, memory_group)
    first = sorted(audio.iter_keys())
    build_presentation(conn, FakeModels(), audio, 1, memory_group)
    assert sorted(audio.iter_keys()) == first


def test_audio_from_a_deleted_memory_is_pruned(conn, memory_group, audio):
    build_presentation(conn, FakeModels(), audio, 1, memory_group)
    audio.write("ee/ff/eeff.wav", b"left over from an older memory")
    build_presentation(conn, FakeModels(), audio, 1, memory_group)
    assert not audio.exists("ee/ff/eeff.wav")


def test_the_signature_records_what_the_film_was_built_against(conn, memory_group, audio):
    from albums.memory_store import current_signature

    result = build_presentation(conn, FakeModels(), audio, 1, memory_group)
    assert result.signature == current_signature(conn, 1)


def test_an_unknown_group_raises(conn, audio):
    with pytest.raises(LookupError):
        build_presentation(conn, FakeModels(), audio, 1, 999)


def test_another_owners_memory_is_not_reachable(conn, memory_group, audio):
    with pytest.raises(LookupError):
        build_presentation(conn, FakeModels(), audio, 2, memory_group)


def test_a_voice_that_raises_reports_the_models_services_own_words(conn, memory_group, audio):
    # "no voice" alone is what made a Russian film's failure take hours to diagnose:
    # Silero refusing a 1 000-symbol script looked identical to a slot being off. The
    # model's own message has to reach the page.
    class Refusing(FakeModels):
        def speak(self, text, language="en", voice=None, segments=None,
              voices=None):
            request = httpx.Request("POST", "http://models/speak")
            raise httpx.HTTPStatusError(
                "500", request=request,
                response=httpx.Response(
                    500, request=request,
                    json={"detail": "Model couldn't generate your text"}),
            )

    with pytest.raises(NarrationFailed):
        build_presentation(conn, Refusing(), audio, 1, memory_group)
    assert "Model couldn't generate your text" in read_presentation(
        conn, memory_group, "en").error


def test_a_missing_soundtrack_records_a_readable_reason(conn, memory_group, audio):
    result = build_presentation(conn, FakeModels(music_off=True), audio, 1, memory_group)
    assert "no music" in result.error


def test_a_complete_film_records_no_reason(conn, memory_group, audio):
    assert build_presentation(conn, FakeModels(), audio, 1, memory_group).error is None


def test_a_narration_the_planner_would_not_translate_is_explained(conn, memory_group, audio):
    # The story pass returns an empty narration when it cannot write in the target
    # language (a Russian voice reading English is worse than silence).
    class EnglishOnly(FakeModels):
        def text_complete(self, model, messages, **kw):
            self.story_prompts.append(messages[-1]["content"])
            return json.dumps({"narration": "Only English.", "cards": [],
                               "music_prompt": "m"})

    with pytest.raises(NarrationFailed):
        build_presentation(conn, EnglishOnly(), audio, 1, memory_group, language="ru")
    stored = read_presentation(conn, memory_group, "ru")
    assert stored.status == "failed" and "would not write this memory in ru" in stored.error


def test_the_viewers_words_are_put_before_every_machine_label(conn, memory_group, audio):
    # Testimony outranks the labels, so it is read FIRST — a prompt that buries it
    # under forty lines of tags gets a story about the tags (§11.3).
    from feed.notes import add_note

    add_note(conn, memory_group, 6.0, "this is my brother Kostya")
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    prompt = dressed(models)
    assert "this is my brother Kostya" in prompt
    assert prompt.index("TOLD YOU") < prompt.index("EACH PHOTOGRAPH")
    assert "EVERY note must be reflected" in prompt
    # And it says what to do when a note and a caption disagree.
    assert "the note is right" in prompt


def test_a_note_is_told_which_photograph_it_was_written_over(conn, memory_group, audio):
    # "This is Julia" is a fact about a PICTURE, not about a second. Without knowing
    # which photograph was on screen, the model can only patch a sentence near the
    # timestamp — which is the failure this exists to prevent (§11.3).
    from feed.notes import add_note

    build_presentation(conn, FakeModels(), audio, 1, memory_group)   # a cut to measure
    add_note(conn, memory_group, 2.5, "this is Julia")
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    prompt = dressed(models)
    assert "looking at photograph" in prompt
    assert "this is Julia" in prompt


def test_the_prompt_demands_the_fact_hold_for_the_whole_film(conn, memory_group, audio):
    from feed.notes import add_note

    add_note(conn, memory_group, 1.0, "this is Julia")
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    prompt = dressed(models)
    assert "FACT ABOUT THE WHOLE MEMORY" in prompt
    assert "BEFORE that point" in prompt          # earlier passages must know it too
    assert "wherever that photograph comes back" in prompt


def test_the_narration_is_still_spoken_as_one_clip(conn, memory_group, audio):
    # Merged, not several files: the player has one <audio> element and one clock.
    from feed.notes import add_note

    add_note(conn, memory_group, 1.0, "[женский голос] x")
    result = build_presentation(conn, FakeModels(), audio, 1, memory_group)
    assert result.narration_key is not None
    assert audio.exists(result.narration_key)


def test_a_cast_the_model_invented_is_discarded(conn, memory_group, audio):
    # An unknown speaker raises inside the TTS child and fails the whole film (§11.1),
    # so it never gets that far.
    class Inventive(FakeModels):
        def text_complete(self, model, messages, *, json_schema=None, **kw):
            raw = super().text_complete(model, messages, json_schema=json_schema, **kw)
            if "EACH PHOTOGRAPH" not in messages[-1]["content"]:
                return raw
            payload = json.loads(raw)
            payload["narrator"] = "count_dracula"
            return json.dumps(payload)

    models = Inventive()
    build_presentation(conn, models, audio, 1, memory_group)
    from feed.voice import offered

    assert models.spoken_voice in {s.name for s in offered("en")}


def test_a_cast_the_model_chose_is_used(conn, memory_group, audio):
    class Casting(FakeModels):
        def text_complete(self, model, messages, *, json_schema=None, **kw):
            raw = super().text_complete(model, messages, json_schema=json_schema, **kw)
            if "EACH PHOTOGRAPH" not in messages[-1]["content"]:
                return raw
            payload = json.loads(raw)
            payload["narrator"] = "bm_george"
            payload["script"][1]["voice"] = "af_bella"
            return json.dumps(payload)

    bracket_the_second_passage(conn, memory_group, audio)
    models = Casting()
    build_presentation(conn, models, audio, 1, memory_group)
    assert models.spoken_voice == "bm_george"
    assert models.spoken_voices[1] == "af_bella"
    assert models.spoken_voices[0] == "bm_george"


def test_the_story_pass_is_shown_the_voices_it_may_cast(conn, memory_group, audio):
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    prompt = dressed(models)
    assert "THE VOICES YOU MAY CAST" in prompt
    assert "am_michael — male, deep" in prompt          # straight from the catalog
    assert "Use these names EXACTLY" in prompt


def test_the_prompt_demands_the_notes_keep_their_order(conn, memory_group, audio):
    # Notes are a sequence of moments, not a bag of facts: one at 0:07 happened before
    # one at 0:11, and a story that doubles back contradicts itself (§11.3).
    from feed.notes import add_note

    add_note(conn, memory_group, 7.0, "the street was dark")
    add_note(conn, memory_group, 11.0, "she got ready")
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    prompt = dressed(models)
    assert "THAT ORDER IS THE ORDER OF THE DAY" in prompt
    # And they are listed in time order for it to follow.
    assert prompt.index("the street was dark") < prompt.index("she got ready")


def test_the_notes_reach_every_pass_whole(conn, memory_group, audio):
    # Nothing is parsed out of them: the story pass and the conductor are each shown
    # the note verbatim and read any instruction themselves (§11.3).
    from feed.notes import add_note

    add_note(conn, memory_group, 1.0, "[страшных хриплый мужкской голос] тут было жутко")
    models = FakeModels()
    build_presentation(conn, models, audio, 1, memory_group)
    story_prompt = dressed(models)
    conductor_prompt = next(p for p in models.story_prompts
                            if "THE PHOTOGRAPHS available to you" in p)
    for prompt in (story_prompt, conductor_prompt):
        assert "[страшных хриплый мужкской голос]" in prompt
        assert "SQUARE BRACKETS" in prompt          # told they are instructions


def test_the_film_uses_the_cast_the_model_chose(conn, memory_group, audio):
    class Casting(FakeModels):
        def text_complete(self, model, messages, *, json_schema=None, **kw):
            raw = super().text_complete(model, messages, json_schema=json_schema, **kw)
            if "EACH PHOTOGRAPH" not in messages[-1]["content"]:
                return raw
            payload = json.loads(raw)
            payload["narrator"] = "af_heart"
            payload["script"][1]["voice"] = "am_michael"
            return json.dumps(payload)

    bracket_the_second_passage(conn, memory_group, audio)
    models = Casting()
    build_presentation(conn, models, audio, 1, memory_group)
    assert models.spoken_voice == "af_heart"
    # ONE passage repainted, the rest in the film's own voice.
    assert models.spoken_voices == ["af_heart", "am_michael", "af_heart"]


def test_an_invented_cast_never_reaches_the_voice_model(conn, memory_group, audio):
    class Inventive(FakeModels):
        def text_complete(self, model, messages, *, json_schema=None, **kw):
            raw = super().text_complete(model, messages, json_schema=json_schema, **kw)
            if "EACH PHOTOGRAPH" not in messages[-1]["content"]:
                return raw
            payload = json.loads(raw)
            payload["narrator"] = "count_dracula"
            payload["script"][0]["voice"] = "also_not_real"
            return json.dumps(payload)

    from feed.voice import offered

    models = Inventive()
    build_presentation(conn, models, audio, 1, memory_group)
    real = {s.name for s in offered("en")}
    assert models.spoken_voice in real
    assert all(v in real for v in models.spoken_voices if v)


def test_a_bracket_that_gets_no_voice_makes_the_pass_try_again(conn, memory_group,
                                                               audio):
    # The model ignoring a `[voice]` bracket is silent: the film looks fine and one
    # thing the person asked for is simply gone. Code notices the bracket exists and
    # was not answered — it never reads what it MEANS (§11.3).
    from feed.notes import add_note

    class Stubborn(FakeModels):
        """Casts nothing the first time, then obeys."""

        def __init__(self):
            super().__init__()
            self.story_calls = 0

        def text_complete(self, model, messages, *, json_schema=None, **kw):
            prompt = messages[-1]["content"]
            raw = super().text_complete(model, messages, json_schema=json_schema, **kw)
            if "EACH PHOTOGRAPH" not in prompt:
                # The story call, the one-sentence rewrite and the "carry it on" call
                # are different jobs; only the CUT-AND-DRESS call casts (§11.1).
                return raw
            self.story_calls += 1
            payload = json.loads(raw)
            if self.story_calls > 1:
                payload["script"][1]["voice"] = "am_michael"
            return json.dumps(payload)

    add_note(conn, memory_group, 1.0, "[страшных хриплый мужкской голос] тут было жутко")
    models = Stubborn()
    build_presentation(conn, models, audio, 1, memory_group)
    assert models.story_calls == 2, "it should have been asked again"
    assert "am_michael" in models.spoken_voices
    # And the complaint says what was wrong without saying what the bracket means.
    complaint = [p for p in models.story_prompts if "EACH PHOTOGRAPH" in p][1]
    assert "CAST NO VOICE" in complaint


def test_a_note_with_no_bracket_never_triggers_a_retry(conn, memory_group, audio):
    from feed.notes import add_note

    class Counting(FakeModels):
        def __init__(self):
            super().__init__()
            self.story_calls = 0

        def text_complete(self, model, messages, *, json_schema=None, **kw):
            prompt = messages[-1]["content"]
            # Only a full TELLING of the story counts. The conductor, the one-sentence
            # note rewrite and the "carry it on" call are different jobs (§11.1/§11.3).
            if "EACH PHOTOGRAPH" in prompt:
                self.story_calls += 1
            return super().text_complete(model, messages, json_schema=json_schema, **kw)

    add_note(conn, memory_group, 1.0, FakeModels.PROSE_EN.split(". ")[0] + ".")
    models = Counting()
    build_presentation(conn, models, audio, 1, memory_group)
    assert models.story_calls == 1


def test_the_film_records_every_choice_it_made(conn, memory_group, audio):
    # "Which voice read this?" must have an answer after the build thread has ended.
    models = FakeModels()
    result = build_presentation(conn, models, audio, 1, memory_group)
    d = result.decisions
    assert d["language"] == "en" and d["planner_model"]
    assert d["cast"]["narrator_used"] in d["cast"]["offered"]
    assert d["score"]["sections"] and d["score"]["seed"]
    assert d["film"]["runtime"] > 0
    # And it survives the round trip through SQLite.
    from feed.store import read_presentation

    assert read_presentation(conn, memory_group, "en").decisions == d


def test_a_note_asking_for_a_voice_that_was_never_cast_is_reported(conn, memory_group, audio):
    # The bug this exists for: a bracket asking for a stern male voice, silently read
    # by whoever the fallback picked, with nothing anywhere saying it was ignored.
    from feed.notes import add_note

    add_note(conn, memory_group, 0.0, "[суровый мужской голос] это было утро")
    result = build_presentation(conn, FakeModels(), audio, 1, memory_group)
    assert result.decisions["cast"]["notes_asked_for_a_voice"] is True
    assert "asked for a different voice" in (result.error or "")


def test_only_a_bracketed_passage_keeps_the_models_extra_voice():
    # The model fills in EVERY passage now that `voice` is required (measured: aidar,
    # baya, kseniya over three passages when one note asked for one voice). Only the
    # passage a bracket points at may override the narrator.
    from feed.build import _bracketed_passages, _passage_voices
    from feed.notes import Note
    from feed.story import Story

    story = Story(narration="a b c", cards=(), music_prompt="",
                  script=((0, "a"), (1, "b"), (2, "c")),
                  voices=("aidar", "baya", "kseniya"))
    notes = [Note(id=1, group_id=1, at_seconds=0.0, text="[суровый мужской голос] утро",
                  created_at="now")]
    bracketed = _bracketed_passages(notes, film_seconds=30.0, passages=3)
    assert bracketed == {0}
    assert _passage_voices(story, "ru", "xenia", bracketed) == ["aidar", "xenia", "xenia"]


def test_a_note_with_no_bracket_marks_no_passage():
    from feed.build import _bracketed_passages
    from feed.notes import Note

    notes = [Note(id=1, group_id=1, at_seconds=10.0, text="это Юля", created_at="now")]
    assert _bracketed_passages(notes, film_seconds=30.0, passages=3) == set()


def test_without_any_bracket_the_model_casts_no_passage():
    # Measured: no notes, the model filled every passage with `me`, and a film the user
    # had given to `iulia` was spoken entirely by `me`.
    from feed.build import _bracketed_passages, _passage_voices
    from feed.story import Story

    story = Story(narration="a b", cards=(), music_prompt="",
                  script=((0, "a"), (1, "b")), voices=("aidar", "aidar"))
    for notes, seconds in (([], 0.0), ([], 30.0)):
        bracketed = _bracketed_passages(notes, film_seconds=seconds, passages=2)
        assert _passage_voices(story, "ru", "xenia", bracketed) == []


def test_without_a_previous_cut_nothing_is_gated():
    # No cut to measure against means no note has a known position (§11.3).
    from feed.build import _bracketed_passages
    from feed.notes import Note

    notes = [Note(id=1, group_id=1, at_seconds=0.0, text="[x] a", created_at="now")]
    assert _bracketed_passages(notes, film_seconds=0.0, passages=3) is None


# --- the long stage reports its own progress (§11.1) ---------------------------------


def test_the_music_stage_moves_while_it_waits():
    # A number that does not move for five minutes is indistinguishable from a hang.
    import time

    from feed.build import _Ticking

    seen = []
    with _Ticking(lambda stage, fraction: seen.append((stage, fraction)),
                  "composing the music", (0.92, 0.97), expect=4.0):
        time.sleep(2.2)
    assert len(seen) >= 2, "the bar never ticked"
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions), "the bar went backwards"
    assert all(0.92 <= f < 0.97 for f in fractions), fractions
    assert all("composing the music" in stage for stage, _ in seen)


def test_the_music_stage_counts_down_and_never_reaches_the_next_one():
    import time

    from feed.build import _Ticking

    seen = []
    # An estimate that has already run out: it must hold, not overshoot.
    with _Ticking(lambda stage, fraction: seen.append((stage, fraction)),
                  "composing the music", (0.92, 0.97), expect=0.5):
        time.sleep(1.4)
    assert seen
    assert max(f for _, f in seen) < 0.97
    assert any("nearly there" in stage for stage, _ in seen)


def test_a_failing_progress_callback_never_breaks_the_film():
    import time

    from feed.build import _Ticking

    def explode(stage, fraction):
        raise RuntimeError("the page went away")

    with _Ticking(explode, "composing the music", (0.92, 0.97), expect=2.0):
        time.sleep(1.2)      # the thread dies quietly; the build carries on


def test_no_progress_callback_at_all_is_fine():
    import time

    from feed.build import _Ticking

    with _Ticking(None, "composing the music", (0.92, 0.97), expect=2.0):
        time.sleep(0.1)


ADAPTER = b"adapter-safetensors-v1"
WORDS = "The words I said in this take."
FAKE_ASR = "fake-asr@fake"
NOT_HEARD_TEXT = ("The words of «Mum»'s English takes are not heard yet. Open «Mum» under Voices "
                  "and let it hear them, or choose another narrator.")


class SavedModels(FakeModels):
    """A models service that speaks saved voices: `speak` takes `references` and answers
    with a revision and, per token, the conditioning it built from the take (`conditioning`),
    as `POST /speak` does (§5.1, D86). `conditioning` replaces a token's entry; None drops it.
    `during_story` runs as the story pass casts the film."""

    def __init__(self, *, narrator: str = "", cast: str = "", conditioning: dict | None = None,
                 refuse: str | None = None, during_speak=None, during_music=None,
                 during_story=None, speak_seconds: float = 0.0) -> None:
        super().__init__()
        self.narrator, self.cast = narrator, cast
        self.conditioning = conditioning or {}
        self.refuse = refuse
        self.during_speak, self.during_music = during_speak, during_music
        self.during_story = during_story
        self.speak_seconds = speak_seconds
        self.speak_calls = 0
        self.references: dict | None = None

    def text_complete(self, model, messages, *, json_schema=None, **kw):
        raw = super().text_complete(model, messages, json_schema=json_schema, **kw)
        if "EACH PHOTOGRAPH" not in messages[-1]["content"]:
            return raw
        if self.during_story is not None:
            self.during_story()
        payload = json.loads(raw)
        payload["narrator"] = self.narrator
        payload["script"][1]["voice"] = self.cast
        return json.dumps(payload)

    def speak(self, text, language="en", voice=None, segments=None, voices=None,
              references=None):
        self.speak_calls += 1
        self.references = references
        if self.refuse is not None:
            request = httpx.Request("POST", "http://models/speak")
            raise httpx.HTTPStatusError("503", request=request, response=httpx.Response(
                503, request=request, json={"detail": self.refuse}))
        reply = super().speak(text, language, voice, segments, voices)
        if self.speak_seconds:
            import time

            time.sleep(self.speak_seconds)
        if self.during_speak is not None:
            self.during_speak()
        conditioning = {name: {"state": "built", "model": "voxcpm2", "revision": "rev-1",
                               "reason": None, "blob": None}
                        for name in (references or {})}
        for name, entry in self.conditioning.items():
            if entry is None:
                conditioning.pop(name, None)
            else:
                conditioning[name] = entry
        return reply | {"model": "voxcpm2", "revision": "rev-1", "conditioning": conditioning,
                        "adapters": {name: {"state": "none", "sha256": None} for name in (references or {})}}

    def music(self, prompt="", seconds=30.0, seed=None, **kw):
        reply = super().music(prompt, seconds, seed, **kw)
        if self.during_music is not None:
            self.during_music()
        return reply


class CountingStorage(LocalStorage):
    """Film audio storage that counts writes and deletes, and the stage each write ran in."""

    def __init__(self, root):
        super().__init__(root)
        self.writes: list[str] = []
        self.deletes: list[str] = []
        self.stage = None

    def write(self, key, data):
        self.writes.append(self.stage)
        super().write(key, data)

    def delete(self, key):
        self.deletes.append(key)
        super().delete(key)

    def reset(self):
        self.writes.clear()
        self.deletes.clear()

    def progress(self, stage, fraction):
        self.stage = stage


def clip(freq: float) -> bytes:
    """A valid example: six seconds of tone, mono PCM-16 at 48 kHz (D21)."""
    t = np.arange(6 * 48000) / 48000
    return encode_wav((0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32), 48000)


@pytest.fixture
def voices(tmp_path):
    return LocalStorage(tmp_path / "voices")


def bake(conn, voices, voice_id, *, data=ADAPTER, revision=None):
    """Store a voice model of `voice_id`, learned from its current takes, as a finished
    bake in `app` does (D72): fake ASR hears the waiting takes, the run is recorded,
    and its snapshot is stored. `revision` other than the catalog's pin makes it unusable."""
    from feed.saved_voices import (
        bake_snapshot,
        read_bake_run,
        save_voice_model,
        set_heard_words,
        start_bake_run,
        takes_to_hear,
    )
    from models import catalog

    for take in takes_to_hear(conn, 1, voice_id):
        set_heard_words(conn, 1, voice_id, take.id, text=WORDS, model=FAKE_ASR)
    bake_key = start_bake_run(conn, 1, voice_id, bake_snapshot(conn, 1, voice_id))
    run = read_bake_run(conn, 1, bake_key)
    entry = catalog.get("voice", "voxcpm2")
    status = {"state": "done", "model": "voxcpm2", "revision": revision or entry.source.revision,
              "lora": json.dumps(entry.adapter.lora, sort_keys=True), "config": {"steps": 4},
              "dataset_sha256": run.dataset_hash}
    return save_voice_model(conn, voices, 1, voice_id, bake_key=bake_key, data=data,
                            status=status, takes=run.takes, dataset_hash=run.dataset_hash)


def hear(conn, example, words=WORDS):
    """What the hearing job stores on a take (§11.1)."""
    from feed.saved_voices import set_heard_words

    return set_heard_words(conn, 1, example.voice_id, example.id, text=words, model=FAKE_ASR)


def saved_voice(conn, voices, name, *, language="en", freq=220.0, heard=True):
    """A saved voice with one take in `language`; `heard` stores the words ASR heard in it."""
    from feed.saved_voices import add_example, create_voice

    voice = create_voice(conn, 1, name)
    example = add_example(conn, voices, 1, voice.id, language=language, source="uploaded",
                          data=clip(freq))
    if heard:
        example = hear(conn, example)
    return voice.id, example


def second_example(conn, voices, voice_id, *, freq=330.0, language="en"):
    from feed.saved_voices import add_example

    return add_example(conn, voices, 1, voice_id, language=language, source="uploaded",
                       data=clip(freq))


def narrate_with(conn, group_id, voice_id, language="en"):
    from feed.saved_voices import set_narrator

    set_narrator(conn, group_id, language, voice_id)


def delete_now(conn, voices, voice_id=None, example=None):
    """What the Voices page and the drain do: enqueue, then cascade (D33)."""
    from feed.saved_voices import (
        process_voice_deletions,
        request_example_deletion,
        request_voice_deletion,
    )

    if example is not None:
        assert request_example_deletion(conn, 1, example.voice_id, example.id)
    else:
        assert request_voice_deletion(conn, 1, voice_id)
    assert process_voice_deletions(conn, voices, 1) == 0


def failed_with(conn, group_id, text, language="en"):
    stored = read_presentation(conn, group_id, language)
    assert stored.status == "failed", stored
    assert text in stored.error, stored.error
    assert stored.narration_key is None and stored.music_key is None
    return stored


def test_a_saved_narrator_film_shows_the_story_pass_no_voice(conn, memory_group, audio,
                                                             voices):
    mum, _ = saved_voice(conn, voices, "Mum")
    saved_voice(conn, voices, "Dad", freq=330.0)
    narrate_with(conn, memory_group, mum)
    models = SavedModels()
    build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    prompt = dressed(models)
    assert "THE VOICES YOU MAY CAST" not in prompt
    assert "a saved voice" not in prompt
    assert "am_michael" not in prompt and "af_heart" not in prompt   # no preset at all


def test_the_saved_narrator_speaks_from_its_newest_heard_take(conn, memory_group, audio,
                                                              voices):
    from feed.saved_voices import reference_for, token

    mum, first = saved_voice(conn, voices, "Mum")
    newest = hear(conn, second_example(conn, voices, mum))
    narrate_with(conn, memory_group, mum)
    models = SavedModels()
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    assert result.status == "ready"
    assert models.speak_calls == 1
    assert models.spoken_voice == token(mum)
    assert models.spoken_voices == []                      # no bracket, one voice
    assert models.references == {token(mum): reference_for(conn, voices, newest)}
    sent = models.references[token(mum)]
    assert (sent["sha256"], sent["transcript"], sent["language"]) == (newest.content_hash, WORDS, "en")
    assert "adapter" not in sent and "use_clip" not in sent
    assert first.id != newest.id


def test_a_saved_film_records_the_take_it_used(conn, memory_group, audio, voices):
    from feed.saved_voices import token

    mum, example = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    result = build_presentation(conn, SavedModels(), audio, 1, memory_group, voices=voices)
    voice = result.decisions["voice"]
    assert voice["saved"] is True
    assert voice["model"] == "voxcpm2" and voice["revision"] == "rev-1"
    assert voice["takes"] == {token(mum): {"example_id": example.id, "sha256": example.content_hash,
                                           "state": "built"}}
    assert "models" not in voice and "conditioning" not in voice
    assert voice["wall_seconds"] >= 0 and voice["narration_seconds"] == 6.0
    assert voice["seed"] == "unsupported (voxcpm 2.0.3)"
    assert result.decisions["cast"]["narrator_used"] == token(mum)
    assert result.decisions["cast"]["narrator_model"] == f"spoken from take #{example.id}"
    assert read_presentation(conn, memory_group, "en").decisions == result.decisions


def test_a_saved_film_stores_no_conditioning(conn, memory_group, audio, tmp_path):
    watched = WatchedStorage(tmp_path / "watched")
    mum, _ = saved_voice(conn, watched, "Mum")
    narrate_with(conn, memory_group, mum)
    watched.lock = threading.Lock()
    build_presentation(conn, SavedModels(), audio, 1, memory_group, voices=watched)
    assert [kind for kind, _ in watched.held] == ["read"]
    assert conn.execute("SELECT COUNT(*) FROM voice_conditionings").fetchone()[0] == 0


@pytest.mark.parametrize("words", [None, "♪ ♪"], ids=["waiting", "unusable"])
def test_a_newer_take_with_no_usable_words_yet_is_skipped(conn, memory_group, audio, voices,
                                                          words):
    from feed.saved_voices import token

    mum, first = saved_voice(conn, voices, "Mum")
    newer = second_example(conn, voices, mum)
    if words is not None:
        hear(conn, newer, words)
    narrate_with(conn, memory_group, mum)
    models = SavedModels()
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    assert result.status == "ready"
    assert models.references[token(mum)]["sha256"] == first.content_hash


def test_a_bracket_voice_with_no_heard_take_is_never_offered(conn, memory_group, audio,
                                                            voices):
    from feed.saved_voices import token

    mum, _ = saved_voice(conn, voices, "Mum")
    saved_voice(conn, voices, "Dad", freq=330.0, heard=False)
    narrate_with(conn, memory_group, mum)
    models = SavedModels(cast="Dad")
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    assert "Dad" not in dressed(models)
    assert models.spoken_voices == []
    assert list(models.references) == [token(mum)]
    assert result.decisions["cast"]["offered"] == ["Mum"]


def test_a_preset_bracket_in_a_saved_film_falls_back_to_the_narrator(conn, memory_group,
                                                                    audio, voices):
    # Saved voices and presets never mix in one film (D17).
    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    models = SavedModels(cast="am_michael")
    build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    assert models.spoken_voices == []
    assert all(name.startswith("saved:") for name in models.references)


def test_the_users_choice_wins_over_the_story_passs_narrator(conn, memory_group, audio,
                                                             voices):
    from feed.saved_voices import token

    mum, _ = saved_voice(conn, voices, "Mum")
    saved_voice(conn, voices, "Dad", freq=330.0)
    narrate_with(conn, memory_group, mum)
    models = SavedModels(narrator="Dad")
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    assert models.spoken_voice == token(mum)
    assert result.decisions["cast"]["narrator_chosen_by"] == "user"
    assert result.decisions["cast"]["offered"] == ["Dad", "Mum"]


def test_a_saved_voice_whose_takes_are_not_heard_fails_the_film(conn, memory_group, audio,
                                                                voices):
    mum, _ = saved_voice(conn, voices, "Mum", heard=False)
    narrate_with(conn, memory_group, mum)
    models = SavedModels()
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group, NOT_HEARD_TEXT)
    assert models.speak_calls == 0 and models.story_prompts == []   # never Automatic


def test_a_saved_voice_whose_takes_have_no_usable_words_fails_the_film(conn, memory_group,
                                                                      audio, voices):
    mum, example = saved_voice(conn, voices, "Mum", heard=False)
    hear(conn, example, "♪ ♪")
    narrate_with(conn, memory_group, mum)
    models = SavedModels()
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group, "No English take of «Mum» has usable words. Record a take "
                "where you speak, or choose another narrator.")
    assert models.speak_calls == 0


def test_a_saved_voice_with_no_english_take_fails_an_english_film(conn, memory_group,
                                                                  audio, voices):
    mum, _ = saved_voice(conn, voices, "Mum", language="ru")
    narrate_with(conn, memory_group, mum)
    models = SavedModels()
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group, "«Mum» has no English take yet. Record one under Voices, "
                "or choose another narrator.")
    assert models.speak_calls == 0


def test_a_saved_voice_needs_no_voice_model(conn, memory_group, audio, voices):
    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    assert conn.execute("SELECT COUNT(*) FROM voice_models").fetchone()[0] == 0
    result = build_presentation(conn, SavedModels(), audio, 1, memory_group, voices=voices)
    assert result.status == "ready"


def test_a_saved_film_without_the_voices_store_fails_the_film(conn, memory_group, audio,
                                                             voices):
    mum, example = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    models = SavedModels()
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group)
    failed_with(conn, memory_group, f"Take #{example.id} of «Mum» could not be read")
    assert models.speak_calls == 0


def test_a_refusal_fails_the_film_with_one_speak_call(conn, memory_group, audio, voices):
    # Never a second call with a preset (D17).
    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    models = SavedModels(refuse="voice copy is not available: no cloning model on jetson")
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group, "voice copy is not available: no cloning model on jetson")
    assert models.speak_calls == 1


def test_a_missing_take_file_fails_the_film(conn, memory_group, audio, voices):
    mum, example = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    voices.delete(example.storage_key)
    models = SavedModels()
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group, f"Take #{example.id} of «Mum» could not be read (")
    assert models.speak_calls == 0


@pytest.mark.parametrize("entry", [
    None,                                                   # the reply names no take for it
    {"state": "failed", "model": "voxcpm2"},                # no conditioning built or restored
])
def test_a_reply_that_did_not_speak_from_the_take_fails_the_film(
        conn, memory_group, voices, tmp_path, entry):
    from feed.saved_voices import token

    audio = CountingStorage(tmp_path / "audio")
    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    models = SavedModels(conditioning={token(mum): entry})
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group, "The models service did not speak «Mum» from its take.")
    assert audio.writes == []


def test_only_the_saved_narrators_take_must_be_in_the_reply(conn, memory_group, audio,
                                                           voices):
    from feed.saved_voices import token

    mum, _ = saved_voice(conn, voices, "Mum")
    dad, _ = saved_voice(conn, voices, "Dad", freq=330.0)
    bracket_the_second_passage(conn, memory_group, audio)
    narrate_with(conn, memory_group, mum)
    models = SavedModels(cast="Dad", conditioning={token(dad): None})
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    assert result.status == "ready"
    assert set(result.decisions["voice"]["takes"]) == {token(mum)}


def test_a_restored_conditioning_is_a_reply_from_the_take(conn, memory_group, audio, voices):
    from feed.saved_voices import token

    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    models = SavedModels(conditioning={token(mum): {"state": "restored", "blob": None}})
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    assert result.status == "ready"
    assert result.decisions["voice"]["takes"][token(mum)]["state"] == "restored"


def test_the_saved_voice_stage_moves_while_it_waits(conn, memory_group, audio, voices):
    # Voice copy can take minutes (D28): the bar must move on its estimate.
    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    seen = []
    build_presentation(conn, SavedModels(speak_seconds=1.3), audio, 1, memory_group,
                       voices=voices, progress=lambda stage, f: seen.append((stage, f)))
    ticks = [(stage, f) for stage, f in seen if stage.startswith("recording the voice —")]
    assert ticks, seen
    assert all(0.30 <= f < 0.92 for _, f in ticks)


def test_a_narrator_being_deleted_fails_the_film(conn, memory_group, audio, voices):
    from feed.saved_voices import request_voice_deletion

    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    assert request_voice_deletion(conn, 1, mum)          # pending, not drained
    models = SavedModels()
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group,
                "The saved voice «Mum» is being deleted. Choose another narrator.")
    assert models.speak_calls == 0                       # and never Automatic


def test_a_take_deleted_between_the_cast_reads_fails_the_film_with_its_language(
        conn, memory_group, audio, voices, monkeypatch):
    from feed import build
    from feed.saved_voices import castable_takes

    mum, english = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)

    def delete_english_then_cast(conn, owner_id, language):
        delete_now(conn, voices, example=english)
        hear(conn, second_example(conn, voices, mum, language="ru"))
        return castable_takes(conn, owner_id, language)

    monkeypatch.setattr(build, "castable_takes", delete_english_then_cast)
    models = SavedModels()
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group,
                "«Mum» has no English take yet. Record one under Voices, or choose another narrator.")
    assert models.speak_calls == 0


def test_a_deleted_narrator_fails_the_film(conn, memory_group, audio, voices):
    from feed.saved_voices import NarratorChoice, read_narrator

    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    delete_now(conn, voices, mum)
    assert read_narrator(conn, memory_group, "en") == NarratorChoice(None)
    models = SavedModels()
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group,
                "The saved voice chosen for this film was deleted. Choose another narrator.")
    assert models.speak_calls == 0


# Deleted while the film is made: the ready save checks every used voice under the same
# write lock as the WAV writes, so a miss writes no film audio at all (D33, D34).


def test_a_narrator_deleted_during_the_build_fails_the_film(conn, memory_group, audio,
                                                           voices):
    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    models = SavedModels(during_speak=lambda: delete_now(conn, voices, mum))
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group, "The saved voice «Mum» was deleted while this film was made.")


def test_a_voice_the_story_pass_cast_deleted_during_the_build_leaves_the_film_ready(
        conn, memory_group, audio, voices):
    from feed.saved_voices import token

    mum, _ = saved_voice(conn, voices, "Mum")
    dad, _ = saved_voice(conn, voices, "Dad", freq=330.0)
    bracket_the_second_passage(conn, memory_group, audio)
    narrate_with(conn, memory_group, mum)
    models = SavedModels(cast="Dad", during_speak=lambda: delete_now(conn, voices, dad))
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    assert result.status == "ready"
    assert models.spoken_voice == token(mum) and models.spoken_voices == []


def test_the_spoken_take_deleted_during_the_build_leaves_the_film_ready(conn, memory_group,
                                                                       audio, voices):
    mum, example = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    models = SavedModels(during_speak=lambda: delete_now(conn, voices, example=example))
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    assert result.status == "ready"
    assert read_presentation(conn, memory_group, "en").status == "ready"
    assert result.decisions["cast"]["narrator_model"] == f"spoken from take #{example.id}"


@pytest.mark.parametrize("when", ["narrator after synthesis", "narrator after music"])
def test_a_failed_saved_film_writes_and_deletes_no_film_audio(conn, memory_group, voices,
                                                              tmp_path, when):
    from feed.saved_voices import token

    audio = CountingStorage(tmp_path / "audio")
    # Another READY film, whose WAVs carry the same bytes this film would write: the
    # fake speaks the same silence for the same three passages.
    other = build_presentation(conn, FakeModels(), audio, 1, memory_group, language="ru")
    assert other.status == "ready"
    bracket_the_second_passage(conn, memory_group, audio)
    # Another film IN FLIGHT, between its WAV write and its ready row.
    in_flight = "ab/cd/abcd-in-flight.wav"
    audio.write(in_flight, b"another build's narration")
    audio.reset()

    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    delete = {
        "narrator after synthesis": {"during_speak": lambda: delete_now(conn, voices, mum)},
        "narrator after music": {"during_music": lambda: delete_now(conn, voices, mum)},
    }[when]
    models = SavedModels(**delete)
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group, "The saved voice «Mum» was deleted while this film was made.")
    assert set(models.references) == {token(mum)}
    assert audio.writes == [] and audio.deletes == []
    assert audio.exists(other.narration_key) and audio.exists(other.music_key)
    assert audio.exists(in_flight)
    assert read_presentation(conn, memory_group, "ru").status == "ready"


def test_a_finished_saved_film_keeps_its_sound_after_the_delete(conn, memory_group, audio,
                                                                voices):
    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    result = build_presentation(conn, SavedModels(), audio, 1, memory_group, voices=voices)
    delete_now(conn, voices, mum)
    stored = read_presentation(conn, memory_group, "en")
    assert stored.status == "ready"
    assert audio.exists(result.narration_key) and audio.exists(result.music_key)


@pytest.mark.parametrize("saved", [False, True])
def test_film_audio_is_written_only_when_the_film_is_saved(conn, memory_group, voices,
                                                            tmp_path, saved):
    audio = CountingStorage(tmp_path / "audio")
    models = FakeModels()
    if saved:
        mum, _ = saved_voice(conn, voices, "Mum")
        narrate_with(conn, memory_group, mum)
        models = SavedModels()
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                                progress=audio.progress)
    assert result.status == "ready"
    assert audio.writes == ["saving the film", "saving the film"]
    assert audio.exists(result.narration_key) and audio.exists(result.music_key)


def test_the_film_and_its_tests_import_no_one_take_name():
    # These names leave the store once nothing imports them (D89).
    import ast
    from pathlib import Path

    removed = {"FilmExample", "film_example", "set_film_example", "castable", "save_conditioning",
               "VoicePreview", "save_preview", "previews", "read_preview_audio"}
    root = Path(__file__).resolve().parents[1]
    for path in ("feed/build.py", "feed/voice_preview.py", "tests/test_feed_build.py",
                 "tests/test_voice_preview.py", "tests/test_feed_store.py"):
        tree = ast.parse((root / path).read_text())
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                    for alias in node.names}
        assert imported.isdisjoint(removed), (path, imported & removed)


class PresetModels(FakeModels):
    """The fake story names `narrator` and casts `cast` at passage 2. `speak` is
    FakeModels' own, which takes no `references` keyword."""

    def __init__(self, *, narrator: str = "", cast: str = "") -> None:
        super().__init__()
        self.narrator, self.cast = narrator, cast

    def text_complete(self, model, messages, *, json_schema=None, **kw):
        raw = super().text_complete(model, messages, json_schema=json_schema, **kw)
        if "EACH PHOTOGRAPH" not in messages[-1]["content"]:
            return raw
        payload = json.loads(raw)
        payload["narrator"] = self.narrator
        payload["script"][1]["voice"] = self.cast
        return json.dumps(payload)


def narrate_with_preset(conn, group_id, speaker, language="en"):
    from feed.saved_voices import set_narrator

    set_narrator(conn, group_id, language, preset_speaker=speaker)


def test_a_chosen_preset_narrates_over_the_story_passs_narrator(conn, memory_group, audio):
    narrate_with_preset(conn, memory_group, "af_bella")
    models = PresetModels(narrator="af_heart")
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert result.status == "ready"
    assert models.spoken_voice == "af_bella"
    cast = result.decisions["cast"]
    assert cast["narrator_used"] == "af_bella"
    assert cast["narrator_chosen_by"] == "user"
    assert cast["narrator_name"] == "af_bella"
    assert cast["narrator_label"] == "af_bella (female, bright)"


@pytest.mark.parametrize(("chosen", "expected"), [("af_bella", "user"), (None, "story")])
def test_who_chose_the_same_speaker_is_recorded(conn, memory_group, audio, chosen,
                                                expected):
    if chosen is not None:
        narrate_with_preset(conn, memory_group, chosen)
    models = PresetModels(narrator="" if chosen else "af_bella")
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert models.spoken_voice == "af_bella"
    assert result.decisions["cast"]["narrator_chosen_by"] == expected


@pytest.mark.parametrize("asked", ["", "count_dracula"])
def test_an_automatic_narrator_the_story_did_not_name_comes_from_the_mood(
        conn, memory_group, audio, asked):
    from feed.signals import read_signals
    from feed.voice import pick_speaker

    result = build_presentation(conn, PresetModels(narrator=asked), audio, 1, memory_group)
    cast = result.decisions["cast"]
    assert cast["narrator_used"] == pick_speaker("en", read_signals(conn, memory_group))
    assert cast["narrator_chosen_by"] == "mood"


def test_a_preset_film_shows_the_story_pass_no_voice(conn, memory_group, audio, voices):
    saved_voice(conn, voices, "Mum")
    narrate_with_preset(conn, memory_group, "af_bella")
    models = PresetModels()
    result = build_presentation(conn, models, audio, 1, memory_group, voices=None)
    assert result.status == "ready"
    prompt = dressed(models)
    assert "THE VOICES YOU MAY CAST" not in prompt
    assert "am_michael — male, deep" not in prompt
    assert "Mum" not in prompt and "a saved voice" not in prompt
    assert result.decisions["voice"]["saved"] is False
    assert "af_bella" in result.decisions["cast"]["offered"]


def test_an_invented_bracket_voice_in_a_preset_film_lists_no_other_voice(conn, memory_group,
                                                                         audio):
    from feed.store import film_narrator

    narrate_with_preset(conn, memory_group, "af_bella")
    models = PresetModels(cast="also_not_real")
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert models.spoken_voices == []
    assert film_narrator(result.decisions).others == ()


def test_a_preset_voice_no_bracket_points_at_lists_no_other_voice(conn, memory_group, audio):
    from feed.notes import add_note
    from feed.store import film_narrator

    build_presentation(conn, FakeModels(), audio, 1, memory_group)   # the previous cut
    add_note(conn, memory_group, 0.0, "[low man's voice] the morning")  # passage 1 only
    narrate_with_preset(conn, memory_group, "af_bella")
    models = PresetModels(cast="am_michael")                           # at passage 2
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert models.spoken_voice == "af_bella"
    assert "am_michael" not in models.spoken_voices
    assert film_narrator(result.decisions).others == ()


def test_a_chosen_preset_the_catalog_does_not_offer_fails_the_film(conn, memory_group, audio):
    narrate_with_preset(conn, memory_group, "xenia")    # a Russian speaker, English film
    models = PresetModels()
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group)
    failed_with(conn, memory_group, "The preset voice «xenia» is not offered for English "
                "films. Choose another narrator.")
    assert models.spoken == []


def test_a_saved_film_records_the_users_choice_and_its_label(conn, memory_group, audio,
                                                             voices):
    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    result = build_presentation(conn, SavedModels(), audio, 1, memory_group, voices=voices)
    cast = result.decisions["cast"]
    assert cast["narrator_chosen_by"] == "user"
    assert cast["narrator_name"] == "Mum"
    assert cast["narrator_label"] == "Mum (saved voice)"


WAITING = "waiting for the voice model"


@pytest.mark.parametrize("saved", [False, True])
def test_a_film_speaks_only_after_the_voice_lock_is_free(conn, memory_group, audio, voices,
                                                         saved):
    lock = threading.Lock()
    lock.acquire()
    models = FakeModels()
    if saved:
        mum, _ = saved_voice(conn, voices, "Mum")
        narrate_with(conn, memory_group, mum)
        models = SavedModels()
    spoken_while_waiting = []

    def progress(stage, fraction):
        if stage == WAITING and lock.locked():
            spoken_while_waiting.append(list(models.spoken))
            lock.release()

    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                                progress=progress, voice_lock=lock)
    assert spoken_while_waiting == [[]]
    assert len(models.spoken) == 1 and result.status == "ready"
    assert lock.acquire(blocking=False)


def test_a_cancel_while_waiting_for_the_voice_lock_cancels_the_film(conn, memory_group,
                                                                    audio, voices):
    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    lock = threading.Lock()
    lock.acquire()
    waited = threading.Event()
    models = SavedModels()
    with pytest.raises(Cancelled):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                           progress=lambda stage, f: stage == WAITING and waited.set(),
                           cancelled=waited.is_set, voice_lock=lock)
    assert models.speak_calls == 0
    assert read_presentation(conn, memory_group, "en").status != "failed"


def test_a_take_heard_while_the_film_waited_is_the_one_it_speaks_from(
        conn, memory_group, audio, voices):
    from feed.saved_voices import token

    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    lock = threading.Lock()
    lock.acquire()
    heard = []

    def progress(stage, fraction):
        if stage == WAITING and lock.locked():
            heard.append(hear(conn, second_example(conn, voices, mum)))   # the hearing job held the lock
            lock.release()

    models = SavedModels()
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                                progress=progress, voice_lock=lock)
    assert result.status == "ready"
    assert models.references[token(mum)]["sha256"] == heard[0].content_hash
    assert result.decisions["voice"]["takes"][token(mum)]["example_id"] == heard[0].id
    assert result.decisions["cast"]["narrator_model"] == f"spoken from take #{heard[0].id}"


def test_a_take_heard_while_an_automatic_film_waited_is_the_one_it_speaks_from(
        conn, memory_group, audio, voices):
    from feed.saved_voices import token

    grandma, _ = saved_voice(conn, voices, "Grandma")
    lock = threading.Lock()
    lock.acquire()
    heard = []

    def progress(stage, fraction):
        if stage == WAITING and lock.locked():
            heard.append(hear(conn, second_example(conn, voices, grandma)))
            lock.release()

    models = SavedModels(narrator="Grandma")
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                                progress=progress, voice_lock=lock, profile="mac")
    assert result.status == "ready"
    assert models.references[token(grandma)]["sha256"] == heard[0].content_hash
    assert result.decisions["cast"]["narrator_model"] == f"spoken from take #{heard[0].id}"


def test_a_take_deleted_while_an_automatic_film_waited_fails_it(
        conn, memory_group, audio, voices):
    grandma, english = saved_voice(conn, voices, "Grandma")
    lock = threading.Lock()
    lock.acquire()

    def progress(stage, fraction):
        if stage == WAITING and lock.locked():
            delete_now(conn, voices, example=english)
            hear(conn, second_example(conn, voices, grandma, language="ru"))
            lock.release()

    models = SavedModels(narrator="Grandma")
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                           progress=progress, voice_lock=lock, profile="mac")
    failed_with(conn, memory_group,
                "«Grandma» has no English take yet. Record one under Voices, or choose another narrator.")
    assert models.speak_calls == 0


def test_a_turn_cancelled_before_it_waits_leaves_the_lock_free():
    lock = threading.Lock()
    ran = []
    with pytest.raises(Cancelled), voice_turn(lock, None, lambda: True, 0.3):
        ran.append(True)
    assert ran == []
    assert lock.acquire(blocking=False)


def test_a_turn_cancelled_as_it_acquires_releases_the_lock():
    lock = threading.Lock()
    answers = iter([False, True])
    ran = []
    with pytest.raises(Cancelled), voice_turn(lock, None, lambda: next(answers), 0.3):
        ran.append(True)
    assert ran == []
    assert lock.acquire(blocking=False)


def test_a_turn_whose_body_raises_releases_the_lock():
    lock = threading.Lock()
    with pytest.raises(OSError), voice_turn(lock, None, lambda: False, 0.3):
        raise OSError("boom")
    assert lock.acquire(blocking=False)


def test_a_turn_whose_body_returns_releases_the_lock():
    lock = threading.Lock()
    with voice_turn(lock, None, lambda: False, 0.3):
        assert lock.locked()
    assert lock.acquire(blocking=False)


class WatchedStorage(LocalStorage):
    """The saved-voice store, recording whether `lock` was held at each read and write
    once `lock` is set; `unreadable` makes that key's read raise."""

    def __init__(self, root):
        super().__init__(root)
        self.lock: threading.Lock | None = None
        self.held: list[tuple[str, bool]] = []
        self.unreadable: str | None = None

    def read(self, key):
        if self.lock is not None:
            self.held.append(("read", self.lock.locked()))
        if key == self.unreadable:
            raise OSError("disk unplugged")
        return super().read(key)

    def write(self, key, data):
        if self.lock is not None:
            self.held.append(("write", self.lock.locked()))
        super().write(key, data)


def test_the_films_turn_covers_the_read_and_the_call(conn, memory_group, audio, tmp_path):
    watched = WatchedStorage(tmp_path / "watched")
    mum, _ = saved_voice(conn, watched, "Mum")
    narrate_with(conn, memory_group, mum)
    lock = threading.Lock()
    watched.lock = lock
    held_at_speak = []
    models = SavedModels(during_speak=lambda: held_at_speak.append(lock.locked()))
    build_presentation(conn, models, audio, 1, memory_group, voices=watched, voice_lock=lock)
    assert watched.held == [("read", True)]                      # the take's audio, nothing written
    assert held_at_speak == [True]
    assert lock.acquire(blocking=False)


def test_a_failed_take_read_releases_the_voice_lock(conn, memory_group, audio, tmp_path):
    watched = WatchedStorage(tmp_path / "watched")
    mum, example = saved_voice(conn, watched, "Mum")
    narrate_with(conn, memory_group, mum)
    watched.unreadable = example.storage_key
    lock = threading.Lock()
    models = SavedModels()
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=watched,
                           voice_lock=lock)
    failed_with(conn, memory_group, f"Take #{example.id} of «Mum» could not be read (disk unplugged).")
    assert models.speak_calls == 0
    assert lock.acquire(blocking=False)


ANOTHER_VOICE_NOTE = "[another voice] the light on the water"


@pytest.mark.parametrize("kind", ["preset", "saved"])
@pytest.mark.parametrize("notes", ["none", "bracket after a cut", "bracket with no cut"])
def test_a_chosen_narrator_speaks_every_passage(conn, memory_group, audio, tmp_path, kind,
                                                notes):
    from feed.notes import add_note
    from feed.saved_voices import token

    if notes == "bracket after a cut":
        bracket_the_second_passage(conn, memory_group, audio)
    elif notes == "bracket with no cut":
        add_note(conn, memory_group, 1.0, ANOTHER_VOICE_NOTE)
    store = WatchedStorage(tmp_path / "watched")
    if kind == "preset":
        narrate_with_preset(conn, memory_group, "af_bella")
        models = PresetModels(narrator="af_heart", cast="am_michael")
        expected, conflicting = "af_bella", "am_michael"
    else:
        mum, _ = saved_voice(conn, store, "Mum")
        saved_voice(conn, store, "Dad", freq=330.0)
        narrate_with(conn, memory_group, mum)
        models = SavedModels(narrator="Dad", cast="Dad")
        expected, conflicting = token(mum), "Dad"
    store.lock = threading.Lock()
    result = build_presentation(conn, models, audio, 1, memory_group, voices=store)
    assert result.status == "ready"
    assert models.spoken_voice == expected
    assert models.spoken_voices == []                     # one voice for every passage
    cast = result.decisions["cast"]
    assert cast["narrator_used"] == expected
    assert cast["narrator_chosen_by"] == "user"
    assert cast["passages"] == []
    assert cast["model_cast"][1] == conflicting           # what the model said, kept
    assert cast["notes_asked_for_a_voice"] is (notes != "none")
    prompt = dressed(models)
    assert "THE VOICES YOU MAY CAST" not in prompt
    assert "cast a speaker like that" not in prompt
    assert sum("EACH PHOTOGRAPH" in p for p in models.story_prompts) == 1   # no retry
    assert "asked for a different voice" not in (result.error or "")
    if kind == "saved":
        assert set(models.references) == {expected}
        assert set(result.decisions["voice"]["takes"]) == {expected}
        assert store.held == [("read", False)]              # the narrator's take alone
    else:
        assert store.held == []


GRANDMA_LINE = "  Grandma — a saved voice: this person's own voice"


@pytest.fixture
def story_speakers(monkeypatch):
    """The `speakers` each story pass of a build was shown, in order."""
    from feed import build

    seen = []
    real = build.write_story

    def recording(*args, **kwargs):
        seen.append(kwargs["speakers"])
        return real(*args, **kwargs)

    monkeypatch.setattr(build, "write_story", recording)
    return seen


def test_automatic_may_pick_a_ready_saved_voice(conn, memory_group, audio, voices,
                                               story_speakers):
    from feed.saved_voices import reference_for, token
    from feed.voice import catalogue_for_prompt, offered

    grandma, example = saved_voice(conn, voices, "Grandma")
    models = SavedModels(narrator="Grandma")
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                                profile="mac")
    assert result.status == "ready"
    shown = story_speakers[-1].splitlines()
    assert set(catalogue_for_prompt("en").splitlines()) <= set(shown)
    assert GRANDMA_LINE in shown
    assert models.speak_calls == 1
    assert models.spoken_voice == token(grandma)
    assert models.references == {token(grandma): reference_for(conn, voices, example)}
    assert "adapter" not in models.references[token(grandma)]
    cast = result.decisions["cast"]
    assert cast["narrator_chosen_by"] == "story"
    assert cast["narrator_label"] == "Grandma (saved voice)"
    assert cast["narrator_model"] == f"spoken from take #{example.id}"
    assert cast["offered"] == [*(s.name for s in offered("en")), "Grandma"]
    assert result.decisions["voice"]["saved"] is True


def test_an_automatic_saved_pick_the_reply_did_not_speak_from_fails_the_film(
        conn, memory_group, audio, voices):
    from feed.saved_voices import token

    grandma, _ = saved_voice(conn, voices, "Grandma")
    models = SavedModels(narrator="Grandma", conditioning={token(grandma): None})
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices, profile="mac")
    failed_with(conn, memory_group, "The models service did not speak «Grandma» from its take.")


def test_a_preset_bracket_voice_in_an_automatic_saved_film_reads_as_the_narrator(
        conn, memory_group, audio, voices):
    from feed.saved_voices import token

    bracket_the_second_passage(conn, memory_group, audio)
    grandma, _ = saved_voice(conn, voices, "Grandma")
    models = SavedModels(narrator="Grandma", cast="am_michael")
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                                profile="mac")
    assert result.status == "ready"
    assert models.spoken_voice == token(grandma)
    assert models.spoken_voices == []
    assert set(models.references) == {token(grandma)}


def test_a_saved_bracket_voice_in_an_automatic_saved_film_speaks_its_passage(
        conn, memory_group, audio, voices):
    from feed.saved_voices import token

    bracket_the_second_passage(conn, memory_group, audio)
    grandma, _ = saved_voice(conn, voices, "Grandma")
    dad, _ = saved_voice(conn, voices, "Dad", freq=330.0)
    models = SavedModels(narrator="Grandma", cast="Dad")
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                                profile="mac")
    assert result.status == "ready"
    assert models.spoken_voices == [token(grandma), token(dad), token(grandma)]
    assert set(models.references) == {token(grandma), token(dad)}


def test_a_saved_bracket_voice_in_an_automatic_preset_film_reads_as_the_narrator(
        conn, memory_group, audio, tmp_path):
    bracket_the_second_passage(conn, memory_group, audio)
    store = WatchedStorage(tmp_path / "watched")
    saved_voice(conn, store, "Grandma")
    store.lock = threading.Lock()
    models = SavedModels(narrator="af_heart", cast="Grandma")
    result = build_presentation(conn, models, audio, 1, memory_group, voices=store,
                                profile="mac")
    assert result.status == "ready"
    assert models.spoken_voice == "af_heart"
    assert models.spoken_voices == []
    assert models.references is None
    assert store.held == []                                  # no take read


def test_an_automatic_story_that_names_no_voice_gets_the_mood_preset(conn, memory_group,
                                                                    audio, voices):
    from feed.signals import read_signals
    from feed.voice import pick_speaker

    saved_voice(conn, voices, "Grandma")
    models = SavedModels(narrator="")
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                                profile="mac")
    assert result.status == "ready"
    assert models.spoken_voice == pick_speaker("en", read_signals(conn, memory_group))
    assert result.decisions["cast"]["narrator_chosen_by"] == "mood"
    assert models.references is None


@pytest.mark.parametrize(("grandma", "profile", "with_voices"), [
    ("none", "mac", True),
    ("russian takes only", "mac", True),
    ("no heard take", "mac", True),
    ("ready", None, True),
    ("ready", "jetson", True),
    ("ready", "mac", False),
])
def test_automatic_without_a_saved_candidate_casts_presets_only(
        conn, memory_group, audio, voices, story_speakers, grandma, profile, with_voices):
    from feed.signals import read_signals
    from feed.voice import catalogue_for_prompt, pick_speaker

    if grandma == "russian takes only":
        saved_voice(conn, voices, "Grandma", language="ru")
    elif grandma == "no heard take":
        saved_voice(conn, voices, "Grandma", heard=False)
    elif grandma == "ready":
        saved_voice(conn, voices, "Grandma")
    models = SavedModels(narrator="Grandma")
    result = build_presentation(conn, models, audio, 1, memory_group,
                                voices=voices if with_voices else None, profile=profile)
    assert result.status == "ready"
    assert story_speakers[-1] == catalogue_for_prompt("en")
    assert models.spoken_voice == pick_speaker("en", read_signals(conn, memory_group))
    assert result.decisions["cast"]["narrator_chosen_by"] == "mood"
    assert models.references is None


def test_a_saved_voice_named_like_a_preset_is_never_a_candidate(conn, memory_group, audio,
                                                               voices, story_speakers):
    from feed.voice import catalogue_for_prompt

    saved_voice(conn, voices, "af_heart")
    models = SavedModels(narrator="af_heart")
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                                profile="mac")
    assert story_speakers[-1] == catalogue_for_prompt("en")
    assert "a saved voice" not in story_speakers[-1]
    assert models.spoken_voice == "af_heart"
    assert models.references is None
    assert result.decisions["voice"]["saved"] is False


def test_an_automatic_pick_deleted_before_the_speak_fails_the_film(conn, memory_group, audio,
                                                                  voices):
    grandma, _ = saved_voice(conn, voices, "Grandma")
    models = SavedModels(narrator="Grandma",
                         during_story=lambda: delete_now(conn, voices, grandma))
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices, profile="mac")
    failed_with(conn, memory_group,
                "The saved voice «Grandma» is being deleted. Choose another narrator.")
    assert models.speak_calls == 0


def test_an_automatic_pick_deleted_after_the_speak_fails_the_film(conn, memory_group, voices,
                                                                 tmp_path):
    audio = CountingStorage(tmp_path / "audio")
    grandma, _ = saved_voice(conn, voices, "Grandma")
    models = SavedModels(narrator="Grandma",
                         during_speak=lambda: delete_now(conn, voices, grandma))
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices, profile="mac")
    failed_with(conn, memory_group,
                "The saved voice «Grandma» was deleted while this film was made.")
    assert audio.writes == []


@pytest.mark.parametrize("kind", ["preset", "saved"])
def test_a_chosen_narrator_still_builds_beside_the_automatic_candidates(
        conn, memory_group, audio, voices, kind):
    from feed.saved_voices import token

    mum, _ = saved_voice(conn, voices, "Mum")
    saved_voice(conn, voices, "Grandma", freq=330.0)
    if kind == "preset":
        narrate_with_preset(conn, memory_group, "af_bella")
    else:
        narrate_with(conn, memory_group, mum)
    models = SavedModels(narrator="Grandma")
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                                profile="mac")
    assert result.status == "ready"
    assert result.decisions["voice"]["saved"] is (kind == "saved")
    assert models.spoken_voice == ("af_bella" if kind == "preset" else token(mum))


MATCH_MARK = "Which numbered photograph"


class OutOfOrder(FakeModels):
    """Four sentences over three photographs. Stage 2 introduces them out of the memory's
    order; `votes` are the per-sentence matches, 0-based, and with none stage 2 stands."""

    PROSE_EN = ("A warm afternoon by the water with nowhere at all to be. "
                "The light came off the surface of it in long sheets. "
                "Somebody laughed and the gulls rose off the jetty at once. "
                "We stayed there until the whole day went cold.")
    SECONDS_PER_CHUNK = 6.0

    def __init__(self, votes=(), *, voice="", segments_kept=None, **kw):
        super().__init__(**kw)
        self.votes = list(votes)
        self.voice = voice
        self.segments_kept = segments_kept

    def text_complete(self, model, messages, *, json_schema=None, **kw):
        prompt = messages[-1]["content"]
        if MATCH_MARK in prompt and self.votes:
            self.story_prompts.append(prompt)
            return json.dumps({"photo": self.votes.pop(0) + 1})
        if "EACH PHOTOGRAPH" not in prompt:
            return super().text_complete(model, messages, json_schema=json_schema, **kw)
        self.story_prompts.append(prompt)
        voices = ["", "", self.voice, ""]
        return json.dumps({
            "script": [{"sentence": i + 1, "shot": photo + 1, "voice": voice}
                       for i, (photo, voice) in enumerate(zip([2, 0, 2, 1], voices))],
            "narration": "",
            "narrator": "af_heart",
            "cards": [{"sentence": 2, "text": "The turn"}],
            "score": [{"from": 1, "brief": "solo piano"}, {"from": 3, "brief": "strings"}],
            "music_prompt": "warm ambient drone, no vocals",
        })

    def speak(self, text, language="en", voice=None, segments=None, voices=None):
        spoken = super().speak(text, language, voice, segments, voices)
        if self.segments_kept is not None:
            spoken["segments"] = spoken["segments"][:self.segments_kept]
        return spoken


def told_sentences() -> list[str]:
    parts = OutOfOrder.PROSE_EN.split(". ")
    return [part if part.endswith(".") else f"{part}." for part in parts]


def test_a_disordered_film_keeps_every_passage_with_its_own_words(
        conn, memory_group, audio):
    models = OutOfOrder()
    result = build_presentation(conn, models, audio, 1, memory_group)
    shots = result.timeline["shots"]
    assert models.segments == told_sentences()
    # With no votes the selected mapping is stage 2's: photographs 3, 1, 3, 2.
    assert [s["photo_id"] for s in shots] == [3, 1, 3, 2]
    assert [s["at"] for s in shots] == [0.0, 6.0, 12.0, 18.0]
    assert [s["line"] for s in shots] == ["", "The turn", "", ""]
    # The score turns where passage 3 is spoken.
    assert [section["at"] for section in result.decisions["score"]["sections"]] == [0.0, 12.0]
    conductor = next(p for p in models.story_prompts if "THE PHOTOGRAPHS available to you" in p)
    briefs = FakeModels.shot_briefs(conductor)
    assert len(briefs) == 4
    assert [f"caption {pid}" in brief for pid, brief in zip([3, 1, 3, 2], briefs)] == [True] * 4
    assert result.decisions["story"]["memory_order"] == "broken"


def test_a_bracket_gated_voice_speaks_its_own_passage_in_a_disordered_film(
        conn, memory_group, audio):
    bracket_the_second_passage(conn, memory_group, audio)
    models = OutOfOrder(voice="am_michael")
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert models.spoken_voices == ["af_heart", "af_heart", "am_michael", "af_heart"]
    assert [s["photo_id"] for s in result.timeline["shots"]] == [3, 1, 3, 2]


def test_the_sentence_matches_move_the_photographs_and_not_the_words(
        conn, memory_group, audio):
    models = OutOfOrder([0, 1, 0, 2])
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert models.segments == told_sentences()
    assert [s["photo_id"] for s in result.timeline["shots"]] == [1, 2, 1, 3]
    assert result.timeline["narration"] == OutOfOrder.PROSE_EN
    assert [s["line"] for s in result.timeline["shots"]] == ["", "The turn", "", ""]
    assert result.decisions["story"]["memory_order"] == "kept"
    assert result.decisions["story"]["votes"] == [0, 1, 0, 2]
    assert result.decisions["story"]["photos"] == [0, 1, 0, 2]


def test_a_match_out_of_the_memorys_order_stands_and_reads_broken(conn, memory_group, audio):
    models = OutOfOrder([1, 0, 1, 2])
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert [s["photo_id"] for s in result.timeline["shots"]] == [2, 1, 2, 3]
    assert result.decisions["story"]["memory_order"] == "broken"


@pytest.mark.parametrize("segments_kept", [0, 3], ids=["no-segments", "one-fewer"])
def test_a_film_whose_segments_do_not_pair_keeps_the_told_order(
        conn, memory_group, audio, segments_kept):
    # The conductor fake answers the shot numbers in reverse; with a script it never
    # reorders the film, whether or not the segments pair.
    models = OutOfOrder(segments_kept=segments_kept)
    result = build_presentation(conn, models, audio, 1, memory_group)
    shots = result.timeline["shots"]
    assert models.arranged
    assert [s["photo_id"] for s in shots] == [3, 1, 3, 2]
    assert [s["line"] for s in shots] == ["", "The turn", "", ""]
    assert all(b["at"] >= a["at"] + a["seconds"] - 1e-6 for a, b in pairwise(shots))
    assert result.decisions["film"]["conductor_kept_timing"] is False
    passages = result.timeline["passages"]
    assert [p["text"] for p in passages] == told_sentences()
    assert all(p["start"] is None and p["end"] is None for p in passages)


NO_VOICE_CAST = "a note asked for a different voice and none was cast"
CONDUCTOR_MARKER = "THE PHOTOGRAPHS available to you"
CONTEXT_TOKENS = 8192
STORED_PHOTOS = [0, 1, 0, 2]
CORRECTED_PHOTOS = [2, 1, 0, 2]
CORRECTED_PHOTO_IDS = [3, 2, 1, 3]
EDITED_TEXT = "A cold morning by the water."


def edit_passage(conn, group_id, n, text, language="en"):
    """What the panel posts for one passage: the rows it last read, then the change."""
    from feed.narration import narration_rows, save_passage

    rows = narration_rows(conn, group_id, language)
    return save_passage(conn, group_id, language, n, text, source=rows.source,
                        draft=rows.draft or "", revision=rows.revision)


def draft_row(conn, group_id, language="en"):
    row = conn.execute("SELECT * FROM narration_drafts WHERE group_id = ? AND language = ?",
                       (group_id, language)).fetchone()
    return dict(row) if row is not None else None


def no_story_pass(models) -> bool:
    """The model was asked only which photograph a sentence shows, and to conduct."""
    return all(MATCH_MARK in prompt or CONDUCTOR_MARKER in prompt
               for prompt in models.story_prompts)


def match_prompts(models) -> list[str]:
    return [prompt for prompt in models.story_prompts if MATCH_MARK in prompt]


def test_a_films_passages_are_what_the_voice_spoke(conn, memory_group, audio):
    models = OutOfOrder()
    result = build_presentation(conn, models, audio, 1, memory_group)
    passages = result.timeline["passages"]
    assert [p["text"] for p in passages] == models.segments == told_sentences()
    assert [(p["start"], p["end"]) for p in passages] == [(0.0, 6.0), (6.0, 12.0),
                                                          (12.0, 18.0), (18.0, 24.0)]
    assert [p["n"] for p in passages] == [0, 1, 2, 3]
    assert (passages[0]["photo"], passages[0]["photo_id"]) == (2, 3)
    assert [p["photo_id"] for p in passages] == [
        s["photo_id"] for s in result.timeline["shots"]] == [3, 1, 3, 2]


def test_a_films_script_carries_a_fresh_token_and_its_replay_inputs(conn, memory_group,
                                                                     audio):
    first = build_presentation(conn, OutOfOrder(), audio, 1, memory_group)
    script = first.timeline["script"]
    assert len(script["token"]) == 32 and int(script["token"], 16) >= 0
    assert script["cards"] == [{"n": 1, "text": "The turn"}]
    assert script["score_plan"] == [{"from": 0, "brief": "solo piano"},
                                    {"from": 2, "brief": "strings"}]
    assert script["music_prompt"] == "warm ambient drone, no vocals"
    assert script["narrator_asked"] == "af_heart"
    assert script["narrator_used"] == first.decisions["cast"]["narrator_used"]
    second = build_presentation(conn, OutOfOrder(), audio, 1, memory_group)
    assert second.timeline["script"]["token"] != script["token"]


def test_a_failed_film_carries_no_passages(conn, memory_group, audio):
    with pytest.raises(NarrationFailed):
        build_presentation(conn, FakeModels(voice_off=True), audio, 1, memory_group)
    timeline = read_presentation(conn, memory_group, "en").timeline
    assert "passages" not in timeline and "script" not in timeline


def test_a_draft_is_spoken_as_written_with_no_story_pass(conn, memory_group, audio):
    build_presentation(conn, FakeModels(), audio, 1, memory_group)
    edit_passage(conn, memory_group, 1, "The light was silver.")
    draft = draft_row(conn, memory_group)
    texts = [p["text"] for p in json.loads(draft["passages_json"])]
    models = FakeModels()
    models.text_context = lambda: CONTEXT_TOKENS
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert result.status == "ready"
    assert no_story_pass(models)
    assert any(CONDUCTOR_MARKER in prompt for prompt in models.story_prompts)
    assert models.spoken[-1][0] == " ".join(texts)
    assert models.segments == texts
    assert texts[1] == "The light was silver."
    # This story planned no score turn, so the stored sentence colours every section.
    assert "warm ambient drone, no vocals" in models.prompted[-1][0]
    story = result.decisions["story"]
    assert story["source"] == "draft"
    assert story["draft_revision"] == draft["revision"] == 1
    assert story["context_tokens"] == CONTEXT_TOKENS


def test_a_film_made_from_a_draft_rebases_it(conn, memory_group, audio):
    from feed.narration import narration_rows

    build_presentation(conn, FakeModels(), audio, 1, memory_group)
    edit_passage(conn, memory_group, 0, "A cold morning by the water.")
    before = draft_row(conn, memory_group)
    result = build_presentation(conn, FakeModels(), audio, 1, memory_group)
    after = draft_row(conn, memory_group)
    assert after["draft_token"] == before["draft_token"]
    assert after["revision"] == before["revision"] + 1
    assert after["source_token"] == result.timeline["script"]["token"]
    assert json.loads(after["passages_json"]) == result.timeline["passages"]
    assert ([p["text"] for p in json.loads(after["passages_json"])]
            == [p["text"] for p in json.loads(before["passages_json"])])
    rows = narration_rows(conn, memory_group, "en")
    assert (rows.state, rows.seekable, rows.changed) == ("draft", True, False)
    again = FakeModels()
    build_presentation(conn, again, audio, 1, memory_group)
    assert no_story_pass(again)
    assert again.segments == [p["text"] for p in json.loads(after["passages_json"])]


def test_a_draft_film_corrects_a_wrong_stored_photograph(conn, memory_group, audio):
    build_presentation(conn, OutOfOrder(STORED_PHOTOS, voice="am_michael"), audio, 1,
                       memory_group)
    edit_passage(conn, memory_group, 0, EDITED_TEXT)
    before = json.loads(draft_row(conn, memory_group)["passages_json"])
    assert [p["photo_id"] for p in before] == [1, 2, 1, 3]

    models = OutOfOrder(CORRECTED_PHOTOS)
    result = build_presentation(conn, models, audio, 1, memory_group)

    passages = result.timeline["passages"]
    assert [p["photo_id"] for p in passages] == CORRECTED_PHOTO_IDS
    assert [s["photo_id"] for s in result.timeline["shots"]] == CORRECTED_PHOTO_IDS
    assert [p["text"] for p in passages] == models.segments == [p["text"] for p in before]
    assert [p["voice_asked"] for p in passages] == [p["voice_asked"] for p in before]
    assert no_story_pass(models)
    assert [s["line"] for s in result.timeline["shots"]] == ["", "The turn", "", ""]
    story = result.decisions["story"]
    assert story["narrator_asked"] == "af_heart"
    assert story["score_plan"] == [{"from": 0, "brief": "solo piano"},
                                   {"from": 2, "brief": "strings"}]
    assert story["music_prompt"] == "warm ambient drone, no vocals"


def test_a_draft_film_records_its_votes_and_the_final_order(conn, memory_group, audio):
    build_presentation(conn, OutOfOrder(STORED_PHOTOS), audio, 1, memory_group)
    edit_passage(conn, memory_group, 0, EDITED_TEXT)

    models = OutOfOrder(CORRECTED_PHOTOS)
    story = build_presentation(conn, models, audio, 1, memory_group).decisions["story"]

    assert story["votes"] == story["photos"] == CORRECTED_PHOTOS
    assert story["memory_order"] == "broken"
    prompts = match_prompts(models)
    assert len(prompts) == 4
    assert EDITED_TEXT in prompts[0]
    assert "3. caption 3" in prompts[0]


def test_a_passage_with_no_vote_keeps_its_stored_photograph(conn, memory_group, audio):
    build_presentation(conn, OutOfOrder(STORED_PHOTOS), audio, 1, memory_group)
    edit_passage(conn, memory_group, 0, EDITED_TEXT)

    result = build_presentation(conn, OutOfOrder([2]), audio, 1, memory_group)

    assert result.decisions["story"]["votes"] == [2, None, None, None]
    assert [p["photo_id"] for p in result.timeline["passages"]] == [3, 2, 1, 3]


def test_a_match_call_that_fails_keeps_the_stored_photographs(conn, memory_group, audio):
    class MatchDown(FakeModels):
        def text_complete(self, model, messages, *, json_schema=None, **kw):
            if MATCH_MARK in messages[-1]["content"]:
                raise RuntimeError("planner is down")
            return super().text_complete(model, messages, json_schema=json_schema, **kw)

    build_presentation(conn, OutOfOrder(STORED_PHOTOS), audio, 1, memory_group)
    edit_passage(conn, memory_group, 0, EDITED_TEXT)

    result = build_presentation(conn, MatchDown(), audio, 1, memory_group)

    assert result.status == "ready"
    assert result.decisions["story"]["votes"] == [None, None, None, None]
    assert [p["photo_id"] for p in result.timeline["passages"]] == [1, 2, 1, 3]


def test_a_ready_draft_film_rebases_the_corrected_photographs(conn, memory_group, audio):
    build_presentation(conn, OutOfOrder(STORED_PHOTOS), audio, 1, memory_group)
    edit_passage(conn, memory_group, 0, EDITED_TEXT)

    result = build_presentation(conn, OutOfOrder(CORRECTED_PHOTOS), audio, 1, memory_group)

    stored = json.loads(draft_row(conn, memory_group)["passages_json"])
    assert stored == result.timeline["passages"]
    assert [p["photo_id"] for p in stored] == CORRECTED_PHOTO_IDS
    again = build_presentation(conn, FakeModels(), audio, 1, memory_group)
    assert [p["photo_id"] for p in again.timeline["passages"]] == CORRECTED_PHOTO_IDS


def test_a_failed_draft_film_keeps_the_stored_photographs(conn, memory_group, audio):
    build_presentation(conn, OutOfOrder(STORED_PHOTOS), audio, 1, memory_group)
    edit_passage(conn, memory_group, 0, EDITED_TEXT)
    before = draft_row(conn, memory_group)
    models = OutOfOrder(CORRECTED_PHOTOS, voice_off=True)

    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group)

    assert len(match_prompts(models)) == 4
    assert draft_row(conn, memory_group) == before


def test_cancel_during_the_match_leaves_the_draft(conn, memory_group, audio):
    build_presentation(conn, OutOfOrder(STORED_PHOTOS), audio, 1, memory_group)
    edit_passage(conn, memory_group, 0, EDITED_TEXT)
    before = draft_row(conn, memory_group)
    models = OutOfOrder(CORRECTED_PHOTOS)

    with pytest.raises(Cancelled):
        build_presentation(conn, models, audio, 1, memory_group,
                           cancelled=lambda: bool(match_prompts(models)))

    assert len(match_prompts(models)) == 1
    assert not models.spoken
    assert draft_row(conn, memory_group) == before


@pytest.mark.parametrize(("edited", "lines", "cards"), [
    (1, ["", "", "", ""], []),
    (0, ["", "The turn", "", ""], [{"n": 1, "text": "The turn"}]),
])
def test_only_the_edited_rows_card_leaves_the_film(conn, memory_group, audio, edited, lines,
                                                   cards):
    build_presentation(conn, OutOfOrder(STORED_PHOTOS), audio, 1, memory_group)
    edit_passage(conn, memory_group, edited, EDITED_TEXT)

    result = build_presentation(conn, FakeModels(), audio, 1, memory_group)

    assert [s["line"] for s in result.timeline["shots"]] == lines
    assert result.timeline["script"]["cards"] == cards


@pytest.mark.parametrize(("rewritten", "lines", "cards"), [
    (1, ["", "", "", ""], []),
    (2, ["", "The turn", "", ""], [{"n": 1, "text": "The turn"}]),
])
def test_only_a_rewritten_rows_card_leaves_the_film(conn, memory_group, audio, rewritten, lines,
                                                    cards):
    from feed.narration import narration_rows, rewrite_passages

    build_presentation(conn, OutOfOrder(STORED_PHOTOS), audio, 1, memory_group)
    rows = narration_rows(conn, memory_group, "en")
    rewrite_passages(conn, memory_group, "en", {rewritten: EDITED_TEXT}, source=rows.source,
                     draft="", revision=rows.revision)

    result = build_presentation(conn, FakeModels(), audio, 1, memory_group)

    assert [s["line"] for s in result.timeline["shots"]] == lines
    assert result.timeline["script"]["cards"] == cards


def test_a_draft_that_lost_a_photographs_introduction_keeps_its_pairs(conn, memory_group,
                                                                      audio):
    from feed.narration import remove_passage

    build_presentation(conn, OutOfOrder([0, 1, 0, 2]), audio, 1, memory_group)
    edit_passage(conn, memory_group, 1, "The light came off it in long sheets.")
    rows = edit_passage(conn, memory_group, 1, "The light came off it in long, slow sheets.")
    remove_passage(conn, memory_group, "en", 0, source=rows.source, draft=rows.draft,
                   revision=rows.revision)
    kept = [(p["photo_id"], p["text"]) for p in json.loads(draft_row(conn, memory_group)
                                                           ["passages_json"])]
    assert [photo_id for photo_id, _ in kept] == [2, 1, 3]
    models = FakeModels()
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert no_story_pass(models)
    assert [(p["photo_id"], p["text"]) for p in result.timeline["passages"]] == kept
    assert [s["photo_id"] for s in result.timeline["shots"]] == [2, 1, 3]
    assert result.decisions["story"]["memory_order"] == "broken"
    assert len(match_prompts(models)) == 3
    assert result.decisions["story"]["votes"] == [None, None, None]
    assert result.decisions["story"]["photos"] == [1, 0, 2]


def test_a_draft_with_a_chosen_preset_is_read_by_it_alone(conn, memory_group, audio):
    build_presentation(conn, PresetModels(narrator="am_adam", cast="am_michael"), audio, 1,
                       memory_group)
    edit_passage(conn, memory_group, 0, "A cold morning by the water.")
    narrate_with_preset(conn, memory_group, "af_bella")
    models = FakeModels()
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert models.spoken_voice == "af_bella"
    assert models.spoken_voices == []
    assert result.decisions["cast"]["narrator_chosen_by"] == "user"


def test_an_automatic_draft_is_read_by_the_narrator_its_story_asked_for(conn, memory_group,
                                                                       audio):
    build_presentation(conn, PresetModels(narrator="am_adam"), audio, 1, memory_group)
    edit_passage(conn, memory_group, 0, "A cold morning by the water.")
    models = FakeModels()
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert models.spoken_voice == "am_adam"
    assert result.decisions["cast"]["narrator_chosen_by"] == "story"


@pytest.mark.parametrize("grandma", ["ready", "deleted"])
def test_an_automatic_draft_asking_for_a_saved_voice(conn, memory_group, audio, voices,
                                                     grandma):
    from feed.saved_voices import reference_for, token
    from feed.signals import read_signals
    from feed.voice import pick_speaker

    voice_id, example = saved_voice(conn, voices, "Grandma")
    build_presentation(conn, SavedModels(narrator="Grandma"), audio, 1, memory_group,
                       voices=voices, profile="mac")
    edit_passage(conn, memory_group, 0, "A cold morning by the water.")
    if grandma == "deleted":
        delete_now(conn, voices, voice_id)
    models = SavedModels()
    result = build_presentation(conn, models, audio, 1, memory_group, voices=voices,
                                profile="mac")
    assert result.status == "ready"
    assert no_story_pass(models)
    if grandma == "ready":
        assert models.spoken_voice == token(voice_id)
        assert models.references == {token(voice_id): reference_for(conn, voices, example)}
        assert result.decisions["voice"]["saved"] is True
    else:
        assert models.spoken_voice == pick_speaker("en", read_signals(conn, memory_group))
        assert models.references is None
        assert result.decisions["cast"]["narrator_chosen_by"] == "mood"


@pytest.mark.parametrize("bracket", ["at its passage", "elsewhere"])
def test_a_drafts_passage_voice_needs_a_current_bracket(conn, memory_group, audio, bracket):
    from feed.notes import add_note

    build_presentation(conn, PresetModels(narrator="af_heart", cast="am_adam"), audio, 1,
                       memory_group)
    seconds = read_presentation(conn, memory_group, "en").timeline["seconds"]
    at = seconds / 2 if bracket == "at its passage" else 0.0
    add_note(conn, memory_group, at, ANOTHER_VOICE_NOTE)
    edit_passage(conn, memory_group, 0, "A cold morning by the water.")
    models = FakeModels()
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert no_story_pass(models)
    if bracket == "at its passage":
        assert models.spoken_voices == ["af_heart", "am_adam", "af_heart"]
    else:
        assert models.spoken_voices == []
        assert NO_VOICE_CAST in (result.error or "")


@pytest.mark.parametrize(("removed", "lines"), [
    (0, ["The turn", "", ""]),
    (1, ["", "", ""]),
])
def test_a_drafts_card_stays_on_its_passage(conn, memory_group, audio, removed, lines):
    from feed.narration import remove_passage

    build_presentation(conn, OutOfOrder([0, 1, 0, 2]), audio, 1, memory_group)
    rows = edit_passage(conn, memory_group, 3, "We stayed until the whole day went cold.")
    remove_passage(conn, memory_group, "en", removed, source=rows.source, draft=rows.draft,
                   revision=rows.revision)
    result = build_presentation(conn, FakeModels(), audio, 1, memory_group)
    assert [s["line"] for s in result.timeline["shots"]] == lines


def test_a_failed_draft_film_leaves_the_draft_and_a_retry_speaks_it(conn, memory_group,
                                                                    audio):
    build_presentation(conn, OutOfOrder([0, 1, 0, 2]), audio, 1, memory_group)
    edit_passage(conn, memory_group, 0, "A cold morning by the water.")
    before = draft_row(conn, memory_group)
    with pytest.raises(NarrationFailed):
        build_presentation(conn, FakeModels(voice_off=True), audio, 1, memory_group)
    failed = read_presentation(conn, memory_group, "en")
    assert failed.status == "failed" and "passages" not in failed.timeline
    assert draft_row(conn, memory_group) == before
    models = FakeModels()
    result = build_presentation(conn, models, audio, 1, memory_group)
    texts = [p["text"] for p in json.loads(before["passages_json"])]
    assert result.status == "ready"
    assert no_story_pass(models)
    assert models.spoken[-1][0] == " ".join(texts)
    assert [s["line"] for s in result.timeline["shots"]] == ["", "The turn", "", ""]
    assert result.decisions["story"]["score_plan"] == [{"from": 0, "brief": "solo piano"},
                                                       {"from": 2, "brief": "strings"}]
    assert "solo piano" in models.prompted[-1][0] and "strings" in models.prompted[-1][0]
    assert result.decisions["story"]["music_prompt"] == "warm ambient drone, no vocals"


def test_a_cancelled_draft_film_leaves_the_draft_and_the_film(conn, memory_group, audio):
    film = build_presentation(conn, FakeModels(), audio, 1, memory_group)
    edit_passage(conn, memory_group, 0, "A cold morning by the water.")
    before = draft_row(conn, memory_group)
    models = FakeModels()
    with pytest.raises(Cancelled):
        build_presentation(conn, models, audio, 1, memory_group,
                           cancelled=lambda: bool(models.spoken))
    assert models.spoken
    assert draft_row(conn, memory_group) == before
    stored = read_presentation(conn, memory_group, "en")
    assert stored.status == "ready"
    assert stored.timeline["script"]["token"] == film.timeline["script"]["token"]


def test_an_edit_saved_during_the_build_survives_it(conn, memory_group, audio):
    from feed.narration import narration_rows

    film = build_presentation(conn, FakeModels(), audio, 1, memory_group)
    edit_passage(conn, memory_group, 0, "A cold morning by the water.")
    spoken_revision = draft_row(conn, memory_group)["revision"]
    newer = {}

    def edit_meanwhile():
        edit_passage(conn, memory_group, 2, "We left when it went cold.")
        newer.update(draft_row(conn, memory_group))

    models = SavedModels(during_speak=edit_meanwhile)
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert result.status == "ready"
    assert models.segments[0] == "A cold morning by the water."
    assert "We left when it went cold." not in models.segments
    assert result.decisions["story"]["draft_revision"] == spoken_revision
    assert draft_row(conn, memory_group) == newer
    assert newer["revision"] == spoken_revision + 1
    assert newer["source_token"] == film.timeline["script"]["token"]
    rows = narration_rows(conn, memory_group, "en")
    assert (rows.state, rows.seekable, rows.changed) == ("draft", False, True)


def test_an_old_build_never_writes_into_a_draft_seeded_again(conn, memory_group, audio):
    from feed.narration import discard_draft

    build_presentation(conn, FakeModels(), audio, 1, memory_group)
    rows = edit_passage(conn, memory_group, 0, "A cold morning by the water.")
    old_token = rows.draft
    seeded = {}

    def write_a_new_story_and_edit_again():
        current = edit_passage(conn, memory_group, 0, "A colder morning by the water.")
        discard_draft(conn, memory_group, "en", source=current.source, draft=current.draft,
                      revision=current.revision)
        edit_passage(conn, memory_group, 1, "The light was silver.")
        seeded.update(draft_row(conn, memory_group))

    models = SavedModels(during_speak=write_a_new_story_and_edit_again)
    result = build_presentation(conn, models, audio, 1, memory_group)
    assert result.status == "ready"
    assert seeded["draft_token"] != old_token and seeded["revision"] == 1
    assert draft_row(conn, memory_group) == seeded


def test_a_saved_voice_deleted_before_a_draft_films_save_leaves_the_draft(
        conn, memory_group, voices, tmp_path):
    audio = CountingStorage(tmp_path / "audio")
    mum, _ = saved_voice(conn, voices, "Mum")
    narrate_with(conn, memory_group, mum)
    build_presentation(conn, SavedModels(), audio, 1, memory_group, voices=voices)
    edit_passage(conn, memory_group, 0, "A cold morning by the water.")
    before = draft_row(conn, memory_group)
    audio.reset()
    models = SavedModels(during_speak=lambda: delete_now(conn, voices, mum))
    with pytest.raises(NarrationFailed):
        build_presentation(conn, models, audio, 1, memory_group, voices=voices)
    failed_with(conn, memory_group, "The saved voice «Mum» was deleted while this film was made.")
    assert audio.writes == []
    assert draft_row(conn, memory_group) == before


def test_with_no_draft_a_films_passages_are_not_counted_as_notes(conn, memory_group, audio):
    from feed.notes import add_note
    from feed.story import narration_words

    build_presentation(conn, FakeModels(), audio, 1, memory_group)
    add_note(conn, memory_group, 1.0, "This is Julia")
    models = FakeModels()
    result = build_presentation(conn, models, audio, 1, memory_group)
    low, high = narration_words(3, 1)
    assert f"Write {low}-{high} words" in told(models)
    assert "source" not in result.decisions["story"]
    assert draft_row(conn, memory_group) is None
