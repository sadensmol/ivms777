"""The film prepare job (design §11.1) — offline generation, fully faked."""

import json

import httpx
import numpy as np
import pytest

from albums.memory_store import Memory, replace_memories
from feed.build import NarrationFailed, build_presentation
from feed.store import read_presentation
from storage.local import LocalStorage
from storage.wav import encode_wav
from tests.factories import add_photo


class FakeModels:
    """The calls the job makes, with recorders for what it asked for."""

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
            "cards": [{"shot": 1, "text": "The water"}],
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
        each, timings, cursor = 2.0, [], 0.0
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
    assert set(edit) == {"shots", "grade", "grades", "seconds", "narration", "effects"}
    move = edit["shots"][0]["move"]
    assert move["scale_from"] != move["scale_to"]
    assert edit["shots"][0]["transition"] == "none"


def test_the_film_follows_the_memorys_photo_order(conn, memory_group, audio):
    # The ORDER belongs to the memory: it was decided when the memory was built, the
    # album shows it, and every film of that memory follows it until the memory is
    # rebuilt (§11, §11.1). The story pass still chooses which photograph carries each
    # passage and may return to one — a repeat is not a reorder — but it cannot
    # introduce photograph 2 before photograph 1.
    edit = build_presentation(conn, FakeModels(), audio, 1, memory_group).timeline
    shown = [s["photo_id"] for s in edit["shots"]]
    first_seen = [pid for i, pid in enumerate(shown) if pid not in shown[:i]]
    assert first_seen == sorted(first_seen), shown
    # The fake's script uses photo 1 twice and never photo 3.
    assert shown == [1, 1, 2]
    assert 3 not in shown


def test_a_photograph_the_story_never_uses_stays_out(conn, memory_group, audio):
    edit = build_presentation(conn, FakeModels(), audio, 1, memory_group).timeline
    assert 3 not in [s["photo_id"] for s in edit["shots"]]


def test_every_shot_starts_when_its_own_words_are_spoken(conn, memory_group, audio):
    # The sync, and it is arithmetic rather than judgement: passage `i` was handed to
    # the voice as its own chunk, so segment `i` IS the words for shot `i`.
    models = FakeModels()
    edit = build_presentation(conn, models, audio, 1, memory_group).timeline
    assert [s["at"] for s in edit["shots"]] == [0.0, 2.0, 4.0]   # the fake speaks 2s each
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


def test_without_a_previous_cut_nothing_is_gated():
    # No cut to measure against means no note has a known position (§11.3).
    from feed.build import _bracketed_passages
    from feed.notes import Note

    notes = [Note(id=1, group_id=1, at_seconds=0.0, text="[x] a", created_at="now")]
    assert _bracketed_passages(notes, film_seconds=0.0, passages=3) is None
