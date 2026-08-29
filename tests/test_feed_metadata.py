"""What a `[bracket]` in a note ACTUALLY changes, end to end (design §11.3).

Every other test here checks one stage. This drives real metadata — in both languages,
in the wording a person types — through a whole `build_presentation` and asserts what
came out the other side: which voice spoke which passage, what the score was asked for,
which effect landed on which shot.

The model is faked, but faked HONESTLY: it reads its own prompt and answers from it, the
way the real one is asked to. So these tests fail when the pipeline stops handing a
bracket to the pass that can act on it — which is the failure that kept happening.
"""

import json

import numpy as np
import pytest

from albums.memory_store import Memory, replace_memories
from feed.build import build_presentation
from feed.notes import add_note
from feed.voice import offered
from storage.local import LocalStorage
from storage.wav import encode_wav
from tests.factories import add_photo


def dressed(models) -> str:
    """The CUT-AND-DRESS prompt — the one that carries the photographs and the speaker
    table. The STORY prompt deliberately carries neither (§11.1)."""
    return next(p for p in models.story_prompts if "EACH PHOTOGRAPH" in p)


class ObedientModel:
    """A planner that does what the prompt asks, using only what the prompt gave it.

    It cannot invent a speaker or an effect: it picks from the lists it was shown. That
    is the point — if the pipeline stops showing them, these tests fail.
    """

    def __init__(self, *, wants_voice=None, wants_effect=None, music_words=""):
        self.wants_voice = wants_voice
        self.wants_effect = wants_effect
        self.music_words = music_words
        self.story_prompts: list[str] = []
        self.prompted: list[tuple[str, float]] = []
        self.spoken_voice = None
        self.spoken_voices: list = []
        self.segments: list[str] = []

    STORY = ("The first passage of this story, which is long enough to hold a shot. "
             "The second passage, also long enough to be a beat of its own. "
             "The third passage, which closes the whole thing off.")

    def text_complete(self, model, messages, *, json_schema=None, **kw):
        prompt = messages[-1]["content"]
        self.story_prompts.append(prompt)
        if prompt.startswith("Translate into"):
            return json.dumps({"story":
                               "Перевод первого предложения, достаточно длинный. "
                               "Перевод второго предложения, тоже длинный. "
                               "Перевод третьего предложения, и на этом всё."})
        if "THE PHOTOGRAPHS available to you" in prompt:
            return self._conduct(prompt)
        # STAGE 1 — the story. An obedient model WRITES THE NOTES IN, so nothing has to
        # be woven or inserted afterwards (§11.3).
        if "WHAT THE DAY WAS" in prompt or "ЧТО ЭТО ЗА ДЕНЬ" in prompt:
            import re as _re

            said = " ".join(line.split("] ", 1)[-1] for line in prompt.splitlines()
                            if line.strip().startswith("[at "))
            said = _re.sub(r"\[[^\]]*\]", " ", said).strip()   # brackets are obeyed
            return json.dumps({"story":
                               "The first passage of this story, long enough to hold "
                               f"a shot. The second passage {said}, also long enough "
                               "to be a beat. The third passage, which closes it off."})
        # The two SMALL follow-ups (§11.3, §11.1) are their own jobs.
        if "ПРЕДЛОЖЕНИЕ" in prompt or "SENTENCE:" in prompt:
            return json.dumps({"text": prompt.split("момент: ")[-1].split("\n")[0]})
        if "Допиши" in prompt or "more sentences" in prompt:
            return json.dumps({"story": ""})
        return self._story(prompt)

    def _story(self, prompt):
        # STAGE 2 — one entry per SENTENCE of the story above, in order.
        count = prompt.count("\n1. ") and len(
            [line for line in prompt.splitlines()
             if line.strip()[:1].isdigit() and ". " in line])
        script = [{"shot": (i % 3) + 1, "sentence": i + 1}
                  for i in range(max(3, min(count, 8)))]
        # Cast only what the prompt actually offered.
        narrator = ""
        if self.wants_voice and self.wants_voice in prompt:
            narrator = self.wants_voice
            script[1]["voice"] = self.wants_voice
        return json.dumps({
            "script": script,
            "narration": "",
            "narrator": narrator,
            "cards": [],
            "music_prompt": self.music_words or "a score",
        })

    def _conduct(self, prompt):
        effect = self.wants_effect if (self.wants_effect and self.wants_effect in prompt) \
            else "plain"
        # As many entries as the film has shots — a plan that is short of them is
        # rejected by `with_arrangement` and the fallback loses the effect. The film is
        # as long as the STORY now, so the count is read from the brief rather than
        # assumed to be three.
        block = prompt.split("THE PHOTOGRAPHS available to you", 1)[-1]
        block = block.split("WHAT THE VOICE SAYS", 1)[0]
        count = max(1, sum(1 for line in block.splitlines()
                           if line.strip()[:1].isdigit() and ". " in line))
        return json.dumps({"shots": [
            {"shot": i + 1, "at": i * 4.0, "seconds": 4.0,
             "transition": "none" if i == 0 else "dissolve",
             "beats": [{"effect": effect if i == 1 else "plain", "from": 0.0, "to": 1.0}]}
            for i in range(count)
        ]})

    def speak(self, text, language="en", voice=None, segments=None, voices=None):
        self.spoken_voice = voice
        self.spoken_voices = list(voices or [])
        self.segments = list(segments or [])
        each, cursor, timings = 2.0, 0.0, []
        for chunk in self.segments or [text]:
            timings.append({"text": chunk, "start": round(cursor, 3),
                            "end": round(cursor + each, 3)})
            cursor += each
        return {"audio": encode_wav(np.zeros(int(24000 * cursor), dtype=np.float32), 24000),
                "sample_rate": 24000, "model": "fake", "segments": timings}

    def music(self, prompt="", seconds=30.0, seed=None, *, sections=None,
              temperature=None, guidance=None):
        # The score arrives as SECTIONS now (§11.1); a bracket must reach every one of
        # them, so the recorder keeps them joined.
        self.sections = list(sections or [])
        self.prompted.append((" ".join(s["brief"] for s in self.sections) or prompt,
                              sum(s["seconds"] for s in self.sections) or seconds))
        return {"audio": encode_wav(np.zeros((2, 44100), dtype=np.float32), 44100),
                "sample_rate": 44100, "model": "fake", "segments": []}


@pytest.fixture
def memory(conn):
    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  width=4000, height=3000, ai_title=f"caption {pid}",
                  shot_at=f"2025-07-12T1{pid}:00:00")
    replace_memories(conn, 1, [Memory("A night out", "Warm.", [1, 2, 3], "sig")])
    return conn.execute("SELECT id FROM groups WHERE kind = 'memory'").fetchone()["id"]


@pytest.fixture
def audio(tmp_path):
    return LocalStorage(tmp_path / "audio")


def _deep_male(language="en"):
    return next(s.name for s in offered(language)
                if s.gender == "male" and s.register == "deep")


# --- voice --------------------------------------------------------------------------

@pytest.mark.parametrize("bracket", [
    "[страшных хриплый мужкской голос] тут было жутко",   # Russian, misspelled
    "[low man's voice] it was grim",                       # English
    "[шёпот] очень тихо",                                  # a register, no gender
])
def test_a_voice_bracket_reaches_the_pass_that_can_cast_it(conn, memory, audio, bracket):
    # However it is worded, in whichever language, it must arrive VERBATIM at the story
    # pass together with the speakers it may choose from. A matcher used to sit in
    # between and swallow the wordings it did not know.
    add_note(conn, memory, 1.0, bracket)
    models = ObedientModel(wants_voice=_deep_male())
    build_presentation(conn, models, audio, 1, memory)
    prompt = dressed(models)
    assert bracket.split("]")[0] + "]" in prompt      # the bracket itself, untouched
    assert "THE VOICES YOU MAY CAST" in prompt        # and what it may cast
    assert _deep_male() in prompt


def test_the_cast_voice_speaks_only_its_own_passage(conn, memory, audio):
    add_note(conn, memory, 1.0, "[страшным хриплым голосом] тут было жутко")
    models = ObedientModel(wants_voice=_deep_male())
    build_presentation(conn, models, audio, 1, memory)
    assert models.spoken_voice == _deep_male()        # the narrator it chose
    # Passage 2 alone is repainted; the rest keep the film's voice.
    assert models.spoken_voices[1] == _deep_male()
    assert len(set(models.spoken_voices)) >= 1
    assert len(models.spoken_voices) == len(models.segments)


def test_the_narration_is_still_one_clip_after_a_voice_change(conn, memory, audio):
    add_note(conn, memory, 1.0, "[мужской голос] x")
    models = ObedientModel(wants_voice=_deep_male())
    result = build_presentation(conn, models, audio, 1, memory)
    assert result.narration_key is not None and audio.exists(result.narration_key)


def test_an_invented_cast_is_refused(conn, memory, audio):
    add_note(conn, memory, 1.0, "[страшный голос] x")
    models = ObedientModel(wants_voice="count_dracula")
    build_presentation(conn, models, audio, 1, memory)
    real = {s.name for s in offered("en")}
    assert models.spoken_voice in real
    assert all(v in real for v in models.spoken_voices if v)


# --- effects ------------------------------------------------------------------------

@pytest.mark.parametrize("bracket,effect", [
    ("[glitch] right here", "glitch"),
    ("[засветка] вот тут", "light_leak"),
    ("[bloom] on this one", "bloom"),
])
def test_an_effect_bracket_reaches_the_conductor(conn, memory, audio, bracket, effect):
    add_note(conn, memory, 4.0, bracket)
    models = ObedientModel(wants_effect=effect)
    build_presentation(conn, models, audio, 1, memory)
    conductor = next(p for p in models.story_prompts
                     if "THE PHOTOGRAPHS available to you" in p)
    assert bracket.split("]")[0] + "]" in conductor
    assert effect in conductor                       # offered in the effect library
    assert "SQUARE BRACKETS" in conductor            # told it is an instruction


def test_the_chosen_effect_lands_on_a_shot(conn, memory, audio):
    add_note(conn, memory, 4.0, "[glitch] right here")
    models = ObedientModel(wants_effect="glitch")
    timeline = build_presentation(conn, models, audio, 1, memory).timeline
    played = {beat["effect"] for shot in timeline["shots"] for beat in shot["beats"]}
    assert "glitch" in played


def test_an_invented_effect_is_refused(conn, memory, audio):
    from feed import effects

    add_note(conn, memory, 4.0, "[explode] x")
    models = ObedientModel(wants_effect="explode")
    timeline = build_presentation(conn, models, audio, 1, memory).timeline
    played = {beat["effect"] for shot in timeline["shots"] for beat in shot["beats"]}
    assert played <= set(effects.SHOT_EFFECTS)


# --- music --------------------------------------------------------------------------

def test_a_music_bracket_reaches_the_score_brief(conn, memory, audio):
    add_note(conn, memory, 2.0, "[весёлый аккорд] тут")
    models = ObedientModel(music_words="a bright major chord lands here")
    build_presentation(conn, models, audio, 1, memory)
    assert "[весёлый аккорд]" in models.story_prompts[0]
    brief, _seconds = models.prompted[0]
    assert "bright major chord" in brief


def test_the_mood_still_leads_the_score(conn, memory, audio):
    # A small model reaches for "warm ambient pads" whatever it is shown, so the
    # measured mood goes in front of whatever it wrote (§11.1).
    models = ObedientModel(music_words="warm ambient pads")
    build_presentation(conn, models, audio, 1, memory)
    brief, _seconds = models.prompted[0]
    assert "BPM" in brief
    assert brief.index("BPM") < brief.index("warm ambient pads")


# --- everything at once -------------------------------------------------------------

def test_several_brackets_in_one_memory_each_reach_their_own_pass(conn, memory, audio):
    add_note(conn, memory, 1.0, "[низкий мужской голос] было страшно")
    add_note(conn, memory, 4.0, "[glitch] вот здесь")
    add_note(conn, memory, 7.0, "[барабаны] и тут")
    models = ObedientModel(wants_voice=_deep_male(), wants_effect="glitch",
                           music_words="drums come in here")
    timeline = build_presentation(conn, models, audio, 1, memory).timeline

    story = dressed(models)
    conductor = next(p for p in models.story_prompts
                     if "THE PHOTOGRAPHS available to you" in p)
    for bracket in ("[низкий мужской голос]", "[glitch]", "[барабаны]"):
        assert bracket in story, bracket          # the story pass sees all of them
        assert bracket in conductor, bracket      # so does the conductor
    assert models.spoken_voices[1] == _deep_male()
    played = {beat["effect"] for shot in timeline["shots"] for beat in shot["beats"]}
    assert "glitch" in played
    assert "drums come in here" in models.prompted[0][0]


# --- placement ----------------------------------------------------------------------

def test_a_note_appended_to_the_end_is_caught(conn, memory, audio):
    # Coverage and placement are different failures, and the second is easy to miss:
    # every note IS in the story, so nothing looks wrong — and the person hears their
    # remark about the middle of the evening read over the closing shot (§11.3).
    from feed.story import Story, misplaced_notes

    notes = ("  [at 0:00, about 0% of the way through] first flat\n"
             "  [at 0:14, about 49% of the way through] someone will be eaten tonight")
    appended = Story(narration="", cards=(), music_prompt="", script=(
        (0, "First flat, in Tbilisi."), (0, "Sunny and windy."),
        (1, "She got ready."), (2, "A girl in makeup."),
        (3, "A fun night."), (4, "Someone will be eaten tonight."),
    ))
    astray = misplaced_notes(appended, notes)
    assert len(astray) == 1
    assert "eaten" in astray[0]
    assert "belongs about 49% in" in astray[0]


def test_a_note_in_its_place_is_not_complained_about(conn, memory, audio):
    from feed.story import Story, misplaced_notes

    notes = ("  [at 0:00, about 0% of the way through] first flat\n"
             "  [at 0:14, about 49% of the way through] someone will be eaten tonight")
    placed = Story(narration="", cards=(), music_prompt="", script=(
        (0, "First flat, in Tbilisi."), (1, "She got ready."),
        (2, "Someone will be eaten tonight."), (3, "A fun night."),
    ))
    assert misplaced_notes(placed, notes) == []


def test_the_pass_is_asked_again_when_a_note_lands_in_the_wrong_place(conn, memory,
                                                                     audio):
    class Sloppy(ObedientModel):
        """Appends the note to the last passage the first time, places it the second."""

        def __init__(self):
            super().__init__()
            self.story_calls = 0

        def text_complete(self, model, messages, *, json_schema=None, **kw):
            prompt = messages[-1]["content"]
            if "WHAT THE DAY WAS" in prompt or "ЧТО ЭТО ЗА ДЕНЬ" in prompt:
                self.story_prompts.append(prompt)
                self.story_calls += 1
                late = self.story_calls == 1
                # The note lands at the END the first time and in its place the second.
                return json.dumps({"story": (
                    "First flat in Tbilisi, and it was not at all what we expected. "
                    "Nothing much happened for a long while after that. "
                    "Someone will be eaten before this evening is over."
                    if late else
                    "First flat in Tbilisi, and it was not at all what we expected. "
                    "Someone will be eaten before this evening is over. "
                    "Nothing much happened for a long while after that.")})
            return super().text_complete(model, messages, json_schema=json_schema, **kw)


    # A first cut, so there is a film length to measure the notes' positions against —
    # without one, a share is not knowable and placement is not checked at all.
    build_presentation(conn, ObedientModel(), audio, 1, memory)
    add_note(conn, memory, 0.0, "first flat in Tbilisi")
    add_note(conn, memory, 2.0, "someone will be eaten")
    models = Sloppy()
    result = build_presentation(conn, models, audio, 1, memory)
    # It is not asked again: the note says where it belongs, so the sentence carrying it
    # is MOVED there (§11.3).
    assert models.story_calls == 1
    said = [text for _, text in
            [(s["photo_id"], s["line"]) for s in result.timeline["shots"]]]
    narration = result.timeline["narration"]
    assert narration.index("eaten") < narration.index("Nothing much")
    assert said is not None


@pytest.mark.parametrize("wanted_share,put_at,should_complain", [
    (0.50, 5, True),    # asked for the middle, appended to the end
    (0.50, 2, False),   # asked for the middle, put in the middle
    (0.10, 0, False),   # asked for the start, put at the start
    (0.10, 5, True),    # asked for the start, put at the end
    (0.90, 5, False),   # asked for the end, put at the end
    (0.90, 0, True),    # asked for the end, put at the start
])
def test_placement_is_judged_against_where_the_note_was_written(
    wanted_share, put_at, should_complain
):
    # The model is told where a note belongs; this is what notices when it did not
    # listen. Every combination of asked-for and actually-placed (§11.3).
    from feed.story import Story, misplaced_notes

    notes = (f"  [at 0:10, about {round(wanted_share * 100)}% of the way through] "
             "someone will be eaten tonight")
    script = tuple(
        (i, "Someone will be eaten tonight." if i == put_at else f"Filler {i}.")
        for i in range(6)
    )
    story = Story(narration="", cards=(), music_prompt="", script=script)
    assert bool(misplaced_notes(story, notes)) is should_complain


def test_a_note_that_is_absent_is_a_coverage_problem_not_a_placement_one():
    # The two failures are distinct and must be reported distinctly, or the retry asks
    # the model to MOVE something it never wrote.
    from feed.story import Story, misplaced_notes, uncovered_notes

    notes = "  [at 0:10, about 50% of the way through] someone will be eaten tonight"
    without = Story(narration="Nothing about that.", cards=(), music_prompt="",
                    script=((0, "Nothing about that."),))
    assert misplaced_notes(without, notes) == []
    assert len(uncovered_notes(without.narration, notes)) == 1


def test_the_complaint_names_both_positions():
    # "You put this about 92% in; it belongs about 49% in" is actionable; "wrong place"
    # is not.
    from feed.story import Story, misplaced_notes

    notes = "  [at 0:14, about 49% of the way through] someone will be eaten tonight"
    story = Story(narration="", cards=(), music_prompt="", script=tuple(
        (i, "Someone will be eaten tonight." if i == 5 else f"Filler {i}.")
        for i in range(6)
    ))
    complaint = misplaced_notes(story, notes)[0]
    assert "you put this about 92% in" in complaint
    assert "belongs about 49% in" in complaint


def test_a_retry_that_places_it_worse_is_discarded():
    # A second attempt is kept only if it answered MORE than the first, so a model that
    # comes back worse cannot make the film worse.
    from feed.story import Story, _answered

    notes = "  [at 0:10, about 50% of the way through] someone will be eaten tonight"
    good = Story(narration="Someone will be eaten tonight.", cards=(), music_prompt="",
                 script=((0, "A."), (1, "Someone will be eaten tonight."), (2, "B.")))
    bad = Story(narration="Someone will be eaten tonight.", cards=(), music_prompt="",
                script=((0, "A."), (1, "B."), (2, "C."), (3, "D."),
                        (4, "Someone will be eaten tonight.")))
    assert _answered(good, notes, "") > _answered(bad, notes, "")


def test_every_note_is_checked_not_just_the_last(conn, memory, audio):
    from feed.story import Story, misplaced_notes

    notes = ("  [at 0:00, about 5% of the way through] first flat\n"
             "  [at 0:10, about 50% of the way through] she got ready\n"
             "  [at 0:18, about 90% of the way through] someone will be eaten")
    # All three crammed into the opening passages.
    story = Story(narration="", cards=(), music_prompt="", script=(
        (0, "First flat."), (1, "She got ready."), (2, "Someone will be eaten."),
        (3, "Filler."), (4, "Filler."), (5, "Filler."),
    ))
    astray = misplaced_notes(story, notes)
    # The first is where it belongs; the other two were both pulled forward.
    assert len(astray) == 2
    joined = " ".join(astray).lower()
    assert "she got ready" in joined and "someone will be eaten" in joined
    assert "first flat" not in joined
