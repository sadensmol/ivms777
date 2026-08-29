"""The story pass (design §11.1) — TWO stages.

1. **Tell**: prose, in the film's language, with a beginning, a middle and an end.
   No photographs, no cards, no cast — that output shape was what turned every film
   into one caption per picture.
2. **Cut and dress**: which photograph carries each sentence, plus the title cards, the
   narrator, any per-passage voice and the score plan. It never rewrites a word.

Between them the prose is repaired against what was asked for: the film's language, its
length, and every one of the viewer's notes.
"""

import json

import pytest

from feed.director import direct
from feed.language import DEFAULT, LANGUAGES, get_language, set_language
from feed.signals import ShotSignal
from feed.story import narration_words, sentences, write_story

# Which call a prompt is — the fake has to answer four different ones.
TELL_MARKS = ("ЧТО ЭТО ЗА ДЕНЬ", "WHAT THE DAY WAS")
EXPAND_MARKS = ("Допиши", "more sentences")
WEAVE_MARKS = ("ПРЕДЛОЖЕНИЕ", "SENTENCE:")

# Sentences long enough to be their own BEAT — a passage holds a photograph for as long
# as it is spoken, so short clauses are joined (`_MIN_SENTENCE_CHARS`).
PROSE = ("Мы приехали в Тбилиси поздним ветреным вечером и сразу пошли гулять. "
         "Было очень холодно, но город казался невероятно красивым. "
         "Мы гуляли допоздна и вернулись домой совершенно счастливыми.")
ENGLISH_PROSE = ("We got into Tbilisi late in the evening and went straight out. "
                 "It was bitterly cold, and the whole city looked beautiful. "
                 "We walked until midnight and came home very happy.")


def stage(prompt: str) -> str:
    if prompt.startswith("Translate into"):
        return "translate"
    for name, marks in (("tell", TELL_MARKS), ("expand", EXPAND_MARKS),
                        ("weave", WEAVE_MARKS)):
        if any(mark in prompt for mark in marks):
            return name
    return "dress"


class FakeClient:
    """Answers every stage. `payload` is the cut-and-dress answer."""

    def __init__(self, payload, prose=PROSE, woven="", extra="", translated=None):
        self.payload = payload
        self.prose = prose
        self.woven = woven
        self.extra = extra
        self.translated = translated
        self.calls = []

    def text_complete(self, model, messages, *, json_schema=None, **kw):
        prompt = messages[-1]["content"]
        which = stage(prompt)
        self.calls.append({"model": model, "messages": messages, "schema": json_schema,
                           "stage": which, "prompt": prompt})
        if isinstance(self.payload, Exception):
            raise self.payload
        if which == "tell":
            return json.dumps({"story": self.prose})
        if which == "translate":
            return json.dumps({"story": self.prose if self.translated is None
                               else self.translated})
        if which == "expand":
            return json.dumps({"story": self.extra})
        if which == "weave":
            return json.dumps({"text": self.woven})
        return self.payload

    def prompt_of(self, which):
        return next(c["prompt"] for c in self.calls if c["stage"] == which)

    def staged(self, which):
        return [c for c in self.calls if c["stage"] == which]


def sig(pid):
    return ShotSignal(photo_id=pid, width=4000, height=3000, shot_at=None,
                      caption=f"caption {pid}", description=f"a full description {pid}",
                      place=None, tags={}, scores={}, vector=None)


EDIT = direct([sig(1), sig(2)])
# The dress answer: sentence -> photograph, plus the film's furniture.
GOOD = json.dumps({
    "script": [{"sentence": 1, "shot": 1, "voice": ""},
               {"sentence": 2, "shot": 2, "voice": ""},
               {"sentence": 3, "shot": 1, "voice": ""}],
    "narration": "",
    "narrator": "xenia",
    "cards": [{"shot": 1, "text": "Golden light"}],
    "music_prompt": "warm ambient drone",
})


# --- stage 1: the story ---------------------------------------------------------------


def test_the_narration_is_the_story_that_was_told():
    story = write_story(FakeClient(GOOD), "gemma", title="Beach day",
                        description="Warm.", edit=EDIT, language="ru")
    assert story.narration == PROSE


def test_the_story_is_asked_for_as_a_story_not_as_captions():
    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru")
    told = client.prompt_of("tell")
    assert "РАССКАЗ, А НЕ ПОДПИСИ К ФОТО" in told
    assert "начало, середина и КОНЕЦ" in told
    # The photographs are NOT in this call: their bookkeeping is what produced captions.
    assert "photograph" not in told.lower()


def test_the_story_call_carries_the_memorys_own_facts():
    client = FakeClient(GOOD)
    write_story(client, "gemma", title="Halloween", description="In Tbilisi.",
                edit=EDIT, language="ru", digest="3 photographs. Where: Tbilisi.")
    told = client.prompt_of("tell")
    assert "Halloween" in told and "In Tbilisi." in told and "Where: Tbilisi." in told


def test_english_asks_in_english():
    client = FakeClient(GOOD, prose=ENGLISH_PROSE)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="en")
    assert "WHAT THE DAY WAS" in client.prompt_of("tell")


def test_english_is_the_default_language():
    client = FakeClient(GOOD, prose=ENGLISH_PROSE)
    write_story(client, "gemma", title="T", description="D", edit=EDIT)
    assert "WHAT THE DAY WAS" in client.prompt_of("tell")


def test_a_story_that_cannot_be_written_leaves_a_wordless_film():
    client = FakeClient(GOOD, prose="")
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru")
    assert story.narration == ""
    assert story.music_prompt                      # the film still gets a score brief


def test_a_raising_client_degrades_rather_than_propagating():
    story = write_story(FakeClient(RuntimeError("boom")), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert story.narration == ""


def test_sentences_split_the_story_into_passages():
    assert sentences(PROSE) == [
        "Мы приехали в Тбилиси поздним ветреным вечером и сразу пошли гулять.",
        "Было очень холодно, но город казался невероятно красивым.",
        "Мы гуляли допоздна и вернулись домой совершенно счастливыми.",
    ]


def test_short_sentences_are_joined_into_one_beat():
    # A passage holds a photograph for as long as it is spoken; four-word sentences
    # would cut the film every two seconds.
    assert sentences("Было темно. Дул ветер. Мы стояли и молчали, не понимая ничего.") == [
        "Было темно. Дул ветер. Мы стояли и молчали, не понимая ничего.",
    ]


# --- the language guarantee ------------------------------------------------------------


def test_a_story_told_in_the_wrong_language_is_translated():
    # Composing in a second language is hard for a small model; translating what it can
    # already see is not.
    client = FakeClient(GOOD, prose="It was a cold evening in Tbilisi and we walked.",
                        translated=PROSE)
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru")
    assert story.narration == PROSE
    assert client.staged("translate")


def test_an_untranslatable_story_plays_silent_rather_than_in_the_wrong_language():
    # A Russian voice reading English word by word is worse than no narration (§11.1).
    client = FakeClient(GOOD, prose="Still English.", translated="Still English.")
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru")
    assert story.narration == ""


def test_english_is_never_translated():
    client = FakeClient(GOOD, prose=ENGLISH_PROSE)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="en")
    assert not client.staged("translate")


# --- the length guarantee --------------------------------------------------------------


SHORT_PROSE = "Мы приехали."


def test_a_story_far_short_of_its_band_is_carried_on():
    client = FakeClient(GOOD, prose=SHORT_PROSE,
                        extra="Потом мы вышли на улицу. Вечер закончился тихо.")
    story = write_story(client, "gemma", title="T", description="D",
                        edit=direct([sig(i) for i in range(1, 6)]), language="ru")
    assert "вышли на улицу" in story.narration
    assert client.staged("expand")


def test_the_extension_never_repeats_a_sentence():
    client = FakeClient(GOOD, prose=SHORT_PROSE,
                        extra="Мы приехали. Вечер закончился тихо.")
    story = write_story(client, "gemma", title="T", description="D",
                        edit=direct([sig(i) for i in range(1, 6)]), language="ru")
    assert story.narration.count("Мы приехали.") == 1


def test_a_story_already_long_enough_is_not_extended():
    client = FakeClient(GOOD, prose="Это было длинное и подробное предложение. " * 20)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru")
    assert not client.staged("expand")


def test_the_word_budget_pays_for_every_note():
    from feed.story import WORDS_PER_NOTE

    bare, with_notes = narration_words(3), narration_words(3, 2)
    assert with_notes[0] - bare[0] == 2 * WORDS_PER_NOTE
    assert with_notes[1] - bare[1] == 2 * WORDS_PER_NOTE


def test_the_story_call_asks_for_the_bigger_budget_when_notes_are_present():
    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru",
                notes="  [at 0:00, about 0% of the way through] это Юля")
    low, high = narration_words(len(EDIT.shots), 1)
    assert f"{low}-{high}" in client.prompt_of("tell")


# --- the note guarantee (§11.3) --------------------------------------------------------


NOTE = ("  [at 0:14, about 66% of the way through, looking at photograph 3] вдруг Юле "
        "показалось что ее отражение в зеркале сверкает красными глазами")


def test_a_note_the_story_ignored_is_woven_into_its_sentence():
    client = FakeClient(GOOD, woven="Юле показалось, что отражение сверкает красными "
                                    "глазами.")
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru", notes=NOTE)
    assert "красными глазами" in story.narration
    assert len(story.script) == 3                        # rewritten, not appended


def test_the_rewrite_is_asked_for_in_the_films_language():
    client = FakeClient(GOOD, woven="Юле показалось, что отражение сверкает красными "
                                    "глазами.")
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru",
                notes=NOTE)
    assert "ПРЕДЛОЖЕНИЕ" in client.prompt_of("weave")


def test_a_rewrite_that_does_not_carry_the_note_falls_back_to_the_verbatim_insert():
    client = FakeClient(GOOD, woven="Совершенно другое предложение.")
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru", notes=NOTE)
    assert "красными глазами" in story.narration
    assert len(story.script) == 4                        # inserted


def test_the_insert_lands_where_the_viewer_was_standing():
    client = FakeClient(GOOD, woven="")
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru", notes=NOTE)
    said = sentences(story.narration)
    assert any("красными" in text for text in said[1:])   # 66% of three sentences


def test_a_bracket_is_never_narrated():
    notes = "  [at 0:00, about 0% of the way through] [суровый мужской голос] это утро"
    client = FakeClient(GOOD, woven="")
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru", notes=notes)
    assert "это утро" in story.narration
    assert "суровый" not in story.narration


def test_a_note_in_another_language_is_left_to_the_model():
    # A Russian voice reading English phonetically is worse than a missing sentence.
    notes = "  [at 0:14, about 66% of the way through] her eyes were red in the mirror"
    client = FakeClient(GOOD, woven="")
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru", notes=notes)
    assert "her eyes" not in story.narration


def test_a_covered_note_is_left_alone():
    covered = ("  [at 0:00, about 0% of the way through] мы приехали в Тбилиси "
               "поздним ветреным вечером")
    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru",
                notes=covered)
    assert not client.staged("weave")


# --- stage 2: the cut and the dressing -------------------------------------------------


def test_every_sentence_gets_a_photograph():
    story = write_story(FakeClient(GOOD), "gemma", title="T", description="D",
                        edit=EDIT, language="ru")
    assert len(story.script) == 3
    assert all(0 <= shot < len(EDIT.shots) for shot, _ in story.script)


def test_the_words_are_ours_not_the_matching_calls():
    # The matching call cannot rewrite the story: it only points at sentences.
    payload = json.loads(GOOD)
    payload["script"][0]["text"] = "SOMETHING THE MODEL MADE UP"
    story = write_story(FakeClient(json.dumps(payload)), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert "MADE UP" not in story.narration


def test_a_sentence_the_matching_call_forgot_is_still_in_the_film():
    payload = json.loads(GOOD)
    payload["script"] = [{"sentence": 1, "shot": 1, "voice": ""}]
    story = write_story(FakeClient(json.dumps(payload)), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert len(story.script) == 3
    assert story.narration == PROSE


def test_an_unusable_match_still_lays_the_story_over_the_photographs():
    payload = json.loads(GOOD)
    payload["script"] = "nonsense"
    story = write_story(FakeClient(json.dumps(payload)), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert len(story.script) == 3
    assert story.narration == PROSE


def test_the_matching_call_is_shown_the_photographs_and_the_sentences():
    client = FakeClient(GOOD)
    edit = direct([sig(1), sig(2)])
    from feed.director import with_lines

    edit = with_lines(edit, ["caption 1", "caption 2"])
    write_story(client, "gemma", title="T", description="D", edit=edit, language="ru")
    dressed = client.prompt_of("dress")
    assert "2 shots" in dressed
    assert "caption 1" in dressed and "caption 2" in dressed
    assert "1. Мы приехали в Тбилиси поздним ветреным вечером" in dressed


def test_a_schema_is_always_sent():
    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru")
    schema = next(c["schema"] for c in client.calls if c["stage"] == "dress")
    assert schema["required"] == ["script", "cards", "music_prompt", "narrator"]
    assert schema["properties"]["script"]["items"]["required"] == [
        "shot", "sentence", "voice",
    ]


def test_title_cards_are_sparse_and_bounded():
    from feed.story import MAX_TITLE_CARDS

    payload = json.loads(GOOD)
    payload["cards"] = [{"shot": 1, "text": f"card {i}"} for i in range(20)]
    story = write_story(FakeClient(json.dumps(payload)), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert len(story.cards) <= MAX_TITLE_CARDS


def test_a_card_pointing_outside_the_film_is_dropped():
    payload = json.loads(GOOD)
    payload["cards"] = [{"shot": 99, "text": "nowhere"}]
    story = write_story(FakeClient(json.dumps(payload)), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert story.cards == ()


def test_lines_for_leaves_untitled_shots_blank():
    story = write_story(FakeClient(GOOD), "gemma", title="T", description="D",
                        edit=EDIT, language="ru")
    assert story.lines_for(3) == ["Golden light", "", ""]


def test_the_fallback_music_prompt_is_still_atmospheric():
    payload = json.loads(GOOD)
    payload["music_prompt"] = ""
    story = write_story(FakeClient(json.dumps(payload)), "gemma", title="Birthday",
                        description="D", edit=EDIT, language="ru")
    assert "Birthday" in story.music_prompt


def test_the_score_plan_is_parsed_as_zero_based_sentences():
    payload = json.loads(GOOD)
    payload["score"] = [{"from": 1, "brief": "solo piano"},
                        {"from": 3, "brief": "strings swell"}]
    story = write_story(FakeClient(json.dumps(payload)), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert story.score == ((0, "solo piano"), (2, "strings swell"))


def test_a_score_change_pointing_past_the_script_is_dropped():
    payload = json.loads(GOOD)
    payload["score"] = [{"from": 1, "brief": "piano"}, {"from": 99, "brief": "brass"}]
    story = write_story(FakeClient(json.dumps(payload)), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert story.score == ((0, "piano"),)


def test_the_matching_prompt_asks_where_the_music_changes():
    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru")
    assert "WHERE THE MUSIC CHANGES" in client.prompt_of("dress")


def test_the_narrator_is_carried_through():
    story = write_story(FakeClient(GOOD), "gemma", title="T", description="D",
                        edit=EDIT, language="ru")
    assert story.narrator == "xenia"


def test_a_bracket_with_no_voice_cast_makes_the_matching_call_ask_again():
    client = FakeClient(GOOD, woven="")
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru",
                notes="  [at 0:00, about 0% of the way through] [хриплый голос] тут",
                speakers="  aidar — male, deep: low and rough")
    assert len(client.staged("dress")) == 2


def test_a_cast_voice_survives_the_match():
    payload = json.loads(GOOD)
    payload["script"][0]["voice"] = "aidar"
    story = write_story(FakeClient(json.dumps(payload)), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert story.voices[0] == "aidar"


# --- the prompt fits the server (§11.1) ------------------------------------------------


def _long_edit(count=40):
    return direct([sig(i) for i in range(1, count + 1)], max_shots=count)


def _briefs(count=40):
    return [f"A very long photograph brief number {i}. " * 20 for i in range(1, count + 1)]


def test_the_matching_prompt_is_cut_to_what_the_server_accepts():
    from feed.story import CHARS_PER_TOKEN, RESERVED_TOKENS

    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=_long_edit(),
                shots=_briefs(), language="ru", context_tokens=8192)
    prompt = client.prompt_of("dress")
    assert len(prompt) / CHARS_PER_TOKEN + RESERVED_TOKENS <= 8192


def test_every_photograph_is_still_listed_when_the_prompt_is_cut():
    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=_long_edit(),
                shots=_briefs(), language="ru", context_tokens=8192)
    prompt = client.prompt_of("dress")
    assert all(f"\n{i}. " in prompt for i in range(1, 41))


def test_an_unknown_context_trims_nothing():
    client = FakeClient(GOOD)
    briefs = _briefs()
    write_story(client, "gemma", title="T", description="D", edit=_long_edit(),
                shots=briefs, language="ru", context_tokens=None)
    assert briefs[0] in client.prompt_of("dress")


def test_a_brief_never_shrinks_below_the_floor():
    from feed.story import MIN_BRIEF_CHARS, brief_chars

    assert brief_chars(40, fixed_chars=100_000, context_tokens=2048) == MIN_BRIEF_CHARS


# --- malformed answers -----------------------------------------------------------------


def test_a_fenced_json_answer_is_parsed():
    client = FakeClient(f"```json\n{GOOD}\n```")
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru")
    assert story.cards == ((0, "Golden light"),)


def test_json_with_chatter_around_it_is_recovered():
    client = FakeClient(f"Sure! Here it is:\n{GOOD}\nHope that helps.")
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru")
    assert story.cards == ((0, "Golden light"),)


def test_an_unparseable_match_still_leaves_the_story():
    story = write_story(FakeClient("not json at all"), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert story.narration == PROSE
    assert len(story.script) == 3


# --- the film's language setting -------------------------------------------------------


def test_language_defaults_to_english(conn):
    assert get_language(conn, 1) == DEFAULT


def test_language_round_trips(conn):
    for language in LANGUAGES:
        set_language(conn, 1, language)
        assert get_language(conn, 1) == language


def test_an_unknown_language_is_rejected(conn):
    with pytest.raises(ValueError):
        set_language(conn, 1, "de")


def test_a_stored_junk_language_degrades_to_the_default(conn):
    from db.settings import set_setting

    set_setting(conn, 1, "feed_language", "klingon")
    assert get_language(conn, 1) == DEFAULT


def test_a_bracket_the_story_copied_in_is_never_narrated():
    # Measured: the story pass put "[суровый мужской голос]" in its first sentence and
    # the voice read it aloud (§11.3).
    client = FakeClient(GOOD, prose="[суровый мужской голос] " + PROSE)
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru")
    assert "суровый" not in story.narration
    assert story.narration.startswith("Мы приехали")
