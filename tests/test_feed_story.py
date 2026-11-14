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
from feed.story import Story, match_photographs, narration_words, sentences, write_story

# Which call a prompt is — the fake has to answer five different ones.
TELL_MARKS = ("ЧТО ЭТО ЗА ДЕНЬ", "WHAT THE DAY WAS")
EXPAND_MARKS = ("Допиши", "more sentences")
WEAVE_MARKS = ("ПРЕДЛОЖЕНИЕ", "SENTENCE:")
MATCH_MARKS = ("Which numbered photograph",)

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
                        ("weave", WEAVE_MARKS), ("match", MATCH_MARKS)):
        if any(mark in prompt for mark in marks):
            return name
    return "dress"


class FakeClient:
    """Answers every stage. `payload` is the cut-and-dress answer, and `votes` the
    per-sentence matches in turn, each a raw reply or an Exception to raise."""

    def __init__(self, payload, prose=PROSE, woven="", extra="", translated=None, votes=()):
        self.payload = payload
        self.votes = list(votes)
        self.prose = prose
        self.woven = woven
        self.extra = extra
        self.translated = translated
        self.calls = []

    def text_complete(self, model, messages, *, json_schema=None, **kw):
        prompt = messages[-1]["content"]
        which = stage(prompt)
        self.calls.append({"model": model, "messages": messages, "schema": json_schema,
                           "stage": which, "prompt": prompt, "kw": kw})
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
        if which == "match":
            return self.vote()
        return self.payload

    def vote(self):
        answer = self.votes.pop(0) if self.votes else "{}"
        if isinstance(answer, Exception):
            raise answer
        return answer

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
    "cards": [{"sentence": 1, "text": "Golden light"}],
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
                edit=EDIT, language="ru", facts="3 photographs. Where: Tbilisi.")
    told = client.prompt_of("tell")
    assert "Halloween" in told and "In Tbilisi." in told and "Where: Tbilisi." in told


def test_the_story_is_told_from_what_the_photographs_show():
    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru",
                photos=["Five young men around a campfire with guitars.",
                        "Two men by a light-coloured SUV at the edge of the trees."])
    told = client.prompt_of("tell")
    assert "1. Five young men around a campfire with guitars." in told
    assert "2. Two men by a light-coloured SUV" in told


def test_the_story_call_is_not_shown_the_guessed_labels():
    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru",
                digest="Season and weather: rain.", facts="2 photographs.")
    told = client.prompt_of("tell")
    assert "2 photographs." in told
    assert "rain" not in told


def test_more_story_is_asked_for_from_the_same_photographs():
    client = FakeClient(GOOD, prose="Было темно.", extra="")
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru",
                photos=["Five young men around a campfire with guitars."])
    assert "campfire with guitars" in client.prompt_of("expand")


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


# A real Russian film's narration: fourteen sentences, thirteen of them under 45
# characters. The old join glued every one onto the one before and returned TWO
# passages, so a six-photo memory played as a two-photo film.
SHORT_SENTENCES = (
    "Помнишь тот день? Мы были в лесу. Дорожка грязная, а дождь хлестал по листве. "
    "Мы собрались вместе. Гитары лежали, огонь потрескивал. Это было то время. "
    "Чувствовалось то знакомое, тягучее счастье. Было немного скучно, но это было "
    "хорошо. Мы смеялись, спорили о чем-то пустом. Обнимались, и тепло от огня "
    "обволакивало. В этом хаосе была вся наша привязанность. Время текло медленно, "
    "как густой мёд. Мы просто были. И в этом моменте, среди сырости и темноты, всё "
    "казалось совершенным.")


def test_a_join_stops_once_the_passage_is_long_enough():
    passages = sentences(SHORT_SENTENCES)
    assert len(passages) == 7
    assert " ".join(passages) == SHORT_SENTENCES


def test_a_short_last_sentence_joins_the_passage_before_it():
    long = "Мы приехали в Тбилиси поздним ветреным вечером и сразу пошли гулять."
    assert sentences(f"{long} Было темно.") == [f"{long} Было темно."]


def test_the_story_is_never_cut_into_fewer_passages_than_photographs():
    passages = sentences(SHORT_SENTENCES, at_least=12)
    assert len(passages) >= 12
    assert " ".join(passages) == SHORT_SENTENCES


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


def test_a_photograph_the_matching_call_left_out_stays_out_of_the_film():
    # GOOD puts sentences 1-3 over photographs 1, 2, 1 — photograph 3 is never shown.
    edit = direct([sig(1), sig(2), sig(3)])
    story = write_story(FakeClient(GOOD), "gemma", title="T", description="D",
                        edit=edit, language="ru")
    assert [shot for shot, _ in story.script] == [0, 1, 0]
    assert story.narration == PROSE


def test_a_story_with_fewer_sentences_than_photographs_is_extended():
    # Long enough in WORDS, so the length repair stays quiet — but three sentences
    # cannot carry four photographs, so the follow-up is asked for more.
    prose = ("Мы приехали в старый Тбилиси поздним ветреным вечером, бросили вещи в "
             "маленькой гостинице и сразу пошли гулять по узким улицам. "
             "Было очень холодно, ветер дул с реки, но город с его огнями и балконами "
             "казался нам невероятно красивым и совсем родным. "
             "Мы гуляли допоздна, заходили в крошечные кафе, пили горячее вино и "
             "вернулись домой совершенно счастливыми и уставшими.")
    extra = "Под утро город затих, и мы долго смотрели на пустые мокрые улицы."
    edit = direct([sig(i) for i in range(1, 5)])
    client = FakeClient(GOOD, prose=prose, extra=extra)
    story = write_story(client, "gemma", title="T", description="D", edit=edit,
                        language="ru")
    assert len(client.staged("expand")) == 1
    assert len(story.script) == 4


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
    payload["cards"] = [{"sentence": 1, "text": f"card {i}"} for i in range(20)]
    story = write_story(FakeClient(json.dumps(payload)), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert len(story.cards) <= MAX_TITLE_CARDS


def test_a_card_pointing_outside_the_film_is_dropped():
    payload = json.loads(GOOD)
    payload["cards"] = [{"sentence": 99, "text": "nowhere"}]
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


VOICE_NOTE = "  [at 0:00, about 0% of the way through] [хриплый голос] тут"
CASTING_CLAUSE = "cast a speaker like that for THAT passage"


def test_a_film_with_speakers_is_told_to_cast_a_bracket():
    client = FakeClient(GOOD, woven="")
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru",
                notes=VOICE_NOTE, speakers="  aidar — male, deep: low and rough")
    prompt = client.prompt_of("dress")
    assert CASTING_CLAUSE in prompt
    assert "THE VOICES YOU MAY CAST" in prompt


def test_a_chosen_narrator_reads_every_passage_and_a_bracket_casts_no_one():
    client = FakeClient(GOOD, woven="")
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru",
                notes=VOICE_NOTE)
    prompt = client.prompt_of("dress")
    assert CASTING_CLAUSE not in prompt
    assert "THE VOICES YOU MAY CAST" not in prompt
    assert "reads ALL passages" in prompt
    assert "ANYTHING IN [SQUARE BRACKETS] IS AN INSTRUCTION" in prompt
    assert "A bracket about the music belongs in `music_prompt`" in prompt
    assert len(client.staged("dress")) == 1                 # no retry for a cast


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


#
# The expected photographs are always a passage's own vote, else the SELECTED stage-2
# answer's. Nothing relabels a photograph.

TOLD = ["Sentence A.", "Sentence B.", "Sentence C.", "Sentence D."]
EDIT3 = direct([sig(1), sig(2), sig(3)])
SPEAKERS = "  aidar — male, deep: low and rough"


class DressClient(FakeClient):
    """Answers each stage-2 call with the next payload; an Exception is raised."""

    def __init__(self, *dresses, votes=()):
        super().__init__(None, votes=votes)
        self.dresses = list(dresses)

    def text_complete(self, model, messages, *, json_schema=None, **kw):
        prompt = messages[-1]["content"]
        self.calls.append({"stage": stage(prompt), "prompt": prompt, "schema": json_schema,
                           "kw": kw})
        if stage(prompt) == "match":
            return self.vote()
        answer = self.dresses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def dress_answer(photos, voices=None, **extra):
    """A stage-2 answer: sentence i+1 over photograph photos[i]+1."""
    voices = voices or [""] * len(photos)
    return json.dumps({
        "script": [{"sentence": i + 1, "shot": photo + 1, "voice": voice}
                   for i, (photo, voice) in enumerate(zip(photos, voices))],
        "narration": "",
        **extra,
    })


def votes(*photos):
    """Per-sentence matches: 0-based photographs, as the 1-based replies a model gives."""
    return [json.dumps({"photo": photo + 1}) for photo in photos]


FIRST = dress_answer([2, 0, 2, 1], ["", "aidar", "", ""], narrator="xenia",
                     cards=[{"sentence": 2, "text": "The turn"}],
                     score=[{"from": 1, "brief": "solo piano"}],
                     music_prompt="warm strings")
# The first answer as the selected Story: its words, photographs and furniture.
SELECTED = {
    "narration": " ".join(TOLD),
    "cards": ((1, "The turn"),),
    "music_prompt": "warm strings",
    "narrator": "xenia",
    "script": ((2, TOLD[0]), (0, TOLD[1]), (2, TOLD[2]), (1, TOLD[3])),
    "voices": ("", "aidar", "", ""),
    "score": ((0, "solo piano"),),
}


def dress(client, told=TOLD, edit=EDIT3, notes="", speakers="", photos=None):
    from feed.story import _dress

    return _dress(client, "gemma", told, "T", "D", edit, None, "en", "", notes, "",
                  speakers, None, photos)


@pytest.mark.parametrize(("photos", "expected"), [
    ([0, 1, 0, 2], True),
    ([2, 0, 2, 1], False),
    ([0, 0, 2, 1], False),
    ([], True),
])
def test_introduces_in_order_checks_first_appearances_only(photos, expected):
    from feed.story import introduces_in_order

    assert introduces_in_order(photos) is expected


def test_the_script_is_parsed_in_told_order_with_each_voice_on_its_sentence():
    from feed.story import _script

    told = ["A", "B", "C", "D", "E"]
    raw = [
        {"sentence": 3, "shot": 2, "voice": "v3"},
        "junk",
        {"sentence": 2, "shot": 3, "voice": "v2"},
        {"sentence": 3, "shot": 1, "voice": "duplicate"},
        {"sentence": "x", "shot": 1, "voice": "malformed"},
        {"sentence": 5, "shot": 1, "voice": "v5"},
    ]
    script, voices, matched = _script(raw, 3, told)
    # Sentence 1 is missing: it takes the next accepted sentence's photograph (2's).
    # Sentence 4 is missing: it takes the photograph of the sentence before it (3's).
    assert script == ((2, "A"), (2, "B"), (1, "C"), (1, "D"), (0, "E"))
    assert voices == ("", "v2", "v3", "", "v5")
    assert matched is True


@pytest.mark.parametrize("raw", [
    "nonsense",
    [],
    ["junk", {"sentence": 9, "shot": 1}, {"shot": 1, "voice": ""}],
])
def test_a_script_with_nothing_accepted_is_the_even_fallback(raw):
    from feed.story import _script

    script, voices, matched = _script(raw, 3, TOLD)
    assert [photo for photo, _ in script] == [0, 0, 1, 2]
    assert [text for _, text in script] == TOLD
    assert voices == ("", "", "", "")
    assert matched is False


def test_a_card_names_a_sentence_and_lands_on_that_passage():
    from feed.story import MAX_TITLE_CARDS, _cards

    assert _cards([{"sentence": 3, "text": "Home"}], 4) == ((2, "Home"),)
    assert _cards([{"sentence": 5, "text": "Past the end"}], 4) == ()
    assert _cards([{"shot": 1, "text": "Old key"}], 4) == ()
    many = [{"sentence": 1, "text": f"card {i}"} for i in range(20)]
    assert len(_cards(many, 4)) == MAX_TITLE_CARDS


def test_a_card_on_a_returning_passage_stays_on_that_passage():
    payload = json.loads(GOOD)
    payload["cards"] = [{"sentence": 3, "text": "Back again"}]
    story = write_story(FakeClient(json.dumps(payload)), "gemma", title="T",
                        description="D", edit=EDIT, language="ru")
    assert story.script[2][0] == story.script[0][0]
    assert story.lines_for(len(story.script)) == ["", "", "Back again"]


def test_the_schema_and_prompt_ask_for_a_cards_sentence_and_no_order_or_coverage():
    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru")
    call = next(c for c in client.calls if c["stage"] == "dress")
    assert call["schema"]["properties"]["cards"]["items"]["required"] == ["sentence", "text"]
    assert '{"sentence": <1-based sentence number>, "text": "..."}' in call["prompt"]
    assert "EVERY photograph at least once" not in call["prompt"]
    assert "Introduce the photographs" not in call["prompt"]


def test_an_answer_in_the_memorys_order_is_kept_with_no_second_stage_2_call():
    client = FakeClient(GOOD)                         # photographs [0, 1, 0]
    story = write_story(client, "gemma", title="T", description="D", edit=EDIT,
                        language="ru")
    assert [photo for photo, _ in story.script] == [0, 1, 0]
    assert len(client.staged("dress")) == 1
    assert story.memory_order == "kept"


def test_a_sentence_moves_to_the_photograph_it_describes():
    # The film of 2026-11-04: the two cats' sentence was on photograph 3, the one cat.
    told = ["The cat and her.", "The girl.", "The blanket.", "The light.",
            "Two cats in a basket.", "The end."]
    client = DressClient(dress_answer([0, 2, 1, 3, 2, 3]), votes=votes(0, 2, 1, 3, 3, 3))
    story = dress(client, told=told, edit=direct([sig(i) for i in range(1, 5)]))
    assert story.script[4] == (3, "Two cats in a basket.")
    assert story.votes == (0, 2, 1, 3, 3, 3)
    assert story.memory_order == "broken"


def test_each_passage_is_matched_on_its_own_at_temperature_0():
    client = DressClient(FIRST)
    dress(client, photos=["Two cats.", "A girl.", "One cat."])
    calls = client.staged("match")
    assert len(calls) == len(TOLD)
    for call, sentence in zip(calls, TOLD):
        assert [text for text in TOLD if text in call["prompt"]] == [sentence]
        assert "1. Two cats.\n2. A girl.\n3. One cat." in call["prompt"]
        assert "order" not in call["prompt"].lower()
        assert call["kw"]["temperature"] == 0
        assert call["schema"]["properties"] == {"photo": {"type": "integer"}}
        assert call["schema"]["required"] == ["photo"]


@pytest.mark.parametrize(("photos", "listed"), [
    (["DESCRIPTION 1", "DESCRIPTION 2"], "1. DESCRIPTION 1\n2. DESCRIPTION 2"),
    (None, "1. BRIEF 1\n2. BRIEF 2"),
], ids=["descriptions", "briefs"])
def test_a_sentence_is_matched_against_the_descriptions_else_the_briefs(photos, listed):
    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=EDIT, language="ru",
                shots=["BRIEF 1", "BRIEF 2"], photos=photos)
    assert listed in client.prompt_of("match")


def test_every_matching_prompt_fits_the_server_and_lists_every_photograph():
    from feed.story import CHARS_PER_TOKEN, RESERVED_TOKENS

    client = FakeClient(GOOD)
    write_story(client, "gemma", title="T", description="D", edit=_long_edit(),
                shots=_briefs(), language="ru", context_tokens=8192)
    prompts = [call["prompt"] for call in client.staged("match")]
    assert prompts
    for prompt in prompts:
        assert len(prompt) / CHARS_PER_TOKEN + RESERVED_TOKENS <= 8192
        assert "\n40. A very long" in prompt


def test_a_correct_return_stays_with_an_unused_third_photograph():
    client = DressClient(dress_answer([0, 2, 1]), votes=votes(0, 1, 0))
    story = dress(client, told=TOLD[:3])
    assert [photo for photo, _ in story.script] == [0, 1, 0]
    assert story.memory_order == "kept"


def test_stage_2s_repeat_stays_with_an_unused_third_photograph():
    story = dress(DressClient(dress_answer([0, 1, 0])), told=TOLD[:3])
    assert [photo for photo, _ in story.script] == [0, 1, 0]
    assert story.votes == (None, None, None)


def test_a_photograph_no_sentence_is_about_is_logged(caplog):
    with caplog.at_level("INFO", logger="feed.story"):
        dress(DressClient(dress_answer([0, 1, 0])), told=TOLD[:3])
    assert "no sentence is about photographs [2]" in caplog.text


def test_out_of_order_matches_stay_with_no_further_call():
    client = DressClient(dress_answer([0, 1, 0, 2]), votes=votes(2, 0, 2, 1))
    story = dress(client)
    assert [photo for photo, _ in story.script] == [2, 0, 2, 1]
    assert [text for _, text in story.script] == TOLD
    assert story.memory_order == "broken"
    assert len(client.staged("dress")) == 1


def test_a_match_changes_only_the_photographs():
    story = dress(DressClient(FIRST, votes=votes(0, 1, 0, 2)))
    assert story == Story(**{**SELECTED,
                             "script": ((0, TOLD[0]), (1, TOLD[1]), (0, TOLD[2]),
                                        (2, TOLD[3]))},
                          votes=(0, 1, 0, 2), memory_order="kept")


def test_a_voice_cast_by_the_voice_retry_survives_the_match():
    client = DressClient(
        dress_answer([0, 1, 0, 2]),                              # no voice cast
        dress_answer([2, 0, 2, 1], ["", "aidar", "", ""]),       # cast
        votes=votes(0, 1, 0, 2),
    )
    story = dress(client, notes=VOICE_NOTE, speakers=SPEAKERS)
    assert len(client.staged("dress")) == 2
    assert [photo for photo, _ in story.script] == [0, 1, 0, 2]
    assert story.voices == ("", "aidar", "", "")


@pytest.mark.parametrize("unusable", [
    "this is not json",
    "{}",
    "[2]",
    '{"photo": 0}',
    '{"photo": 4}',
    '{"photo": null}',
    '{"photo": "x"}',
    '{"photo": true}',
    '{"photo": 1.5}',
])
def test_an_unusable_answer_keeps_stage_2s_photograph_and_the_calls_go_on(unusable):
    client = DressClient(FIRST, votes=[unusable, *votes(1, 0, 2)])
    story = dress(client)
    assert [photo for photo, _ in story.script] == [2, 1, 0, 2]
    assert story.votes == (None, 1, 0, 2)
    assert len(client.staged("match")) == 4


def test_a_raised_match_stops_the_calls_and_stage_2_stands_from_there():
    client = DressClient(FIRST, votes=[*votes(1), RuntimeError("the server went away"),
                                       *votes(0, 0)])
    story = dress(client)
    assert [photo for photo, _ in story.script] == [1, 0, 2, 1]
    assert story.votes == (1, None, None, None)
    assert len(client.staged("match")) == 2


def test_a_film_with_no_votes_is_stage_2s_film():
    story = dress(DressClient(FIRST))
    assert story == Story(**SELECTED, votes=(None,) * 4, memory_order="broken")


@pytest.mark.parametrize("answer", ["this is not json", RuntimeError("gone")],
                         ids=["unparsable", "raised"])
def test_a_vote_replaces_the_even_fallback_and_it_stays_marked(answer):
    story = dress(DressClient(answer, votes=votes(2, 1, 0, 0)))
    assert [photo for photo, _ in story.script] == [2, 1, 0, 0]
    assert story.matched is False


def test_no_vote_leaves_the_even_fallback_marked():
    story = dress(DressClient("this is not json"))
    assert [photo for photo, _ in story.script] == [0, 0, 1, 2]
    assert story.matched is False
    assert story.memory_order == "kept"


def test_cancel_stops_the_photograph_calls_that_are_left():
    class Counting:
        calls = 0

        def text_complete(self, *_args, **_kwargs):
            self.calls += 1
            return '{"photo": 2}'

    client = Counting()
    told = Story(narration="a b c", cards=(), music_prompt="",
                 script=((0, "a"), (0, "b"), (0, "c")))

    matched = match_photographs(client, "m", told, ["one", "two"], 2, None,
                                lambda: client.calls >= 1)

    assert client.calls == 1
    assert matched.votes == (1, None, None)
    assert [index for index, _text in matched.script] == [1, 0, 0]
