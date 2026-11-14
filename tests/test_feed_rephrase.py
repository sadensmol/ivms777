"""`feed/rephrase.py` — the film's context, the prompt and the checked proposal (design §11.3)."""

import json

import pytest

from albums.memory_store import Memory, replace_memories
from feed.build import Cancelled
from feed.narration import (
    MAX_SCRIPT_CHARS,
    NarrationRefused,
    narration_rows,
    save_passage,
)
from feed.notes import add_note
from feed.rephrase import (
    MAX_HISTORY_MESSAGES,
    MAX_REQUEST_CHARS,
    SCHEMA,
    TIMEOUT_SECONDS,
    Ask,
    Message,
    RephraseRefused,
    build_prompt,
    check_fit,
    clean_request,
    output_tokens,
    propose,
    read_changes,
    read_film,
    read_history,
)
from feed.store import Presentation, save_presentation
from tests.factories import add_photo

FILM_TOKEN = "f" * 32
MODEL = "a-text-model"
TEXTS = ("A warm afternoon by the water.", "The light came off it in sheets.",
         "We stayed until the day went cold.")
RUSSIAN = ("Тёплый день у воды.", "Свет лежал на ней полосами.", "Мы остались до холода.")
SHOT = {"photo_id": 2, "at": 4.0, "seconds": 4.0, "effect": "ember", "params": [0.25, 0.75],
        "beats": [{"effect": "drift-beat", "at": 4.5}], "transition": "dissolve",
        "transition_seconds": 0.6, "grade": "amber", "line": "", "hero": True,
        "move": {"scale_from": 1.0, "scale_to": 1.2, "ease": "out-sine"}}
SCRIPT = {"token": FILM_TOKEN, "narrator_asked": "af_bella", "narrator_used": "af_heart",
          "cards": [{"n": 1, "text": "The water card"}],
          "score_plan": [{"from": 2, "brief": "lift brief"}], "music_prompt": "warm guitar"}
DECISIONS = {
    "story": {"memory_order": "kept", "music_prompt": "warm guitar"},
    "cast": {"narrator_used": "af_heart", "narrator_chosen_by": "mood"},
    "voice": {"model": "kokoro-voice", "narration_seconds": 11.5},
    "score": {"seed": 4242, "energy": "calm", "model": "musicgen-score",
              "sections": [{"role": "open", "at": 0.0, "seconds": 12.0, "brief": "opening strings"}]},
    "film": {"runtime": 12.5, "shots": 3},
    "secret": {"never": "shown-in-the-prompt"},
}


class Models:
    """`text_complete` and `text_context`, as the models service answers them."""

    def __init__(self, *, reply="", context=None):
        self.reply = reply
        self.context = context
        self.calls: list[tuple] = []

    def text_context(self):
        if isinstance(self.context, Exception):
            raise self.context
        return self.context

    def text_complete(self, model, messages, **options):
        self.calls.append((model, messages, options))
        return self.reply


@pytest.fixture
def group_id(conn):
    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  ai_description=f"description of photo {pid}", width=4000, height=3000)
    replace_memories(conn, 1, [Memory("Beach day", "Warm afternoon.", [1, 2, 3], "sig")])
    return conn.execute("SELECT id FROM groups WHERE kind = 'memory'").fetchone()["id"]


def _film(conn, group_id, language="en", *, texts=TEXTS, status="ready", token=FILM_TOKEN):  # pragma: allowlist secret
    passages = [{"n": i, "text": text, "start": 4.0 * i, "end": 4.0 * i + 3.5, "photo": i,
                 "photo_id": i + 1, "voice_asked": "am_adam" if i == 1 else "",
                 "voice_used": "af_heart"} for i, text in enumerate(texts)]
    save_presentation(conn, Presentation(
        group_id=group_id, language=language, signature="sig",
        timeline={"seconds": 12.5, "grade": "amber", "shots": [SHOT],
                  "effects": {"ember": 7}, "grades": {"amber": [1, 2, 3]},
                  "passages": passages, "script": {**SCRIPT, "token": token}},
        music_key="m.wav", narration_key="n.wav" if status == "ready" else None,
        status=status, error=None, decisions=DECISIONS))


def _posted(conn, group_id, language="en") -> tuple[str, str, int]:
    rows = narration_rows(conn, group_id, language)
    return rows.source, rows.draft or "", rows.revision


def _ask(conn, group_id, *, language="en", request="Make it warmer.", history=(), proposal=None):
    return Ask(group_id, language, _posted(conn, group_id, language), request, history,
               proposal or {}, MODEL)


def _answer(*changes, reply="I changed it.") -> str:
    return json.dumps({"reply": reply, "changes": [{"n": n, "text": text} for n, text in changes]})


def _propose(conn, ask, models):
    return propose(conn, models, ask, progress=lambda _stage, _fraction: None,
                   cancelled=lambda: False)


def _prompt(conn, group_id, **options) -> str:
    ask = _ask(conn, group_id, **options)
    return build_prompt(read_film(conn, group_id, ask.language, ask.posted), ask)


def test_the_prompt_carries_the_whole_record_of_the_film(conn, group_id):
    _film(conn, group_id)
    prompt = _prompt(conn, group_id)
    for recorded in ("ember", "drift-beat", "0.25", "dissolve", "out-sine", "amber", "hero",
                     "The water card", "lift brief", "warm guitar", "opening strings", "4242",
                     "kokoro-voice", "musicgen-score", "narrator_chosen_by", "12.5"):
        assert recorded in prompt


def test_the_prompt_keeps_the_voices_asked_apart_from_the_voices_used(conn, group_id):
    _film(conn, group_id)
    prompt = _prompt(conn, group_id)
    assert f"voice used af_heart | {TEXTS[1]}" in prompt
    assert "am_adam" not in prompt  # a raw request is never shown as a voice
    assert "raw requests, not validated" in prompt
    assert '"narrator_asked":"af_bella"' in prompt and '"narrator_used":"af_heart"' in prompt


def test_todays_details_are_labelled_and_tags_are_guesses(conn, group_id):
    _film(conn, group_id)
    add_note(conn, group_id, 4.5, "This is Julia")
    prompt = _prompt(conn, group_id)
    memory = prompt.index("THE MEMORY, AS IT IS TODAY")
    film = prompt.index("THE FILM, THE EXACT RECORD")
    assert memory < prompt.index("Beach day") < film
    assert memory < prompt.index("This is Julia") < film
    assert memory < prompt.index("description of photo 2") < film
    assert "automatic guesses, not facts" in prompt
    # 4.5 s stands on the shot of photo id 2, the second photograph of the listing.
    assert "looking at photograph 2] This is Julia" in prompt


def test_the_prompt_holds_no_table_vector_or_other_decision(conn, group_id):
    _film(conn, group_id)
    prompt = _prompt(conn, group_id)
    assert '"ember":7' not in prompt and "[1, 2, 3]" not in prompt and "[1,2,3]" not in prompt
    assert "shown-in-the-prompt" not in prompt
    assert "vector" not in prompt.lower()


def test_a_draft_with_no_ready_film_has_only_its_script_and_says_so(conn, group_id):
    _film(conn, group_id)
    save_passage(conn, group_id, "en", 0, "Afternoon.", source=FILM_TOKEN, draft="", revision=0)
    conn.execute("UPDATE memory_presentations SET status = 'failed', narration_key = NULL"
                 " WHERE group_id = ?", (group_id,))
    prompt = _prompt(conn, group_id)
    assert "No ready film of this language exists now" in prompt
    assert "The water card" in prompt
    for film_fact in ("ember", "dissolve", "opening strings", "kokoro-voice"):
        assert film_fact not in prompt


def test_a_draft_of_an_earlier_film_says_its_times_are_not_this_timeline(conn, group_id):
    _film(conn, group_id, token="e" * 32)
    save_passage(conn, group_id, "en", 0, "Afternoon.", source="e" * 32, draft="", revision=0)
    _film(conn, group_id)
    assert "came from an earlier film" in _prompt(conn, group_id)


def test_the_proposal_on_screen_is_kept_apart_from_the_saved_rows(conn, group_id):
    _film(conn, group_id)
    prompt = _prompt(conn, group_id, proposal={1: "Unsaved words."},
                     history=(Message("user", "First ask."), Message("ai", "First reply.")))
    assert prompt.index("n=1 | at 0:04") < prompt.index("THE PROPOSAL ON SCREEN")
    assert prompt.index("THE PROPOSAL ON SCREEN") < prompt.index("n=1: Unsaved words.")
    assert "USER: First ask.\nYOU: First reply." in prompt
    assert prompt.index("First reply.") < prompt.index("THE USER ASKS NOW\nMake it warmer.")


def test_reading_the_film_refuses_rows_that_changed_and_keeps_no_transaction(conn, group_id):
    _film(conn, group_id)
    old = _posted(conn, group_id)
    save_passage(conn, group_id, "en", 0, "Afternoon.", source=FILM_TOKEN, draft="", revision=0)
    with pytest.raises(NarrationRefused) as refused:
        read_film(conn, group_id, "en", old)
    assert refused.value.code == "stale"
    assert not conn.in_transaction
    read_film(conn, group_id, "en", _posted(conn, group_id))
    assert not conn.in_transaction


def test_a_legacy_film_and_no_film_cannot_be_read(conn, group_id):
    with pytest.raises(NarrationRefused):
        read_film(conn, group_id, "en", ("", "", 0))
    save_presentation(conn, Presentation(
        group_id=group_id, language="en", signature="sig", timeline={"shots": []},
        music_key=None, narration_key="n.wav", status="ready"))
    with pytest.raises(NarrationRefused):
        read_film(conn, group_id, "en", ("", "", 0))


def test_a_prompt_over_the_context_is_refused_and_no_call_is_made(conn, group_id):
    _film(conn, group_id)
    models = Models(reply=_answer(), context=500)
    with pytest.raises(RephraseRefused, match="Nothing was cut"):
        _propose(conn, _ask(conn, group_id), models)
    assert models.calls == []


@pytest.mark.parametrize("context", [None, RuntimeError("no service")])
def test_a_server_that_does_not_say_its_context_has_no_budget(conn, group_id, context):
    _film(conn, group_id)
    models = Models(reply=_answer(reply="Fine."), context=context)
    assert _propose(conn, _ask(conn, group_id), models).reply == "Fine."
    assert len(models.calls) == 1


def test_the_fit_counts_the_prompt_the_answer_and_the_template():
    prompt = "x" * 300                  # 100 estimated tokens at 3 chars a token
    check_fit(prompt, 50, 100 + 50 + 256)
    with pytest.raises(RephraseRefused):
        check_fit(prompt, 50, 100 + 50 + 256 - 1)


def test_the_room_for_the_answer_grows_with_the_rows_up_to_the_app_limit(conn, group_id):
    _film(conn, group_id, texts=("a" * 300,))
    small = output_tokens(narration_rows(conn, group_id, "en").rows)
    _film(conn, group_id, texts=("a" * 3000,))
    large = output_tokens(narration_rows(conn, group_id, "en").rows)
    _film(conn, group_id, texts=("a" * MAX_SCRIPT_CHARS,))
    limit = output_tokens(narration_rows(conn, group_id, "en").rows)
    assert small < large < limit
    assert limit == output_tokens(narration_rows(conn, group_id, "en").rows * 2)


def test_one_turn_calls_the_model_once_with_the_schema_and_writes_nothing(conn, group_id):
    _film(conn, group_id)
    models = Models(reply=_answer((1, "Bright sheets of light.")))
    before = conn.total_changes
    _propose(conn, _ask(conn, group_id), models)
    (model, messages, options), = models.calls
    assert model == MODEL
    assert messages[0]["role"] == "user" and "Make it warmer." in messages[0]["content"]
    assert options["json_schema"] == SCHEMA and options["timeout"] == TIMEOUT_SECONDS
    assert options["max_tokens"] > 0
    assert conn.total_changes == before
    assert conn.execute("SELECT COUNT(*) c FROM narration_drafts").fetchone()["c"] == 0


def test_only_the_named_passages_change_and_the_count_says_so(conn, group_id):
    _film(conn, group_id)
    proposal = _propose(conn, _ask(conn, group_id), Models(reply=_answer((1, "New words."))))
    payload = proposal.as_json()
    assert payload["summary"] == "Changed 1 of 3 passages."
    assert payload["changes"] == [{"n": 1, "stamp": "0:04", "old": TEXTS[1], "new": "New words."}]
    assert payload["base"] == {"source": FILM_TOKEN, "draft": "", "revision": 0}


def test_a_reply_with_no_change_is_a_valid_turn(conn, group_id):
    _film(conn, group_id)
    proposal = _propose(conn, _ask(conn, group_id), Models(reply=_answer(reply="It is fine.")))
    assert proposal.reply == "It is fine."
    assert proposal.changes == ()
    assert proposal.as_json()["summary"] == "No passage changed."


def test_a_fenced_answer_is_read_and_whitespace_joins(conn, group_id):
    _film(conn, group_id)
    raw = f"```json\n{_answer((0, '  Warm   water\nand  sun. '))}\n```"
    proposal = _propose(conn, _ask(conn, group_id), Models(reply=raw))
    assert proposal.changes[0].new == "Warm water and sun."


@pytest.mark.parametrize(("raw", "reason"), [
    (_answer((7, "New words.")), "does not exist"),
    (_answer((1, "New words."), (1, "Other words.")), "twice"),
    (_answer((1, "")), "empty"),
    (_answer((1, "   ")), "empty"),
    (_answer((1, "New [soft] words.")), "bracket"),
    (_answer((1, "New words]")), "bracket"),
    (_answer((1, "x" * (MAX_SCRIPT_CHARS + 1))), "at most"),
    (_answer(("1", "New words.")), "does not exist"),
    (_answer((True, "New words.")), "does not exist"),
    (json.dumps({"reply": "x", "changes": [{"n": 1, "text": 5}]}), "empty"),
    (json.dumps({"reply": "x", "changes": "all of them"}), "could not be read"),
    (json.dumps({"reply": "x", "changes": [3]}), "could not be read"),
    (json.dumps({"reply": 4, "changes": []}), "could not be read"),
    ("no json here", "could not be read"),
    ("{}", "no answer"),
    (json.dumps({"reply": "", "changes": []}), "no answer"),
])
def test_any_bad_item_rejects_the_whole_proposal_with_the_reason(conn, group_id, raw, reason):
    _film(conn, group_id)
    good = json.loads(raw) if raw.startswith("{") else None
    if good and isinstance(good.get("changes"), list) and good["changes"]:
        good["changes"].insert(0, {"n": 0, "text": "A good change."})
        raw = json.dumps(good)
    with pytest.raises(RephraseRefused, match=reason):
        _propose(conn, _ask(conn, group_id), Models(reply=raw))


def test_a_text_not_in_the_films_language_rejects_the_whole_proposal(conn, group_id):
    _film(conn, group_id, "ru", texts=RUSSIAN)
    raw = _answer((0, "Тёплая вода."), (1, "English words."))
    with pytest.raises(RephraseRefused, match="passage 1 is not in Russian"):
        _propose(conn, _ask(conn, group_id, language="ru"), Models(reply=raw))


def test_words_in_the_films_language_pass(conn, group_id):
    _film(conn, group_id, "ru", texts=RUSSIAN)
    proposal = _propose(conn, _ask(conn, group_id, language="ru"),
                        Models(reply=_answer((0, "Тёплая вода."))))
    assert [change.new for change in proposal.changes] == ["Тёплая вода."]


def test_cancel_before_the_model_call_makes_no_call(conn, group_id):
    _film(conn, group_id)
    models = Models(reply=_answer())
    with pytest.raises(Cancelled):
        propose(conn, models, _ask(conn, group_id), progress=lambda _stage, _fraction: None,
                cancelled=lambda: True)
    assert models.calls == []


def test_the_stages_are_named_and_stay_below_the_end(conn, group_id):
    _film(conn, group_id)
    seen: list[tuple[str, float]] = []
    propose(conn, Models(reply=_answer(reply="Fine.")), _ask(conn, group_id),
            progress=lambda stage, fraction: seen.append((stage, fraction)),
            cancelled=lambda: False)
    assert [stage for stage, _ in seen] == ["reading the film", "checking the words"]
    assert all(fraction < 1.0 for _, fraction in seen)


def test_a_follow_up_keeps_the_earlier_unsaved_change(conn, group_id):
    _film(conn, group_id)
    first = _propose(conn, _ask(conn, group_id), Models(reply=_answer((0, "First new."))))
    on_screen = {change.n: change.new for change in first.changes}

    second = _propose(conn, _ask(conn, group_id, proposal=on_screen),
                      Models(reply=_answer((2, "Third new."))))

    assert [(change.n, change.old, change.new) for change in second.changes] == [
        (0, TEXTS[0], "First new."), (2, TEXTS[2], "Third new.")]
    assert second.as_json()["summary"] == "Changed 2 of 3 passages."


def test_a_reply_only_follow_up_keeps_the_proposal(conn, group_id):
    _film(conn, group_id)
    proposal = _propose(conn, _ask(conn, group_id, proposal={0: "First new."}),
                        Models(reply=_answer(reply="That is all.")))
    assert [(change.n, change.new) for change in proposal.changes] == [(0, "First new.")]


def test_a_follow_up_on_the_same_passage_replaces_the_new_words_only(conn, group_id):
    _film(conn, group_id)
    proposal = _propose(conn, _ask(conn, group_id, proposal={0: "First new."}),
                        Models(reply=_answer((0, "Second new."))))
    assert [(change.old, change.new) for change in proposal.changes] == [(TEXTS[0], "Second new.")]


def test_a_passage_back_at_its_saved_words_leaves_the_patch(conn, group_id):
    _film(conn, group_id)
    proposal = _propose(conn, _ask(conn, group_id, proposal={0: "First new."}),
                        Models(reply=_answer((0, TEXTS[0]))))
    assert proposal.changes == ()


def test_the_limit_counts_the_complete_narration_with_the_proposal(conn, group_id):
    _film(conn, group_id)
    room = MAX_SCRIPT_CHARS - len(TEXTS[2])
    first_half = "x" * (room // 2)
    second_half = "y" * (room - len(first_half))
    fits = _ask(conn, group_id, proposal={0: first_half})
    assert len(_propose(conn, fits, Models(reply=_answer((1, second_half)))).changes) == 2
    over = _ask(conn, group_id, proposal={0: f"{first_half}x"})
    with pytest.raises(RephraseRefused, match="at most"):
        _propose(conn, over, Models(reply=_answer((1, second_half))))


def test_a_proposal_for_a_passage_that_is_gone_is_refused(conn, group_id):
    _film(conn, group_id)
    with pytest.raises(RephraseRefused, match="Ask again"):
        _propose(conn, _ask(conn, group_id, proposal={9: "Gone."}), Models(reply=_answer()))


def test_a_request_has_a_length_and_cannot_be_blank():
    assert clean_request("  Warmer.  ") == "Warmer."
    assert len(clean_request("x" * MAX_REQUEST_CHARS)) == MAX_REQUEST_CHARS
    for bad in ("   ", "x" * (MAX_REQUEST_CHARS + 1)):
        with pytest.raises(RephraseRefused):
            clean_request(bad)


def test_the_history_is_the_last_messages_only():
    messages = [{"role": "user" if i % 2 == 0 else "ai", "text": str(i)} for i in range(9)]
    kept = read_history(json.dumps(messages))
    assert [message.text for message in kept] == [str(i) for i in range(3, 9)]
    assert len(kept) == MAX_HISTORY_MESSAGES


@pytest.mark.parametrize("raw", ["not json", '{"role": "user"}', '[{"role": "system", "text": "x"}]',
                                 '[{"role": "user"}]', "[3]"])
def test_an_unreadable_history_is_refused(raw):
    with pytest.raises(RephraseRefused):
        read_history(raw)


def test_a_posted_patch_names_each_passage_once():
    assert read_changes('[{"n": 2, "text": "Words."}]') == {2: "Words."}
    assert read_changes("") == {}
    for raw in ('[{"n": 2, "text": "a"}, {"n": 2, "text": "b"}]', '[{"n": "2", "text": "a"}]',
                '[{"n": true, "text": "a"}]', '[{"n": 2}]', "[3]", "{}", "nope"):
        with pytest.raises(RephraseRefused):
            read_changes(raw)
