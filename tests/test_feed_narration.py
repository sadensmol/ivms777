"""`feed/narration.py` — the film's passages and the user's edited draft (design §6, §11.1, §11.3)."""

import json
import re
from dataclasses import replace

import pytest

from albums.memory_store import Memory, replace_memories
from db.connection import migrate
from feed.narration import (
    MAX_SCRIPT_CHARS,
    Draft,
    NarrationRefused,
    Passage,
    discard_draft,
    draft_languages,
    draft_story,
    narration_rows,
    read_draft,
    rebase_consumed,
    remove_passage,
    rewrite_passages,
    save_passage,
    timeline_script,
)
from feed.notes import notes_for_prompt, read_notes
from feed.store import Presentation, save_presentation
from feed.story import Story
from tests.factories import add_photo

FILM_TOKEN = "f" * 32
OTHER_TOKEN = "e" * 32
TEXTS = ("A warm afternoon by the water.", "The light came off it in sheets.",
         "We stayed until the day went cold.")
FILM_SCRIPT = {
    "token": FILM_TOKEN,
    "narrator_asked": "af_heart",
    "narrator_used": "af_heart",
    "cards": [{"n": 1, "text": "The water"}],
    "score_plan": [{"from": 0, "brief": "calm"}, {"from": 2, "brief": "lift"}],
    "music_prompt": "warm acoustic guitar",
}
HEX32 = re.compile(r"[0-9a-f]{32}")


def _film_passages(texts=TEXTS) -> list[dict]:
    return [{"n": i, "text": text, "start": 4.0 * i, "end": 4.0 * i + 3.5, "photo": i,
             "photo_id": i + 1, "voice_asked": "", "voice_used": "af_heart"}
            for i, text in enumerate(texts)]


@pytest.fixture
def group_id(conn):
    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    replace_memories(conn, 1, [Memory("Beach day", "Warm afternoon.", [1, 2, 3], "sig")])
    return conn.execute("SELECT id FROM groups WHERE kind = 'memory'").fetchone()["id"]


def _save_film(conn, group_id, language="en", *, status="ready", passages=True, token=FILM_TOKEN):  # pragma: allowlist secret
    timeline: dict = {"shots": [{"photo_id": 1}]}
    if passages:
        timeline["passages"] = _film_passages()
        timeline["script"] = {**FILM_SCRIPT, "token": token}
    save_presentation(conn, Presentation(
        group_id=group_id, language=language, signature="sig", timeline=timeline,
        music_key="aa/bb/aabb.wav",
        narration_key="cc/dd/ccdd.wav" if status == "ready" else None,
        status=status, error=None if status == "ready" else "it failed",
    ))


def _post(rows) -> dict:
    return {"source": rows.source, "draft": rows.draft or "", "revision": rows.revision}


def _draft_row(conn, group_id, language="en"):
    return conn.execute("SELECT * FROM narration_drafts WHERE group_id = ? AND language = ?",
                        (group_id, language)).fetchone()


def _story(script, **over) -> Story:
    fields = {"narration": " ".join(text for _, text in script), "cards": (),
              "music_prompt": "strings", "narrator": "af_heart", "script": tuple(script)}
    fields.update(over)
    return Story(**fields)


def _draft(passages, script=None) -> Draft:
    return Draft(group_id=1, language="en", draft_token="d" * 32, source_token=FILM_TOKEN,
                 revision=3, script=script or {"narrator_asked": "am_adam", "narrator_used": "am_adam",
                                               "cards": [], "score_plan": [], "music_prompt": "piano"},
                 passages=tuple(passages))


def _passage(n, text, photo_id, photo=0, voice_asked="") -> Passage:
    return Passage(n=n, text=text, start=None, end=None, photo=photo, photo_id=photo_id,
                   voice_asked=voice_asked, voice_used=None)


def test_timeline_script_pairs_chunk_i_with_segment_i_in_telling_order():
    story = _story([(2, "A"), (0, "B"), (1, "C")], voices=("", "am_adam", ""))
    segments = [{"text": "A", "start": 0.0, "end": 2.0}, {"text": "B", "start": 2.0, "end": 4.0},
                {"text": "C", "start": 4.0, "end": 6.0}]

    passages, _ = timeline_script(story, segments, ["af_heart", "am_adam", "af_heart"], "af_heart",
                                  [10, 20, 30], token=FILM_TOKEN)  # pragma: allowlist secret

    assert [p["text"] for p in passages] == ["A", "B", "C"]
    assert [p["n"] for p in passages] == [0, 1, 2]
    assert [(p["start"], p["end"]) for p in passages] == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    assert [p["photo"] for p in passages] == [2, 0, 1]
    assert [p["photo_id"] for p in passages] == [30, 10, 20]
    assert [p["voice_asked"] for p in passages] == ["", "am_adam", ""]
    assert [p["voice_used"] for p in passages] == ["af_heart", "am_adam", "af_heart"]


def test_with_no_passage_voices_the_speaker_reads_every_passage():
    story = _story([(0, "A"), (1, "B")], voices=("am_adam", ""))
    passages, _ = timeline_script(story, [], [], "bf_emma", [10, 20], token=FILM_TOKEN)  # pragma: allowlist secret
    assert [p["voice_used"] for p in passages] == ["bf_emma", "bf_emma"]
    assert [p["voice_asked"] for p in passages] == ["am_adam", ""]


def test_segments_that_do_not_pair_leave_every_time_empty_and_keep_the_texts():
    story = _story([(0, "A"), (1, "B"), (2, "C")])
    segments = [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 4.0}]

    passages, _ = timeline_script(story, segments, [], "af_heart", [10, 20, 30], token=FILM_TOKEN)  # pragma: allowlist secret

    assert [p["text"] for p in passages] == ["A", "B", "C"]
    assert all(p["start"] is None and p["end"] is None for p in passages)


def test_the_script_names_each_card_by_its_passage_and_drops_one_past_the_last():
    story = _story([(0, "A"), (1, "B"), (2, "C")], cards=((2, "Late"), (1, "The water"), (5, "Gone")),
                   score=((0, "calm"), (2, "lift")), music_prompt="warm guitar", narrator="am_adam")

    _, script = timeline_script(story, [], [], "af_heart", [10, 20, 30], token=FILM_TOKEN)  # pragma: allowlist secret

    assert script == {
        "token": FILM_TOKEN,
        "narrator_asked": "am_adam",
        "narrator_used": "af_heart",
        "cards": [{"n": 2, "text": "Late"}, {"n": 1, "text": "The water"}],
        "score_plan": [{"from": 0, "brief": "calm"}, {"from": 2, "brief": "lift"}],
        "music_prompt": "warm guitar",
    }


def test_a_ready_film_with_passages_shows_its_own_rows(conn, group_id):
    _save_film(conn, group_id)

    rows = narration_rows(conn, group_id, "en")

    assert (rows.state, rows.source, rows.draft, rows.revision) == ("film", FILM_TOKEN, None, 0)
    assert rows.seekable and not rows.changed
    assert [row.text for row in rows.rows] == list(TEXTS)
    assert rows.as_json()["rows"][1] == {"n": 1, "text": TEXTS[1], "start": 4.0, "end": 7.5,
                                         "stamp": "0:04"}


def test_a_ready_film_without_passages_is_legacy(conn, group_id):
    _save_film(conn, group_id, passages=False)
    rows = narration_rows(conn, group_id, "en")
    assert (rows.state, rows.source, rows.draft, rows.rows) == ("legacy", None, None, ())


@pytest.mark.parametrize("status", ["failed", "running"])
def test_a_film_that_is_not_ready_shows_nothing(conn, group_id, status):
    _save_film(conn, group_id, status=status)
    rows = narration_rows(conn, group_id, "en")
    assert (rows.state, rows.source, rows.draft, rows.rows) == ("none", None, None, ())


def test_no_film_shows_nothing(conn, group_id):
    assert narration_rows(conn, group_id, "en").state == "none"


def test_a_passage_with_no_time_has_no_stamp():
    assert _passage(0, "A", 1).stamp == "–:––"


def test_the_first_change_seeds_every_passage_and_the_script(conn, group_id):
    _save_film(conn, group_id)

    rows = save_passage(conn, group_id, "en", 1, "The light was gold.",
                        **_post(narration_rows(conn, group_id, "en")))

    stored = _draft_row(conn, group_id)
    assert stored["revision"] == 1
    assert HEX32.fullmatch(stored["draft_token"])
    assert stored["source_token"] == FILM_TOKEN
    script = dict(FILM_SCRIPT, cards=[])
    del script["token"]
    assert json.loads(stored["script_json"]) == script
    expected = _film_passages()
    expected[1]["text"] = "The light was gold."
    assert json.loads(stored["passages_json"]) == expected
    assert (rows.state, rows.draft, rows.revision, rows.source) == (
        "draft", stored["draft_token"], 1, FILM_TOKEN)
    assert rows.seekable and rows.changed


def test_the_seed_keeps_the_card_of_a_row_that_did_not_change(conn, group_id):
    _save_film(conn, group_id)

    save_passage(conn, group_id, "en", 0, "A cold morning.",
                 **_post(narration_rows(conn, group_id, "en")))

    assert read_draft(conn, group_id, "en").script["cards"] == FILM_SCRIPT["cards"]


def test_a_later_change_drops_only_the_card_of_the_row_it_changed(conn, group_id):
    _save_film(conn, group_id)
    first = save_passage(conn, group_id, "en", 0, "A cold morning.",
                         **_post(narration_rows(conn, group_id, "en")))
    second = save_passage(conn, group_id, "en", 2, "We left at dusk.", **_post(first))
    assert read_draft(conn, group_id, "en").script["cards"] == FILM_SCRIPT["cards"]

    save_passage(conn, group_id, "en", 1, "The light was gold.", **_post(second))

    expected = {key: value for key, value in FILM_SCRIPT.items() if key != "token"}
    assert read_draft(conn, group_id, "en").script == {**expected, "cards": []}


def test_putting_the_old_words_back_does_not_bring_the_card_back(conn, group_id):
    _save_film(conn, group_id)
    first = save_passage(conn, group_id, "en", 1, "The light was gold.",
                         **_post(narration_rows(conn, group_id, "en")))

    save_passage(conn, group_id, "en", 1, TEXTS[1], **_post(first))

    assert read_draft(conn, group_id, "en").script["cards"] == []


def test_the_next_change_raises_the_revision_and_putting_the_text_back_is_not_changed(conn, group_id):
    _save_film(conn, group_id)
    first = save_passage(conn, group_id, "en", 1, "The light was gold.",
                         **_post(narration_rows(conn, group_id, "en")))

    back = save_passage(conn, group_id, "en", 1, TEXTS[1], **_post(first))

    assert back.revision == 2
    assert back.draft == first.draft
    assert back.state == "draft" and back.seekable and not back.changed


@pytest.mark.parametrize("wrong", [
    {"source": OTHER_TOKEN},
    {"draft": "0" * 32},
    {"revision": 2},
    {"revision": 0},
])
def test_a_save_on_rows_that_changed_is_stale_and_writes_nothing(conn, group_id, wrong):
    _save_film(conn, group_id)
    seeded = save_passage(conn, group_id, "en", 0, "Afternoon.",
                          **_post(narration_rows(conn, group_id, "en")))
    before = dict(_draft_row(conn, group_id))

    with pytest.raises(NarrationRefused) as refused:
        save_passage(conn, group_id, "en", 1, "Something else.", **{**_post(seeded), **wrong})

    assert refused.value.code == "stale"
    assert refused.value.message == ("Not saved: this narration changed in another window,"
                                     " or a new film replaced it.")
    assert dict(_draft_row(conn, group_id)) == before


def test_a_save_on_the_film_rows_with_a_wrong_source_is_stale(conn, group_id):
    _save_film(conn, group_id)
    with pytest.raises(NarrationRefused) as refused:
        save_passage(conn, group_id, "en", 0, "Afternoon.", source=OTHER_TOKEN, draft="", revision=0)
    assert refused.value.code == "stale"
    assert _draft_row(conn, group_id) is None


@pytest.mark.parametrize("status", [None, "failed"])
def test_any_save_with_no_rows_is_stale(conn, group_id, status):
    if status:
        _save_film(conn, group_id, status=status)
    with pytest.raises(NarrationRefused) as refused:
        save_passage(conn, group_id, "en", 0, "Afternoon.", source="", draft="", revision=0)
    assert refused.value.code == "stale"
    assert _draft_row(conn, group_id) is None


def test_a_new_seed_after_write_a_new_story_has_a_new_draft_token(conn, group_id):
    _save_film(conn, group_id)
    old = save_passage(conn, group_id, "en", 0, "Afternoon.",
                       **_post(narration_rows(conn, group_id, "en")))
    film = discard_draft(conn, group_id, "en", **_post(old))

    new = save_passage(conn, group_id, "en", 0, "Evening.", **_post(film))

    assert new.revision == 1 and new.draft != old.draft
    with pytest.raises(NarrationRefused) as refused:
        save_passage(conn, group_id, "en", 1, "Late.", **_post(old))
    assert refused.value.code == "stale"


def test_a_long_passage_is_stored_whole(conn, group_id):
    _save_film(conn, group_id)
    long_text = "word " * 199 + "lasts"

    rows = save_passage(conn, group_id, "en", 0, long_text,
                        **_post(narration_rows(conn, group_id, "en")))

    assert len(rows.rows[0].text) == 1000
    assert rows.rows[0].text == long_text


def test_a_script_over_the_app_limit_is_refused(conn, group_id):
    _save_film(conn, group_id)
    text = "a" * MAX_SCRIPT_CHARS
    total = MAX_SCRIPT_CHARS + len(TEXTS[1]) + len(TEXTS[2])

    with pytest.raises(NarrationRefused) as refused:
        save_passage(conn, group_id, "en", 0, text, **_post(narration_rows(conn, group_id, "en")))

    assert refused.value.code == "too_long"
    assert refused.value.message == (f"The narration is longer than this app allows: {total}"
                                     " characters, at most 10000.")
    assert _draft_row(conn, group_id) is None


def test_a_bracket_is_refused(conn, group_id):
    _save_film(conn, group_id)
    with pytest.raises(NarrationRefused) as refused:
        save_passage(conn, group_id, "en", 0, "[whisper] it was late",
                     **_post(narration_rows(conn, group_id, "en")))
    assert refused.value.code == "bracket"
    assert _draft_row(conn, group_id) is None


def test_whitespace_is_joined_to_single_spaces(conn, group_id):
    _save_film(conn, group_id)
    rows = save_passage(conn, group_id, "en", 0, "  two\n\nlines  ",
                        **_post(narration_rows(conn, group_id, "en")))
    assert rows.rows[0].text == "two lines"


def test_an_unchanged_text_writes_nothing(conn, group_id):
    _save_film(conn, group_id)
    rows = save_passage(conn, group_id, "en", 0, f" {TEXTS[0]} ",
                        **_post(narration_rows(conn, group_id, "en")))
    assert rows.state == "film"
    assert _draft_row(conn, group_id) is None


def test_an_empty_text_removes_the_passage_and_keeps_the_others_numbers(conn, group_id):
    _save_film(conn, group_id)
    rows = save_passage(conn, group_id, "en", 1, "  ", **_post(narration_rows(conn, group_id, "en")))
    assert [(row.n, row.text) for row in rows.rows] == [(0, TEXTS[0]), (2, TEXTS[2])]


def test_removing_the_last_passage_is_refused(conn, group_id):
    _save_film(conn, group_id)
    rows = remove_passage(conn, group_id, "en", 0, **_post(narration_rows(conn, group_id, "en")))
    rows = remove_passage(conn, group_id, "en", 1, **_post(rows))
    before = dict(_draft_row(conn, group_id))

    with pytest.raises(NarrationRefused) as refused:
        remove_passage(conn, group_id, "en", 2, **_post(rows))

    assert refused.value.code == "empty"
    assert dict(_draft_row(conn, group_id)) == before


def test_an_unknown_passage_is_missing(conn, group_id):
    _save_film(conn, group_id)
    with pytest.raises(NarrationRefused) as refused:
        save_passage(conn, group_id, "en", 7, "Late.", **_post(narration_rows(conn, group_id, "en")))
    assert refused.value.code == "missing"


def test_write_a_new_story_brings_the_film_rows_back(conn, group_id):
    _save_film(conn, group_id)
    seeded = save_passage(conn, group_id, "en", 0, "Afternoon.",
                          **_post(narration_rows(conn, group_id, "en")))

    rows = discard_draft(conn, group_id, "en", **_post(seeded))

    assert rows.state == "film" and [row.text for row in rows.rows] == list(TEXTS)
    assert _draft_row(conn, group_id) is None


def test_write_a_new_story_after_a_failed_build_shows_nothing(conn, group_id):
    _save_film(conn, group_id)
    save_passage(conn, group_id, "en", 0, "Afternoon.", **_post(narration_rows(conn, group_id, "en")))
    _save_film(conn, group_id, status="failed", passages=False)
    draft = narration_rows(conn, group_id, "en")
    assert draft.state == "draft" and not draft.seekable and draft.changed

    rows = discard_draft(conn, group_id, "en", **_post(draft))

    assert rows.state == "none"
    assert _draft_row(conn, group_id) is None


def test_a_stale_discard_keeps_the_draft(conn, group_id):
    _save_film(conn, group_id)
    seeded = save_passage(conn, group_id, "en", 0, "Afternoon.",
                          **_post(narration_rows(conn, group_id, "en")))
    with pytest.raises(NarrationRefused) as refused:
        discard_draft(conn, group_id, "en", **{**_post(seeded), "revision": 0})
    assert refused.value.code == "stale"
    assert _draft_row(conn, group_id) is not None


def test_languages_are_separate(conn, group_id):
    _save_film(conn, group_id, "en")
    _save_film(conn, group_id, "ru")
    ru_before = narration_rows(conn, group_id, "ru")

    save_passage(conn, group_id, "en", 0, "Afternoon.", **_post(narration_rows(conn, group_id, "en")))
    assert narration_rows(conn, group_id, "ru") == ru_before
    assert draft_languages(conn, group_id) == ["en"]

    en_before = narration_rows(conn, group_id, "en")
    save_passage(conn, group_id, "ru", 0, "День.", **_post(ru_before))
    assert narration_rows(conn, group_id, "en") == en_before
    assert draft_languages(conn, group_id) == ["en", "ru"]


def test_a_narration_row_is_never_a_note(conn, group_id):
    _save_film(conn, group_id)
    rows = save_passage(conn, group_id, "en", 0, "Afternoon light.",
                        **_post(narration_rows(conn, group_id, "en")))
    remove_passage(conn, group_id, "en", 1, **_post(rows))

    assert conn.execute("SELECT COUNT(*) c FROM memory_notes").fetchone()["c"] == 0
    prompt = notes_for_prompt(read_notes(conn, group_id))
    assert "Afternoon light." not in prompt and TEXTS[2] not in prompt


def test_the_draft_becomes_the_story():
    draft = _draft([_passage(0, "A", 20, voice_asked="am_adam"), _passage(1, "B", 10),
                    _passage(2, "C", 30)])

    story = draft_story(draft, [10, 20, 30])

    assert story.script == ((1, "A"), (0, "B"), (2, "C"))
    assert story.narration == "A B C"
    assert story.voices == ("am_adam", "", "")
    assert story.narrator == "am_adam"
    assert story.music_prompt == "piano"
    assert story.votes == ()


def test_a_photograph_gone_from_the_memory_takes_the_previous_passages():
    draft = _draft([_passage(0, "A", 99), _passage(1, "B", 20), _passage(2, "C", 98)])
    assert [photo for photo, _ in draft_story(draft, [10, 20, 30]).script] == [0, 1, 1]


def test_the_score_plan_moves_to_the_first_remaining_passage_at_or_after_it():
    script = {"narrator_asked": "", "cards": [], "music_prompt": "piano",
              "score_plan": [{"from": 0, "brief": "calm"}, {"from": 2, "brief": "lift"},
                             {"from": 4, "brief": "end"}]}
    draft = _draft([_passage(0, "A", 10), _passage(1, "B", 20), _passage(3, "D", 30)], script)

    assert draft_story(draft, [10, 20, 30]).score == ((0, "calm"), (2, "lift"))


def test_a_card_stays_on_its_passage_and_a_removed_passages_card_is_dropped():
    script = {"narrator_asked": "", "music_prompt": "piano", "score_plan": [],
              "cards": [{"n": 1, "text": "Gone"}, {"n": 3, "text": "Kept"}]}
    draft = _draft([_passage(0, "A", 10), _passage(2, "C", 20), _passage(3, "D", 30)], script)

    assert draft_story(draft, [10, 20, 30]).cards == ((2, "Kept"),)


def test_memory_order_is_checked_on_the_drafts_photographs():
    out_of_order = _draft([_passage(1, "B", 20), _passage(2, "C", 10)])
    in_order = _draft([_passage(0, "A", 10), _passage(1, "B", 20), _passage(2, "C", 10)])

    broken = draft_story(out_of_order, [10, 20])

    assert broken.memory_order == "broken"
    assert broken.script == ((1, "B"), (0, "C"))
    assert draft_story(in_order, [10, 20]).memory_order == "kept"


def _new_film(texts) -> tuple[list[dict], dict]:
    passages = [{"n": i, "text": text, "start": 5.0 * i, "end": 5.0 * i + 4.0, "photo": i,
                 "photo_id": i + 1, "voice_asked": "", "voice_used": "am_adam"}
                for i, text in enumerate(texts)]
    return passages, {**FILM_SCRIPT, "token": OTHER_TOKEN, "narrator_used": "am_adam"}


def test_the_ready_save_rebases_the_draft_it_consumed(conn, group_id):
    _save_film(conn, group_id)
    rows = save_passage(conn, group_id, "en", 1, "", **_post(narration_rows(conn, group_id, "en")))
    draft = read_draft(conn, group_id, "en")
    passages, script = _new_film([row.text for row in rows.rows])

    assert rebase_consumed(conn, draft, passages, script) is True

    stored = _draft_row(conn, group_id)
    assert stored["draft_token"] == draft.draft_token
    assert stored["revision"] == draft.revision + 1
    assert stored["source_token"] == OTHER_TOKEN
    assert json.loads(stored["passages_json"]) == passages
    assert json.loads(stored["script_json"]) == {k: v for k, v in script.items() if k != "token"}
    assert [p["text"] for p in json.loads(stored["passages_json"])] == [TEXTS[0], TEXTS[2]]


def test_a_rebase_never_touches_a_newer_revision_or_another_draft(conn, group_id):
    _save_film(conn, group_id)
    first = save_passage(conn, group_id, "en", 0, "Afternoon.",
                         **_post(narration_rows(conn, group_id, "en")))
    consumed = read_draft(conn, group_id, "en")
    save_passage(conn, group_id, "en", 1, "Gold.", **_post(first))
    newer = dict(_draft_row(conn, group_id))
    passages, script = _new_film(["Afternoon.", TEXTS[1], TEXTS[2]])

    assert rebase_consumed(conn, consumed, passages, script) is False
    other = replace(consumed, draft_token="0" * 32, revision=newer["revision"])
    assert rebase_consumed(conn, other, passages, script) is False
    assert dict(_draft_row(conn, group_id)) == newer


def test_a_rewrite_changes_only_the_words_of_the_named_passages(conn, group_id):
    _save_film(conn, group_id)
    before = narration_rows(conn, group_id, "en")

    rows = rewrite_passages(conn, group_id, "en", {0: "New first.", 2: "New  third."},
                            **_post(before))

    assert [row.text for row in rows.rows] == ["New first.", TEXTS[1], "New third."]
    for was, now in zip(before.rows, rows.rows, strict=True):
        assert replace(now, text=was.text) == was
    draft = read_draft(conn, group_id, "en")
    assert draft.script == {key: value for key, value in FILM_SCRIPT.items() if key != "token"}


def test_a_rewrite_drops_the_card_of_a_row_it_changed(conn, group_id):
    _save_film(conn, group_id)

    rewrite_passages(conn, group_id, "en", {1: "New second."},
                     **_post(narration_rows(conn, group_id, "en")))

    assert read_draft(conn, group_id, "en").script["cards"] == []


def test_a_rewrite_keeps_the_card_of_a_named_row_with_the_same_words(conn, group_id):
    _save_film(conn, group_id)

    rewrite_passages(conn, group_id, "en", {0: "New first.", 1: TEXTS[1]},
                     **_post(narration_rows(conn, group_id, "en")))

    assert read_draft(conn, group_id, "en").script["cards"] == FILM_SCRIPT["cards"]


def test_a_rewrite_of_a_draft_drops_the_card_of_a_row_it_changed(conn, group_id):
    _save_film(conn, group_id)
    seeded = save_passage(conn, group_id, "en", 0, "A cold morning.",
                          **_post(narration_rows(conn, group_id, "en")))

    rewrite_passages(conn, group_id, "en", {1: "New second."}, **_post(seeded))

    assert read_draft(conn, group_id, "en").script["cards"] == []


def test_a_rewrite_is_one_write(conn, group_id):
    _save_film(conn, group_id)
    first = save_passage(conn, group_id, "en", 0, "Afternoon.",
                         **_post(narration_rows(conn, group_id, "en")))

    rows = rewrite_passages(conn, group_id, "en", {1: "B.", 2: "C."}, **_post(first))

    assert rows.revision == first.revision + 1
    assert rows.draft == first.draft
    assert [row.text for row in rows.rows] == ["Afternoon.", "B.", "C."]


@pytest.mark.parametrize("wrong", [{"source": OTHER_TOKEN}, {"draft": "d" * 32}, {"revision": 4}])
def test_a_rewrite_on_rows_that_changed_is_stale_and_writes_nothing(conn, group_id, wrong):
    _save_film(conn, group_id)
    posted = _post(narration_rows(conn, group_id, "en")) | wrong

    with pytest.raises(NarrationRefused) as refused:
        rewrite_passages(conn, group_id, "en", {0: "New.", 1: "Newer."}, **posted)

    assert refused.value.code == "stale"
    assert _draft_row(conn, group_id) is None


@pytest.mark.parametrize(("bad", "code"), [
    ({1: ""}, "empty_text"), ({1: "   "}, "empty_text"), ({1: "With [a] bracket."}, "bracket"),
    ({9: "Gone."}, "missing"), ({1: "x" * (MAX_SCRIPT_CHARS + 1)}, "too_long"),
])
def test_one_bad_item_refuses_the_whole_rewrite(conn, group_id, bad, code):
    _save_film(conn, group_id)
    rows = narration_rows(conn, group_id, "en")

    with pytest.raises(NarrationRefused) as refused:
        rewrite_passages(conn, group_id, "en", {0: "A good change."} | bad, **_post(rows))

    assert refused.value.code == code
    assert _draft_row(conn, group_id) is None
    assert narration_rows(conn, group_id, "en") == rows


def test_the_length_limit_counts_the_complete_narration(conn, group_id):
    _save_film(conn, group_id)
    fits = MAX_SCRIPT_CHARS - len(TEXTS[1]) - len(TEXTS[2])
    rows = narration_rows(conn, group_id, "en")

    with pytest.raises(NarrationRefused) as refused:
        rewrite_passages(conn, group_id, "en", {0: "x" * (fits + 1)}, **_post(rows))
    assert refused.value.code == "too_long"

    saved = rewrite_passages(conn, group_id, "en", {0: "x" * fits}, **_post(rows))
    assert sum(len(row.text) for row in saved.rows) == MAX_SCRIPT_CHARS


@pytest.mark.parametrize("patch", [{}, {0: TEXTS[0]}, {0: f"  {TEXTS[0]} "}])
def test_a_rewrite_with_no_difference_makes_no_draft(conn, group_id, patch):
    _save_film(conn, group_id)
    rows = narration_rows(conn, group_id, "en")

    assert rewrite_passages(conn, group_id, "en", patch, **_post(rows)) == rows
    assert _draft_row(conn, group_id) is None


def test_deleting_the_memory_deletes_its_drafts(conn, group_id):
    _save_film(conn, group_id)
    save_passage(conn, group_id, "en", 0, "Afternoon.", **_post(narration_rows(conn, group_id, "en")))

    conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))

    assert conn.execute("SELECT COUNT(*) c FROM narration_drafts").fetchone()["c"] == 0


def test_migrate_creates_the_table_when_it_is_missing(conn):
    conn.execute("DROP TABLE narration_drafts")
    migrate(conn)
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'narration_drafts'").fetchone()
