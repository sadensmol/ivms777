"""Viewer notes on a memory (design §11.3) — plain SQL, no model."""

from albums.memory_store import Memory, replace_memories
from feed.notes import add_note, delete_note, notes_for_prompt, read_notes
from tests.factories import add_photo


def _memory(conn):
    for pid in (1, 2):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    replace_memories(conn, 1, [Memory("Beach day", "Warm.", [1, 2], "sig")])
    return conn.execute("SELECT id FROM groups WHERE kind = 'memory'").fetchone()["id"]


def test_a_note_records_what_was_said_and_when(conn):
    gid = _memory(conn)
    note = add_note(conn, gid, 12.5, "  это мой брат Костя  ")
    assert note.text == "это мой брат Костя"          # trimmed, any language
    assert note.at_seconds == 12.5
    assert note.stamp == "0:12"


def test_notes_come_back_in_film_order(conn):
    gid = _memory(conn)
    add_note(conn, gid, 30.0, "third")
    add_note(conn, gid, 5.0, "first")
    add_note(conn, gid, 12.0, "second")
    assert [n.text for n in read_notes(conn, gid)] == ["first", "second", "third"]


def test_an_empty_note_is_not_stored(conn):
    gid = _memory(conn)
    assert add_note(conn, gid, 1.0, "   \n  ") is None
    assert read_notes(conn, gid) == []


def test_a_note_is_capped_rather_than_rejected(conn):
    gid = _memory(conn)
    note = add_note(conn, gid, 1.0, "x" * 5000)
    assert 0 < len(note.text) <= 600


def test_deleting_is_scoped_to_the_memory(conn):
    gid = _memory(conn)
    note = add_note(conn, gid, 1.0, "mine")
    assert delete_note(conn, gid + 999, note.id) is False
    assert delete_note(conn, gid, note.id) is True
    assert read_notes(conn, gid) == []


def test_notes_die_with_their_memory(conn):
    gid = _memory(conn)
    add_note(conn, gid, 1.0, "about a memory that is about to go")
    # Rebuilding memories replaces them entirely (§11); a note about one that no
    # longer exists describes nothing.
    replace_memories(conn, 1, [])
    assert conn.execute("SELECT COUNT(*) c FROM memory_notes").fetchone()["c"] == 0


def test_the_prompt_form_carries_the_timestamp(conn):
    gid = _memory(conn)
    add_note(conn, gid, 65.0, "this is where we got lost")
    text = notes_for_prompt(read_notes(conn, gid))
    assert "1:05" in text and "this is where we got lost" in text


def test_a_notes_position_is_given_as_a_share_of_the_film(conn):
    # A model writing a list of passages has no idea how long 0:12 is, but it knows
    # exactly what "halfway" means for a list — and a note answered in the wrong place
    # is a note ignored (§11.3).
    gid = _memory(conn)
    add_note(conn, gid, 12.0, "we got completely lost")
    text = notes_for_prompt(read_notes(conn, gid), film_seconds=24.0)
    assert "0:12" in text and "50%" in text


def test_no_share_is_quoted_without_a_cut_to_measure_against(conn):
    # Falling back to the notes' own span made the LAST note "100% of the way through"
    # every time — it is the one that defines the span, so that is an artefact, not a
    # position. Without a film length, the timestamp says all that is known (§11.3).
    gid = _memory(conn)
    add_note(conn, gid, 5.0, "early")
    add_note(conn, gid, 20.0, "late")
    text = notes_for_prompt(read_notes(conn, gid))
    assert "%" not in text
    assert "0:05" in text and "0:20" in text


def test_a_note_reaches_the_models_verbatim(conn):
    # Nothing is parsed out of a note. Every pass is shown it whole and reads any
    # instruction itself — a matcher only recognised phrasings someone thought to add
    # to it, and `[страшных хриплый мужкской голос]` was not one of them (§11.3).
    gid = _memory(conn)
    add_note(conn, gid, 4.0, "[страшных хриплый мужкской голос]и кажется кто-то будет съеден")
    text = notes_for_prompt(read_notes(conn, gid))
    assert "[страшных хриплый мужкской голос]" in text
    assert "кажется кто-то будет съеден" in text
