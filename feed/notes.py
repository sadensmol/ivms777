"""What the viewer said about their own memory (design §11.3).

A film is generated from what the machine could work out: labels, timestamps, a vision
model's sentences. The one thing it cannot know is what the day MEANT to the person who
lived it — who that is in the photograph, why that evening mattered, what the model got
wrong. A note is that, typed at the exact second of the film where it belongs.

Plain SQL, no model, no HTTP. Notes are INPUT to the next generation, never output: the
story pass is shown them with their timestamps and asked to work them into the
narration, translated into whatever language the film is in.

They cascade with the memory, like a presentation — a note about a memory that no
longer exists describes nothing.
"""

import sqlite3
from dataclasses import dataclass

NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%SZ','now')"
# Long enough for a real thought, short enough that it stays a note on a film rather
# than an essay the narration cannot absorb.
MAX_LENGTH = 600


@dataclass(frozen=True)
class Note:
    id: int
    group_id: int
    at_seconds: float
    text: str
    created_at: str

    @property
    def stamp(self) -> str:
        """`m:ss`, for showing next to the note."""
        total = max(int(self.at_seconds), 0)
        return f"{total // 60}:{total % 60:02d}"


def add_note(
    conn: sqlite3.Connection, group_id: int, at_seconds: float, text: str
) -> Note | None:
    """Record one note. None when there is nothing to record.

    Whitespace-only text is dropped rather than stored: an empty note would take up a
    line in the memory view and a line in the next story prompt, and say nothing in
    either.
    """
    cleaned = " ".join(text.split())[:MAX_LENGTH]
    if not cleaned:
        return None
    at = max(float(at_seconds or 0.0), 0.0)
    cursor = conn.execute(
        "INSERT INTO memory_notes(group_id, at_seconds, text, created_at)"
        f" VALUES (?, ?, ?, {NOW_SQL})",
        (group_id, at, cleaned),
    )
    return read_note(conn, int(cursor.lastrowid))


def read_note(conn: sqlite3.Connection, note_id: int) -> Note | None:
    row = conn.execute(
        "SELECT id, group_id, at_seconds, text, created_at FROM memory_notes"
        " WHERE id = ?",
        (note_id,),
    ).fetchone()
    return _note(row) if row is not None else None


def read_notes(conn: sqlite3.Connection, group_id: int) -> list[Note]:
    """Every note on a memory, in FILM order — which is the order the story pass needs
    them in, and the order they make sense to read in."""
    return [
        _note(row)
        for row in conn.execute(
            "SELECT id, group_id, at_seconds, text, created_at FROM memory_notes"
            " WHERE group_id = ? ORDER BY at_seconds, id",
            (group_id,),
        )
    ]


def delete_note(conn: sqlite3.Connection, group_id: int, note_id: int) -> bool:
    """Remove one note. Scoped by memory so a stale id from another page cannot
    delete someone else's line."""
    cursor = conn.execute(
        "DELETE FROM memory_notes WHERE id = ? AND group_id = ?", (note_id, group_id)
    )
    return cursor.rowcount > 0


def notes_for_prompt(
    notes: list[Note], film_seconds: float = 0.0,
    photo_at=None,
) -> str:
    """Every note, VERBATIM, with where it belongs — for a model to interpret.

    Nothing is parsed out of them here. A note may be plain prose ("this is Julia"), an
    instruction in square brackets ("[страшным хриплым голосом]"), or both in one line,
    and deciding which is which — and what a hoarse frightening voice means for a
    speaker, an effect or a score — is exactly the judgement a model makes better than
    a table of keywords. A matcher needed the phrase written its way; people write
    theirs.

    So the text goes through untouched and every prompt that receives it is told that
    brackets are instructions to obey rather than sentences to narrate.

    Position is given three ways, because each is useful to a different reader: the
    timestamp, the share of the way through (a model writing a list of passages knows
    what "40% in" means and has no idea how long 0:12 is), and which PHOTOGRAPH was on
    screen — which is what makes a note a fact about a picture rather than a moment.
    """
    # The share is quoted ONLY when a previous cut says how long the film was. Falling
    # back to the notes' own span made the last note "100% of the way through" every
    # time — it is the one that defines the span — which is not a position, it is an
    # artefact. Without a cut to measure against, the timestamp and the photograph say
    # everything that is actually known.
    span = float(film_seconds or 0.0)
    lines = []
    for note in notes:
        where = f"at {note.stamp}"
        if span > 0:
            where += f", about {round(100 * note.at_seconds / span)}% of the way through"
        photo = photo_at(note.at_seconds) if photo_at is not None else None
        if photo is not None:
            where += f", looking at photograph {photo}"
        lines.append(f"  [{where}] {note.text}")
    return "\n".join(lines)


def _note(row: sqlite3.Row) -> Note:
    return Note(
        id=int(row["id"]),
        group_id=int(row["group_id"]),
        at_seconds=float(row["at_seconds"]),
        text=row["text"],
        created_at=row["created_at"],
    )
