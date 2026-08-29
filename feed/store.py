"""Stored memory presentations (design §6, §11.1).

A presentation is derived data: a deterministic timeline plus the storage keys of two
generated WAVs. It is keyed by the memory's `group_id` and cascades with it, so
rebuilding memories (§11) throws every presentation away — which is correct, because
the memory each one described no longer exists.

No models, no HTTP: this module is a thin client (CLAUDE.md § "One model process").
"""

import json
import sqlite3
from dataclasses import dataclass

from storage.base import Storage

NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%SZ','now')"


@dataclass(frozen=True)
class Presentation:
    group_id: int
    language: str
    signature: str
    timeline: dict
    music_key: str | None
    narration_key: str | None
    status: str          # "ready" | "failed"
    error: str | None = None
    # What the build DECIDED (design §11.1) — the cast, the score, the seed, the
    # sampling, the model context. Stored so a finished film can be asked what it did
    # rather than guessed at; `{}` for a row written before this existed.
    decisions: dict | None = None


def save_presentation(conn: sqlite3.Connection, presentation: Presentation) -> None:
    """Upsert: preparing a memory twice replaces the previous feed rather than
    failing on the primary key."""
    conn.execute(
        "INSERT INTO memory_presentations"
        "(group_id, language, signature, timeline_json, music_key, narration_key,"
        f" status, error, decisions_json, created_at)"
        f" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {NOW_SQL})"
        " ON CONFLICT(group_id, language) DO UPDATE SET"
        " signature = excluded.signature, timeline_json = excluded.timeline_json,"
        " music_key = excluded.music_key, narration_key = excluded.narration_key,"
        " status = excluded.status, error = excluded.error,"
        f" decisions_json = excluded.decisions_json, created_at = {NOW_SQL}",
        (
            presentation.group_id,
            presentation.language,
            presentation.signature,
            json.dumps(presentation.timeline),
            presentation.music_key,
            presentation.narration_key,
            presentation.status,
            presentation.error,
            json.dumps(presentation.decisions or {}),
        ),
    )


def read_presentation(
    conn: sqlite3.Connection, group_id: int, language: str = "en"
) -> Presentation | None:
    row = conn.execute(
        "SELECT group_id, language, signature, timeline_json, music_key, narration_key,"
        " status, error, decisions_json"
        " FROM memory_presentations WHERE group_id = ? AND language = ?",
        (group_id, language),
    ).fetchone()
    if row is None:
        return None
    return Presentation(
        group_id=int(row["group_id"]),
        language=row["language"],
        signature=row["signature"],
        timeline=json.loads(row["timeline_json"]),
        music_key=row["music_key"],
        narration_key=row["narration_key"],
        status=row["status"],
        error=row["error"],
        decisions=json.loads(row["decisions_json"] or "{}"),
    )


def prune_orphan_audio(conn: sqlite3.Connection, audio: Storage) -> int:
    """Delete every audio blob no presentation references, and return how many went.

    The table cascades away with its memories but the FILES do not, so a rebuild would
    otherwise leave every previous feed's WAVs on disk forever. Called at the end of a
    prepare run — the one moment we know the table is authoritative.
    """
    referenced = {
        key
        for row in conn.execute("SELECT music_key, narration_key FROM memory_presentations")
        for key in (row["music_key"], row["narration_key"])
        if key
    }
    removed = 0
    for key in list(audio.iter_keys()):
        if key not in referenced:
            audio.delete(key)
            removed += 1
    return removed
