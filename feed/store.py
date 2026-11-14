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

from feed.language import LANGUAGES
from feed.voice import voice_label
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


@dataclass(frozen=True)
class FilmNarrator:
    """Who narrates a finished film, as its stored cast says (never the current choice)."""

    name: str
    label: str
    chosen_by: str                        # "user" | "story" | "mood" | "auto" (Automatic, reason not recorded)
    others: tuple[tuple[int, str], ...]   # (passage number from 1, label) per passage voice that is not the narrator
    model: str | None = None


@dataclass(frozen=True)
class FilmState:
    language: str
    status: str                   # the stored row: "ready" | "failed" | "running"
    ready: bool                   # status "ready" with a narration: the film plays
    narrator: FilmNarrator | None  # None = the voice was not recorded
    error: str | None


def film_narrator(decisions: dict | None) -> FilmNarrator | None:
    """Read the narrator from a film's `decisions["cast"]` (D37, D42). A film made before
    the cast stored its labels falls back to `voice_label`; one that did not record who
    chose the voice reads by `narrator_asked_was_real`, else "auto". A saved-voice film
    names the voice model that spoke (D86), or says it was made from one take before
    voice models (D81)."""
    cast = (decisions or {}).get("cast") or {}
    used = cast.get("narrator_used")
    if not used:
        return None
    language = decisions.get("language", "")
    if cast.get("narrator_name") and cast.get("narrator_label"):
        name, label = cast["narrator_name"], cast["narrator_label"]
    else:
        fallback = voice_label(used, language)
        name, label = fallback.name, fallback.label
    chosen_by = cast.get("narrator_chosen_by")
    if chosen_by not in ("user", "mood"):
        asked_was_real = cast.get("narrator_asked_was_real")
        chosen_by = "auto" if asked_was_real is None else "story" if asked_was_real else "mood"
    # `asked` and `model_cast` are never read: the build may have rejected those names.
    others = tuple(
        (entry["passage"] + 1, entry.get("used_label") or voice_label(entry["used"], language).label)
        for entry in cast.get("passages") or []
        if entry.get("used") and entry["used"] != used
    )
    voice = decisions.get("voice") or {}
    model = cast.get("narrator_model")
    if model is None and voice.get("saved") and not voice.get("models"):
        model = "made from one take, before voice models"
    return FilmNarrator(name=name, label=label, chosen_by=chosen_by, others=others, model=model)


def film_states(conn: sqlite3.Connection, group_id: int) -> dict[str, FilmState]:
    """The stored film of each language of one memory, in `LANGUAGES` order; a language
    with no row is absent."""
    rows = {
        row["language"]: row
        for row in conn.execute(
            "SELECT language, status, narration_key, error, decisions_json"
            " FROM memory_presentations WHERE group_id = ?",
            (group_id,),
        )
    }
    return {
        language: FilmState(
            language=language,
            status=rows[language]["status"],
            ready=rows[language]["status"] == "ready" and rows[language]["narration_key"] is not None,
            narrator=film_narrator(json.loads(rows[language]["decisions_json"] or "{}")),
            error=rows[language]["error"],
        )
        for language in LANGUAGES
        if language in rows
    }


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
