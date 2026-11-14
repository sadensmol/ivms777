"""The film's narration as passages, and the user's edited draft of it (design §6, §11.1, §11.3).

A ready film keeps its narration in `timeline.passages`: one passage per sentence of the
story, in telling order, with its time in the film's audio, its photograph and its voice.
The first change the user makes to those words seeds a draft (`narration_drafts`) with
ALL passages and every replay input, so a later build speaks the draft as written and
never needs the film it came from.

A narration row is never a note: nothing here reads or writes `memory_notes`.

Plain SQL, no model, no HTTP.
"""

import json
import secrets
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace

from feed.language import LANGUAGES
from feed.store import Presentation, read_presentation
from feed.story import Story, introduces_in_order

NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%SZ','now')"
MAX_SCRIPT_CHARS = 10_000   # a provisional app limit (D115)

_REFUSALS = {
    "stale": "Not saved: this narration changed in another window, or a new film replaced it.",
    "missing": "This passage is no longer in the narration.",
    "empty": "A film needs narration. Keep at least one passage.",
    "empty_text": "A passage cannot be left empty here. Nothing changed.",
    "bracket": "The narration is read aloud word for word. Put a [bracket] instruction in a note.",
}


@dataclass(frozen=True)
class Passage:
    n: int                      # its number in the film whose times it carries; a change never renumbers it
    text: str
    start: float | None         # seconds in that film's audio; None = the voice did not pair the segments
    end: float | None
    photo: int                  # the story's photograph for it: an index of that film's photo_ids
    photo_id: int | None
    voice_asked: str            # the story pass's voice for it ("" = the narrator)
    voice_used: str | None      # what spoke it in that film (a preset name or a saved-voice token)

    @property
    def stamp(self) -> str:
        if self.start is None:
            return "–:––"
        total = max(int(self.start), 0)
        return f"{total // 60}:{total % 60:02d}"


@dataclass(frozen=True)
class Draft:
    group_id: int
    language: str
    draft_token: str
    source_token: str
    revision: int
    script: dict
    passages: tuple[Passage, ...]


@dataclass(frozen=True)
class NarrationRows:
    state: str                  # "film" | "draft" | "legacy" | "none"
    source: str | None          # the film token (film) or the draft's source_token (draft); None otherwise
    draft: str | None           # the draft's draft_token; None for every other state
    revision: int               # the draft's revision; 0 for the film's own rows
    seekable: bool              # the times locate the ready film of this language
    changed: bool               # a draft whose words are not the words of the ready film (D122)
    rows: tuple[Passage, ...]

    @property
    def triple(self) -> tuple[str | None, str, int]:
        """What a change must post to be taken (D114)."""
        return self.source, self.draft or "", self.revision

    def as_json(self) -> dict:
        return {
            "state": self.state,
            "source": self.source,
            "draft": self.draft,
            "revision": self.revision,
            "seekable": self.seekable,
            "changed": self.changed,
            "rows": [{"n": row.n, "text": row.text, "start": row.start, "end": row.end,
                      "stamp": row.stamp} for row in self.rows],
        }


class NarrationRefused(ValueError):
    def __init__(self, code: str, message: str | None = None):
        super().__init__(message or _REFUSALS[code])
        self.code = code
        self.message = message or _REFUSALS[code]


def timeline_script(story: Story, segments: Sequence[dict], passage_voices: Sequence[str | None],
                    speaker: str, photo_ids: Sequence[int], *, token: str) -> tuple[list[dict], dict]:
    """The film's `timeline.passages` and `timeline.script` (§2.19 A).

    Passage `i` is chunk `i` of the story, which is segment `i` of the audio. When the
    segments do not pair one to one, every time is None and the texts stay (D112)."""
    chunks = story.chunks
    timed = [segment for segment in segments if isinstance(segment, dict) and "start" in segment]
    paired = len(timed) == len(chunks)
    passages = []
    for i, (photo, text) in enumerate(story.script):
        voice = passage_voices[i] if i < len(passage_voices) else None
        passages.append({
            "n": i,
            "text": text,
            "start": float(timed[i]["start"]) if paired else None,
            "end": float(timed[i]["end"]) if paired and timed[i].get("end") is not None else None,
            "photo": photo,
            "photo_id": photo_ids[photo] if 0 <= photo < len(photo_ids) else None,
            "voice_asked": story.voices[i] if i < len(story.voices) else "",
            "voice_used": voice or speaker,
        })
    script = {
        "token": token,
        "narrator_asked": story.narrator,
        "narrator_used": speaker,
        "cards": [{"n": n, "text": text} for n, text in story.cards if 0 <= n < len(chunks)],
        "score_plan": [{"from": index, "brief": brief} for index, brief in story.score],
        "music_prompt": story.music_prompt,
    }
    return passages, script


def read_draft(conn: sqlite3.Connection, group_id: int, language: str) -> Draft | None:
    row = conn.execute(
        "SELECT group_id, language, draft_token, source_token, revision, script_json, passages_json"
        " FROM narration_drafts WHERE group_id = ? AND language = ?",
        (group_id, language),
    ).fetchone()
    if row is None:
        return None
    return Draft(
        group_id=int(row["group_id"]),
        language=row["language"],
        draft_token=row["draft_token"],
        source_token=row["source_token"],
        revision=int(row["revision"]),
        script=json.loads(row["script_json"]),
        passages=tuple(_to_passage(item) for item in json.loads(row["passages_json"])),
    )


def narration_rows(conn: sqlite3.Connection, group_id: int, language: str) -> NarrationRows:
    """What the panel shows for one memory and language (§2.19 B.1)."""
    presentation = read_presentation(conn, group_id, language)
    ready = (presentation is not None and presentation.status == "ready"
             and presentation.narration_key is not None)
    film = _film_passages(presentation) if ready else None
    draft = read_draft(conn, group_id, language)
    if draft is not None:
        film_token, film_rows = film if film is not None else (None, ())
        seekable = film_token == draft.source_token
        changed = not (seekable and _words(draft.passages) == _words(film_rows))
        return NarrationRows("draft", draft.source_token, draft.draft_token, draft.revision,
                             seekable, changed, draft.passages)
    if film is not None:
        token, rows = film
        return NarrationRows("film", token, None, 0, True, False, rows)
    if ready:
        return NarrationRows("legacy", None, None, 0, False, False, ())
    return NarrationRows("none", None, None, 0, False, False, ())


def save_passage(conn: sqlite3.Connection, group_id: int, language: str, n: int, text: str, *,
                 source: str, draft: str, revision: int) -> NarrationRows:
    """Change the words of passage `n`; an empty text removes it (§2.19 B.3-B.5)."""
    cleaned = " ".join(text.split())

    def change(rows: tuple[Passage, ...]) -> tuple[Passage, ...] | None:
        current = next((row for row in rows if row.n == n), None)
        if current is None:
            raise NarrationRefused("missing")
        if cleaned == current.text:
            return None
        if not cleaned:
            kept = tuple(row for row in rows if row.n != n)
            if not kept:
                raise NarrationRefused("empty")
            return kept
        if "[" in cleaned or "]" in cleaned:
            raise NarrationRefused("bracket")
        changed = tuple(replace(row, text=cleaned) if row.n == n else row for row in rows)
        total = sum(len(row.text) for row in changed)
        if total > MAX_SCRIPT_CHARS:
            raise NarrationRefused(
                "too_long",
                f"The narration is longer than this app allows: {total} characters,"
                f" at most {MAX_SCRIPT_CHARS}.",
            )
        return changed

    return _locked_change(conn, group_id, language, (source, draft, revision), change)


def rewrite_passages(conn: sqlite3.Connection, group_id: int, language: str,
                     changes: Mapping[int, str], *, source: str, draft: str,
                     revision: int) -> NarrationRows:
    """Change the words of several passages, all or none (§11.3). In `save_passage` an
    empty text removes the passage; here it is refused, so a rewrite never drops one."""
    cleaned = {n: " ".join(text.split()) for n, text in changes.items()}

    def change(rows: tuple[Passage, ...]) -> tuple[Passage, ...] | None:
        if not cleaned.keys() <= {row.n for row in rows}:
            raise NarrationRefused("missing")
        if not all(cleaned.values()):
            raise NarrationRefused("empty_text")
        if any("[" in text or "]" in text for text in cleaned.values()):
            raise NarrationRefused("bracket")
        changed = tuple(replace(row, text=cleaned.get(row.n, row.text)) for row in rows)
        if changed == rows:
            return None
        total = sum(len(row.text) for row in changed)
        if total > MAX_SCRIPT_CHARS:
            raise NarrationRefused(
                "too_long",
                f"The narration is longer than this app allows: {total} characters,"
                f" at most {MAX_SCRIPT_CHARS}.",
            )
        return changed

    return _locked_change(conn, group_id, language, (source, draft, revision), change)


def remove_passage(conn: sqlite3.Connection, group_id: int, language: str, n: int, *,
                   source: str, draft: str, revision: int) -> NarrationRows:
    return save_passage(conn, group_id, language, n, "", source=source, draft=draft,
                        revision=revision)


def discard_draft(conn: sqlite3.Connection, group_id: int, language: str, *,
                  source: str, draft: str, revision: int) -> NarrationRows:
    """"Write a new story": the stale rule, then the draft row goes (§2.19 B.6)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        _check_fresh(narration_rows(conn, group_id, language), (source, draft, revision))
        conn.execute("DELETE FROM narration_drafts WHERE group_id = ? AND language = ?",
                     (group_id, language))
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return narration_rows(conn, group_id, language)


def draft_languages(conn: sqlite3.Connection, group_id: int) -> list[str]:
    stored = {row["language"] for row in conn.execute(
        "SELECT language FROM narration_drafts WHERE group_id = ?", (group_id,))}
    return [language for language in LANGUAGES if language in stored]


def draft_story(draft: Draft, photo_ids: Sequence[int]) -> Story:
    """The Story a build speaks from a draft, with no model call (§2.19 C.2)."""
    photos: list[int] = []
    for passage in draft.passages:
        if passage.photo_id in photo_ids:
            photos.append(photo_ids.index(passage.photo_id))
        else:
            photos.append(photos[-1] if photos else 0)
    numbers = [passage.n for passage in draft.passages]
    position = {n: k for k, n in enumerate(numbers)}
    texts = [passage.text for passage in draft.passages]
    cards = tuple((position[card["n"]], card["text"])
                  for card in draft.script.get("cards") or () if card["n"] in position)
    score: dict[int, str] = {}
    for entry in draft.script.get("score_plan") or ():
        k = next((k for k, n in enumerate(numbers) if n >= entry["from"]), None)
        if k is not None:
            score.setdefault(k, entry["brief"])
    return Story(
        narration=" ".join(texts),
        cards=cards,
        music_prompt=draft.script.get("music_prompt") or "",
        narrator=draft.script.get("narrator_asked") or "",
        script=tuple(zip(photos, texts)),
        voices=tuple(passage.voice_asked for passage in draft.passages),
        score=tuple(sorted(score.items())),
        memory_order="kept" if introduces_in_order(photos) else "broken",
    )


def rebase_consumed(conn: sqlite3.Connection, draft: Draft, passages: list[dict], script: dict) -> bool:
    """Give the draft a build consumed the new film's passages, token and script, keeping
    its words (§2.19 B.7). The caller holds the transaction. True when a row changed."""
    cursor = conn.execute(
        "UPDATE narration_drafts SET passages_json = ?, source_token = ?, script_json = ?,"
        f" revision = revision + 1, updated_at = {NOW_SQL}"
        " WHERE group_id = ? AND language = ? AND draft_token = ? AND revision = ?",
        (json.dumps(passages, ensure_ascii=False), script["token"],
         json.dumps(_without_token(script), ensure_ascii=False),
         draft.group_id, draft.language, draft.draft_token, draft.revision),
    )
    return cursor.rowcount == 1


def _locked_change(conn: sqlite3.Connection, group_id: int, language: str,
                   posted: tuple[str, str, int],
                   change: Callable[[tuple[Passage, ...]], tuple[Passage, ...] | None]) -> NarrationRows:
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = narration_rows(conn, group_id, language)
        _check_fresh(rows, posted)
        changed = change(rows.rows)
        if changed is not None:
            _write(conn, group_id, language, rows, changed)
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return narration_rows(conn, group_id, language)


def _check_fresh(rows: NarrationRows, posted: tuple[str, str, int]) -> None:
    if rows.state == "none" or posted != rows.triple:
        raise NarrationRefused("stale")


def _write(conn: sqlite3.Connection, group_id: int, language: str, rows: NarrationRows,
           passages: tuple[Passage, ...]) -> None:
    """Store the change. A row whose text changed loses its card: the card was written for
    the old words (§11.1)."""
    passages_json = json.dumps([asdict(passage) for passage in passages], ensure_ascii=False)
    before = {row.n: row.text for row in rows.rows}
    rewritten = {passage.n for passage in passages if passage.text != before[passage.n]}
    script = (read_draft(conn, group_id, language).script if rows.state == "draft"
              else read_presentation(conn, group_id, language).timeline["script"])
    cards = [card for card in script.get("cards") or () if card["n"] not in rewritten]
    script_json = json.dumps(_without_token({**script, "cards": cards}), ensure_ascii=False)
    if rows.state == "draft":
        conn.execute(
            "UPDATE narration_drafts SET passages_json = ?, script_json = ?, revision = revision + 1,"
            f" updated_at = {NOW_SQL} WHERE group_id = ? AND language = ?",
            (passages_json, script_json, group_id, language),
        )
        return
    conn.execute(
        "INSERT INTO narration_drafts(group_id, language, draft_token, source_token, revision,"
        f" script_json, passages_json, updated_at) VALUES (?, ?, ?, ?, 1, ?, ?, {NOW_SQL})",
        (group_id, language, secrets.token_hex(16), rows.source, script_json, passages_json),
    )


def _film_passages(presentation: Presentation) -> tuple[str, tuple[Passage, ...]] | None:
    """A film's token and passages; None for a film made before it carried them."""
    passages = presentation.timeline.get("passages")
    token = (presentation.timeline.get("script") or {}).get("token")  # pragma: allowlist secret
    if not passages or not token:
        return None
    return token, tuple(_to_passage(item) for item in passages)


def _words(passages: Sequence[Passage]) -> list[tuple[int, str]]:
    return [(passage.n, passage.text) for passage in passages]


def _without_token(script: dict) -> dict:
    return {key: value for key, value in script.items() if key != "token"}


def _to_passage(item: dict) -> Passage:
    return Passage(
        n=int(item["n"]),
        text=item["text"],
        start=item.get("start"),
        end=item.get("end"),
        photo=int(item["photo"]),
        photo_id=item.get("photo_id"),
        voice_asked=item.get("voice_asked") or "",
        voice_used=item.get("voice_used"),
    )
