"""Ask AI to rephrase the narration (design §11.3).

A model proposes new words for the narration rows, and only the user's Use this changes
anything (`narration.rewrite_passages`). This module writes nothing: it reads the film,
asks one model call, and checks the answer. The model names only the passages it changes,
so every other passage keeps its words by identity.

What the film WAS is exact: `timeline` and `decisions` of the ready row. What the memory
IS, its title, notes and photograph details, is today's: no build keeps a copy of them, and
the prompt says so. Nothing is cut to fit: a prompt over the model's context is refused.

No audio, no vector, no effect or grade table reaches the prompt.
"""

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from feed.build import Cancelled, _context_tokens, _previous_cut, _Ticking
from feed.narration import (
    MAX_SCRIPT_CHARS,
    NarrationRows,
    Passage,
    _check_fresh,
    narration_rows,
    read_draft,
)
from feed.notes import notes_for_prompt, read_notes
from feed.signals import digest, read_signals
from feed.store import Presentation, read_presentation
from feed.story import (
    _LANGUAGE_NAME,
    CHARS_PER_TOKEN,
    _estimated_tokens,
    _is_in_language,
    _loads,
)

MAX_REQUEST_CHARS = 1_000
MAX_HISTORY_MESSAGES = 6
TIMEOUT_SECONDS = 300.0
TEMPERATURE = 0.5

# Room for the words, the JSON around them and the reply. A rewrite may run longer than the
# rows, up to the app's limit.
GROWTH = 2
REPLY_TOKENS = 400
TEMPLATE_TOKENS = 256
# An estimate for the moving bar only. One call of gemma4-E2B on this mac read about 3 400
# prompt tokens and wrote a short answer in 11 s. The output rate is not measured.
PROMPT_TOKENS_PER_SECOND = 350.0
OUTPUT_TOKENS_PER_SECOND = 40.0

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["n", "text"],
            },
        },
    },
    "required": ["reply", "changes"],
}


class RephraseRefused(ValueError):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class Message:
    role: str                   # "user" | "ai"
    text: str


@dataclass(frozen=True)
class Ask:
    group_id: int
    language: str
    posted: tuple[str, str, int]        # the (source, draft, revision) the page shows
    request: str
    history: tuple[Message, ...]
    proposal: Mapping[int, str]         # the unsaved patch on screen, against the saved rows
    model: str


@dataclass(frozen=True)
class Film:
    rows: NarrationRows
    title: str
    description: str
    day: str                    # the hard facts of the memory, from `signals.digest`
    notes: str
    briefs: tuple[tuple[int, str], ...]     # (photo_id, brief) in the memory's order
    script: dict
    ready: Presentation | None  # the ready film of this language


@dataclass(frozen=True)
class Change:
    n: int
    stamp: str
    old: str
    new: str


@dataclass(frozen=True)
class Proposal:
    base: tuple[str, str, int]
    reply: str
    changes: tuple[Change, ...]         # cumulative, against the saved rows
    total: int

    def as_json(self) -> dict:
        source, draft, revision = self.base
        count = len(self.changes)
        return {
            "base": {"source": source, "draft": draft, "revision": revision},
            "reply": self.reply,
            "summary": (f"Changed {count} of {self.total} passages." if count
                        else "No passage changed."),
            "total": self.total,
            "changes": [{"n": change.n, "stamp": change.stamp, "old": change.old,
                         "new": change.new} for change in self.changes],
        }


def clean_request(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise RephraseRefused("Ask for a change first.")
    if len(cleaned) > MAX_REQUEST_CHARS:
        raise RephraseRefused(f"A request is at most {MAX_REQUEST_CHARS} characters.")
    return cleaned


def read_history(raw: str) -> tuple[Message, ...]:
    """The last messages the page sent. It keeps the whole transcript itself."""
    items = _posted_list(raw)
    if not all(isinstance(item, dict) and item.get("role") in ("user", "ai")
               and isinstance(item.get("text"), str) for item in items):
        raise RephraseRefused("The conversation could not be read.")
    return tuple(Message(item["role"], item["text"]) for item in items[-MAX_HISTORY_MESSAGES:])


def read_changes(raw: str) -> dict[int, str]:
    """A posted patch: `[{"n": 3, "text": "…"}]`, each passage once."""
    changes: dict[int, str] = {}
    for item in _posted_list(raw):
        n = item.get("n") if isinstance(item, dict) else None
        text = item.get("text") if isinstance(item, dict) else None
        if type(n) is not int or not isinstance(text, str) or n in changes:
            raise RephraseRefused("The proposal could not be read.")
        changes[n] = text
    return changes


def read_film(conn: sqlite3.Connection, group_id: int, language: str,
              posted: tuple[str, str, int]) -> Film:
    """The rows, their triple and the film's context, from ONE snapshot of the database.

    The transaction ends here, so no lock is held during the model call."""
    conn.execute("BEGIN")
    try:
        rows = narration_rows(conn, group_id, language)
        _check_fresh(rows, posted)
        presentation = read_presentation(conn, group_id, language)
        ready = (presentation if presentation is not None and presentation.status == "ready"
                 and presentation.narration_key is not None else None)
        memory = conn.execute(
            "SELECT name, COALESCE(description, '') AS description FROM groups WHERE id = ?",
            (group_id,),
        ).fetchone()
        signals = read_signals(conn, group_id)
        seconds, photo_at = (_previous_cut(conn, group_id, language, signals)
                             if ready is not None else (0.0, None))
        draft = read_draft(conn, group_id, language)
        return Film(
            rows=rows,
            title=memory["name"],
            description=memory["description"],
            day=digest(signals, guesses=False),
            notes=notes_for_prompt(read_notes(conn, group_id), seconds, photo_at=photo_at),
            briefs=tuple((signal.photo_id, signal.brief()) for signal in signals),
            script=draft.script if draft is not None else ready.timeline["script"],
            ready=ready,
        )
    finally:
        conn.execute("COMMIT")


def build_prompt(film: Film, ask: Ask) -> str:
    language = _LANGUAGE_NAME.get(ask.language, ask.language)
    sections = [
        ("You edit the spoken narration of a personal film. You change only its words."
         " Photographs, voices, times, title cards and music stay as they are."),
        _memory_section(film),
        _film_section(film),
        f"THE NARRATION, AS SAVED. One line per passage; n is its number.\n{_rows_section(film)}",
    ]
    if ask.proposal:
        lines = "\n".join(f"n={n}: {text}" for n, text in sorted(ask.proposal.items()))
        sections.append(
            "THE PROPOSAL ON SCREEN. Not saved. These are your earlier words, which the user"
            f" is still refining. For these n, they replace the saved words:\n{lines}")
    if ask.history:
        lines = "\n".join(f"{'USER' if message.role == 'user' else 'YOU'}: {message.text}"
                          for message in ask.history)
        sections.append(f"THE CONVERSATION SO FAR\n{lines}")
    sections.append(f"THE USER ASKS NOW\n{ask.request}")
    sections.append(
        "RULES\n"
        f"- Write every text in {language}.\n"
        "- Change only what the request needs. In `changes`, list ONLY the passages you"
        " change, each once, with its complete new text. Every other passage keeps its"
        " words.\n"
        "- Use each n exactly as listed. Never add, remove, merge or reorder passages.\n"
        "- The text is read aloud word for word: no [brackets], no stage directions, no"
        " empty text.\n"
        "- Keep to the details above. Invent no person, place or event.\n"
        "- `reply` is one or two sentences to the user, saying what you did. When the"
        " request needs no change of words, or asks for something other than words, say so"
        " there and leave `changes` empty.\n"
        'Answer with JSON only: {"reply": "…", "changes": [{"n": 0, "text": "…"}]}')
    return "\n\n".join(sections)


def output_tokens(rows: Sequence[Passage]) -> int:
    chars = min(GROWTH * sum(len(row.text) for row in rows), MAX_SCRIPT_CHARS)
    return int(chars / CHARS_PER_TOKEN) + REPLY_TOKENS


def check_fit(prompt: str, output: int, context_tokens: int | None) -> None:
    """Refuse a prompt the server would reject. `None` = the server does not say: no budget."""
    needed = _estimated_tokens(prompt) + output + TEMPLATE_TOKENS
    if context_tokens and needed > context_tokens:
        raise RephraseRefused(
            f"This film's details are too long for the model: about {needed} tokens, and it"
            f" takes {context_tokens}. Nothing was cut and nothing was sent.")


def repair(raw: str) -> tuple[str, list[tuple[object, object]]]:
    """Tidy what a small model emits, without changing what it said: a fenced block is
    unwrapped and whitespace joins, as a saved passage does. Anything else that is wrong
    is for `check_changes` to reject."""
    try:
        payload = _loads(raw)
    except json.JSONDecodeError as error:
        raise RephraseRefused("Not used: the AI's answer could not be read.") from error
    reply = payload.get("reply")
    changes = payload.get("changes")
    if changes is None:
        changes = []
    if (not isinstance(changes, list) or not isinstance(reply, str | None)
            or not all(isinstance(item, dict) for item in changes)):
        raise RephraseRefused("Not used: the AI's answer could not be read.")
    tidied = [(item.get("n"), " ".join(item["text"].split())
               if isinstance(item.get("text"), str) else item.get("text"))
              for item in changes]
    return " ".join((reply or "").split()), tidied


def check_changes(rows: Sequence[Passage], proposal: Mapping[int, str],
                  changes: Sequence[tuple[object, object]], language: str) -> dict[int, str]:
    """The cumulative patch against the saved rows, or the reason the WHOLE turn is rejected.

    New changes go on top of the proposal on screen, so a follow-up keeps earlier unsaved
    words. A passage back at its saved words leaves the patch."""
    saved = {row.n: row.text for row in rows}
    seen: set[int] = set()
    for n, text in changes:
        if type(n) is not int or n not in saved:
            raise RephraseRefused(f"Not used: the AI named a passage that does not exist ({n}).")
        if n in seen:
            raise RephraseRefused(f"Not used: the AI changed passage {n} twice.")
        seen.add(n)
        if not isinstance(text, str) or not text:
            raise RephraseRefused(f"Not used: passage {n} came back empty.")
        if "[" in text or "]" in text:
            raise RephraseRefused(f"Not used: passage {n} has a [bracket].")
        if not _is_in_language(text, language):
            raise RephraseRefused(
                f"Not used: passage {n} is not in {_LANGUAGE_NAME.get(language, language)}.")
    if not proposal.keys() <= saved.keys():
        raise RephraseRefused("The proposal does not fit the saved narration. Ask again.")
    words = {**saved, **proposal, **dict(changes)}
    total = sum(len(text) for text in words.values())
    if total > MAX_SCRIPT_CHARS:
        raise RephraseRefused(
            f"Not used: the narration would be {total} characters, at most {MAX_SCRIPT_CHARS}.")
    return {n: text for n, text in words.items() if text != saved[n]}


def propose(conn: sqlite3.Connection, client, ask: Ask, *,
            progress: Callable[[str, float], None],
            cancelled: Callable[[], bool]) -> Proposal:
    """One turn: read the film, ask the model once, check the answer. Writes nothing."""
    progress("reading the film", 0.02)
    film = read_film(conn, ask.group_id, ask.language, ask.posted)
    prompt = build_prompt(film, ask)
    output = output_tokens(film.rows.rows)
    check_fit(prompt, output, _context_tokens(client))
    if cancelled():
        raise Cancelled
    expect = (_estimated_tokens(prompt) / PROMPT_TOKENS_PER_SECOND
              + output / OUTPUT_TOKENS_PER_SECOND)
    with _Ticking(progress, "writing new words", (0.1, 0.9), expect):
        raw = client.text_complete(
            ask.model, [{"role": "user", "content": prompt}], json_schema=SCHEMA,
            timeout=TIMEOUT_SECONDS, temperature=TEMPERATURE, max_tokens=output)
    progress("checking the words", 0.92)
    reply, changes = repair(raw)
    if not reply and not changes:
        raise RephraseRefused("Not used: the AI gave no answer.")
    patch = check_changes(film.rows.rows, ask.proposal, changes, ask.language)
    return Proposal(
        base=ask.posted,
        reply=reply,
        changes=tuple(Change(row.n, row.stamp, row.text, patch[row.n])
                      for row in film.rows.rows if row.n in patch),
        total=len(film.rows.rows),
    )


def _posted_list(raw: str) -> list:
    try:
        items = json.loads(raw or "[]")
    except json.JSONDecodeError as error:
        raise RephraseRefused("The request could not be read.") from error
    if not isinstance(items, list):
        raise RephraseRefused("The request could not be read.")
    return items


def _memory_section(film: Film) -> str:
    photos = "\n".join(f"Photograph {i} (id {photo_id}): {brief}"
                       for i, (photo_id, brief) in enumerate(film.briefs, start=1))
    return (
        "THE MEMORY, AS IT IS TODAY. These details may have changed since the film was made."
        " The viewer's notes come first: they say what the day meant.\n"
        f"Title: {film.title}\n"
        f"Description: {film.description or '(none)'}\n"
        f"What the library knows:\n{film.day}\n"
        f"The viewer's notes, with the second and photograph each was written at:\n"
        f"{film.notes or '  (none)'}\n"
        "The photographs. The key=value tags (vibe=, subject=, emotion= and the like) are"
        f" automatic guesses, not facts:\n{photos or '(none)'}"
    )


def _film_section(film: Film) -> str:
    script = {key: film.script.get(key) for key in
              ("narrator_asked", "narrator_used", "cards", "score_plan", "music_prompt")}
    record = f"Script (a card's n is its passage): {_compact(script)}"
    if film.ready is None:
        return ("THE FILM. No ready film of this language exists now, so there is no timeline"
                f" and no record of its music or voices. Only this script remains.\n{record}")
    timeline = film.ready.timeline
    shots = "\n".join(_compact(shot) for shot in timeline.get("shots") or ())
    decisions = film.ready.decisions or {}
    recorded = "\n".join(f"{name}: {_compact(decisions.get(name))}"
                         for name in ("story", "cast", "voice", "score", "film"))
    if film.rows.state == "film":
        note = "The rows below are this film's own words."
    elif film.rows.seekable:
        note = ("The rows below are the user's edit of this film. Each time is where the"
                " film speaks that passage; new words get new timing at the next build.")
    else:
        note = ("The rows below came from an earlier film, not this one: their times do not"
                " belong to this timeline.")
    return (
        f"THE FILM, THE EXACT RECORD OF WHAT WAS COMPOSED. {note}\n"
        f"{record}\n"
        f"Length {timeline.get('seconds')} s, grade {timeline.get('grade')}.\n"
        f"Shots, in film order:\n{shots or '(none)'}\n"
        "What the build decided. In `cast`, `model_cast` and every `asked` field are the"
        " model's raw requests, not validated and not voices used. The voices used are"
        f" `narrator_used` and each row's voice used:\n{recorded}"
    )


def _rows_section(film: Film) -> str:
    return "\n".join(
        f"n={row.n} | at {row.stamp} | photo id {row.photo_id}"
        f" | voice used {row.voice_used}"
        f" | {row.text}"
        for row in film.rows.rows
    )


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
