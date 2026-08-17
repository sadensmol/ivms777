import json
import logging
import sqlite3

from albums.memory_store import Memory
from albums.seeds import Candidate
from inference.client import ChatMessage, InferenceClient
from search.semantic import similar_photos

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You compose ONE memory from a set of photos taken close together in time and "
    "place. Read their summaries (date, place, caption, tags). Decide if they form "
    "one coherent memory — a place, a day, an occasion. "
    "Reply with ONLY a JSON object, no prose. To ask for more context first, use "
    '{\"action\":\"expand\",\"tool\":\"similar|facets|nearby\",\"photo_id\":<id>}. '
    "When ready, answer with "
    '{\"action\":\"answer\",\"keep\":<bool>,\"title\":\"...\",\"description\":\"...\",'
    '\"drop_photo_ids\":[...]}. '
    "You are writing for the person whose life this is — they should read it and "
    "want to open it. "
    'TITLE: short and warm, and it NAMES THE PLACE when there is one — "A winter '
    'day in Borjomi" is the SHAPE to follow, not words to copy. Never "Activities", '
    'never "a location" or "an indoor setting", never coordinates. Put nothing in '
    "the title that the summaries do not show — if they never say the photos are at "
    'home, in a cafe, or in a museum, the title does not say it either. '
    "DESCRIPTION: 2-4 sentences of warm, plain storytelling about WHAT IS IN THESE "
    "PHOTOS — the place, the day, the things that are actually there. Write about "
    "the DAY as one whole, never frame by frame: no \"another photo shows\", no "
    '"a third view", no "these photos capture", no "There were scenes of…". Say it '
    "as it happened — a girl by the window in the morning, later blankets and a "
    "seascape — never the same sentence shape twice. "
    "PEOPLE — the hard rule. Mention only people the captions mention, described the "
    'way the captions describe them ("a young girl", "two people"). If no caption '
    "mentions a person, THERE ARE NO PEOPLE IN THIS MEMORY: write about the cake, "
    "the street, the light — and never about friends, family, guests, everyone, we, "
    "or you. Two photos of a cake are two photos of a cake, however warm the day. "
    "Warmth comes from the real place, light, season and occasion — never from "
    "people, feelings or doings you added yourself. "
    "Tags are hints, not words to quote: when a tag contradicts the date or the "
    'captions (a "summer" tag on a November day), trust the date and the captions. '
    "Invent no names and no relationships. When a real town or country is named you "
    "MAY add one short, well-known touch about it (its mountains, its old town, what "
    "it is known for) — that is the only thing you may bring from your own "
    "knowledge. "
    "keep=true whenever the photos hang together — then write the title and the "
    "description. Photos from the same day in the same town are one memory unless "
    "they are truly unrelated; WHEN IN DOUBT, KEEP. To skip, reply exactly "
    '{"action":"answer","keep":false} and nothing else — never an empty value like '
    '"title":, which is not valid JSON. '
    "drop_photo_ids is USUALLY EMPTY: it lists only the odd photo that does not "
    "belong. The memory keeps every photo you do not drop, so NEVER list them all — "
    "listing every id throws the whole memory away."
)

# The prompt must fit the SMALLEST context we run: jetson's `llm_ctx: 2048` (§3.1,
# deliberately small to shrink gemma's resident footprint on the 8 GB board). At
# ~40 tokens per photo line, 24 lines plus the system prompt and the agent's own
# tool rounds leaves comfortable room; unbounded, a large cluster produced a 3326-
# token request that llama-server rejected outright. Captions are the long part of
# a line, so they are clipped too — a caption is one sentence in practice, and the
# cap only bites on pathological ones.
MAX_SUMMARY_PHOTOS = 24
MAX_CAPTION_CHARS = 160

# Tag dimensions that describe the PICTURE, not the day. Fed "sharp, top-down,
# pastel" the model wrote them into the story ("a moment filled with sharp, joyful
# holiday cheer"), and they crowd out the four tag slots that carry meaning. The
# agent gets subject/setting/occasion/season/emotion/vibe/light only.
SKIPPED_TAG_DIMENSIONS = ("composition", "palette", "quality")


def compose_memory(
    conn: sqlite3.Connection,
    client: InferenceClient,
    model: str,
    owner_id: int,
    candidate: Candidate,
    *,
    signature: str,
    max_rounds: int = 3,
    use_captions: bool = True,
) -> Memory | None:
    """Curate + narrate one candidate via a bounded agent loop (§11, §9.1).

    Returns a Memory when the agent keeps the candidate, or None when it skips
    it, drops it below two photos, or emits output we cannot use.
    """
    facts = _cluster_facts(conn, owner_id, candidate.photo_ids)
    messages: list[ChatMessage] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": facts + _summarise(conn, owner_id, candidate.photo_ids)},
    ]
    for round_no in range(max_rounds + 1):
        force_answer = round_no == max_rounds
        turn = _complete(client, model, messages, force_answer)
        if turn is None:
            return None
        if turn.get("action") == "expand" and not force_answer:
            messages.append({"role": "assistant", "content": json.dumps(turn)})
            messages.append(
                {"role": "user", "content": _run_tool(conn, owner_id, turn, use_captions)}
            )
            continue
        if turn.get("action") != "answer" or not turn.get("keep"):
            return None
        dropped = set(turn.get("drop_photo_ids") or [])
        kept = [pid for pid in candidate.photo_ids if pid not in dropped]
        title = (turn.get("title") or "").strip()
        description = (turn.get("description") or "").strip()
        if len(kept) < 2 or not title or not description:
            return None
        return Memory(title, description, kept, signature)
    return None


def _summarise(
    conn: sqlite3.Connection, owner_id: int, photo_ids: list[int], *, limit: int = MAX_SUMMARY_PHOTOS
) -> str:
    """One compact line per photo: id, date, place, caption, top tags (~40 tokens).

    **Capped at `limit` lines.** A seed candidate is every photo in a time/place
    cluster and has no upper bound — a busy day can hold a hundred. At ~40 tokens
    each that overruns the jetson's 2048-token context and `llama-server` rejects
    the whole request ("request (3326 tokens) exceeds the available context size"),
    which used to abort the entire rebuild. The agent only needs enough of the
    cluster to judge whether it is one coherent memory, so the summary is truncated
    and says so — the omitted photos are still kept in the memory, they just do not
    each get a line in the prompt.
    """
    lines: list[str] = []
    omitted = max(0, len(photo_ids) - limit)
    shown = photo_ids[:limit]
    places = _place_names(conn, shown)
    for photo_id in shown:
        photo = conn.execute(
            "SELECT shot_at, caption FROM photos WHERE id = ? AND owner_id = ?",
            (photo_id, owner_id),
        ).fetchone()
        if photo is None:
            continue
        place = places.get(photo_id, "place unknown")
        tags = ", ".join(
            r["label"] for r in conn.execute(
                "SELECT t.label FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
                f" WHERE pt.photo_id = ? AND t.dimension NOT IN ({_marks(list(SKIPPED_TAG_DIMENSIONS))})"
                " ORDER BY pt.score DESC LIMIT 4",
                (photo_id, *SKIPPED_TAG_DIMENSIONS),
            )
        )
        caption = (photo["caption"] or "")[:MAX_CAPTION_CHARS]
        lines.append(
            f"[{photo_id}] {photo['shot_at'] or 'no date'} · {place} · "
            f"{caption}" + (f" · tags: {tags}" if tags else "")
        )
    summary = "Photos:\n" + "\n".join(lines)
    if omitted:
        # Say it outright: the agent must not read the truncation as "the cluster
        # is only this big" and judge coherence on a partial view.
        summary += f"\n(+{omitted} more photos in this cluster, not listed)"
    return summary


def _cluster_facts(conn: sqlite3.Connection, owner_id: int, photo_ids: list[int]) -> str:
    """The 'When / Where / Camera' spine of the whole cluster, from EXIF (§6.2).

    The per-photo lines say what is IN each frame; this says what the day WAS —
    the date(s), the weekday and the part of day, the town, the camera. It is read
    from the EXIF facets the ingest stage already derives, which the agent
    otherwise never sees, and it is what turns "photos of a street" into "a Sunday
    afternoon in Tbilisi". One line per cluster, so it costs ~30 tokens.
    """
    dates = sorted(
        {
            row["shot_at"][:10]
            for row in conn.execute(
                f"SELECT shot_at FROM photos WHERE owner_id = ?"
                f" AND id IN ({_marks(photo_ids)}) AND shot_at IS NOT NULL",
                (owner_id, *photo_ids),
            )
        }
    )
    facets = _facet_values(conn, photo_ids, ("weekday", "time_of_day", "camera_model"))
    parts = []
    if dates:
        when = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
        if len(dates) == 1:
            # A single day earns its weekday and its part of the day; a multi-day
            # cluster does not — "Friday, afternoon" would be a lie about the rest.
            when += "".join(
                f", {facets[key][0]}" for key in ("weekday", "time_of_day") if facets.get(key)
            )
        parts.append(f"When: {when}")
    places = sorted(set(_place_names(conn, photo_ids).values()))
    if places:
        parts.append(f"Where: {'; '.join(places[:3])}")
    if facets.get("camera_model"):
        parts.append(f"Camera: {'; '.join(facets['camera_model'][:2])}")
    return ("Facts — " + " · ".join(parts) + "\n") if parts else ""


def _marks(values: list[int]) -> str:
    return ",".join("?" * len(values))


def _facet_values(
    conn: sqlite3.Connection, photo_ids: list[int], keys: tuple[str, ...]
) -> dict[str, list[str]]:
    """key -> its distinct values across the cluster, most common first."""
    if not photo_ids:
        return {}
    rows = conn.execute(
        f"SELECT key, value_text, count(*) AS n FROM photo_facets"
        f" WHERE photo_id IN ({_marks(photo_ids)})"
        f" AND key IN ({','.join('?' * len(keys))}) AND value_text IS NOT NULL"
        " AND value_text != '' GROUP BY key, value_text ORDER BY n DESC",
        (*photo_ids, *keys),
    )
    values: dict[str, list[str]] = {}
    for row in rows:
        values.setdefault(row["key"], []).append(row["value_text"])
    return values


def _place_names(conn: sqlite3.Connection, photo_ids: list[int]) -> dict[int, str]:
    """photo_id -> "Borjomi, Georgia" from the reverse-geocoded facets (§6.2, §11).

    **The agent is never given raw coordinates.** The facets stage already resolves
    every GPS point to a real place offline, and a small model cannot: fed
    `41.68,44.86` it wrote titles like "Activities in a location on December 1" and
    descriptions ending "at coordinates 42.17, 42.93". Design §11 is explicit that a
    place is a name a person recognises and bare lat/long belongs on `/photo` only,
    so a photo whose place cannot be named is simply "place unknown" here.
    """
    if not photo_ids:
        return {}
    marks = ",".join("?" * len(photo_ids))
    rows = conn.execute(
        "SELECT photo_id, key, value_text FROM photo_facets"
        f" WHERE photo_id IN ({marks}) AND key IN ('place_city', 'place_country')",
        tuple(photo_ids),
    )
    parts: dict[int, dict[str, str]] = {}
    for row in rows:
        if row["value_text"]:
            parts.setdefault(row["photo_id"], {})[row["key"]] = row["value_text"]
    names: dict[int, str] = {}
    for photo_id, place in parts.items():
        ordered = [place[key] for key in ("place_city", "place_country") if key in place]
        names[photo_id] = ", ".join(ordered)
    return names


def _run_tool(
    conn: sqlite3.Connection, owner_id: int, turn: dict, use_captions: bool = True
) -> str:
    """Run one read-only expansion tool; always returns a short text block."""
    tool = turn.get("tool")
    photo_id = turn.get("photo_id")
    if tool == "similar" and isinstance(photo_id, int):
        ids = [
            r["id"]
            for r in similar_photos(conn, owner_id, photo_id, k=5, use_captions=use_captions)
        ]
        return "Similar photos: " + (_summarise(conn, owner_id, ids) if ids else "(none)")
    if tool == "facets" and isinstance(photo_id, int):
        facts = [
            f"{r['key']}={r['value_text'] if r['value_text'] is not None else r['value_num']}"
            for r in conn.execute(
                "SELECT key, value_text, value_num FROM photo_facets WHERE photo_id = ?",
                (photo_id,),
            )
        ]
        return f"Facets for [{photo_id}]: " + (", ".join(facts) or "(none)")
    if tool == "nearby" and isinstance(photo_id, int):
        near = [
            r["id"] for r in conn.execute(
                "SELECT p2.id FROM photos p1 JOIN photos p2 ON p2.owner_id = p1.owner_id"
                " WHERE p1.id = ? AND p2.id != p1.id AND p2.shot_at IS NOT NULL"
                " AND abs(julianday(p2.shot_at) - julianday(p1.shot_at)) < 0.25"
                " ORDER BY p2.shot_at LIMIT 5",
                (photo_id,),
            )
        ]
        return "Nearby in time: " + (_summarise(conn, owner_id, near) if near else "(none)")
    return "(no result)"


def _complete(
    client: InferenceClient, model: str, messages: list[ChatMessage], force_answer: bool
) -> dict | None:
    """One model turn -> parsed JSON dict, or None if the output is unusable.

    No strict json_schema (Ollama's strict mode is unreliable with the mixed
    expand/answer shape); the reply is expected to be a bare JSON object and
    parsed leniently. Anything unparseable skips the candidate (§11).

    A FAILED CALL SKIPS THIS CANDIDATE — it never aborts the build. The catch used
    to list only the parse errors, so an `httpx.HTTPStatusError` (a 500 from
    `/text/complete`, e.g. a prompt over the context limit) propagated out through
    `build_memories` and killed the background thread: one bad cluster and the
    whole rebuild died with nothing stored. The failure is logged rather than
    swallowed silently, because a rebuild that quietly keeps zero memories is
    indistinguishable from a library with no memories in it.
    """
    turn_messages = messages
    if force_answer:
        turn_messages = [
            *messages,
            {"role": "user", "content": "Answer now with a keep/skip decision (action=answer)."},
        ]
    try:
        raw = client.complete(model, turn_messages, timeout=60.0)
    except Exception:  # one candidate must never abort the build
        logger.exception("memory composition call failed; skipping this candidate")
        return None
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return None
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError, KeyError):
        return None
