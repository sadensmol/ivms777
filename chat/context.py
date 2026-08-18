import sqlite3

# EXIF facts worth carrying into an answer (§10 step 3); order is the display order.
_FACT_KEYS = ("camera_model", "lens", "iso", "aperture", "shutter_speed",
              "focal_length", "place_city", "place_country")

# The prompt must fit the SMALLEST context we run: jetson's `llm_ctx: 2048` (§3.1,
# kept small so gemma's resident footprint fits the 8 GB board). A photo block is
# ~50-80 tokens, so 12 of them plus the system prompt and the question overran it
# and `llama-server` rejected the request outright — "request (2323 tokens) exceeds
# the available context size (2048 tokens)" — which kills the answer stream
# mid-flight. ~3500 chars ≈ 900 tokens leaves room for the system prompt, the
# question, and the generated answer.
MAX_CONTEXT_CHARS = 3500
# A caption is one sentence in practice; the cap only bites on pathological ones.
MAX_CAPTION_CHARS = 200


def build_context(
    conn: sqlite3.Connection,
    photo_ids: list[int],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
    strengths: dict[int, float] | None = None,
) -> str:
    """A compact, grounded block per photo for the chat prompt (§10 step 3).

    **Bounded.** Blocks are added best-match-first until `max_chars` is reached, and
    the rest are dropped with a note — the model can only cite what it can see, so a
    truncated tail costs recall on the weakest hits, whereas an overflowed prompt
    costs the entire answer.
    """
    if not photo_ids:
        return "No photos matched."
    blocks: list[str] = []
    used = 0
    shown = 0
    for photo_id in photo_ids:
        photo = conn.execute(
            "SELECT id, shot_at, camera, caption FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
        if photo is None:
            continue
        facts = {
            row["key"]: (
                row["value_text"] if row["value_text"] is not None else row["value_num"]
            )
            for row in conn.execute(
                "SELECT key, value_text, value_num FROM photo_facets WHERE photo_id = ?",
                (photo_id,),
            )
        }
        tags = [
            row["label"] for row in conn.execute(
                "SELECT t.label FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
                " WHERE pt.photo_id = ? ORDER BY pt.score DESC LIMIT 6",
                (photo_id,),
            )
        ]
        parts = [f"[photo:{photo['id']}]"]
        if strengths is not None and photo_id in strengths:
            # What the IMAGE model saw, which the caption often cannot say. Rank
            # plus strength RELATIVE to the best hit — an absolute score would be
            # meaningless here (see search_photos_scored) and, at SigLIP's true
            # scale of ~1 %, would read as "no match" and get the photo discarded.
            parts.append(
                f"visual match: rank {shown + 1}, "
                f"{round(strengths[photo_id] * 100)}% as close as the best match"
            )
        if photo["shot_at"]:
            parts.append(f"date: {photo['shot_at']}")
        if photo["caption"]:
            parts.append(f"caption: {photo['caption'][:MAX_CAPTION_CHARS]}")
        if tags:
            parts.append("tags: " + ", ".join(tags))
        if photo["camera"]:
            parts.append(f"camera: {photo['camera']}")
        facts_line = ", ".join(f"{k}={facts[k]}" for k in _FACT_KEYS if k in facts)
        if facts_line:
            parts.append(facts_line)
        block = "\n".join(parts)
        # Always emit at least one block: a single oversized photo must still produce
        # a usable prompt rather than an empty one.
        if shown and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block) + 2
        shown += 1
    omitted = len(photo_ids) - shown
    context = "\n\n".join(blocks)
    if omitted > 0:
        context += f"\n\n(+{omitted} lower-ranked photos not shown)"
    return context
