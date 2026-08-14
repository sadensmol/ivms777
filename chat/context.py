import sqlite3

# EXIF facts worth carrying into an answer (§10 step 3); order is the display order.
_FACT_KEYS = ("camera_model", "lens", "iso", "aperture", "shutter_speed",
              "focal_length", "place_city", "place_country")


def build_context(conn: sqlite3.Connection, photo_ids: list[int]) -> str:
    """A compact, grounded block per photo for the chat prompt (§10 step 3)."""
    if not photo_ids:
        return "No photos matched."
    blocks: list[str] = []
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
        if photo["shot_at"]:
            parts.append(f"date: {photo['shot_at']}")
        if photo["caption"]:
            parts.append(f"caption: {photo['caption']}")
        if tags:
            parts.append("tags: " + ", ".join(tags))
        if photo["camera"]:
            parts.append(f"camera: {photo['camera']}")
        facts_line = ", ".join(f"{k}={facts[k]}" for k in _FACT_KEYS if k in facts)
        if facts_line:
            parts.append(facts_line)
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)
