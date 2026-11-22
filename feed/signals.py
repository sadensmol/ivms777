"""What the library already knows about each shot (design §11.1).

Pure SQL, no model, no HTTP. Everything here was computed by the ingest pipeline long
before a film was asked for — the director's whole job is to read it well, not to add
a new analysis pass.
"""

import sqlite3
from dataclasses import dataclass

from embedding.store import read_vectors
from ingest.people import people_in_photos

# Every dimension the taxonomy produces. The director reads a handful of them; the
# CONDUCTOR reads all of them, because its job is to put the photograph of the cat under
# the sentence about the cat, and `subject`/`setting`/`occasion` are how it knows which
# photograph that is (§11.1).
DIMENSIONS: tuple[str, ...] = (
    "vibe", "palette", "light", "quality", "subject", "setting",
    "composition", "emotion", "occasion", "season_weather",
)

# EXIF facets worth putting in front of the conductor. Not the whole set: `megapixels`
# and `has_gps` say nothing about what a photograph IS, and every useless line in the
# brief costs attention that should go to the picture.
FACETS: tuple[str, ...] = (
    "time_of_day", "hour", "year", "month", "weekday",
    "place_city", "place_country", "focal_length", "aperture", "iso",
)


@dataclass(frozen=True)
class ShotSignal:
    photo_id: int
    width: int
    height: int
    shot_at: str | None
    caption: str        # `ai_title` — the short label, for on-screen use
    description: str    # `ai_description` — the full sentence, for the story pass
    place: str | None
    tags: dict[str, str]      # dimension -> the top-scoring label
    scores: dict[str, float]  # dimension -> that label's score
    vector: list[float] | None
    camera: str | None = None
    facets: dict[str, str] | None = None   # EXIF, as `FACETS` names them
    people: tuple[str, ...] = ()           # names of the people recognised in the photo (§11.1)

    def brief(self) -> str:
        """One line describing this photograph to the conductor.

        It exists so the model can answer "which picture is this sentence about?".
        Everything the library knows goes in — what is in the frame, where, when, how
        it was shot — because the alternative is showing the car while the voice talks
        about the cat.
        """
        parts = [self.description or self.caption or "(no description)"]
        labels = [f"{dim}={self.tags[dim]}" for dim in DIMENSIONS if dim in self.tags]
        if labels:
            parts.append(" ".join(labels))
        where = self.place or (self.facets or {}).get("place_city")
        if where:
            parts.append(f"place={where}")
        when = self.shot_at or (self.facets or {}).get("time_of_day")
        if when:
            parts.append(f"when={when}")
        shot_with = [
            f"{key}={(self.facets or {})[key]}"
            for key in ("focal_length", "aperture", "iso")
            if (self.facets or {}).get(key)
        ]
        if self.camera:
            shot_with.insert(0, f"camera={self.camera}")
        if shot_with:
            parts.append(" ".join(shot_with))
        if self.people:
            parts.append(f"people={', '.join(self.people)}")
        parts.append("portrait" if is_portrait(self) else "landscape")
        return " · ".join(parts)


def digest(signals: list["ShotSignal"], *, guesses: bool = True) -> str:
    """What the MEMORY as a whole is, in a few lines.

    `guesses=False` keeps only the hard facts — count, time, place, camera — and drops
    the zero-shot labels. The story WRITER gets that version: the labels are guesses at
    0.2-0.6 confidence, and it narrated them as facts ("it rained", "it was boring")
    over photographs that showed neither (§11.1).

    The per-shot briefs say what each photograph contains; this says what the day was.
    A writer given only three captions writes three captions — it needs to know that
    these are all one snowy holiday evening in Tbilisi in January before it can write
    about the evening rather than about the pictures.

    Pure: labels in, prose out. No model, no I/O.
    """
    if not signals:
        return "(an empty memory)"
    lines = [f"{len(signals)} photographs."]
    stamps = sorted(s.shot_at for s in signals if s.shot_at)
    if stamps:
        span = stamps[0][:16].replace("T", " ")
        lines.append(f"Taken {span}" + (f" through {stamps[-1][11:16]}"
                                        if stamps[-1][:10] == stamps[0][:10]
                                        else f" through {stamps[-1][:10]}") + ".")
    places = _common([s.place or (s.facets or {}).get("place_city") for s in signals])
    countries = _common([(s.facets or {}).get("place_country") for s in signals])
    where = ", ".join(places + [c for c in countries if c not in places])
    if where:
        lines.append(f"Where: {where}.")
    for dimension, label in () if not guesses else (
        ("occasion", "What kind of occasion"), ("vibe", "Mood"),
        ("emotion", "Feeling"), ("season_weather", "Season and weather"),
        ("light", "Light"), ("setting", "Setting"), ("subject", "Mostly of"),
        ("palette", "Colour"),
    ):
        values = _common([s.tags.get(dimension) for s in signals])
        if values:
            lines.append(f"{label}: {', '.join(values)}.")
    cameras = _common([s.camera for s in signals])
    if cameras:
        lines.append(f"Shot on: {', '.join(cameras)}.")
    return "\n".join(lines)


def _common(values: list[str | None], limit: int = 3) -> list[str]:
    """The most frequent non-empty values, commonest first, ties broken by name so a
    memory always digests the same way."""
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked[:limit]]


def is_portrait(signal: ShotSignal) -> bool:
    """Taller than wide. Drives the motion axis: the camera should travel along the
    photo's LONG edge, or the move reveals nothing (§11.1)."""
    return signal.height > signal.width


def read_signals(conn: sqlite3.Connection, group_id: int) -> list[ShotSignal]:
    rows = conn.execute(
        "SELECT p.id, p.width, p.height, p.shot_at,"
        " COALESCE(p.ai_title, p.caption, '') AS caption,"
        # The story pass needs the RICH description, not the four-word label:
        # "People by Flower Arch" cannot carry a film, while "Three individuals are
        # posing behind a colorful arrangement of flowers…" can.
        " COALESCE(p.ai_description, p.caption, '') AS description, p.camera, g.owner_id"
        " FROM group_photos gp JOIN photos p ON p.id = gp.photo_id"
        " JOIN groups g ON g.id = gp.group_id"
        " WHERE gp.group_id = ? AND p.thumb_key IS NOT NULL"
        " ORDER BY gp.rank",
        (group_id,),
    ).fetchall()
    ids = [int(r["id"]) for r in rows]
    if not ids:
        return []
    tags, scores = _top_tags(conn, ids)
    places = _places(conn, ids)
    facets = _facets(conn, ids)
    vectors = read_vectors(conn, ids)
    people = people_in_photos(conn, int(rows[0]["owner_id"]), ids)
    return [
        ShotSignal(
            photo_id=int(r["id"]),
            width=int(r["width"] or 0),
            height=int(r["height"] or 0),
            shot_at=r["shot_at"],
            caption=r["caption"],
            description=r["description"],
            place=places.get(int(r["id"])),
            tags=tags.get(int(r["id"]), {}),
            scores=scores.get(int(r["id"]), {}),
            vector=vectors.get(int(r["id"])),
            camera=r["camera"],
            facets=facets.get(int(r["id"]), {}),
            people=tuple(people.get(int(r["id"]), [])),
        )
        for r in rows
    ]


def _top_tags(
    conn: sqlite3.Connection, ids: list[int]
) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, float]]]:
    """The single best label per dimension per photo.

    A photo carries several labels in one dimension with different scores, and the
    same (dimension, label) can appear MORE THAN ONCE because `source` is part of
    `photo_tags`' primary key. So this takes MAX(score) per label first, then the
    highest-scoring label per dimension.
    """
    placeholders = ", ".join("?" for _ in ids)
    dims = ", ".join("?" for _ in DIMENSIONS)
    rows = conn.execute(
        "SELECT pt.photo_id, t.dimension, t.label, MAX(pt.score) AS score"
        " FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
        f" WHERE pt.photo_id IN ({placeholders}) AND t.dimension IN ({dims})"
        " GROUP BY pt.photo_id, t.dimension, t.label"
        " ORDER BY pt.photo_id, t.dimension, score DESC",
        (*ids, *DIMENSIONS),
    ).fetchall()
    tags: dict[int, dict[str, str]] = {}
    scores: dict[int, dict[str, float]] = {}
    for row in rows:
        photo_id, dimension = int(row["photo_id"]), row["dimension"]
        # Rows arrive score-descending within a dimension, so the FIRST one wins.
        if dimension in tags.setdefault(photo_id, {}):
            continue
        tags[photo_id][dimension] = row["label"]
        scores.setdefault(photo_id, {})[dimension] = float(row["score"])
    return tags, scores


def _facets(conn: sqlite3.Connection, ids: list[int]) -> dict[int, dict[str, str]]:
    """The EXIF the conductor is shown, as text. One query, `FACETS` only."""
    placeholders = ", ".join("?" for _ in ids)
    keys = ", ".join("?" for _ in FACETS)
    out: dict[int, dict[str, str]] = {}
    for row in conn.execute(
        "SELECT photo_id, key, COALESCE(value_text, CAST(value_num AS TEXT)) AS value"
        f" FROM photo_facets WHERE photo_id IN ({placeholders}) AND key IN ({keys})",
        (*ids, *FACETS),
    ):
        if row["value"]:
            out.setdefault(int(row["photo_id"]), {})[row["key"]] = str(row["value"])
    return out


def _places(conn: sqlite3.Connection, ids: list[int]) -> dict[int, str]:
    placeholders = ", ".join("?" for _ in ids)
    return {
        int(r["photo_id"]): r["value_text"]
        for r in conn.execute(
            "SELECT photo_id, value_text FROM photo_facets"
            f" WHERE key = 'place_city' AND photo_id IN ({placeholders})",
            ids,
        )
        if r["value_text"]
    }
