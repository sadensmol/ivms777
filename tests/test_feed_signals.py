"""Per-shot signals for the director (design §11.1) — pure SQL, no model."""

import pytest

from albums.memory_store import Memory, replace_memories
from embedding.store import write_vector
from feed.signals import is_portrait, read_signals
from tests.factories import add_photo


def _tag(conn, photo_id, dimension, label, score, source="siglip"):
    conn.execute("INSERT OR IGNORE INTO tags(dimension, label) VALUES (?, ?)",
                 (dimension, label))
    tag_id = conn.execute("SELECT id FROM tags WHERE dimension = ? AND label = ?",
                          (dimension, label)).fetchone()["id"]
    conn.execute("INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?,?,?,?)",
                 (photo_id, tag_id, score, source))


@pytest.fixture
def group(conn):
    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg", width=4000,
              height=3000, caption="a beach", ai_title="Beach at noon",
              shot_at="2025-07-12T10:00:00")
    add_photo(conn, photo_id=2, content_hash="h2", thumb_key="2.jpg", width=3000,
              height=4000, caption="a portrait", ai_title="Girl in the sun",
              shot_at="2025-07-12T10:01:00")
    _tag(conn, 1, "palette", "warm", 0.7)
    _tag(conn, 1, "palette", "cool", 0.2)       # loses: lower score
    _tag(conn, 1, "quality", "sharp", 0.95)
    _tag(conn, 2, "subject", "person", 0.83)
    conn.execute("INSERT INTO photo_facets(photo_id, key, value_text) VALUES (?,?,?)",
                 (1, "place_city", "Borjomi"))
    write_vector(conn, 1, [0.1] * 1152)
    replace_memories(conn, 1, [Memory("Beach day", "Warm.", [1, 2], "sig")])
    return conn.execute("SELECT id FROM groups WHERE kind='memory'").fetchone()["id"]


def test_returns_one_signal_per_photo_in_rank_order(conn, group):
    assert [s.photo_id for s in read_signals(conn, group)] == [1, 2]


def test_top_scoring_label_wins_per_dimension(conn, group):
    first = read_signals(conn, group)[0]
    assert first.tags["palette"] == "warm"      # 0.7 beats cool's 0.2
    assert first.scores["palette"] == pytest.approx(0.7)
    assert first.tags["quality"] == "sharp"


def test_the_same_label_from_two_sources_is_not_counted_twice(conn, group):
    # `source` is part of photo_tags' primary key, so one label can appear twice.
    _tag(conn, 1, "palette", "warm", 0.9, source="manual")
    first = read_signals(conn, group)[0]
    assert first.tags["palette"] == "warm"
    assert first.scores["palette"] == pytest.approx(0.9)   # MAX across sources


def test_place_comes_from_the_facet(conn, group):
    assert read_signals(conn, group)[0].place == "Borjomi"
    assert read_signals(conn, group)[1].place is None


def test_orientation_is_read_from_the_stored_dimensions(conn, group):
    landscape, portrait = read_signals(conn, group)
    assert is_portrait(landscape) is False
    assert is_portrait(portrait) is True


def test_the_caption_prefers_the_ai_title(conn, group):
    assert read_signals(conn, group)[0].caption == "Beach at noon"


def test_vectors_are_attached_when_present(conn, group):
    signals = read_signals(conn, group)
    assert signals[0].vector is not None and len(signals[0].vector) == 1152
    assert signals[1].vector is None            # never embedded


def test_photos_without_a_thumbnail_are_skipped(conn, group):
    conn.execute("UPDATE photos SET thumb_key = NULL WHERE id = 2")
    assert [s.photo_id for s in read_signals(conn, group)] == [1]


def test_an_empty_memory_returns_nothing(conn, group):
    conn.execute("DELETE FROM group_photos WHERE group_id = ?", (group,))
    assert read_signals(conn, group) == []


def test_the_rich_description_is_carried_separately_from_the_label(conn, group):
    # The story pass needs "Three individuals posing behind a colourful arrangement of
    # flowers…", not "People by Flower Arch" — a film cannot be written from a label.
    conn.execute("UPDATE photos SET ai_description = ? WHERE id = 1",
                 ("Three people pose behind a colourful arrangement of flowers.",))
    first = read_signals(conn, group)[0]
    assert first.caption == "Beach at noon"                      # short, for the screen
    assert first.description.startswith("Three people pose")     # rich, for the story


def test_the_description_falls_back_to_the_caption(conn, group):
    conn.execute("UPDATE photos SET ai_description = NULL WHERE id = 1")
    assert read_signals(conn, group)[0].description == "a beach"


def test_the_digest_says_what_the_whole_memory_is(conn):
    # The per-shot briefs say what each photograph contains; the digest says what the
    # DAY was. A writer shown only captions writes captions (§11.1).
    from feed.signals import ShotSignal, digest

    signals = [
        ShotSignal(photo_id=i, width=4000, height=3000,
                   shot_at=f"2023-01-01T18:5{i}:00", caption="c", description="d",
                   place="Tbilisi",
                   tags={"vibe": "festive", "occasion": "holiday",
                         "season_weather": "snow", "light": "low light"},
                   scores={}, vector=None, camera="iPhone SE",
                   facets={"place_country": "Georgia"})
        for i in range(1, 4)
    ]
    text = digest(signals)
    assert "3 photographs" in text
    assert "Tbilisi" in text and "Georgia" in text
    assert "holiday" in text and "festive" in text and "snow" in text
    assert "2023-01-01" in text
    assert "iPhone SE" in text


def test_the_digest_ranks_by_how_common_a_label_is(conn):
    from feed.signals import ShotSignal, digest

    def sig(pid, vibe):
        return ShotSignal(photo_id=pid, width=1, height=1, shot_at=None, caption="",
                          description="", place=None, tags={"vibe": vibe}, scores={},
                          vector=None)

    text = digest([sig(1, "calm"), sig(2, "calm"), sig(3, "chaotic")])
    assert text.index("calm") < text.index("chaotic")


def test_an_empty_memory_digests_without_raising(conn):
    from feed.signals import digest

    assert digest([])


def test_the_brief_carries_every_signal_the_library_has(conn):
    # The conductor answers "which photograph is this sentence about?" from this line,
    # so everything the library knows has to be in it.
    from feed.signals import ShotSignal

    signal = ShotSignal(
        photo_id=1, width=3000, height=4000, shot_at="2023-01-01T18:58:19",
        caption="Cat on a Sofa", description="A ginger cat asleep on a green sofa.",
        place="Tbilisi",
        tags={"vibe": "calm", "subject": "animal", "occasion": "everyday"},
        scores={}, vector=None, camera="iPhone SE",
        facets={"focal_length": "3.99", "aperture": "1.8", "iso": "640"},
    )
    brief = signal.brief()
    assert "ginger cat" in brief
    assert "subject=animal" in brief and "vibe=calm" in brief
    assert "place=Tbilisi" in brief
    assert "2023-01-01" in brief
    assert "iPhone SE" in brief and "aperture=1.8" in brief
    assert "portrait" in brief
