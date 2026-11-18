"""Face grouping (design §6, §8): nearest group by mean vector, one threshold, pinned faces
never move. `ingest/faces.py` over the real schema."""

import math

from ingest.faces import GroupMean, assign_to_group, nearest_group
from tests.factories import add_face, add_face_group, add_photo

THRESHOLD = 0.5
ROOT_HALF = math.sqrt(0.5)


def _group_of(conn, face_id: int) -> int | None:
    return conn.execute("SELECT group_id FROM faces WHERE id = ?", (face_id,)).fetchone()[0]


def test_the_closest_group_by_cosine_wins():
    groups = [GroupMean(group_id=1, mean=(1.0, 0.0, 0.0)),
              GroupMean(group_id=2, mean=(0.8, 0.6, 0.0))]
    assert nearest_group((0.9, math.sqrt(1 - 0.81), 0.0), groups, THRESHOLD) == 2


def test_a_similarity_exactly_at_the_threshold_joins():
    groups = [GroupMean(group_id=7, mean=(1.0, 0.0))]
    assert nearest_group((0.5, math.sqrt(0.75)), groups, 0.5) == 7


def test_the_similarity_is_the_cosine_of_an_unnormalised_mean():
    groups = [GroupMean(group_id=3, mean=(0.5, 0.5))]
    assert nearest_group((ROOT_HALF, ROOT_HALF), groups, 0.9) == 3


def test_a_new_face_joins_the_owners_group_with_the_closest_mean(conn):
    photo = add_photo(conn, content_hash="p1")
    near = add_face_group(conn)
    far = add_face_group(conn)
    add_face(conn, photo_id=photo, group_id=near, vector=(1.0, 0.0, 0.0, 0.0))
    add_face(conn, photo_id=photo, group_id=near, vector=(1.0, 0.0, 0.0, 0.0))
    add_face(conn, photo_id=photo, group_id=far, vector=(0.0, 1.0, 0.0, 0.0))
    new = add_face(conn, photo_id=photo, group_id=None, vector=(0.95, math.sqrt(1 - 0.9025), 0, 0))

    assert assign_to_group(conn, new, THRESHOLD) == near
    assert _group_of(conn, new) == near


def test_the_group_mean_decides_not_its_closest_member(conn):
    photo = add_photo(conn, content_hash="p1")
    mixed = add_face_group(conn)
    other = add_face_group(conn)
    add_face(conn, photo_id=photo, group_id=mixed, vector=(1.0, 0.0))
    add_face(conn, photo_id=photo, group_id=mixed, vector=(0.0, 1.0))
    add_face(conn, photo_id=photo, group_id=other, vector=(0.8, -0.6))
    new = add_face(conn, photo_id=photo, group_id=None, vector=(ROOT_HALF, ROOT_HALF))

    assert assign_to_group(conn, new, 0.9) == mixed


def test_a_face_under_the_threshold_of_every_group_starts_a_new_group(conn):
    photo = add_photo(conn, content_hash="p1")
    existing = add_face_group(conn)
    add_face(conn, photo_id=photo, group_id=existing, vector=(1.0, 0.0))
    new = add_face(conn, photo_id=photo, group_id=None, vector=(0.0, 1.0))

    group = assign_to_group(conn, new, THRESHOLD)

    assert group is not None and group != existing
    row = conn.execute("SELECT owner_id, person_id FROM face_groups WHERE id = ?",
                       (group,)).fetchone()
    assert (row["owner_id"], row["person_id"]) == (1, None)
    assert _group_of(conn, new) == group


def test_a_group_of_another_owner_is_never_joined(conn):
    mine = add_photo(conn, content_hash="p1", owner_id=1)
    theirs = add_photo(conn, content_hash="p2", owner_id=2)
    foreign = add_face_group(conn, owner_id=2)
    add_face(conn, photo_id=theirs, group_id=foreign, vector=(1.0, 0.0))
    new = add_face(conn, photo_id=mine, group_id=None, vector=(1.0, 0.0))

    group = assign_to_group(conn, new, THRESHOLD)

    assert group is not None and group != foreign
    assert conn.execute("SELECT owner_id FROM face_groups WHERE id = ?",
                        (group,)).fetchone()[0] == 1


def test_a_pinned_face_stays_while_an_unpinned_twin_joins(conn):
    photo = add_photo(conn, content_hash="p1")
    target = add_face_group(conn)
    own = add_face_group(conn)
    add_face(conn, photo_id=photo, group_id=target, vector=(1.0, 0.0))
    pinned = add_face(conn, photo_id=photo, group_id=own, vector=(0.96, 0.28), pinned=True)
    twin = add_face(conn, photo_id=photo, group_id=None, vector=(1.0, 0.0))

    assert assign_to_group(conn, twin, THRESHOLD) == target
    assert assign_to_group(conn, pinned, THRESHOLD) is None
    assert _group_of(conn, pinned) == own
