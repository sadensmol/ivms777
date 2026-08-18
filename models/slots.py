"""Which model each slot holds right now (design §4.1).

Resolution order is **stored choice → env override → profile default**. `app` owns
the stored choice (`app_settings`); the `models` service holds no database, so it
calls the same resolver with `conn=None` and gets the profile defaults.

A stored or configured key that is not a catalog entry *for that slot on this
profile* is ignored rather than raising: a downgraded install, a removed catalog
entry, or a leftover `IVMS777_CAPTION_MODEL=ollama-…` must still boot on the
defaults. Silently doing the right thing beats a service that will not start.

No torch, no HTTP — `app`, `worker` AND the `models` service import this. `db` and
`ingest` are therefore imported lazily, inside the `conn`-taking paths only: the
service ships without them (Dockerfile.models), and a top-level import would stop it
booting.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from models import catalog
from models.catalog import ModelEntry

SETTING_PREFIX = "model_slot."

# The env/config field that overrides each slot when nothing is stored. The value
# must be a catalog key; anything else (a bare HF id, an old Ollama tag) is ignored.
_ENV_FIELD: dict[str, str] = {
    "image_embed": "embed_model_name",
    "text_embed": "text_embed_model",
    "caption": "caption_model",
    "planner": "planner_model",
}


def setting_key(slot: str) -> str:
    return f"{SETTING_PREFIX}{slot}"


def _offered(slot: str, key: str | None, profile: str) -> bool:
    if not key or not catalog.has(slot, key):
        return False
    return profile in catalog.get(slot, key).profiles


def resolve_key(conn: sqlite3.Connection | None, settings, slot: str) -> str:
    profile = settings.profile
    if conn is not None and catalog.is_switchable(slot, profile):
        from db.settings import get_setting

        stored = get_setting(conn, settings.owner_id, setting_key(slot))
        if _offered(slot, stored, profile):
            return stored
    override = getattr(settings, _ENV_FIELD[slot], None)
    if _offered(slot, override, profile):
        return override
    return catalog.default_key(slot, profile)


def resolve_keys(conn: sqlite3.Connection | None, settings) -> dict[str, str]:
    return {slot: resolve_key(conn, settings, slot) for slot in catalog.SLOTS}


def resolve(conn: sqlite3.Connection | None, settings) -> dict[str, ModelEntry]:
    return {
        slot: catalog.get(slot, key) for slot, key in resolve_keys(conn, settings).items()
    }


@dataclass(frozen=True)
class SwitchResult:
    """What a switch did (or, from `preview`, would do) — the popup states this
    before it asks for confirmation (design §13)."""

    slot: str
    key: str
    photos_requeued: int
    stages: tuple[str, ...]
    vectors_dropped: bool


def _stages_for(slot: str) -> tuple[str, ...]:
    rng = catalog.INVALIDATES[slot]
    if rng is None:
        return ()
    from ingest.jobs import STAGES

    first, last = rng
    return STAGES[STAGES.index(first) : STAGES.index(last) + 1]


def _validate(settings, slot: str, key: str) -> ModelEntry:
    if slot not in catalog.SLOTS:
        raise ValueError(f"unknown slot: {slot}")
    if not catalog.is_switchable(slot, settings.profile):
        raise ValueError(f"{slot} is not switchable on {settings.profile}")
    if not _offered(slot, key, settings.profile):
        raise ValueError(f"{key} is not offered for {slot} on {settings.profile}")
    return catalog.get(slot, key)


def _photo_count(conn: sqlite3.Connection, owner_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM photos WHERE owner_id = ?", (owner_id,)
    ).fetchone()["n"]


def preview(conn: sqlite3.Connection, settings, slot: str, key: str) -> SwitchResult:
    """What switching would cost. Writes nothing."""
    from db.vectors import vec_dim

    entry = _validate(settings, slot, key)
    if resolve_key(conn, settings, slot) == key:
        return SwitchResult(slot, key, 0, (), False)
    stages = _stages_for(slot)
    return SwitchResult(
        slot=slot,
        key=key,
        photos_requeued=_photo_count(conn, settings.owner_id) if stages else 0,
        stages=stages,
        vectors_dropped=slot == "image_embed" and entry.dim != vec_dim(conn),
    )


def switch(conn: sqlite3.Connection, settings, slot: str, key: str) -> SwitchResult:
    """Point a slot at a different model and re-index what that invalidates
    (design §4.1).

    ONE transaction: the stored choice, the vector table, and the requeued jobs
    move together or not at all. The connection is autocommit
    (`isolation_level=None`), so the `BEGIN IMMEDIATE` here is what makes that
    true — without it a failure halfway leaves a library in a state the design
    says never exists (a new model against the old model's vectors).
    """
    from db.settings import set_setting
    from db.vectors import ensure_vec_dim
    from ingest.jobs import reprocess

    result = preview(conn, settings, slot, key)
    if resolve_key(conn, settings, slot) == key:
        return result

    conn.execute("BEGIN IMMEDIATE")
    try:
        set_setting(conn, settings.owner_id, setting_key(slot), key)
        if result.vectors_dropped:
            ensure_vec_dim(conn, catalog.get(slot, key).dim)
        if result.stages:
            reprocess(conn, settings.owner_id, result.stages[0], result.stages[-1])
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return result
