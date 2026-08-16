import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ingest.jobs import STAGES, claim_next, complete, fail, requeue_running
from ingest.thumbs import make_thumbnails
from storage.base import Storage


class StageHandler(Protocol):
    def __call__(self, conn: sqlite3.Connection, photo_id: int) -> None: ...


class Preempted(Exception):
    """An interactive workload asked ingest to yield the models (design §8.1)."""


def drain(
    conn: sqlite3.Connection,
    handlers: dict[str, StageHandler],
    stages: tuple[str, ...] = STAGES,
    *,
    should_preempt: Callable[[], bool] = lambda: False,
) -> dict[str, int]:
    completed: dict[str, int] = {}
    for stage in stages:
        handler = handlers.get(stage)
        if handler is None:
            continue
        done = 0
        attempted: set[int] = set()
        while True:
            if should_preempt():
                raise Preempted()  # yield BEFORE claiming another photo (§8.1)
            photo_id = claim_next(conn, stage, exclude=attempted)
            if photo_id is None:
                break
            attempted.add(photo_id)
            try:
                handler(conn, photo_id)
            except Preempted:
                # A mid-photo yield (§8.1): revert the claimed job to pending (not a
                # failed attempt) so it re-runs next pass, then propagate the yield.
                requeue_running(conn, photo_id, stage)
                raise
            except Exception as error:  # noqa: BLE001 - one bad file must never stall the queue
                fail(conn, photo_id, stage, str(error))
            else:
                complete(conn, photo_id, stage)
                done += 1
        completed[stage] = done
    return completed


def thumbnail_handler(
    originals: Storage,
    derived: Storage,
    grid_px: int,
    detail_px: int,
) -> StageHandler:
    def handle(conn: sqlite3.Connection, photo_id: int) -> None:
        row = conn.execute(
            "SELECT storage_key, content_hash FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
        source: Path | None = originals.local_path(row["storage_key"])
        if source is None or not source.is_file():
            raise FileNotFoundError(row["storage_key"])
        key = make_thumbnails(source, row["content_hash"], derived, grid_px, detail_px)
        conn.execute("UPDATE photos SET thumb_key = ? WHERE id = ?", (key, photo_id))

    return handle
