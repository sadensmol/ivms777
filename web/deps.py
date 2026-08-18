import sqlite3
import threading
from pathlib import Path

from config import Settings
from storage.local import LocalStorage


class AppContext:
    """Shared app state, with a **thread-local** database connection.

    FastAPI runs sync route handlers in a threadpool, so several threads touch
    the DB at once during an upload. A single sqlite3 connection is not safe
    under that: `check_same_thread=False` silences the guard, but the connection
    and its cursors keep per-object state that concurrent `execute()` calls
    corrupt — "bad parameter or other API misuse", plus phantom FK/UNIQUE
    errors. Each thread therefore gets its own connection to the same WAL file;
    SQLite serialises writers across connections via the busy timeout (§5).
    """

    def __init__(
        self,
        settings: Settings,
        db_path: Path,
        derived: LocalStorage,
        originals: LocalStorage,
    ) -> None:
        self.settings = settings
        self._db_path = db_path
        self.derived = derived
        self.originals = originals
        self._local = threading.local()

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            from db.connection import connect

            conn = connect(self._db_path)
            self._local.conn = conn
        return conn


def build_context(settings: Settings) -> AppContext:
    from db.connection import connect, migrate

    # Migrate once on a throwaway bootstrap connection; each thread opens its own
    # afterwards and reads the already-migrated schema from the file.
    boot = connect(settings.db_path)
    migrate(boot)
    _heal_vector_width(boot, settings)
    boot.close()
    return AppContext(
        settings=settings,
        db_path=settings.db_path,
        derived=LocalStorage(settings.thumb_dir),
        originals=LocalStorage(settings.originals_dir),
    )


def _heal_vector_width(conn: sqlite3.Connection, settings: Settings) -> None:
    """Make `photo_vec` match the SELECTED image embedder (design §4.1).

    A switch does this in one transaction, so normally there is nothing to heal.
    This is for the case where that transaction never finished — the process died
    mid-switch, or the stored slot was changed while this app was down. Left alone,
    every KNN query would fail (or worse, rank against a foreign space), so the
    table is rebuilt and the embed stage requeued, exactly as a switch would.
    """
    from db.vectors import ensure_vec_dim
    from ingest.jobs import reprocess
    from models.slots import resolve

    dim = resolve(conn, settings)["image_embed"].dim
    if ensure_vec_dim(conn, dim):
        reprocess(conn, settings.owner_id, "embed", "taxonomy")
