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

    def make_coordinator(self, client, holder: str):
        """Return a `NoopCoordinator` (design §8.1 stub, plan 15 task 7):
        app/worker hold no models anymore — `modelsvc` owns all loading and
        residency — so there is nothing left to coordinate. `client` and
        `holder` are accepted for signature compatibility with every
        call site (`web/app.py`, `ingest/cli.py`); neither is used."""
        from models.coordinator import NoopCoordinator

        return NoopCoordinator()


def build_context(settings: Settings) -> AppContext:
    from db.connection import connect, migrate

    # Migrate once on a throwaway bootstrap connection; each thread opens its own
    # afterwards and reads the already-migrated schema from the file.
    boot = connect(settings.db_path)
    migrate(boot)
    boot.close()
    return AppContext(
        settings=settings,
        db_path=settings.db_path,
        derived=LocalStorage(settings.thumb_dir),
        originals=LocalStorage(settings.originals_dir),
    )
