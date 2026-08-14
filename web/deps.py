import sqlite3
from dataclasses import dataclass

from config import Settings
from storage.local import LocalStorage


@dataclass
class AppContext:
    settings: Settings
    conn: sqlite3.Connection
    derived: LocalStorage
    originals: LocalStorage


def build_context(settings: Settings) -> AppContext:
    from db.connection import connect, migrate

    conn = connect(settings.db_path)
    migrate(conn)
    return AppContext(
        settings=settings,
        conn=conn,
        derived=LocalStorage(settings.thumb_dir),
        originals=LocalStorage(settings.originals_dir),
    )
