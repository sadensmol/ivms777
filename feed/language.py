"""The film's language (design §11.1) — `en` (default) or `ru`.

It selects the language the story pass WRITES in, which is what makes it the
translation as well: the shot lines and the narration are generated in that language
rather than produced in English and translated afterwards.
"""

import sqlite3

LANGUAGES: tuple[str, ...] = ("en", "ru")
DEFAULT = "en"
SETTING_KEY = "feed_language"

DISPLAY: dict[str, str] = {"en": "English", "ru": "Русский"}


def get_language(conn: sqlite3.Connection, owner_id: int) -> str:
    """The stored choice, or the default.

    A stored value that is no longer a known language degrades to the default rather
    than raising — a downgraded install must still play films.
    """
    from db.settings import get_setting

    stored = get_setting(conn, owner_id, SETTING_KEY)
    return stored if stored in LANGUAGES else DEFAULT


def set_language(conn: sqlite3.Connection, owner_id: int, language: str) -> None:
    from db.settings import set_setting

    if language not in LANGUAGES:
        raise ValueError(f"unknown language: {language}")
    set_setting(conn, owner_id, SETTING_KEY, language)
