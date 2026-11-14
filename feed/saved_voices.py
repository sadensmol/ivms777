"""Saved voices (design §6, §11.1): a person's voice, named by the user, and its takes
(the example clips with the words speech recognition heard in them). Films and tests of
the voice speak from ONE take: the newest take of the language whose words were heard
(`film_take`). The voice models a bake stored stay as data for the bake check.

User data, not catalog data: a saved voice is never a catalog Speaker. The clips, the
conditioning blobs, the voice model files and the test WAVs live in `data_dir/voices`,
content-addressed, so two voices may hold the same file and a deleted clip may be added
again. Hence the two rules of this module, for every kind of voice file:

- Deletion is an outbox (`voice_deletions`, the §3.2c pattern). A request only records
  the intent; `process_voice_deletions` deletes the rows, then the files, then the intent.
- A voice file is written behind a durable write-ahead row in `voice_file_deletions`, and
  is removed only by the drain, only when no row references its key. The ref check and
  every write run under the same `BEGIN IMMEDIATE` write lock.

No models, no HTTP: this module is a thin client (CLAUDE.md § "One model process").
"""

import hashlib
import io
import json
import logging
import secrets
import sqlite3
import wave
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NamedTuple

import numpy as np

from feed.language import LANGUAGES
from feed.story import _LANGUAGE_NAME as language_names
from models import catalog
from storage.base import Storage
from storage.keys import content_key
from storage.wav import decode_wav, duration_seconds

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class KitText:
    id: str      # "<lang>-<style>-<n>", or one of the four reading-text ids (D45)
    style: str   # a KIT_STYLES key
    text: str    # the exact words to read; numbers written as words


# style -> (label, instruction shown over the text) (D70)
KIT_STYLES: dict[str, tuple[str, str]] = {
    "normal": ("Normal", "Read in your normal voice."),
    "slow": ("Slow", "Read slowly and clearly, with short pauses."),
    "fast": ("Fast", "Read quickly, as if you are in a hurry, but keep every word clear."),
    "calm": ("Calm", "Read calmly and softly."),
    "happy": ("Happy", "Read with joy, as good news."),
    "sad": ("Sad", "Read sadly and quietly."),
    "serious": ("Serious", "Read seriously, as an important announcement."),
    "excited": ("Excited", "Read with excitement."),
    "question": ("Questions", "Read every question as a real question."),
    "emphasis": ("Emphasis", "Read with energy and stress the words that matter."),
    "artic": ("Articulation", "Read clearly and take your time with each sound."),
    "numbers": ("Numbers and names", "Read the numbers and names as they are written."),
}

# Language -> the kit in reading order. The four "normal" texts come first and keep their
# ids, so an old client that reads the first text reads the same words (D45, D70).
RECORDING_KIT: dict[str, tuple[KitText, ...]] = {
    "en": (
        KitText("en-kitchen", "normal",
                "The quick morning light crept across the kitchen table. I poured a cup of coffee, opened the window, "
                "and listened to the city slowly waking up below."),
        KitText("en-market", "normal",
                "On Sunday mornings the market fills the square. Someone sells warm bread from a cart, a child counts "
                "coins for apples, and the clock above the bakery runs ten minutes slow."),
        KitText("en-rain", "normal",
                "Rain tapped on the roof all afternoon, so we stayed inside. My sister found an old board game in the "
                "cupboard, and we argued about the rules over cups of hot tea."),
        KitText("en-train", "normal",
                "The train was quiet except for the steady rattle of the wheels. A man across the aisle read a thick "
                "paperback, and small towns slid past the window one after another."),
        KitText("en-slow-1", "slow",
                "Every morning the baker opens the shop at six. First the lights come on, then the ovens, and soon "
                "the whole street smells of fresh bread."),
        KitText("en-slow-2", "slow",
                "Take one deep breath. Hold it for a moment. Now let it go, slowly, and listen to the room around "
                "you grow quiet."),
        KitText("en-fast-1", "fast",
                "Quick, grab your coat, the bus leaves in two minutes, and if we miss it we will be late for the "
                "film again!"),
        KitText("en-fast-2", "fast",
                "She packed the bags, locked the door, checked the tickets twice, and ran down the stairs before "
                "anyone could stop her."),
        KitText("en-calm-1", "calm",
                "The garden is quiet in the evening. A light wind moves the leaves, and the last birds sing "
                "somewhere far away."),
        KitText("en-calm-2", "calm",
                "There is no need to hurry. The candle is lit, the book is open, and the whole afternoon belongs "
                "to us."),
        KitText("en-happy-1", "happy",
                "We did it! The letter came this morning, and they said yes. I cannot stop smiling, and I want to "
                "tell everyone."),
        KitText("en-happy-2", "happy",
                "What a wonderful surprise! The whole family is here, the music is playing, and the cake is bigger "
                "than the table."),
        KitText("en-sad-1", "sad",
                "The platform was empty when the last bus left. I waved until I could not see it any more, and "
                "then I walked home alone."),
        KitText("en-sad-2", "sad",
                "Her old chair still stands by the window. Nobody sits in it now, but every evening someone turns "
                "on the lamp beside it."),
        KitText("en-serious-1", "serious",
                "Attention, please. The museum will close in fifteen minutes. Please take your belongings and walk "
                "to the main exit."),
        KitText("en-serious-2", "serious",
                "This decision is not easy, and it will change many things. We have to think about it carefully "
                "before we agree."),
        KitText("en-excited-1", "excited",
                "Look up, look up, it is snowing! The first snow of the year, and it is already covering the cars "
                "and the roofs!"),
        KitText("en-excited-2", "excited",
                "The doors are opening, the crowd is cheering, and here they come, right now, the whole band on "
                "one stage!"),
        KitText("en-question-1", "question",
                "Did you remember the keys? And the tickets? Are you sure we have everything, or should we check "
                "one more time?"),
        KitText("en-question-2", "question",
                "Where did you find this photo? Who is the man in the hat, and why is everyone in the room looking "
                "at him?"),
        KitText("en-emphasis-1", "emphasis",
                "No, not the blue box, the red one! The red box goes on the top shelf, and the blue one stays right "
                "here!"),
        KitText("en-emphasis-2", "emphasis",
                "I said I would be there at noon, and I was there at noon. Not one minute late. Not one!"),
        KitText("en-artic-1", "artic",
                "Six slim swans swam swiftly south, while three thin thrushes sang through the thick green "
                "thistles."),
        KitText("en-artic-2", "artic",
                "A proper copper coffee pot keeps the coffee properly hot; a plain cheap pot does not keep it hot "
                "at all."),
        KitText("en-numbers-1", "numbers",
                "Anna and Peter moved to Lisbon on the third of March, two thousand nineteen, into flat number "
                "forty-seven."),
        KitText("en-numbers-2", "numbers",
                "The recipe needs two hundred grams of flour, three eggs, and one and a half cups of milk, says "
                "Aunt Maria."),
    ),
    "ru": (
        KitText("ru-kitchen", "normal",
                "Утренний свет медленно скользил по кухонному столу. За окном понемногу просыпался город, "
                "а в комнате пахло свежим кофе и тёплым хлебом."),
        KitText("ru-market", "normal",
                "По воскресеньям рынок на площади открывается рано. Продавец предлагает тёплый хлеб, ребёнок "
                "пересчитывает монеты на яблоки, а часы над пекарней отстают на десять минут."),
        KitText("ru-rain", "normal",
                "Весь день по крыше стучал дождь, и мы остались дома. Сестра нашла в шкафу старую настольную игру, "
                "и мы долго спорили о правилах за чашкой горячего чая."),
        KitText("ru-train", "normal",
                "В поезде было тихо, только мерно стучали колёса. Пассажир напротив читал толстую книгу, а за окном "
                "один за другим проплывали маленькие города."),
        KitText("ru-slow-1", "slow",
                "Каждое утро пекарь открывает лавку в шесть часов. Сначала загорается свет, потом печи, и скоро "
                "вся улица пахнет свежим хлебом."),
        KitText("ru-slow-2", "slow",
                "Сделайте глубокий вдох. Задержите дыхание. А теперь медленно выдохните и послушайте, как в "
                "комнате становится тихо."),
        KitText("ru-fast-1", "fast",
                "Быстрее, бери куртку, автобус уходит через две минуты, и если мы опоздаем, то опять пропустим "
                "начало фильма!"),
        KitText("ru-fast-2", "fast",
                "Она собрала сумки, заперла дверь, дважды проверила билеты и сбежала по лестнице, пока никто не "
                "успел её остановить."),
        KitText("ru-calm-1", "calm",
                "Вечером в саду тихо. Лёгкий ветер шевелит листья, и где-то далеко поют последние птицы."),
        KitText("ru-calm-2", "calm",
                "Спешить некуда. Свеча горит, книга открыта на любимой странице, и весь этот длинный день "
                "принадлежит только нам."),
        KitText("ru-happy-1", "happy",
                "У нас получилось! Утром пришло письмо, и нам ответили «да». Невозможно перестать улыбаться, "
                "хочется рассказать всем!"),
        KitText("ru-happy-2", "happy",
                "Какой чудесный сюрприз! Вся семья в сборе, играет музыка, а торт больше, чем сам стол."),
        KitText("ru-sad-1", "sad",
                "Остановка опустела, когда ушёл последний автобус. Мы долго махали вслед, а потом молча побрели "
                "домой."),
        KitText("ru-sad-2", "sad",
                "Её старое кресло всё ещё стоит у окна. Теперь в нём никто не сидит, но каждый вечер кто-то "
                "включает рядом лампу."),
        KitText("ru-serious-1", "serious",
                "Внимание. Музей закрывается через пятнадцать минут. Пожалуйста, возьмите свои вещи и пройдите к "
                "главному выходу."),
        KitText("ru-serious-2", "serious",
                "Это решение непростое, и оно многое изменит в нашей жизни. Нужно спокойно всё обдумать, прежде "
                "чем соглашаться."),
        KitText("ru-excited-1", "excited",
                "Смотри, смотри, пошёл снег! Первый снег в этом году, крупный и мягкий, и он уже засыпает машины, "
                "крыши и скамейки!"),
        KitText("ru-excited-2", "excited",
                "Двери открываются, зал аплодирует, и вот они выходят, прямо сейчас, вся группа на одной сцене!"),
        KitText("ru-question-1", "question",
                "Ключи взяли? А билеты? Паспорта тоже? Точно ничего не забыли, или лучше проверить ещё раз, пока "
                "не поздно?"),
        KitText("ru-question-2", "question",
                "Где нашлась эта фотография? Кто этот человек в шляпе, и почему все в комнате на него смотрят?"),
        KitText("ru-emphasis-1", "emphasis",
                "Нет, не синюю коробку, а красную! Красная идёт на верхнюю полку, а синяя остаётся прямо здесь!"),
        KitText("ru-emphasis-2", "emphasis",
                "Встреча была назначена ровно на полдень, и все пришли ровно в полдень. Ни минутой позже. Ни "
                "одной!"),
        KitText("ru-artic-1", "artic",
                "Шла Саша по шоссе и сосала сушку. Карл у Клары украл кораллы, а Клара у Карла украла кларнет."),
        KitText("ru-artic-2", "artic",
                "От топота копыт пыль по полю летит. Сшит колпак не по-колпаковски, надо колпак переколпаковать."),
        KitText("ru-numbers-1", "numbers",
                "Анна и Пётр переехали в Лиссабон третьего марта две тысячи девятнадцатого года, в квартиру номер "
                "сорок семь."),
        KitText("ru-numbers-2", "numbers",
                "Для пирога нужно двести граммов муки, три яйца и полтора стакана молока, так говорит тётя Мария."),
    ),
}
# Language -> ordered {prompt_id: text}; the first text is the one an old client reads.
# The ids name their language, so an id of the other language is refused (D45).
RECORDING_PROMPTS: dict[str, dict[str, str]] = {
    language: {text.id: text.text for text in kit} for language, kit in RECORDING_KIT.items()
}
MIN_SECONDS, MAX_SECONDS, MAX_BYTES, NAME_MAX = 5.0, 30.0, 4 * 1024 * 1024, 80
SILENCE_RMS = 0.00316          # -50 dBFS
SAMPLE_RATE = 48000            # D21: the only rate the browser and afconvert make
TRANSCRIPT_MAX = 1000         # the models service's TEXT_MAX for the words of a bake take
NO_WORDS = "the speech recognition found no words in it"
TOO_LONG = f"it holds more than {TRANSCRIPT_MAX} characters of words"

SOURCES = ("recorded", "uploaded")
TOKEN_PREFIX = "saved:"


class VoiceError(ValueError):
    """A message shown to the user as it is."""


class VoiceDeleting(VoiceError):
    """The target is being deleted (or was, a moment ago); the route answers 409."""


class VoiceNameTaken(VoiceError):
    """Another saved voice of the owner has this name; the route answers 409."""


class BakeCancelled(VoiceError):
    """The bake was cancelled, replaced or its run already ended: its voice model is not
    stored (D88)."""


@dataclass(frozen=True)
class VoiceExample:
    id: int
    voice_id: int
    language: str
    source: str              # "recorded" | "uploaded"
    transcript: str | None   # the heard words; a take stored before ASR holds its shown text; None = none yet
    storage_key: str
    content_hash: str
    seconds: float
    sample_rate: int
    created_at: str
    deleting: bool           # this example, or its voice, has a pending deletion
    transcript_model: str | None   # '<asr key>@<revision>' that heard it; None = not heard yet
    transcript_error: str | None   # why the heard words are not usable; None otherwise
    kit_id: str | None       # the kit text the take was shown, a suggestion; None for an upload or no kit text

    @property
    def words_state(self) -> str:
        """"waiting" (the hearing job hears it), "heard" (films and tests may speak from it)
        or "unusable" (never spoken from, with `transcript_error`; D94). A shown text is
        never read as words."""
        if self.transcript_model is None:
            return "waiting"
        return "heard" if self.transcript else "unusable"


@dataclass(frozen=True)
class SavedVoice:
    id: int
    name: str
    examples: tuple[VoiceExample, ...]
    deleting: bool


@dataclass(frozen=True)
class NarratorChoice:
    voice_id: int | None                 # a saved voice
    preset_speaker: str | None = None    # a catalog speaker name; both None = the chosen saved voice was deleted


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _rollback(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        conn.execute("ROLLBACK")


def token(voice_id: int) -> str:
    """The voice name `POST /speak` takes for a saved voice."""
    return f"{TOKEN_PREFIX}{voice_id}"


# Pending = an intent with `cascaded_at IS NULL` names the voice, the example, or the
# example's voice. Both fragments expect the aliases `v` (saved_voices) and `e`.

_VOICE_PENDING = (
    "EXISTS (SELECT 1 FROM voice_deletions d WHERE d.owner_id = v.owner_id"
    " AND d.kind = 'voice' AND d.target_id = v.id AND d.cascaded_at IS NULL)"
)
_EXAMPLE_PENDING = (
    "EXISTS (SELECT 1 FROM voice_deletions d WHERE d.owner_id = v.owner_id"
    " AND d.kind = 'example' AND d.target_id = e.id AND d.cascaded_at IS NULL)"
)
_EXAMPLE_COLUMNS = (
    "e.id, e.voice_id, e.language, e.source, e.transcript, e.storage_key, e.content_hash,"
    " e.seconds, e.sample_rate, e.created_at, e.transcript_model, e.transcript_error, e.kit_id,"
    f" ({_VOICE_PENDING} OR {_EXAMPLE_PENDING}) AS deleting"
)


def _example(row: sqlite3.Row) -> VoiceExample:
    return VoiceExample(
        id=row["id"], voice_id=row["voice_id"], language=row["language"], source=row["source"],
        transcript=row["transcript"], storage_key=row["storage_key"],
        content_hash=row["content_hash"], seconds=row["seconds"], sample_rate=row["sample_rate"],
        created_at=row["created_at"], deleting=bool(row["deleting"]),
        transcript_model=row["transcript_model"], transcript_error=row["transcript_error"],
        kit_id=row["kit_id"],
    )


def _voices(conn: sqlite3.Connection, owner_id: int, voice_id: int | None = None) -> list[SavedVoice]:
    where, args = "v.owner_id = ?", [owner_id]
    if voice_id is not None:
        where, args = f"{where} AND v.id = ?", [owner_id, voice_id]
    voices = conn.execute(
        f"SELECT v.id, v.name, {_VOICE_PENDING} AS deleting FROM saved_voices v WHERE {where}",
        args,
    ).fetchall()
    examples: dict[int, list[VoiceExample]] = {}
    for row in conn.execute(
        f"SELECT {_EXAMPLE_COLUMNS} FROM voice_examples e JOIN saved_voices v ON v.id = e.voice_id"
        f" WHERE {where} ORDER BY e.id",
        args,
    ):
        examples.setdefault(row["voice_id"], []).append(_example(row))
    result = [
        SavedVoice(r["id"], r["name"], tuple(examples.get(r["id"], ())), bool(r["deleting"]))
        for r in voices
    ]
    # Python, not COLLATE NOCASE: SQLite folds ASCII only, and names are often Cyrillic.
    return sorted(result, key=lambda voice: (voice.name.casefold(), voice.id))


def list_voices(conn: sqlite3.Connection, owner_id: int) -> list[SavedVoice]:
    """Every saved voice of the owner, by name; pending ones too, with `deleting=True`."""
    return _voices(conn, owner_id)


def get_voice(conn: sqlite3.Connection, owner_id: int, voice_id: int) -> SavedVoice | None:
    found = _voices(conn, owner_id, voice_id)
    return found[0] if found else None


def _voice_state(conn: sqlite3.Connection, owner_id: int, voice_id: int) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT v.id, v.name, {_VOICE_PENDING} AS deleting FROM saved_voices v"
        " WHERE v.owner_id = ? AND v.id = ?",
        (owner_id, voice_id),
    ).fetchone()


def _clean_name(name: str) -> str:
    clean = name.strip()
    if not clean:
        raise VoiceError("Give the voice a name.")
    if len(clean) > NAME_MAX:
        raise VoiceError(f"A voice name can have at most {NAME_MAX} characters.")
    return clean


def _check_name_free(
    conn: sqlite3.Connection, owner_id: int, name: str, own_id: int | None = None
) -> None:
    """Unique per owner by `casefold()` — the table's UNIQUE is case-sensitive, so the
    real check is here, under the caller's write lock."""
    folded = name.casefold()
    for row in conn.execute(
        f"SELECT v.id, v.name, {_VOICE_PENDING} AS deleting FROM saved_voices v WHERE v.owner_id = ?",
        (owner_id,),
    ):
        if row["id"] == own_id or row["name"].casefold() != folded:
            continue
        if row["deleting"]:
            raise VoiceDeleting(f"The voice «{row['name']}» is being deleted. Try again in a moment.")
        raise VoiceNameTaken(f"A saved voice named «{row['name']}» already exists.")


def create_voice(conn: sqlite3.Connection, owner_id: int, name: str) -> SavedVoice:
    clean = _clean_name(name)
    conn.execute("BEGIN IMMEDIATE")
    try:
        _check_name_free(conn, owner_id, clean)
        voice_id = conn.execute(
            "INSERT INTO saved_voices(owner_id, name, created_at) VALUES (?, ?, ?)",
            (owner_id, clean, _now()),
        ).lastrowid
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    return SavedVoice(voice_id, clean, (), False)


def rename_voice(conn: sqlite3.Connection, owner_id: int, voice_id: int, name: str) -> SavedVoice:
    clean = _clean_name(name)
    conn.execute("BEGIN IMMEDIATE")
    try:
        state = _voice_state(conn, owner_id, voice_id)
        if state is None or state["deleting"]:
            raise VoiceDeleting("This voice is being deleted.")
        _check_name_free(conn, owner_id, clean, own_id=voice_id)
        conn.execute("UPDATE saved_voices SET name = ? WHERE id = ?", (clean, voice_id))
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    voice = get_voice(conn, owner_id, voice_id)
    assert voice is not None  # renamed under the lock a moment ago; only the drain deletes
    return voice


def _enqueue(conn: sqlite3.Connection, owner_id: int, kind: str, target_id: int) -> None:
    conn.execute(
        "INSERT INTO voice_deletions(owner_id, kind, target_id, requested_at) VALUES (?, ?, ?, ?)"
        " ON CONFLICT(owner_id, kind, target_id) DO NOTHING",
        (owner_id, kind, target_id, _now()),
    )


def request_voice_deletion(conn: sqlite3.Connection, owner_id: int, voice_id: int) -> bool:
    """Record the intent to delete a voice and all its examples. Idempotent. Deletes no
    row and no file. False = no such voice for this owner (→ 404)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        if _voice_state(conn, owner_id, voice_id) is None:
            conn.execute("ROLLBACK")
            return False
        _enqueue(conn, owner_id, "voice", voice_id)
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    return True


def request_example_deletion(
    conn: sqlite3.Connection, owner_id: int, voice_id: int, example_id: int
) -> bool:
    """Record the intent to delete ONE example. An example of a pending voice is already
    covered: True, and nothing is inserted. False = no such example for this owner."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        state = _voice_state(conn, owner_id, voice_id)
        found = state is not None and conn.execute(
            "SELECT 1 FROM voice_examples WHERE id = ? AND voice_id = ?", (example_id, voice_id)
        ).fetchone() is not None
        if not found:
            conn.execute("ROLLBACK")
            return False
        if not state["deleting"]:
            _enqueue(conn, owner_id, "example", example_id)
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    return True


def pending_voice_deletions(conn: sqlite3.Connection) -> int:
    """Rows the drain still has to work through: intents plus file keys (write-ahead
    rows of in-flight writes included)."""
    row = conn.execute(
        "SELECT (SELECT COUNT(*) FROM voice_deletions) + (SELECT COUNT(*) FROM voice_file_deletions)"
    ).fetchone()
    return int(row[0])


def process_voice_deletions(conn: sqlite3.Connection, storage: Storage, owner_id: int) -> int:
    """Drain the voice deletion outbox (D33): cascade the rows, remove unreferenced
    files, end the finished intents. Each transaction re-reads its row first, so two
    drainers (app and worker) are safe. A failed row is rolled back, logged and left
    for the next run — this never raises for one row, because `drain_pass` has no guard
    and a raise would stop the worker loop. Returns the rows still pending (0 = done)."""
    _cascade_intents(conn, owner_id)
    _remove_files(conn, storage)
    _finish_intents(conn, owner_id)
    return pending_voice_deletions(conn)


def _cascade_intents(conn: sqlite3.Connection, owner_id: int) -> None:
    ids = [
        r["id"] for r in conn.execute(
            "SELECT id FROM voice_deletions WHERE owner_id = ? AND cascaded_at IS NULL ORDER BY id",
            (owner_id,),
        )
    ]
    for deletion_id in ids:
        try:
            conn.execute("BEGIN IMMEDIATE")
            _cascade_one(conn, owner_id, deletion_id)
            conn.execute("COMMIT")
        except Exception:
            _rollback(conn)
            logger.exception("voice deletion %s: cascade failed; retried next pass", deletion_id)


def _cascade_one(conn: sqlite3.Connection, owner_id: int, deletion_id: int) -> None:
    intent = conn.execute(
        "SELECT kind, target_id FROM voice_deletions WHERE id = ? AND cascaded_at IS NULL",
        (deletion_id,),
    ).fetchone()
    if intent is None:  # another drainer cascaded it in the meantime
        return
    if intent["kind"] == "voice":
        scope = "e.voice_id = ? AND v.owner_id = ?"
    else:
        scope = "e.id = ? AND v.owner_id = ?"
    args = (intent["target_id"], owner_id)
    query = (
        "SELECT e.storage_key FROM voice_examples e JOIN saved_voices v ON v.id = e.voice_id"
        f" WHERE {scope}"
        " UNION SELECT c.storage_key FROM voice_conditionings c"
        " JOIN voice_examples e ON e.id = c.example_id JOIN saved_voices v ON v.id = e.voice_id"
        f" WHERE {scope}"
        " UNION SELECT p.storage_key FROM voice_previews p"
        " JOIN voice_examples e ON e.id = p.example_id JOIN saved_voices v ON v.id = e.voice_id"
        f" WHERE {scope}"
    )
    query_args = args + args + args
    if intent["kind"] == "voice":
        # An example delete keeps the voice model and the test: they belong to the voice (D79, D80).
        query += (
            " UNION SELECT m.storage_key FROM voice_models m JOIN saved_voices v ON v.id = m.voice_id"
            " WHERE m.voice_id = ? AND v.owner_id = ?"
            " UNION SELECT t.storage_key FROM voice_tests t JOIN saved_voices v ON v.id = t.voice_id"
            " WHERE t.voice_id = ? AND v.owner_id = ?"
        )
        query_args += args + args
    keys = [r[0] for r in conn.execute(query, query_args)]
    now = _now()
    for key in keys:
        conn.execute(
            "INSERT INTO voice_file_deletions(storage_key, deletion_id, requested_at) VALUES (?, ?, ?)"
            " ON CONFLICT(storage_key, deletion_id) DO NOTHING",
            (key, deletion_id, now),
        )
    if intent["kind"] == "voice":
        # Cascades examples, conditionings, the voice model, its takes, the test and the bake
        # runs; memory_narrators.voice_id becomes NULL.
        conn.execute("DELETE FROM saved_voices WHERE id = ? AND owner_id = ?", args)
    else:
        conn.execute(
            "DELETE FROM voice_examples WHERE id = ?"
            " AND voice_id IN (SELECT id FROM saved_voices WHERE owner_id = ?)",
            args,
        )
    conn.execute("UPDATE voice_deletions SET cascaded_at = ? WHERE id = ?", (now, deletion_id))


def _remove_files(conn: sqlite3.Connection, storage: Storage) -> None:
    keys = [r[0] for r in conn.execute("SELECT DISTINCT storage_key FROM voice_file_deletions")]
    for key in keys:
        try:
            conn.execute("BEGIN IMMEDIATE")
            _remove_one(conn, storage, key)
            conn.execute("COMMIT")
        except Exception:
            _rollback(conn)
            logger.exception("voice file %s: removal failed; retried next pass", key)


def _remove_one(conn: sqlite3.Connection, storage: Storage, key: str) -> None:
    if conn.execute(
        "SELECT 1 FROM voice_file_deletions WHERE storage_key = ? LIMIT 1", (key,)
    ).fetchone() is None:  # another drainer acknowledged it in the meantime
        return
    held = conn.execute(
        "SELECT 1 FROM voice_examples WHERE storage_key = ?"
        " UNION ALL SELECT 1 FROM voice_conditionings WHERE storage_key = ?"
        " UNION ALL SELECT 1 FROM voice_previews WHERE storage_key = ?"
        " UNION ALL SELECT 1 FROM voice_models WHERE storage_key = ?"
        " UNION ALL SELECT 1 FROM voice_tests WHERE storage_key = ? LIMIT 1",
        (key, key, key, key, key),
    ).fetchone()
    if held is None:
        storage.delete(key)
    # The ack: every row of the key, whichever intent or write left it.
    conn.execute("DELETE FROM voice_file_deletions WHERE storage_key = ?", (key,))


def _finish_intents(conn: sqlite3.Connection, owner_id: int) -> None:
    try:
        conn.execute(
            "DELETE FROM voice_deletions WHERE owner_id = ? AND cascaded_at IS NOT NULL"
            " AND NOT EXISTS (SELECT 1 FROM voice_file_deletions f"
            "                 WHERE f.deletion_id = voice_deletions.id)",
            (owner_id,),
        )
    except Exception:
        logger.exception("voice deletions: finishing intents failed; retried next pass")


def _write_ahead(conn: sqlite3.Connection, key: str) -> int:
    """Step 1: a durable cleanup row for `key`, committed BEFORE the file exists."""
    return conn.execute(
        "INSERT INTO voice_file_deletions(storage_key, deletion_id, requested_at) VALUES (?, NULL, ?)",
        (key, _now()),
    ).lastrowid


def _write_voice_file(
    conn: sqlite3.Connection,
    storage: Storage,
    key: str,
    data: bytes,
    parent_ok: Callable[[], bool],
    write_row: Callable[[], object],
) -> tuple[bool, object]:
    """Steps 1-6 of §2.7. `parent_ok` re-reads the parent under the lock; False rolls
    back and returns `(False, None)` with the write-ahead row left for the drain.
    Otherwise the file is written (unless an intact copy is there) and `write_row` runs
    in the same transaction; returns `(True, its result)`."""
    ahead = _write_ahead(conn, key)
    digest = hashlib.sha256(data).hexdigest()
    while True:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if conn.execute(
                "SELECT 1 FROM voice_file_deletions"
                " WHERE id = ? AND storage_key = ? AND deletion_id IS NULL",
                (ahead, key),
            ).fetchone() is None:
                # A drain acknowledged the key between step 1 and the lock: start over.
                conn.execute("ROLLBACK")
                ahead = _write_ahead(conn, key)
                continue
            if not parent_ok():
                conn.execute("ROLLBACK")
                return False, None
            # `LocalStorage.write` truncates in place, so a rewrite could break a file
            # another row holds: skip it when an intact copy is already there.
            if not (storage.exists(key) and hashlib.sha256(storage.read(key)).hexdigest() == digest):
                storage.write(key, data)
            result = write_row()
            conn.execute(
                "DELETE FROM voice_file_deletions"
                " WHERE id = ? AND storage_key = ? AND deletion_id IS NULL",
                (ahead, key),
            )
        except BaseException:
            _rollback(conn)
            raise
        conn.execute("COMMIT")
        return True, result


def _describe(width: int, channels: int, rate: int, comptype: str) -> str:
    layout = {1: "mono", 2: "stereo"}.get(channels, f"{channels}-channel")
    encoding = f"{width * 8}-bit" if comptype == "NONE" else f"compressed ({comptype})"
    return f"{encoding} {layout} at {rate} Hz"


def _check_wav(data: bytes) -> None:
    """D21: an uncompressed, 16-bit, mono WAV at exactly SAMPLE_RATE with every frame
    present. Stdlib `wave` only — `decode_wav` reads any width as int16 and downmixes
    any channel count, so it cannot be the check."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            comptype = handle.getcomptype()
            width = handle.getsampwidth()
            channels = handle.getnchannels()
            rate = handle.getframerate()
            nframes = handle.getnframes()
            payload = handle.readframes(nframes)
    except (wave.Error, EOFError) as error:
        raise VoiceError("This file is not a WAV file.") from error
    if comptype != "NONE" or width != 2 or channels != 1 or rate != SAMPLE_RATE:
        raise VoiceError(
            f"The example must be mono 16-bit PCM at {SAMPLE_RATE} Hz;"
            f" this file is {_describe(width, channels, rate, comptype)}."
        )
    if nframes <= 0:
        raise VoiceError("This WAV file holds no audio.")
    if len(payload) != nframes * 2:
        raise VoiceError("This WAV file is cut short: part of its audio is missing.")


def _measure(data: bytes) -> float:
    samples, rate = decode_wav(data)
    seconds = samples.size / rate
    if seconds < MIN_SECONDS:
        raise VoiceError(f"The example must be at least {MIN_SECONDS:g} seconds; this one is {seconds:.2f} s.")
    if seconds > MAX_SECONDS:
        raise VoiceError(f"The example must be at most {MAX_SECONDS:g} seconds; this one is {seconds:.2f} s.")
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms < SILENCE_RMS:
        raise VoiceError("The example is silent. Record or choose a clip with speech.")
    return seconds


def add_example(
    conn: sqlite3.Connection,
    storage: Storage,
    owner_id: int,
    voice_id: int,
    *,
    language: str,
    source: str,
    data: bytes,
    prompt_id: str | None = None,
) -> VoiceExample:
    """Check a clip (D21, D22), store it as posted and record it (D45, D91, D99). A
    recording stores `prompt_id`, a kit text of its language, as `kit_id`: the text it was
    shown, only a suggestion; with no `prompt_id`, and for an upload, `kit_id` is None. No
    take holds words until the hearing job hears it. `VoiceDeleting` when the voice is
    pending or gone; `VoiceError` for every other refusal."""
    if language not in LANGUAGES:
        raise VoiceError(f"Unknown language: {language!r}.")
    if source not in SOURCES:
        raise VoiceError(f"Unknown example source: {source!r}.")
    kit_id = prompt_id if source == "recorded" else None
    if kit_id is not None and kit_id not in RECORDING_PROMPTS[language]:
        raise VoiceError(f"Unknown reading text for {language_names[language]}: '{prompt_id}'.")
    if len(data) > MAX_BYTES:
        raise VoiceError(f"The example is larger than {MAX_BYTES // (1024 * 1024)} MiB.")
    _check_wav(data)
    seconds = _measure(data)

    content_hash = hashlib.sha256(data).hexdigest()
    key = content_key(content_hash, ".wav")
    now = _now()
    refusal: list[VoiceError] = []

    def parent_ok() -> bool:
        state = _voice_state(conn, owner_id, voice_id)
        if state is None or state["deleting"]:
            refusal.append(VoiceDeleting("This voice is being deleted."))
            return False
        same = conn.execute(
            f"SELECT e.id, v.owner_id, {_EXAMPLE_PENDING} AS deleting"
            " FROM voice_examples e JOIN saved_voices v ON v.id = e.voice_id"
            " WHERE e.voice_id = ? AND e.content_hash = ?",
            (voice_id, content_hash),
        ).fetchone()
        if same is not None:
            refusal.append(
                VoiceDeleting("This example is being deleted. Try again in a moment.")
                if same["deleting"] else VoiceError("This voice already has this example.")
            )
            return False
        return True

    def write_row() -> int:
        return conn.execute(
            "INSERT INTO voice_examples(voice_id, language, source, storage_key, content_hash,"
            " seconds, sample_rate, consent_at, created_at, kit_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (voice_id, language, source, key, content_hash, seconds, SAMPLE_RATE, now, now, kit_id),
        ).lastrowid

    stored, example_id = _write_voice_file(conn, storage, key, data, parent_ok, write_row)
    if not stored:
        raise refusal[0]
    return VoiceExample(
        id=example_id, voice_id=voice_id, language=language, source=source,
        transcript=None, storage_key=key, content_hash=content_hash, seconds=seconds,
        sample_rate=SAMPLE_RATE, created_at=now, deleting=False,
        transcript_model=None, transcript_error=None, kit_id=kit_id,
    )


def heard_words(text: str) -> tuple[str | None, str | None]:
    """`(words, error)` for what ASR heard: whitespace runs become one space. Words with
    no letter and no digit give `NO_WORDS`; more than `TRANSCRIPT_MAX` characters give
    `TOO_LONG` (the models service refuses them, D94)."""
    words = " ".join(text.split())
    if not any(char.isalpha() or char.isdigit() for char in words):
        return None, NO_WORDS
    if len(words) > TRANSCRIPT_MAX:
        return None, TOO_LONG
    return words, None


def not_used_text(takes: Iterable[VoiceExample]) -> str:
    """Every `unusable` take with its own reason, by id: "take #9 — <reason>; take #12 — …"."""
    unusable = sorted((take for take in takes if take.words_state == "unusable"), key=lambda take: take.id)
    return "; ".join(f"take #{take.id} — {take.transcript_error}" for take in unusable)


def takes_to_hear(conn: sqlite3.Connection, owner_id: int, voice_id: int) -> list[VoiceExample]:
    """The takes the hearing job hears: not pending, not heard yet, by id. A take stored
    before ASR is one of them: its shown text is no words."""
    return [take for take in _current_takes(conn, owner_id, voice_id) if take.words_state == "waiting"]


def take_to_hear(
    conn: sqlite3.Connection, owner_id: int, voice_id: int, example_id: int
) -> VoiceExample | None:
    """That take while it still waits for ASR: it and its voice exist and are not pending,
    and it is not heard yet. Else None (the hearing job skips it)."""
    row = conn.execute(
        f"SELECT {_EXAMPLE_COLUMNS} FROM voice_examples e JOIN saved_voices v ON v.id = e.voice_id"
        f" WHERE v.owner_id = ? AND v.id = ? AND e.id = ? AND e.transcript_model IS NULL"
        f" AND NOT {_VOICE_PENDING} AND NOT {_EXAMPLE_PENDING}",
        (owner_id, voice_id, example_id),
    ).fetchone()
    return _example(row) if row is not None else None


def set_heard_words(
    conn: sqlite3.Connection, owner_id: int, voice_id: int, example_id: int, *, text: str, model: str
) -> VoiceExample | None:
    """Store what `model` ('<asr key>@<revision>') heard in one take, once: the words, or
    why they are not usable (`heard_words`). None = nothing written: the take or its voice
    is pending or gone, or the take was heard already (a take is heard once, D94)."""
    words, error = heard_words(text)
    conn.execute("BEGIN IMMEDIATE")
    try:
        if take_to_hear(conn, owner_id, voice_id, example_id) is None:
            conn.execute("ROLLBACK")
            return None
        conn.execute(
            "UPDATE voice_examples SET transcript = ?, transcript_model = ?, transcript_error = ? WHERE id = ?",
            (words, model, error, example_id),
        )
        heard = _example(conn.execute(
            f"SELECT {_EXAMPLE_COLUMNS} FROM voice_examples e JOIN saved_voices v ON v.id = e.voice_id"
            " WHERE e.id = ?",
            (example_id,),
        ).fetchone())
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    return heard


def read_example_audio(storage: Storage, example: VoiceExample) -> bytes:
    return storage.read(example.storage_key)


def example_for(
    conn: sqlite3.Connection, owner_id: int, voice_id: int, language: str
) -> VoiceExample | None:
    """The NEWEST example of the voice in `language` that is not pending (D19)."""
    row = conn.execute(
        f"SELECT {_EXAMPLE_COLUMNS} FROM voice_examples e JOIN saved_voices v ON v.id = e.voice_id"
        f" WHERE v.owner_id = ? AND v.id = ? AND e.language = ?"
        f" AND NOT {_VOICE_PENDING} AND NOT {_EXAMPLE_PENDING}"
        " ORDER BY e.id DESC LIMIT 1",
        (owner_id, voice_id, language),
    ).fetchone()
    return _example(row) if row is not None else None


def read_narrator(conn: sqlite3.Connection, group_id: int, language: str) -> NarratorChoice | None:
    """None = no row = Automatic. `NarratorChoice(None)` = the chosen voice was deleted."""
    row = conn.execute(
        "SELECT voice_id, preset_speaker FROM memory_narrators WHERE group_id = ? AND language = ?",
        (group_id, language),
    ).fetchone()
    return NarratorChoice(row["voice_id"], row["preset_speaker"]) if row is not None else None


def set_narrator(
    conn: sqlite3.Connection,
    group_id: int,
    language: str,
    voice_id: int | None = None,
    *,
    preset_speaker: str | None = None,
) -> None:
    """Store the user's narrator for one memory's film in one language: a saved voice or
    a preset speaker; neither = Automatic (the row is deleted). `VoiceDeleting` when the
    voice, of the memory's owner, is pending or gone. The preset name is not checked
    against the catalog here: the route and the build do (D38)."""
    if language not in LANGUAGES:
        raise VoiceError(f"Unknown language: {language!r}.")
    if voice_id is not None and preset_speaker is not None:
        raise VoiceError("A narrator is a saved voice or a preset voice, not both.")
    if preset_speaker is not None and not preset_speaker.strip():
        raise VoiceError("Choose a preset voice by its name.")
    conn.execute("BEGIN IMMEDIATE")
    try:
        if preset_speaker is not None:
            conn.execute(
                "INSERT INTO memory_narrators(group_id, language, voice_id, preset_speaker) VALUES (?, ?, NULL, ?)"
                " ON CONFLICT(group_id, language) DO UPDATE SET voice_id = NULL,"
                " preset_speaker = excluded.preset_speaker",
                (group_id, language, preset_speaker),
            )
        elif voice_id is None:
            conn.execute(
                "DELETE FROM memory_narrators WHERE group_id = ? AND language = ?",
                (group_id, language),
            )
        else:
            state = conn.execute(
                f"SELECT {_VOICE_PENDING} AS deleting FROM saved_voices v"
                " JOIN groups g ON g.owner_id = v.owner_id WHERE v.id = ? AND g.id = ?",
                (voice_id, group_id),
            ).fetchone()
            if state is None or state["deleting"]:
                raise VoiceDeleting("This saved voice is being deleted. Choose another narrator.")
            conn.execute(
                "INSERT INTO memory_narrators(group_id, language, voice_id, preset_speaker) VALUES (?, ?, ?, NULL)"
                " ON CONFLICT(group_id, language) DO UPDATE SET voice_id = excluded.voice_id,"
                " preset_speaker = NULL",
                (group_id, language, voice_id),
            )
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")


def conditionings(conn: sqlite3.Connection, storage: Storage, example_id: int) -> list[dict]:
    """The stored conditioning blobs of one example, for every model and revision. A blob
    whose file is missing is left out: it is a cache, and the child rebuilds it."""
    found = []
    for row in conn.execute(
        "SELECT model_key, revision, storage_key FROM voice_conditionings WHERE example_id = ?"
        " ORDER BY model_key, revision",
        (example_id,),
    ):
        try:
            blob = storage.read(row["storage_key"])
        except FileNotFoundError:
            logger.warning("conditioning %s is missing; the model rebuilds it", row["storage_key"])
            continue
        found.append({"model": row["model_key"], "revision": row["revision"], "blob": blob})
    return found


def reference_for(conn: sqlite3.Connection, storage: Storage, example: VoiceExample) -> dict:
    """One saved example as `POST /speak` takes it, with its heard words and every
    conditioning blob a model already built from it. A film and a test send the same
    dict for their `film_take`."""
    return {
        "audio": read_example_audio(storage, example),
        "sha256": example.content_hash,
        "language": example.language,
        "transcript": example.transcript,
        "conditioning": conditionings(conn, storage, example.id),
    }


@dataclass(frozen=True)
class VoiceModel:
    voice_id: int
    version: int
    model_key: str
    revision: str
    lora_json: str
    config_json: str
    dataset_hash: str
    takes: int
    seconds: float
    storage_key: str
    sha256: str
    bytes: int
    metrics: dict
    created_at: str


@dataclass(frozen=True)
class ModelChanges:
    added: int      # takes that are not pending, not in voice_model_takes and not `unusable`
    deleted: int    # voice_model_takes rows whose take is gone or pending
    unusable: int   # takes that are not pending and `unusable`


@dataclass(frozen=True)
class VoiceModelState:
    model: VoiceModel | None
    usable: bool                              # True = the bake check can speak through it
    problem: str | None                       # None | "none" (never baked) | "revision" (another model revision or LoRA shape)
    languages: dict[str, tuple[int, float]]   # what it learned: language -> (takes, seconds)
    changes: ModelChanges


@dataclass(frozen=True)
class StoredModel:
    """The stored voice model the bake check speaks through in one language."""

    model: VoiceModel | None   # set only when problem is None
    problem: str | None        # None | "none" | "revision" | "language" | "deleting"


@dataclass(frozen=True)
class FilmTake:
    """THE answer to "can this voice narrate `language`, and from which take" (§11.1)."""

    take: VoiceExample | None   # set only when problem is None
    problem: str | None         # None | "language" | "waiting" | "unusable" | "deleting"


@dataclass(frozen=True)
class SnapshotTake:
    example_id: int
    content_hash: str
    language: str
    seconds: float
    transcript: str
    kit_id: str | None
    fingerprint: str


@dataclass(frozen=True)
class BakeSnapshot:
    takes: tuple[VoiceExample, ...]      # the `heard` takes that are not pending, by id: what a bake sends now
    waiting: tuple[VoiceExample, ...]    # not heard yet
    unusable: tuple[VoiceExample, ...]   # heard, no usable words: left out and named
    seconds: float                       # of `takes`
    by_language: dict[str, float]        # of `takes`
    dataset_hash: str                    # snapshot_hash(takes)


class BakeRun(NamedTuple):
    voice_id: int
    takes: tuple[SnapshotTake, ...]
    dataset_hash: str
    cancel_requested_at: str | None   # set = Cancel, a replace or a voice delete asked to stop it (D88)


def _fingerprint(content_hash: str, language: str, transcript: str | None) -> str:
    return hashlib.sha256(f"{content_hash}\t{language}\t{transcript or ''}".encode()).hexdigest()


def snapshot_hash(takes: Iterable[VoiceExample | SnapshotTake]) -> str:
    """The models service's `dataset_sha256` of these takes: independent of their order."""
    lines = sorted(f"{t.content_hash}\t{t.language}\t{t.transcript or ''}" for t in takes)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _current_takes(conn: sqlite3.Connection, owner_id: int, voice_id: int) -> list[VoiceExample]:
    return [
        _example(row) for row in conn.execute(
            f"SELECT {_EXAMPLE_COLUMNS} FROM voice_examples e JOIN saved_voices v ON v.id = e.voice_id"
            f" WHERE v.owner_id = ? AND v.id = ? AND NOT {_VOICE_PENDING} AND NOT {_EXAMPLE_PENDING}"
            " ORDER BY e.id",
            (owner_id, voice_id),
        )
    ]


def bake_snapshot(conn: sqlite3.Connection, owner_id: int, voice_id: int) -> BakeSnapshot:
    """The takes of the voice that are not pending, by their words: a bake learns from
    every heard one; a waiting one is heard first; an unusable one is left out and named,
    and never blocks a bake (D94)."""
    current = _current_takes(conn, owner_id, voice_id)
    takes = [take for take in current if take.words_state == "heard"]
    by_language: dict[str, float] = {}
    for take in takes:
        by_language[take.language] = by_language.get(take.language, 0.0) + take.seconds
    return BakeSnapshot(
        takes=tuple(takes),
        waiting=tuple(take for take in current if take.words_state == "waiting"),
        unusable=tuple(take for take in current if take.words_state == "unusable"),
        seconds=sum(take.seconds for take in takes),
        by_language=by_language,
        dataset_hash=snapshot_hash(takes),
    )


def start_bake_run(conn: sqlite3.Connection, owner_id: int, voice_id: int, snapshot: BakeSnapshot) -> str:
    """Record the immutable snapshot of a bake BEFORE it starts; returns its `bake_key`.
    `VoiceDeleting` when the voice is pending or gone."""
    bake_key = secrets.token_hex(16)
    snapshot_json = json.dumps([
        {"example_id": t.id, "content_hash": t.content_hash, "language": t.language,
         "seconds": t.seconds, "transcript": t.transcript, "kit_id": t.kit_id,
         "fingerprint": _fingerprint(t.content_hash, t.language, t.transcript)}
        for t in snapshot.takes
    ], ensure_ascii=False)
    conn.execute("BEGIN IMMEDIATE")
    try:
        state = _voice_state(conn, owner_id, voice_id)
        if state is None or state["deleting"]:
            raise VoiceDeleting("This voice is being deleted.")
        conn.execute(
            "INSERT INTO voice_bake_runs(bake_key, owner_id, voice_id, snapshot_json, dataset_hash, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (bake_key, owner_id, voice_id, snapshot_json, snapshot.dataset_hash, _now()),
        )
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    return bake_key


def read_bake_run(conn: sqlite3.Connection, owner_id: int, bake_key: str) -> BakeRun | None:
    """The run as it was written before its bake started; None = it ended or never was."""
    row = conn.execute(
        "SELECT voice_id, snapshot_json, dataset_hash, cancel_requested_at FROM voice_bake_runs"
        " WHERE owner_id = ? AND bake_key = ?",
        (owner_id, bake_key),
    ).fetchone()
    if row is None:
        return None
    takes = tuple(SnapshotTake(**take) for take in json.loads(row["snapshot_json"]))
    return BakeRun(row["voice_id"], takes, row["dataset_hash"], row["cancel_requested_at"])


def end_bake_run(conn: sqlite3.Connection, bake_key: str) -> None:
    """Runs on the caller's connection as it is: inside `save_voice_model`'s transaction,
    or on its own."""
    conn.execute("DELETE FROM voice_bake_runs WHERE bake_key = ?", (bake_key,))


def request_bake_cancel(conn: sqlite3.Connection, bake_key: str) -> bool:
    """Store a cancel request with an open run, so it outlives an app restart (D88).
    False = no run row: the run already ended, its result stored or dropped."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "UPDATE voice_bake_runs SET cancel_requested_at = ? WHERE bake_key = ? AND cancel_requested_at IS NULL",
            (_now(), bake_key),
        )
        found = conn.execute("SELECT 1 FROM voice_bake_runs WHERE bake_key = ?", (bake_key,)).fetchone()
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    return found is not None


def _model(row: sqlite3.Row) -> VoiceModel:
    return VoiceModel(
        voice_id=row["voice_id"], version=row["version"], model_key=row["model_key"],
        revision=row["revision"], lora_json=row["lora_json"], config_json=row["config_json"],
        dataset_hash=row["dataset_hash"], takes=row["takes"], seconds=row["seconds"],
        storage_key=row["storage_key"], sha256=row["sha256"], bytes=row["bytes"],
        metrics=json.loads(row["metrics_json"]), created_at=row["created_at"],
    )


def _model_row(conn: sqlite3.Connection, owner_id: int, voice_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT m.* FROM voice_models m JOIN saved_voices v ON v.id = m.voice_id"
        " WHERE v.owner_id = ? AND m.voice_id = ?",
        (owner_id, voice_id),
    ).fetchone()


def _made_for_the_catalog(model: VoiceModel) -> bool:
    """The model fits the catalog's cloning entry: the same key, Hub revision and LoRA shape."""
    try:
        entry = catalog.get("voice", model.model_key)
    except KeyError:
        return False
    return (entry.adapter is not None and model.revision == entry.source.revision
            and model.lora_json == json.dumps(entry.adapter.lora, sort_keys=True))


def voice_model_for(conn: sqlite3.Connection, owner_id: int, voice_id: int) -> VoiceModelState:
    """The stored voice model of one voice, whether the bake check can use it, what it
    learned, and how the takes changed since it (D79, D100). Films and tests never use it."""
    takes = _current_takes(conn, owner_id, voice_id)
    unusable = sum(1 for take in takes if take.words_state == "unusable")
    row = _model_row(conn, owner_id, voice_id)
    if row is None:
        return VoiceModelState(None, False, "none", {}, ModelChanges(len(takes) - unusable, 0, unusable))
    model = _model(row)
    trained: set[int] = set()
    languages: dict[str, tuple[int, float]] = {}
    for take in conn.execute(
        "SELECT example_id, language, seconds FROM voice_model_takes WHERE voice_id = ?",
        (voice_id,),
    ):
        trained.add(take["example_id"])
        count, seconds = languages.get(take["language"], (0, 0.0))
        languages[take["language"]] = (count + 1, seconds + take["seconds"])
    current = {take.id: take for take in takes}
    changes = ModelChanges(
        added=sum(1 for take_id, take in current.items()
                  if take_id not in trained and take.words_state != "unusable"),
        deleted=sum(1 for take_id in trained if take_id not in current),
        unusable=unusable,
    )
    usable = _made_for_the_catalog(model)
    return VoiceModelState(model, usable, None if usable else "revision", languages, changes)


def stored_model(conn: sqlite3.Connection, owner_id: int, voice_id: int, language: str) -> StoredModel:
    """The stored voice model of the voice in `language`, for the bake check: a usable
    model that learned at least one take in `language`. Films and tests never use it."""
    state = _voice_state(conn, owner_id, voice_id)
    if state is None or state["deleting"]:
        return StoredModel(None, "deleting")
    found = voice_model_for(conn, owner_id, voice_id)
    if not found.usable:
        return StoredModel(None, found.problem)
    if language not in found.languages:
        return StoredModel(None, "language")
    return StoredModel(found.model, None)


def film_take(conn: sqlite3.Connection, owner_id: int, voice_id: int, language: str) -> FilmTake:
    """The ONE answer to "can this voice narrate `language`, and from which take": the
    newest take of `language` whose words were heard. A newer take that waits for its
    words or has none usable is skipped. With no such take the problem says why: no take
    of the language, takes that still wait for their words, or takes with no usable words."""
    state = _voice_state(conn, owner_id, voice_id)
    if state is None or state["deleting"]:
        return FilmTake(None, "deleting")
    takes = [take for take in reversed(_current_takes(conn, owner_id, voice_id)) if take.language == language]
    if not takes:
        return FilmTake(None, "language")
    heard = next((take for take in takes if take.words_state == "heard"), None)
    if heard is not None:
        return FilmTake(heard, None)
    if any(take.words_state == "waiting" for take in takes):
        return FilmTake(None, "waiting")
    return FilmTake(None, "unusable")


def castable_takes(conn: sqlite3.Connection, owner_id: int, language: str) -> list[tuple[SavedVoice, VoiceExample]]:
    """Voices that can narrate a film in `language`, each with its `film_take`: not
    pending, and ready for the language. Sorted as `list_voices` (D89)."""
    cast = []
    for voice in list_voices(conn, owner_id):
        if voice.deleting:
            continue
        found = film_take(conn, owner_id, voice.id, language)
        if found.take is not None:
            cast.append((voice, found.take))
    return cast


def save_voice_model(
    conn: sqlite3.Connection,
    storage: Storage,
    owner_id: int,
    voice_id: int,
    *,
    bake_key: str,
    data: bytes,
    status: dict,
    takes: tuple[SnapshotTake, ...],
    dataset_hash: str,
) -> VoiceModel | None:
    """Store a finished bake as the voice's model: version 1, or the stored version + 1
    in place, with the takes it learned from (D72, D79). A bake already stored returns its
    row and stores nothing. `BakeCancelled` when its run carries a cancel request or has
    already ended (D88); `VoiceError` when the service trained other takes; None = the
    voice went away. A replaced model file goes to the drain."""
    if status.get("dataset_sha256") != dataset_hash:
        raise VoiceError("The voice model does not match the takes that were sent.")
    sha256 = hashlib.sha256(data).hexdigest()
    key = content_key(sha256, ".safetensors")
    outcome: dict[str, object] = {}

    def parent_ok() -> bool:
        stored = conn.execute(
            "SELECT m.* FROM voice_models m JOIN saved_voices v ON v.id = m.voice_id"
            " WHERE m.bake_key = ? AND v.owner_id = ?",
            (bake_key, owner_id),
        ).fetchone()
        if stored is not None:
            outcome["stored"] = _model(stored)
            return False
        run = conn.execute(
            "SELECT cancel_requested_at FROM voice_bake_runs WHERE bake_key = ?", (bake_key,)
        ).fetchone()
        if run is None or run["cancel_requested_at"] is not None:
            outcome["cancelled"] = True
            return False
        state = _voice_state(conn, owner_id, voice_id)
        return state is not None and not state["deleting"]

    def write_row() -> None:
        old = conn.execute(
            "SELECT version, storage_key FROM voice_models WHERE voice_id = ?", (voice_id,)
        ).fetchone()
        now = _now()
        conn.execute(
            "INSERT INTO voice_models(voice_id, version, bake_key, model_key, revision, lora_json,"
            " config_json, dataset_hash, takes, seconds, storage_key, sha256, bytes, metrics_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(voice_id) DO UPDATE SET version = excluded.version, bake_key = excluded.bake_key,"
            " model_key = excluded.model_key, revision = excluded.revision, lora_json = excluded.lora_json,"
            " config_json = excluded.config_json, dataset_hash = excluded.dataset_hash,"
            " takes = excluded.takes, seconds = excluded.seconds, storage_key = excluded.storage_key,"
            " sha256 = excluded.sha256, bytes = excluded.bytes, metrics_json = excluded.metrics_json,"
            " created_at = excluded.created_at",
            (voice_id, old["version"] + 1 if old is not None else 1, bake_key, status["model"],
             status["revision"], status["lora"], json.dumps(status["config"], sort_keys=True),
             dataset_hash, len(takes), sum(take.seconds for take in takes), key, sha256, len(data),
             json.dumps(status, sort_keys=True), now),
        )
        conn.execute("DELETE FROM voice_model_takes WHERE voice_id = ?", (voice_id,))
        conn.executemany(
            "INSERT INTO voice_model_takes(voice_id, example_id, language, seconds, fingerprint)"
            " VALUES (?, ?, ?, ?, ?)",
            [(voice_id, take.example_id, take.language, take.seconds, take.fingerprint) for take in takes],
        )
        end_bake_run(conn, bake_key)
        if old is not None and old["storage_key"] != key:
            conn.execute(
                "INSERT INTO voice_file_deletions(storage_key, deletion_id, requested_at) VALUES (?, NULL, ?)",
                (old["storage_key"], now),
            )

    written, _ = _write_voice_file(conn, storage, key, data, parent_ok, write_row)
    if "stored" in outcome:
        return outcome["stored"]
    if "cancelled" in outcome:
        raise BakeCancelled("The bake was cancelled, so its voice model was not stored.")
    if not written:
        return None
    row = _model_row(conn, owner_id, voice_id)
    assert row is not None  # stored under the lock a moment ago; only the drain deletes
    return _model(row)


def read_voice_model(storage: Storage, model: VoiceModel) -> bytes:
    return storage.read(model.storage_key)


def model_reference(model: VoiceModel, data: bytes, language: str) -> dict:
    """The voice model as `POST /speak` takes it for the bake check's arm C: the adapter
    alone, with no take and no clip (D76). Films and tests never send it."""
    return {"language": language, "adapter": {"blob": data, "sha256": model.sha256}, "use_clip": False}


@dataclass(frozen=True)
class VoiceTest:
    voice_id: int
    language: str
    text_kind: str              # "sentence" | "take" | "custom"
    text_id: str | None         # the sentence id when text_kind == "sentence"
    example_id: int | None      # the take whose words were spoken; None once it is deleted
    text: str
    model_version: int          # the voice model version that spoke; 0 = spoken from a take
    model_sha256: str           # that model's sha256; "" when spoken from a take
    model_key: str              # the reply's model
    revision: str               # the reply's revision
    elapsed_seconds: float
    seconds: float
    storage_key: str
    created_at: str
    likeness: dict | None = None        # the measure (§11.1); None with no error = never measured
    likeness_error: str | None = None   # why there is no measure
    prompt_example_id: int | None = None   # the take that spoke; None once it is deleted, or made through a voice model


def _test(row: sqlite3.Row) -> VoiceTest:
    return VoiceTest(
        voice_id=row["voice_id"], language=row["language"], text_kind=row["text_kind"],
        text_id=row["text_id"], example_id=row["example_id"], text=row["text"],
        model_version=row["model_version"], model_sha256=row["model_sha256"],
        model_key=row["model_key"], revision=row["revision"],
        elapsed_seconds=row["elapsed_seconds"], seconds=row["seconds"],
        storage_key=row["storage_key"], created_at=row["created_at"],
        likeness=json.loads(row["likeness_json"]) if row["likeness_json"] is not None else None,
        likeness_error=row["likeness_error"], prompt_example_id=row["prompt_example_id"],
    )


def save_test(
    conn: sqlite3.Connection,
    storage: Storage,
    owner_id: int,
    voice_id: int,
    *,
    language: str,
    text_kind: str,
    text_id: str | None,
    example_id: int | None,
    text: str,
    prompt: VoiceExample,
    model: str,
    revision: str,
    elapsed_seconds: float,
    audio: bytes,
    likeness: dict | None = None,
    likeness_error: str | None = None,
) -> VoiceTest | None:
    """Store the last test of one voice, spoken from the take `prompt`, replacing the one
    before (D80). None = the voice is gone or pending, or `prompt` is gone or pending:
    nothing is stored. A replaced test WAV's key goes to the drain."""
    key = content_key(hashlib.sha256(audio).hexdigest(), ".wav")
    seconds = duration_seconds(audio)
    created_at = _now()
    likeness_json = json.dumps(likeness, sort_keys=True) if likeness is not None else None
    written: dict[str, int | None] = {}

    def parent_ok() -> bool:
        state = _voice_state(conn, owner_id, voice_id)
        if state is None or state["deleting"]:
            return False
        return conn.execute(
            "SELECT 1 FROM voice_examples e JOIN saved_voices v ON v.id = e.voice_id"
            f" WHERE e.id = ? AND e.voice_id = ? AND NOT {_EXAMPLE_PENDING}",
            (prompt.id, voice_id),
        ).fetchone() is not None

    def write_row() -> None:
        take_id = example_id
        if take_id is not None and conn.execute(
            "SELECT 1 FROM voice_examples WHERE id = ? AND voice_id = ?", (take_id, voice_id)
        ).fetchone() is None:
            take_id = None
        written["example_id"] = take_id
        old = conn.execute("SELECT storage_key FROM voice_tests WHERE voice_id = ?", (voice_id,)).fetchone()
        conn.execute(
            "INSERT INTO voice_tests(voice_id, language, text_kind, text_id, example_id, text, model_version,"
            " model_sha256, model_key, revision, elapsed_seconds, seconds, storage_key, created_at,"
            " likeness_json, likeness_error, prompt_example_id)"
            " VALUES (?, ?, ?, ?, ?, ?, 0, '', ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(voice_id) DO UPDATE SET language = excluded.language,"
            " text_kind = excluded.text_kind, text_id = excluded.text_id, example_id = excluded.example_id,"
            " text = excluded.text, model_version = excluded.model_version,"
            " model_sha256 = excluded.model_sha256, model_key = excluded.model_key,"
            " revision = excluded.revision, elapsed_seconds = excluded.elapsed_seconds,"
            " seconds = excluded.seconds, storage_key = excluded.storage_key, created_at = excluded.created_at,"
            " likeness_json = excluded.likeness_json, likeness_error = excluded.likeness_error,"
            " prompt_example_id = excluded.prompt_example_id",
            (voice_id, language, text_kind, text_id, take_id, text, model, revision, elapsed_seconds,
             seconds, key, created_at, likeness_json, likeness_error, prompt.id),
        )
        if old is not None and old["storage_key"] != key:
            conn.execute(
                "INSERT INTO voice_file_deletions(storage_key, deletion_id, requested_at) VALUES (?, NULL, ?)",
                (old["storage_key"], created_at),
            )

    stored, _ = _write_voice_file(conn, storage, key, audio, parent_ok, write_row)
    if not stored:
        return None
    return VoiceTest(
        voice_id=voice_id, language=language, text_kind=text_kind, text_id=text_id,
        example_id=written["example_id"], text=text, model_version=0, model_sha256="",
        model_key=model, revision=revision, elapsed_seconds=elapsed_seconds, seconds=seconds,
        storage_key=key, created_at=created_at, likeness=likeness, likeness_error=likeness_error,
        prompt_example_id=prompt.id,
    )


def voice_test(conn: sqlite3.Connection, owner_id: int, voice_id: int) -> VoiceTest | None:
    row = conn.execute(
        "SELECT t.* FROM voice_tests t JOIN saved_voices v ON v.id = t.voice_id"
        " WHERE v.owner_id = ? AND t.voice_id = ?",
        (owner_id, voice_id),
    ).fetchone()
    return _test(row) if row is not None else None


def read_test_audio(storage: Storage, test: VoiceTest) -> bytes:
    return storage.read(test.storage_key)
