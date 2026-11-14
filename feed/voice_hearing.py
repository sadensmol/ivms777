"""Hearing the words of a saved voice's takes in `app` (design §11.1 "The words of a take
are heard right after it is saved"): every take of the voice that waits for its words is
sent to the models service's speech recognition, one per `POST /voice/transcribe`, and
the words it heard are stored on the take. Films and tests then speak from the newest
heard take of the language (`film_take`).

The job holds the voice lock for each take, so films and tests wait for it and it waits
for them (D52). A take saved while the job runs is heard in the same job. Cancel is
logical: it is checked before each take, and the call in flight runs to its reply.

No models: this module is a thin client (CLAUDE.md § "One model process").
"""

import logging
import sqlite3
import threading

from feed.build import Cancelled, _reason, voice_turn
from feed.saved_voices import (
    VoiceError,
    read_example_audio,
    set_heard_words,
    take_to_hear,
    takes_to_hear,
)
from storage.base import Storage

logger = logging.getLogger(__name__)

HEARING = "hearing the words"
NOT_OFFERED = "Hearing the words of a take is not offered on this device yet."
BAKE_CHECK = "The models service is baking a voice for the bake check. Hear the words again when it ends."


class HearingFailed(VoiceError):
    """The job stopped before every take was heard, for the reason the page shows."""


def hear_takes(
    conn: sqlite3.Connection,
    client,
    voices: Storage,
    owner_id: int,
    voice_id: int,
    *,
    voice_lock: threading.Lock | None = None,
    progress=None,
    cancelled=None,
) -> int:
    """Hear every take of `voice_id` that waits for its words, also the ones saved while
    this runs, and store what was heard. Returns the takes heard. Raises `Cancelled` and
    `HearingFailed`."""
    heard = skipped = 0
    while True:
        queue = takes_to_hear(conn, owner_id, voice_id)
        if not queue:
            return heard
        total = heard + skipped + len(queue)
        for take in queue:
            done = heard + skipped
            if cancelled is not None and cancelled():
                raise Cancelled(HEARING)
            if progress is not None:
                progress(f"{HEARING} · take {done + 1} of {total}", done / total)
            with voice_turn(voice_lock, progress, cancelled, done / total):
                if take_to_hear(conn, owner_id, voice_id, take.id) is None:
                    skipped += 1
                    continue
                try:
                    audio = read_example_audio(voices, take)
                except OSError as exc:
                    if take_to_hear(conn, owner_id, voice_id, take.id) is None:
                        skipped += 1
                        continue
                    raise HearingFailed(
                        f"The audio of take #{take.id} could not be read ({_reason(exc)}).") from exc
                try:
                    reply = client.transcribe(audio, take.language)
                except Exception as exc:
                    response = getattr(exc, "response", None)
                    code = response.status_code if response is not None else None
                    if code == 409:
                        raise HearingFailed(BAKE_CHECK) from exc
                    if code == 503:
                        raise HearingFailed(NOT_OFFERED) from exc
                    raise HearingFailed(
                        f"Could not hear the words of take #{take.id} ({_reason(exc)}).") from exc
                stored = set_heard_words(conn, owner_id, voice_id, take.id, text=reply["text"],
                                         model=f"{reply['model']}@{reply['revision']}")
            if stored is None:
                skipped += 1
                continue
            heard += 1
            logger.info("heard the words of take %s of voice %s", take.id, voice_id)
