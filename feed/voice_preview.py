"""The test of one saved voice (design §6, §11.1): a text of the user's choice, spoken
from the take films of that language speak from, to judge the person's voice by ear.

A test speaks as a film does: the same `POST /speak` and the same reference, the voice's
newest heard take of the language with its words (`film_take`). One test per voice is
kept (`voice_tests`, D80). A test never changes a take or the narrator films use, and it
never starts a film. Its
speaker similarity is measured against the voice's takes before the one store; a failed
measure never fails the test (`feed/voice_likeness.py`).

Cancel is logical, not a GPU stop. A cancel seen before the call makes no call; a call
already sent runs to its reply, and the test is dropped. The store step runs under the
caller's job lock, so a cancelled or replaced test is never stored after its cancel or
the newer start has returned.

No models: this module is a thin client (CLAUDE.md § "One model process").
"""

import logging
import sqlite3
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass

from feed.build import CLONE_RTF_ESTIMATE, Cancelled, _reason, _Ticking, voice_turn
from feed.saved_voices import (
    SavedVoice,
    VoiceError,
    VoiceTest,
    film_take,
    get_voice,
    reference_for,
    save_test,
    token,
)
from feed.story import _LANGUAGE_NAME as language_names
from feed.voice_likeness import MEASURING, measure_test
from storage.base import Storage

logger = logging.getLogger(__name__)

# The fixed test sentences of `scripts/voice_check.py`: app code cannot import `scripts`
# (D49), so a test keeps the two equal.
PREVIEW_SENTENCES: dict[str, dict[str, str]] = {
    "en": {
        "EN-1": "We drove to the lake early on Saturday, before the fog had lifted.",
        "EN-2": "Julia laughed when the dog jumped into the water for the third time.",
        "EN-3": "It was the fourteenth of July, 2023, and nobody wanted to go home.",
        "EN-4": "Was it really that cold, or do we only remember it that way?",
        "EN-5": ("The road wound through the hills for hours. Every turn showed another valley, "
                 "greener than the last. By evening we reached the old house, tired and happy, "
                 "and sat on the porch until the stars came out. Nobody said much. "
                 "Nobody needed to."),
    },
    "ru": {
        "RU-1": "В субботу рано утром мы поехали к озеру, пока туман ещё не рассеялся.",
        "RU-2": "Юля засмеялась, когда собака в третий раз прыгнула в воду.",
        "RU-3": "Это было четырнадцатое июля 2023 года, и никто не хотел уезжать домой.",
        "RU-4": "Неужели тогда было так холодно, или нам это только кажется?",
        "RU-5": ("Дорога часами петляла между холмами. За каждым поворотом открывалась новая "
                 "долина, зеленее прежней. К вечеру мы добрались до старого дома, усталые и "
                 "счастливые, и сидели на крыльце, пока не показались звёзды. Никто почти "
                 "ничего не говорил. Слова были не нужны."),
    },
}
CUSTOM_TEXT_MAX = 300
TEXT_KINDS = ("sentence", "take", "custom")
# Where the bar stands when the speaking ends and the measure starts, and when it ends.
MEASURE_FROM, MEASURE_TO = 0.85, 0.97


@dataclass(frozen=True)
class PreviewText:
    kind: str                # "sentence" | "take" | "custom"
    text_id: str | None      # the sentence id when kind == "sentence"
    example_id: int | None   # the take whose heard words are spoken when kind == "take"
    text: str


@dataclass(frozen=True)
class PreviewJob:
    voice_id: int
    language: str
    text: PreviewText
    revision: str            # cloning_entry_for(profile, language).source.revision when the test starts


class PreviewFailed(VoiceError):
    """Why the test was not stored, as the test panel shows it."""


def preview_text(voice: SavedVoice, language: str, kind: str, text_id: str, custom: str) -> PreviewText:
    """The words to test `voice` with in `language`. `VoiceError` names each refusal."""
    if kind == "sentence":
        sentence = PREVIEW_SENTENCES.get(language, {}).get(text_id)
        if sentence is None:
            raise VoiceError(
                f"Unknown test sentence for {language_names.get(language, language)}: '{text_id}'.")
        return PreviewText("sentence", text_id, None, sentence)
    if kind == "take":
        take = next((example for example in voice.examples
                     if str(example.id) == text_id and example.language == language
                     and not example.deleting), None)
        if take is None:
            raise VoiceError("Unknown take.")
        if take.words_state == "waiting":
            raise VoiceError(f"The words of take #{take.id} are not heard yet.")
        if take.words_state == "unusable":
            raise VoiceError(f"Take #{take.id} has no words to speak: {take.transcript_error}.")
        return PreviewText("take", None, take.id, take.transcript)
    if kind == "custom":
        text = custom.strip()
        if not text:
            raise VoiceError("Type a text to test.")
        if len(text) > CUSTOM_TEXT_MAX:
            raise VoiceError(f"A test text can have at most {CUSTOM_TEXT_MAX} characters.")
        return PreviewText("custom", None, None, text)
    raise VoiceError("Unknown kind of test text.")


def _cannot_test(problem: str, language: str) -> str:
    """Why the voice cannot be tested in `language`, as `film_take`'s problem names it."""
    language_name = language_names.get(language, language)
    if problem == "language":
        return f"Record a {language_name} take first."
    if problem == "waiting":
        return f"The words of the {language_name} takes are not heard yet."
    if problem == "unusable":
        return f"No {language_name} take has usable words."
    return "This voice is being deleted."


def run_preview(
    conn: sqlite3.Connection,
    client,
    voices: Storage,
    owner_id: int,
    job: PreviewJob,
    *,
    voice_lock: "threading.Lock | None" = None,
    job_lock: "threading.Lock | None" = None,
    on_saved=None,
    progress=None,
    cancelled=None,
) -> VoiceTest:
    """Make, store and return one test (D80).

    `voice_lock` is the app's one voice call at a time (D52): resolving the take, reading
    its audio and the speak call run while it is held, so the test speaks from the take
    it finds there and the hearing job hears no newer one meanwhile. The
    measure runs after it is released, so a film does not wait on the first embedding of
    all the takes. `job_lock` is the caller's job lock, the publication barrier: the
    cancel check, the store and `on_saved(test)` run while it is held, and no model call
    does. `progress` is never called under it, because the caller's progress takes the
    same lock.

    Raises `Cancelled` when `cancelled()` turns True before the store, and `PreviewFailed`
    with the reason the panel shows.
    """
    speaker = token(job.voice_id)
    changed = ("The take of the test was deleted while the test ran, or the voice is being deleted. "
               "Generate it again.")

    def report(stage: str, fraction: float) -> None:
        if progress is not None:
            progress(stage, fraction)

    def stop_if_cancelled(stage: str) -> None:
        if cancelled is not None and cancelled():
            raise Cancelled(stage)

    with voice_turn(voice_lock, progress, cancelled, 0.02):
        report("reading the take", 0.05)
        found = film_take(conn, owner_id, job.voice_id, job.language)
        voice = get_voice(conn, owner_id, job.voice_id)
        if found.problem is not None or voice is None:
            raise PreviewFailed(_cannot_test(found.problem or "deleting", job.language))
        prompt = found.take
        try:
            reference = reference_for(conn, voices, prompt)
        except OSError as exc:
            raise PreviewFailed(
                f"Take #{prompt.id} of «{voice.name}» could not be read ({_reason(exc)}).") from exc

        report("generating the voice", 0.10)
        with _Ticking(progress, "generating the voice", (0.10, MEASURE_FROM),
                      len(job.text.text) / 14 * CLONE_RTF_ESTIMATE):
            # The last step before the call, after every progress report, so a cancel that
            # returned during one makes no model call. Outside the `try` below: `Cancelled`
            # is a `RuntimeError`, and the broad handler would turn it into a failure.
            stop_if_cancelled("generating the voice")
            started = time.monotonic()
            try:
                reply = client.speak(job.text.text, job.language, speaker, None, None,
                                     references={speaker: reference})
            except Exception as exc:  # every reason is reported, none swallowed
                raise PreviewFailed(
                    f"The models service could not make the test ({_reason(exc)}).") from exc
            elapsed = time.monotonic() - started

    if ((reply.get("conditioning") or {}).get(speaker) or {}).get("state") not in ("built", "restored"):
        raise PreviewFailed(f"The models service did not speak «{voice.name}» from its take.")

    def before_embedding(stage: str, done: float) -> None:
        report(stage, MEASURE_FROM + (MEASURE_TO - MEASURE_FROM) * done)
        stop_if_cancelled(stage)

    likeness = likeness_error = None
    try:
        likeness = measure_test(conn, client, voices, owner_id, job.voice_id, language=job.language,
                                audio=reply["audio"], before_embedding=before_embedding)
    except Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - every reason is stored and shown
        logger.warning("voice test of voice %s: %s failed: %s", job.voice_id, MEASURING, _reason(exc))
        likeness_error = _reason(exc)

    report("saving the test", MEASURE_TO)
    with job_lock if job_lock is not None else nullcontext():
        stop_if_cancelled("saving the test")
        if reply.get("revision") != job.revision:
            raise PreviewFailed("The voice-copy model changed while the test ran. Generate it again.")
        test = save_test(
            conn, voices, owner_id, job.voice_id, language=job.language,
            text_kind=job.text.kind, text_id=job.text.text_id, example_id=job.text.example_id,
            text=job.text.text, prompt=prompt,
            model=reply["model"], revision=reply["revision"], elapsed_seconds=elapsed,
            audio=reply["audio"], likeness=likeness, likeness_error=likeness_error,
        )
        if test is None:
            raise PreviewFailed(changed)
        logger.info("voice test of voice %s stored: take #%s, %s %s, %.1fs", job.voice_id,
                    test.prompt_example_id, test.model_key, test.revision, elapsed)
        if on_saved is not None:
            on_saved(test)
    return test
