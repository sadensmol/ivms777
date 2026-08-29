"""Preparing one memory as a film, in one language (design §11.1).

Runs OFFLINE, on a background thread. Four stages, and only one of them touches a
model: read the signals, direct the edit, write the story, generate the audio. What it
produces — an `Edit` and at most two WAVs — is everything the player needs, so playback
is plain WebGL with no model anywhere near it.

**Narration is MANDATORY; music is not.** A film is a story being told — with no voice
it is a slideshow, which is not what this feature is. So a script that cannot be
written in the film's language, or cannot be spoken, FAILS the whole prepare: the row
is stored `status='failed'` carrying the reason, and the page shows that reason instead
of a player. Music is the one artifact that may go missing (a 503 from an `off` slot,
or a refusal) and still leave a watchable, silent film.
"""

import hashlib
import logging
import sqlite3

import httpx

from albums.memory_store import current_signature
from feed.beats import bar_times, estimate_bpm, onset_envelope
from feed.conductor import arrange
from feed.director import (
    direct,
    with_arrangement,
    with_lines,
    with_narration,
    with_narration_timing,
    with_script,
)
from feed.language import DEFAULT
from feed.mood import SAMPLING
from feed.mood import energy as measure_energy
from feed.notes import notes_for_prompt, read_notes
from feed.score import plan_sections
from feed.signals import digest, read_signals
from feed.store import (
    Presentation,
    prune_orphan_audio,
    read_presentation,
    save_presentation,
)
from feed.story import _asks_for_a_voice as asks_for_a_voice
from feed.story import uncovered_notes, write_story
from feed.voice import catalogue_for_prompt, offered, pick_speaker, resolve
from storage.base import Storage
from storage.keys import content_key
from storage.wav import decode_wav, duration_seconds

logger = logging.getLogger(__name__)

# Every stage is named for WHAT IT IS MAKING, not for the code that makes it: the bar
# is the only thing the user can see during a run that takes minutes, so "composing the
# music" and "recording the voice" tell them where they are, while "stage 4/6" does not.
#
# What each stage costs, as a fraction of a whole run. These are the measured shape of
# a real run, not guesses: signals and directing are milliseconds over a few dozen
# rows, the story pass is one LLM call, narration is seconds — and MUSIC DOMINATES,
# because a 30 s clip is 1 500 autoregressive tokens and takes minutes.
_STAGES: tuple[tuple[str, float], ...] = (
    ("reading your photos", 0.02),
    ("planning the shots", 0.05),
    ("writing the narration", 0.18),
    ("recording the voice", 0.30),
    ("composing the music", 0.92),
    ("matching effects to the music", 0.97),
    ("saving the film", 1.00),
)


class Cancelled(RuntimeError):
    """The user pressed Cancel. Raised between stages so a long run stops promptly
    without leaving a half-written presentation."""


class NarrationFailed(RuntimeError):
    """No voice could be produced for this film, so there is no film.

    Raised only AFTER a `status='failed'` row carrying the same message is stored, so
    the page can state the reason the moment the user reloads it.
    """


# The score now covers the film (§11.1) — `feed/score.py` owns its length, and the
# player no longer loops anything.

# A film does not stop on the last spoken word. This is the breath after it, where the
# closing shot is allowed to finish its move.
# The film holds after the last word — long enough to be an ending rather than the
# voice stopping and the picture vanishing with it. The score's own fade-out is 1.2 s
# (`audio/musicgen.py`), so this covers it and leaves the last photograph on screen.
_TAIL_SECONDS = 5.0


def build_presentation(
    conn: sqlite3.Connection,
    client,
    audio: Storage,
    owner_id: int,
    group_id: int,
    *,
    language: str = DEFAULT,
    planner_model: str = "gemma4-E2B",
    base_seconds: float = 4.5,
    max_shots: int = 40,
    progress=None,
    cancelled=None,
) -> Presentation:
    """Generate, store and record one memory's film.

    Raises `LookupError` when `group_id` is not this owner's memory — a missing memory
    is a routing bug, not a degraded film, so unlike the audio calls it is not
    swallowed.
    """
    stages = dict(_STAGES)

    def step(stage: str) -> None:
        """Report a stage as STARTED, so the bar moves when work begins rather than
        only when it ends — and check for cancellation at the same boundary."""
        if cancelled is not None and cancelled():
            raise Cancelled(stage)
        if progress is not None:
            progress(stage, stages[stage])

    # EVERY CHOICE THIS BUILD MAKES, recorded as it is made (§11.1). It is stored with
    # the film and logged as it happens, because "which voice read this?" and "did the
    # bracket work?" had no answer at all once the build thread ended — and a question
    # with no answer gets guessed at instead of fixed.
    decisions: dict = {"language": language, "group_id": group_id,
                       "planner_model": planner_model}

    step("reading your photos")
    title, description = _memory_row(conn, owner_id, group_id)
    _mark_running(conn, group_id, language, current_signature(conn, owner_id))
    shot_signals = read_signals(conn, group_id)
    step("planning the shots")
    edit = direct(shot_signals, base_seconds=base_seconds, max_shots=max_shots)

    step("writing the narration")
    # The viewer's notes, whole. Nothing is parsed out of them: every pass that can act
    # on an instruction is shown the note verbatim and reads it itself (§11.3).
    viewer_notes = read_notes(conn, group_id)
    # A note points at whatever was ON SCREEN when it was written, so the previous cut
    # is what turns "0:12" into "photograph 3" — which is what lets a fact attach to a
    # picture instead of to a moment (§11.3).
    previous_seconds, photo_at = _previous_cut(conn, group_id, language, shot_signals)
    # The RICH briefs, not the bare vision sentence: the story pass is the one WRITING
    # the memory, and it was previously blind to `vibe=festive`, `occasion=holiday`,
    # `season_weather=snow`, the place and the hour — every label that turns a
    # description of pictures into a memory of an evening (§11.1). Plus a digest of
    # what the memory is as a WHOLE, which no per-photo line can carry.
    story = write_story(
        client, planner_model, title=title, description=description, edit=edit,
        shots=[s.brief() for s in shot_signals[: len(edit.shots)]],
        language=language, digest=digest(shot_signals),
        # What the viewer typed into an earlier cut of this same film (§11.3).
        # The PREVIOUS cut's length, so a note's timestamp can be expressed as a
        # position in the script the model is about to write.
        notes=notes_for_prompt(viewer_notes, previous_seconds, photo_at=photo_at),
        # The speakers this language offers, as declared in the catalog: the story pass
        # CASTS the film from them (CLAUDE.md § "A model's own metadata is DECLARED").
        speakers=catalogue_for_prompt(language),
        # What the server will actually take. Asked here, once, so the prompt is cut to
        # fit instead of being rejected whole — which is what an over-long listing did:
        # llama-server refused the call and the film failed for want of narration.
        context_tokens=_context_tokens(client),
    )
    # THE STORY IS THE FILM. The shot list is rebuilt from the script's passages, not
    # from the photo list: a shot is a passage, and a photograph is what illustrates it.
    # So a picture the story returns to appears twice, and one it never needed does not
    # appear at all (§11.1).
    decisions["story"] = {
        "context_tokens": _context_tokens(client),
        "passages": len(story.script),
        "narrator_asked": story.narrator,
        "score_plan": [{"from": index, "brief": brief} for index, brief in story.score],
        "music_prompt": story.music_prompt,
        "notes_uncovered": uncovered_notes(story.narration, viewer_notes and notes_for_prompt(
            viewer_notes, previous_seconds, photo_at=photo_at)),
    }
    logger.info("story: %s", decisions["story"])
    edit = with_script(edit, story.script)
    # Sparse title cards, NOT a caption per shot — and the machine labels the model was
    # shown are cleared, so a failed story pass leaves a wordless film rather than 40
    # lines of "Two people outdoors".
    edit = with_lines(edit, story.lines_for(len(edit.shots)))
    edit = with_narration(edit, story.narration)

    step("recording the voice")
    notes: list[str] = []
    signature = current_signature(conn, owner_id)
    if not story.narration:
        _fail(conn, group_id, language, signature, edit,
              f"No narration: the planner would not write this memory in {language}.",
              decisions)
    # The language picks the MODEL (in the models service) and the memory's mood picks
    # the SPEAKER — the user chooses neither (§11.1).
    # THE MODEL CASTS THE FILM; the code only checks the cast is real. A name it
    # invented would raise inside the TTS child and fail the whole film (§11.1), so
    # every choice goes through `resolve` against the catalog, and anything it did not
    # or could not answer falls back to the deterministic rule.
    speaker = resolve(language, story.narrator) or pick_speaker(language, shot_signals)
    bracketed = _bracketed_passages(viewer_notes, previous_seconds, len(story.script))
    passage_voices = _passage_voices(story, language, speaker, bracketed)
    # WHAT WAS ASKED FOR vs WHAT WAS CAST, per passage. A name the story pass invented
    # is dropped by `resolve` and the passage silently falls back to the narrator — so
    # a viewer's "[суровый мужской голос]" became a woman's voice with nothing anywhere
    # saying so. Now it says so.
    wanted_voice = asks_for_a_voice(notes_for_prompt(
        viewer_notes, previous_seconds, photo_at=photo_at))
    cast = [{"passage": index, "asked": asked, "used": used}
            for index, (asked, used) in enumerate(zip(story.voices, passage_voices))
            if asked or (used and used != speaker)]
    decisions["cast"] = {
        "narrator_used": speaker,
        "narrator_asked_was_real": bool(resolve(language, story.narrator)),
        "notes_asked_for_a_voice": wanted_voice,
        "bracketed_passages": sorted(bracketed) if bracketed is not None else None,
        "model_cast": list(story.voices),
        "passages": cast,
        "offered": [s.name for s in offered(language)],
    }
    logger.info("cast: %s", decisions["cast"])
    if wanted_voice and not cast:
        # Visible on the page, not just in a log: the note asked for something and the
        # film did not do it.
        notes.append("a note asked for a different voice and none was cast")
    try:
        # The script's passages are handed over as the EXACT chunks to speak, so
        # segment `i` of the returned audio is the words for shot `i`. That is what
        # makes the film synchronised by arithmetic instead of by guesswork.
        narration = client.speak(story.narration, language, speaker,
                                 story.chunks or None, passage_voices or None)
    except Exception as exc:  # every reason is reported, none swallowed
        logger.exception("narrating the film failed")
        _fail(conn, group_id, language, signature, edit,
              f"No voice: the models service could not speak {language} "
              f"({_reason(exc)}).", decisions)
    narration_key = _store_audio(audio, narration["audio"])
    # The NARRATION decides how long the film is. It used to be the other way round —
    # a runtime summed from nominal shot holds, with the voice left to fall where it
    # fell — which is precisely how picture and sound ended up as two things happening
    # near each other rather than one thing (§11.1).
    spoken_seconds = duration_seconds(narration["audio"])
    segments = [s for s in narration.get("segments", []) if isinstance(s, dict)]
    runtime = spoken_seconds + _TAIL_SECONDS if spoken_seconds else edit.seconds
    # Every shot now starts exactly when its own passage is spoken.
    edit = with_narration_timing(edit, segments, runtime)

    step("composing the music")
    # A SCORE, not a clip (§11.1): two to four sections that follow the story's own
    # turns, each generated as a continuation of the one before it. The story pass
    # planned where they change (it wrote the passages, so it knows); `feed/score.py`
    # writes each brief from the photographs that stretch actually shows, because a
    # single prompt can only ever describe the average of the whole memory — which is
    # what made every film sound like one bed under different pictures.
    energy = measure_energy(shot_signals)
    sections = plan_sections(
        shot_signals, _passages(segments, story.script, edit), runtime, energy,
        model_plan=story.score, hint=story.music_prompt, title=title,
    )
    decisions["score"] = {
        "energy": energy,
        "seed": _seed(group_id, language, signature),
        **SAMPLING[energy],
        "sections": [{"role": s.role, "at": s.at, "seconds": s.seconds, "brief": s.brief}
                     for s in sections],
    }
    logger.info("score: %s", decisions["score"])
    music = _optional(
        client.music, sections=[{"brief": s.brief, "seconds": s.seconds}
                                for s in sections],
        # The SAME memory scores the same way twice, and a different memory never gets
        # this one's music: the seed is the film's own identity (§11.1).
        seed=_seed(group_id, language, signature), **SAMPLING[energy],
    )
    music_key = _store_audio(audio, music["audio"]) if music is not None else None
    if music is None:
        notes.append("no music — the models service refused to compose")
    decisions["score"]["model"] = (music or {}).get("model")
    decisions["score"]["segments"] = (music or {}).get("segments", [])

    step("matching effects to the music")
    bars, bpm = _pulse(music)
    # ONE model call that places every photograph, effect and cut against the beat grid
    # and the narration's own sentence timings. It never raises: a failure falls back to
    # an even, beat-snapped arrangement (feed/conductor.py).
    # The conductor now chooses EFFECTS and transitions only. Order and timing already
    # belong to the story and the voice, and no model gets to overrule them.
    aligned = bool(segments) and len(segments) == len(edit.shots)
    edit = with_arrangement(edit, arrange(
        client, planner_model, title=title, description=description, edit=edit,
        # The library is thirty effects now, so the conductor is shown the ones this
        # memory's mood can actually use (§11.1) — the whole table is 8 000 tokens and
        # a birthday has no business being offered `soft_focus`.
        energy=energy,
        shots=[shot_signals[i].brief() if i < len(shot_signals) else ""
               for i, _ in (story.script or [(n, "") for n in range(len(edit.shots))])],
        segments=segments, bars=bars, bpm=bpm, total_seconds=runtime,
        digest=digest(shot_signals),
        # The notes themselves, whole: the conductor reads any instruction in them and
        # places the effects and lengths it asks for.
        directions=notes_for_prompt(viewer_notes, previous_seconds, photo_at=photo_at),
    ), keep_timing=aligned)

    step("saving the film")
    decisions["film"] = {"runtime": runtime, "shots": len(edit.shots),
                         "narration_seconds": spoken_seconds,
                         "conductor_kept_timing": aligned}
    logger.info("film: %s", decisions["film"])
    presentation = Presentation(
        group_id=group_id,
        language=language,
        signature=signature,
        timeline=edit.as_dict(),
        music_key=music_key,
        narration_key=narration_key,
        status="ready",
        # `error` is not a failure here — the film IS ready. It records what is
        # MISSING and why, so the page can say "no voice, because…" instead of just
        # playing silent and leaving the user to guess.
        error="; ".join(notes) or None,
        decisions=decisions,
    )
    save_presentation(conn, presentation)
    # The table is authoritative right now — the one safe moment to drop the WAVs a
    # previous memory build left behind when its rows cascaded away (§11.1).
    prune_orphan_audio(conn, audio)
    return presentation


def _passage_voices(
    story, language: str, film_voice: str | None, bracketed: set[int] | None = None,
) -> list[str | None]:
    """One speaker per passage: the film's own, except where the story pass cast another.

    The casting is the model's — it saw the notes, the photographs and the speaker
    table together, which is the only place that judgement can be made. All that
    happens here is `resolve`, which checks each name is real (an invented one raises
    inside the TTS child and narration is mandatory, §11.1), and the BRACKET GATE below.

    There is no fallback that reads the brackets itself. One existed and it needed the
    phrase written its way; a note whose wording it did not know was silently ignored.

    **The gate is positional, never lexical.** `voice` had to become a required schema
    field, because an optional one is simply never emitted and every film was read by
    the fallback speaker (measured). Required, the model fills in EVERY passage — it
    cast `aidar`, `baya`, `kseniya` over three passages when one note asked for one
    voice — and a film that changes narrator every sentence is a different bug. So an
    override is kept only at a passage a viewer's BRACKET points at. Which passage that
    is comes from the note's own position; what the bracket MEANS is still the model's
    to read, and no word of it is inspected here.
    """
    if not story.script:
        return []
    cast = [resolve(language, name) for name in story.voices]
    cast += [None] * (len(story.script) - len(cast))
    if bracketed is not None:
        cast = [name if index in bracketed else None for index, name in enumerate(cast)]
    if not any(cast):
        return []
    return [name or film_voice for name in cast]


def _bracketed_passages(notes, film_seconds: float, passages: int) -> set[int] | None:
    """The passages a viewer's `[bracket]` points at, by POSITION.

    A note carries the second of the previous cut it was written at, so its share of
    that cut is its share of the story — and the passage at that share is the one it is
    about. None when there is no previous cut to measure against (nothing is known about
    where a note belongs, so nothing is gated).
    """
    if not notes or film_seconds <= 0 or passages <= 0:
        return None
    marked = set()
    for note in notes:
        if "[" not in note.text:
            continue
        share = min(max(note.at_seconds / film_seconds, 0.0), 0.999)
        marked.add(int(share * passages))
    return marked


def _previous_cut(
    conn: sqlite3.Connection, group_id: int, language: str, signals: list
) -> tuple[float, object]:
    """`(runtime, at -> photograph number)` for the cut the viewer was WATCHING.

    Their timestamps are relative to that film, so it is the only scale on which
    "0:12" becomes "a fifth of the way through" — and the only way to know which
    photograph they were looking at when they wrote it.

    The number returned is the photograph's position in THIS prompt's listing, not in
    the old cut, because that is the numbering the model is about to read.
    """
    numbering = {signal.photo_id: index + 1 for index, signal in enumerate(signals)}
    previous = read_presentation(conn, group_id, language)
    if previous is None:
        return 0.0, lambda _seconds: None
    try:
        seconds = float(previous.timeline.get("seconds") or 0.0)
        shots = [s for s in previous.timeline.get("shots", []) if isinstance(s, dict)]
    except (TypeError, ValueError, AttributeError):
        return 0.0, lambda _seconds: None

    def photo_at(at: float):
        showing = None
        for shot in shots:
            if float(shot.get("at", 0.0)) <= at:
                showing = numbering.get(shot.get("photo_id"))
            else:
                break
        return showing

    return seconds, photo_at


def _mark_running(
    conn: sqlite3.Connection, group_id: int, language: str, signature: str
) -> None:
    """Leave a durable trace that a run STARTED.

    A prepare runs on a daemon thread. If the process goes away underneath it — a code
    reload, a restart, a crash — the thread dies without a `finally`, nothing is
    written, and the page simply shows the Prepare button again. That is
    indistinguishable from never having clicked it, and it is the exact failure that
    gets mis-diagnosed as "the button does nothing".

    Only written when there is NOTHING to lose: an existing ready film is left alone, so
    a Regenerate that gets interrupted still leaves the previous film playable.
    """
    existing = read_presentation(conn, group_id, language)
    if existing is not None and existing.status == "ready":
        return
    save_presentation(conn, Presentation(
        group_id=group_id, language=language, signature=signature,
        timeline={"shots": []}, music_key=None, narration_key=None,
        status="running", error=None,
    ))


def _pulse(music: dict | None) -> tuple[list[float], float]:
    """The soundtrack's downbeats and tempo, or an empty grid when there is no music.

    Pure signal processing on bytes we just generated — no model, and nothing that
    imports torch (CLAUDE.md § one model process).
    """
    if music is None:
        return [], 72.0
    try:
        bars = bar_times(music["audio"])
        samples, rate = decode_wav(music["audio"])
        envelope, hop = onset_envelope(samples, rate)
        return bars, estimate_bpm(envelope, hop)
    except Exception:  # a film with no beat grid still plays, just on its own pacing
        logger.exception("beat detection failed; pacing the film without a grid")
        return [], 72.0


def _passages(segments: list[dict], script, edit) -> list[tuple[float, int]]:
    """`(second it starts, shot index)` per narration passage — the VOICE's own timings.

    This is what lets the music turn where the story turns rather than on a stopwatch.
    Empty when the voice reported no segments, in which case the score falls back to an
    even split (`feed/score.py`).
    """
    shots = [shot for shot, _ in script] if script else list(range(len(edit.shots)))
    return [(float(segment.get("start", 0.0)), shots[index] if index < len(shots) else 0)
            for index, segment in enumerate(segments)]


def _seed(group_id: int, language: str, signature: str) -> int:
    """A stable seed per film. Not `None` (which leaves the sampler to the clock) and
    not a constant (which would give every memory the same music)."""
    raw = f"{group_id}:{language}:{signature}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _context_tokens(client) -> int | None:
    """How many tokens the text server accepts, or None when it cannot be asked.

    Best-effort: an unreachable models service is about to fail the story call anyway,
    and it must not be this read that raises."""
    try:
        return client.text_context()
    except Exception:  # noqa: BLE001 - a missing budget only means "do not trim"
        logger.warning("could not read the model context; the prompt is not budgeted")
        return None


def _fail(
    conn: sqlite3.Connection, group_id: int, language: str, signature: str, edit,
    reason: str, decisions: dict | None = None,
) -> None:
    """Record this film as FAILED, with the reason, then raise.

    The row is written before raising on purpose: the page reads the presentation, so
    storing the failure is the only way "show the error immediately" survives the
    background thread ending.
    """
    save_presentation(conn, Presentation(
        group_id=group_id,
        language=language,
        signature=signature,
        timeline=edit.as_dict(),
        music_key=None,
        narration_key=None,
        status="failed",
        error=reason,
        # A failed film records what it HAD decided before it failed — the story it
        # got, the cast it resolved — because that is exactly the state a failure is
        # diagnosed from.
        decisions=decisions,
    ))
    logger.error("film failed: %s | decisions: %s", reason, decisions)
    raise NarrationFailed(reason)


def _reason(exc: Exception) -> str:
    """The models service's OWN words, so a failure is diagnosable from the page.

    FastAPI puts the message in `{"detail": …}`; anything else falls back to the
    exception's text.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            detail = response.json().get("detail")
        except Exception:  # noqa: BLE001 — a non-JSON body is a fact, not an error
            detail = None
        if detail:
            return str(detail)
    return str(exc) or type(exc).__name__


def _memory_row(conn: sqlite3.Connection, owner_id: int, group_id: int) -> tuple[str, str]:
    row = conn.execute(
        "SELECT name, COALESCE(description, '') AS description FROM groups"
        " WHERE id = ? AND owner_id = ? AND kind = 'memory'",
        (group_id, owner_id),
    ).fetchone()
    if row is None:
        raise LookupError(f"no memory {group_id} for owner {owner_id}")
    return row["name"], row["description"]


def _optional(call, *args, **kwargs):
    """Run a models-service call for an artifact the film can do without — MUSIC, and
    only music (narration is mandatory, see the module docstring).

    A 503 is the documented "this is turned off / cannot do that" answer (§5.1) and
    yields None. Anything else is logged and ALSO yields None: one missing soundtrack
    must not cost the whole film, which is still perfectly watchable without it.
    """
    try:
        return call(*args, **kwargs)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 503:
            logger.warning("film audio call failed: %s", exc)
        return None
    except Exception:  # a silent film beats no film
        logger.exception("film audio call raised")
        return None


def _store_audio(audio: Storage, data: bytes) -> str:
    """Content-addressed, like every other blob (§6.1): re-preparing a memory whose
    audio came out identical rewrites the same key instead of growing the store."""
    key = content_key(hashlib.sha256(data).hexdigest(), ".wav")
    audio.write(key, data)
    return key
