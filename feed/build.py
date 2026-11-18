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
import secrets
import sqlite3
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager

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
from feed.narration import draft_story, read_draft, rebase_consumed, timeline_script
from feed.notes import notes_for_prompt, read_notes
from feed.saved_voices import (
    castable_takes,
    film_take,
    get_voice,
    read_narrator,
    reference_for,
    token,
)
from feed.score import plan_sections
from feed.signals import digest, read_signals
from feed.store import (
    Presentation,
    prune_orphan_audio,
    read_presentation,
    save_presentation,
)
from feed.story import _LANGUAGE_NAME as language_names
from feed.story import _asks_for_a_voice as asks_for_a_voice
from feed.story import match_photographs, uncovered_notes, write_story
from feed.voice import (
    SavedCast,
    automatic_for_prompt,
    offered,
    pick_speaker,
    resolve,
    resolve_saved,
    voice_label,
)
from models.catalog import cloning_entry_for
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


# Seconds of compute per second of audio, MEASURED on this mac (musicgen at 4.75-4.90x
# realtime, both small and medium). It is only ever used to say how long is left, never
# to decide anything, so a board that runs at a different rate gets a wrong estimate and
# a correct film.
MUSIC_RATE = 4.8

# Seconds of compute per second of saved-voice narration (D28). An ESTIMATE, for the
# moving bar only: VoxCPM2's rate on this mac is not measured yet (§6.1 replaces it).
# Narration runs at about 14 characters a second.
CLONE_RTF_ESTIMATE = 1.0


class _Ticking:
    """A stage that reports its own progress while it waits on ONE long call.

    Composing the score is minutes of work behind a single blocking request, so the bar
    parked at 92% for the whole of it — and a number that does not move is
    indistinguishable from a hung process, which is exactly what CLAUDE.md forbids and
    exactly what happened: the score took five minutes and looked broken for four of
    them.

    The share is an ESTIMATE from the measured rate, and it says so: the label counts
    down in minutes rather than pretending to know. It never goes backwards, never
    reaches the next stage's number, and it holds at the ceiling if the estimate runs
    out before the call returns — a slow board shows "nearly there", not 100%.
    """

    def __init__(self, progress, stage: str, span: tuple[float, float], expect: float):
        self._progress = progress
        self._stage = stage
        self._low, self._high = span
        self._expect = max(expect, 1.0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if self._progress is None:
            return self
        started = time.monotonic()

        def tick() -> None:
            while not self._stop.wait(1.0):
                spent = time.monotonic() - started
                share = min(spent / self._expect, 1.0)
                left = self._expect - spent
                if left > 45:
                    label = f"{self._stage} — about {round(left / 60) or 1} min left"
                elif left > 5:
                    label = f"{self._stage} — about {int(left)}s left"
                else:
                    label = f"{self._stage} — nearly there"
                # Stops just short of the next stage: only the work finishing may
                # advance past this.
                fraction = self._low + (self._high - self._low) * 0.97 * share
                try:
                    self._progress(label, fraction)
                except Exception:  # noqa: BLE001 - a bar must never fail a film
                    return

        self._thread = threading.Thread(target=tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class Cancelled(RuntimeError):
    """The user pressed Cancel. Raised between stages so a long run stops promptly
    without leaving a half-written presentation."""


@contextmanager
def voice_turn(lock: "threading.Lock | None", progress, cancelled, fraction: float):
    """Hold `lock`, the app's one voice call at a time (D52), for the body.

    A caller that waits says so on its bar ("waiting for the voice model" at `fraction`)
    and still honours Cancel: `cancelled()` is checked before the first acquire, at each
    1-second miss and once more after the acquire, so a job cancelled while it waited
    never takes the voice unit. `None` holds nothing."""
    if lock is None:
        yield
        return
    stage = "waiting for the voice model"
    if cancelled is not None and cancelled():
        raise Cancelled(stage)
    reported = False
    while not lock.acquire(timeout=1.0):
        if not reported and progress is not None:
            progress(stage, fraction)
        reported = True
        if cancelled is not None and cancelled():
            raise Cancelled(stage)
    try:
        if cancelled is not None and cancelled():
            raise Cancelled(stage)
        yield
    finally:
        lock.release()


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
    voices: Storage | None = None,
    voice_lock: "threading.Lock | None" = None,
    profile: str | None = None,
) -> Presentation:
    """Generate, store and record one memory's film.

    `voices` is the saved-voice store (`data_dir/voices`). A film narrated by a saved
    voice, chosen or picked by Automatic, needs it. Without it a chosen saved narrator
    fails the film with the reason, and Automatic offers no saved voice.

    `profile` is this app's hardware profile: Automatic offers saved voices only where
    voice copy speaks the film's language on it (D105). None offers none.

    `voice_lock` is the app's one voice call at a time (D52): resolving each used take
    again, reading its audio and the speak call run while it is held (`voice_turn`).

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
    photo_ids = [shot.photo_id for shot in edit.shots]
    # THE USER'S NARRATOR, when they chose one (D17, D18): a preset or one of their saved
    # voices. A saved-voice film is narrated by saved voices only. Neither choice ever
    # becomes Automatic: a narrator that cannot speak fails the film with the reason,
    # before any model call (D33, D38). A row with neither column is a deleted saved voice.
    chosen = read_narrator(conn, group_id, language)
    saved_chosen = chosen is not None and chosen.preset_speaker is None
    preset = None
    if chosen is not None and not saved_chosen:
        preset = resolve(language, chosen.preset_speaker)
        if preset is None:
            _fail(conn, group_id, language, current_signature(conn, owner_id), edit,
                  f"The preset voice «{chosen.preset_speaker}» is not offered for "
                  f"{language_names.get(language, language)} films. Choose another narrator.",
                  decisions)
    if saved_chosen:
        saved = _saved_cast(conn, voices, owner_id, group_id, language, chosen, edit, decisions)
    elif (chosen is None and profile and voices is not None
          and cloning_entry_for(profile, language) is not None):
        # Automatic may pick a saved voice (D105). A name a preset already holds is left
        # out: the story's answer would be ambiguous.
        saved = [(voice, take) for voice, take in castable_takes(conn, owner_id, language)
                 if resolve(language, voice.name) is None]
    else:
        saved = []
    saved_cast = [SavedCast(token(voice.id), voice.name) for voice, _ in saved]

    step("writing the narration")
    # The viewer's notes, whole. Nothing is parsed out of them: every pass that can act
    # on an instruction is shown the note verbatim and reads it itself (§11.3).
    viewer_notes = read_notes(conn, group_id)
    # A note points at whatever was ON SCREEN when it was written, so the previous cut
    # is what turns "0:12" into "photograph 3" — which is what lets a fact attach to a
    # picture instead of to a moment (§11.3).
    previous_seconds, photo_at = _previous_cut(conn, group_id, language, shot_signals)
    # What the server will actually take. Asked here, once, so the prompt is cut to fit
    # instead of being rejected whole — which is what an over-long listing did: llama-server
    # refused the call and the film failed for want of narration.
    context_tokens = _context_tokens(client)
    photos = [s.description or s.caption for s in shot_signals[: len(edit.shots)]]
    # THE USER'S EDITED NARRATION, when there is one: its words are the script, spoken as
    # written, and no story pass runs. Only each passage's photograph is asked for again
    # (§11.1 "Edited narration is spoken as written").
    draft = read_draft(conn, group_id, language)
    if draft is not None:
        story = match_photographs(client, planner_model, draft_story(draft, photo_ids), photos,
                                  len(edit.shots), context_tokens, cancelled)
    else:
        # The RICH briefs, not the bare vision sentence: the story pass is the one WRITING
        # the memory, and it was previously blind to `vibe=festive`, `occasion=holiday`,
        # `season_weather=snow`, the place and the hour — every label that turns a
        # description of pictures into a memory of an evening (§11.1). Plus a digest of
        # what the memory is as a WHOLE, which no per-photo line can carry.
        story = write_story(
            client, planner_model, title=title, description=description, edit=edit,
            shots=[s.brief() for s in shot_signals[: len(edit.shots)]],
            language=language, digest=digest(shot_signals),
            photos=photos,
            facts=digest(shot_signals[: len(edit.shots)], guesses=False),
            cancelled=cancelled,
            # What the viewer typed into an earlier cut of this same film (§11.3).
            # The PREVIOUS cut's length, so a note's timestamp can be expressed as a
            # position in the script the model is about to write.
            notes=notes_for_prompt(viewer_notes, previous_seconds, photo_at=photo_at),
            # The speakers this language offers, as declared in the catalog, and the saved
            # candidates by name: an Automatic film is CAST from them (CLAUDE.md § "A model's
            # own metadata is DECLARED", D106). A chosen narrator reads every passage, so
            # that story pass is shown no voice.
            speakers=automatic_for_prompt(language, saved_cast) if chosen is None else "",
            context_tokens=context_tokens,
        )
    # THE STORY IS THE FILM. The shot list is rebuilt from the script's passages, not
    # from the photo list: a shot is a passage, and a photograph is what illustrates it.
    # So a picture the story returns to appears twice, and a picture no sentence is about
    # does not appear (§11.1).
    decisions["story"] = {
        **({"source": "draft", "draft_revision": draft.revision} if draft is not None else {}),
        "context_tokens": context_tokens,
        "passages": len(story.script),
        # Whether the final photographs keep the memory's order, "kept" or "broken". A
        # diagnostic: nothing acts on it (§11.1).
        "memory_order": story.memory_order,
        "votes": list(story.votes),
        "photos": [index for index, _text in story.script],
        "matched": story.matched,
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
    # The language picks the MODEL (in the models service). The SPEAKER is the user's
    # choice when they made one on /feed, else the story pass's cast, else the memory's
    # mood (§11.1).
    # THE MODEL CASTS THE FILM; the code only checks the cast is real. A name it
    # invented would raise inside the TTS child and fail the whole film (§11.1), so
    # every choice goes through `resolve` against the catalog (or `resolve_saved` against
    # the saved candidates), and anything it did not or could not answer falls back to
    # the deterministic rule.
    bracketed = _bracketed_passages(viewer_notes, previous_seconds, len(story.script))
    saved_pick = None
    if saved_chosen:
        # The user's choice reads EVERY passage, whatever the story pass cast (§11.1).
        speaker = token(chosen.voice_id)
        passage_voices = []
        narrator_asked_was_real = bool(resolve_saved(story.narrator, saved_cast))
    elif chosen is not None:
        speaker = preset
        passage_voices = []
        narrator_asked_was_real = bool(resolve(language, story.narrator))
    else:
        # Automatic: the story pass's narrator, a preset first, then a saved candidate.
        # The narrator's kind decides every passage voice, so one engine speaks (D107).
        preset_pick = resolve(language, story.narrator)
        saved_pick = None if preset_pick else resolve_saved(story.narrator, saved_cast)
        if saved_pick is not None:
            speaker = saved_pick
            passage_voices = _passage_voices(story, language, speaker, bracketed,
                                             saved=saved_cast)
        else:
            speaker = preset_pick or pick_speaker(language, shot_signals)
            passage_voices = _passage_voices(story, language, speaker, bracketed)
        narrator_asked_was_real = bool(preset_pick or saved_pick)
    # A saved pick follows every rule of a saved-voice film below, and never falls back
    # to a preset (D109).
    saved_film = saved_chosen or saved_pick is not None
    # WHAT WAS ASKED FOR vs WHAT WAS CAST, per passage. A name the story pass invented
    # is dropped by `resolve` and the passage silently falls back to the narrator — so
    # a viewer's "[суровый мужской голос]" became a woman's voice with nothing anywhere
    # saying so. Now it says so.
    wanted_voice = asks_for_a_voice(notes_for_prompt(
        viewer_notes, previous_seconds, photo_at=photo_at))
    # The labels are stored with the film, so a later delete or rename of a voice does
    # not change what the film says about itself (D37).
    cast = [{"passage": index, "asked": asked, "used": used,
             "used_label": voice_label(used, language, saved_cast).label if used else None}
            for index, (asked, used) in enumerate(zip(story.voices, passage_voices))
            if asked or (used and used != speaker)]
    narrator = voice_label(speaker, language, saved_cast)
    decisions["cast"] = {
        "narrator_used": speaker,
        "narrator_chosen_by": ("user" if chosen is not None
                               else "story" if narrator_asked_was_real else "mood"),
        "narrator_name": narrator.name if narrator is not None else None,
        "narrator_label": narrator.label if narrator is not None else None,
        "narrator_asked_was_real": narrator_asked_was_real,
        "notes_asked_for_a_voice": wanted_voice,
        "bracketed_passages": sorted(bracketed) if bracketed is not None else None,
        "model_cast": list(story.voices),
        "passages": cast,
        "offered": ([voice.name for voice in saved_cast] if saved_chosen
                    else [s.name for s in offered(language)]
                    + [voice.name for voice in saved_cast]),
    }
    logger.info("cast: %s", decisions["cast"])
    if chosen is None and wanted_voice and not cast:
        # Visible on the page, not just in a log: the note asked for something and the
        # film did not do it.
        notes.append("a note asked for a different voice and none was cast")
    by_token = {token(voice.id): voice for voice, _ in saved}
    used = {name: by_token[name] for name in dict.fromkeys([speaker, *passage_voices])
            if name in by_token}
    # One turn covers resolving each take again, reading its audio and the call: a take
    # whose words were heard while this film waited gives it that take, and the hearing
    # job hears nothing while the film holds the lock (D52, §11.1).
    takes: dict = {}
    with voice_turn(voice_lock, progress, cancelled, stages["recording the voice"]):
        references = None
        if saved_film:
            references = {}
            for name, voice in used.items():
                found = film_take(conn, owner_id, voice.id, language)
                if found.problem is not None:
                    _fail(conn, group_id, language, signature, edit,
                          _cannot_narrate(voice.name, found.problem, language), decisions)
                try:
                    references[name] = reference_for(conn, voices, found.take)
                except Exception as exc:  # a file the drain removed meanwhile, or unreadable
                    logger.exception("reading take %s of %s failed", found.take.id, name)
                    _fail(conn, group_id, language, signature, edit,
                          f"Take #{found.take.id} of «{voice.name}» could not be read "
                          f"({_reason(exc)}).", decisions)
                takes[name] = found.take
        started = time.monotonic()
        try:
            # The script's passages are handed over as the EXACT chunks to speak, so
            # segment `i` of the returned audio is the words for shot `i`. That is what
            # makes the film synchronised by arithmetic instead of by guesswork.
            if references is None:
                narration = client.speak(story.narration, language, speaker,
                                         story.chunks or None, passage_voices or None)
            else:
                # Voice copy takes minutes, so the bar moves on an estimate (D28). One
                # call, and never a second one with a preset if it fails (D17).
                with _Ticking(progress, "recording the voice",
                              (stages["recording the voice"], stages["composing the music"]),
                              len(story.narration) / 14 * CLONE_RTF_ESTIMATE):
                    narration = client.speak(story.narration, language, speaker,
                                             story.chunks or None, passage_voices or None,
                                             references=references)
        except Exception as exc:  # every reason is reported, none swallowed
            logger.exception("narrating the film failed")
            _fail(conn, group_id, language, signature, edit,
                  f"No voice: the models service could not speak {language} "
                  f"({_reason(exc)}).", decisions)
        wall_seconds = time.monotonic() - started
    # The reply is the only proof that each take spoke (D86).
    conditioning = narration.get("conditioning") or {}
    for name in takes:
        if (conditioning.get(name) or {}).get("state") not in ("built", "restored"):
            _fail(conn, group_id, language, signature, edit,
                  f"The models service did not speak «{used[name].name}» from its take.",
                  decisions)
    # The NARRATION decides how long the film is. It used to be the other way round —
    # a runtime summed from nominal shot holds, with the voice left to fall where it
    # fell — which is precisely how picture and sound ended up as two things happening
    # near each other rather than one thing (§11.1).
    spoken_seconds = duration_seconds(narration["audio"])
    segments = [s for s in narration.get("segments", []) if isinstance(s, dict)]
    runtime = spoken_seconds + _TAIL_SECONDS if spoken_seconds else edit.seconds
    # Every shot now starts exactly when its own passage is spoken.
    edit = with_narration_timing(edit, segments, runtime)
    passages, script = timeline_script(story, segments, passage_voices, speaker, photo_ids,
                                       token=secrets.token_hex(16))  # pragma: allowlist secret
    decisions["voice"] = {
        "saved": saved_film,
        "model": narration.get("model"),
        "revision": narration.get("revision"),
        "wall_seconds": round(wall_seconds, 3),
        "narration_seconds": spoken_seconds,
        "seed": "unsupported (voxcpm 2.0.3)" if saved_film else None,
    }
    if saved_film:
        decisions["voice"]["takes"] = {
            name: {"example_id": take.id, "sha256": take.content_hash,
                   "state": conditioning[name]["state"]}
            for name, take in takes.items()}
        decisions["cast"]["narrator_model"] = f"spoken from take #{takes[speaker].id}"
    logger.info("voice: %s", decisions["voice"])

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
    # The one call that takes minutes. It reports its own progress while it runs, from
    # the seconds of audio it is about to generate and the measured rate.
    wanted = sum(section.seconds for section in sections)
    with _Ticking(progress, "composing the music",
                  (stages["composing the music"],
                   stages["matching effects to the music"]),
                  wanted * MUSIC_RATE):
        music = _optional(
            client.music, sections=[{"brief": s.brief, "seconds": s.seconds}
                                    for s in sections],
            # The SAME memory scores the same way twice, and a different memory never
            # gets this one's music: the seed is the film's own identity (§11.1).
            seed=_seed(group_id, language, signature), **SAMPLING[energy],
        )
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
    ), keep_timing=aligned, keep_order=bool(story.script))

    step("saving the film")
    decisions["film"] = {"runtime": runtime, "shots": len(edit.shots),
                         "narration_seconds": spoken_seconds,
                         "conductor_kept_timing": aligned}
    logger.info("film: %s", decisions["film"])

    def save_ready() -> Presentation:
        """Write the film's WAVs and its ready row. The ONLY place film audio is
        written (D34): the steps before used the bytes alone, so a film that fails or
        is cancelled earlier leaves no audio behind, and no failure path prunes."""
        presentation = Presentation(
            group_id=group_id,
            language=language,
            signature=signature,
            timeline={**edit.as_dict(), "passages": passages, "script": script},
            music_key=_store_audio(audio, music["audio"]) if music is not None else None,
            narration_key=_store_audio(audio, narration["audio"]),
            status="ready",
            # `error` is not a failure here — the film IS ready. It records what is
            # MISSING and why, so the page can say "no voice, because…" instead of just
            # playing silent and leaving the user to guess.
            error="; ".join(notes) or None,
            decisions=decisions,
        )
        save_presentation(conn, presentation)
        return presentation

    if draft is None and not saved_film:
        presentation = save_ready()
    else:
        # A saved voice the film used may have been deleted while it was made (D33).
        # The check, the WAV writes, the ready row and the rebase of the draft it spoke
        # share ONE write lock, so no delete can land between them; a miss writes no film
        # audio at all and leaves the draft as it was (D117).
        conn.execute("BEGIN IMMEDIATE")
        try:
            gone = _deleted_voice(conn, owner_id, used) if saved_film else None
            if gone is None:
                presentation = save_ready()
                if draft is not None:
                    rebase_consumed(conn, draft, passages, script)
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        if gone is not None:
            conn.execute("ROLLBACK")
            _fail(conn, group_id, language, signature, edit,
                  f"The saved voice «{gone}» was deleted while this film was made.",
                  decisions)
        conn.execute("COMMIT")
    # The table is authoritative right now — the one safe moment to drop the WAVs a
    # previous memory build left behind when its rows cascaded away (§11.1).
    prune_orphan_audio(conn, audio)
    return presentation


def _saved_cast(
    conn: sqlite3.Connection, voices: Storage | None, owner_id: int, group_id: int,
    language: str, chosen, edit, decisions: dict,
) -> list:
    """`(SavedVoice, VoiceExample)` for every saved voice this film may cast: the voices
    ready for `language` with their `film_take` (§11.1), once the user's chosen narrator
    is known to be one of them. Otherwise the film FAILS with the reason (D33): it is
    never quietly read by Automatic or a preset instead."""
    signature = current_signature(conn, owner_id)
    voice = get_voice(conn, owner_id, chosen.voice_id) if chosen.voice_id is not None else None
    if voice is None:
        _fail(conn, group_id, language, signature, edit,
              "The saved voice chosen for this film was deleted. Choose another narrator.",
              decisions)
    found = film_take(conn, owner_id, voice.id, language)
    if found.problem is not None:
        _fail(conn, group_id, language, signature, edit,
              _cannot_narrate(voice.name, found.problem, language), decisions)
    cast = castable_takes(conn, owner_id, language)
    if not any(candidate.id == voice.id for candidate, _ in cast):
        # A delete landed between the two reads: say what it is now.
        found = film_take(conn, owner_id, voice.id, language)
        if found.problem is not None:
            _fail(conn, group_id, language, signature, edit,
                  _cannot_narrate(voice.name, found.problem, language), decisions)
        cast.append((voice, found.take))
    return cast


def _cannot_narrate(name: str, problem: str, language: str) -> str:
    """Why the saved voice `name` cannot narrate a `language` film, as `film_take`'s
    problem names it (§11.1)."""
    language_name = language_names.get(language, language)
    if problem == "language":
        return f"«{name}» has no {language_name} take yet. Record one under Voices, or choose another narrator."
    if problem == "waiting":
        return (f"The words of «{name}»'s {language_name} takes are not heard yet. Open «{name}» "
                "under Voices and let it hear them, or choose another narrator.")
    if problem == "unusable":
        return (f"No {language_name} take of «{name}» has usable words. Record a take where you "
                "speak, or choose another narrator.")
    return f"The saved voice «{name}» is being deleted. Choose another narrator."


def _deleted_voice(conn: sqlite3.Connection, owner_id: int, used: dict) -> str | None:
    """The name of the first used saved voice that is gone or being deleted, else None.
    A take deleted meanwhile does not count: the film spoke from the take it read.
    Runs under the ready save's write lock."""
    for voice in used.values():
        current = get_voice(conn, owner_id, voice.id)
        if current is None or current.deleting:
            return voice.name
    return None


def _passage_voices(
    story, language: str, film_voice: str | None, bracketed: set[int] | None = None,
    saved: Sequence[SavedCast] | None = None,
) -> list[str | None]:
    """One speaker per passage: the film's own, except where the story pass cast another.

    The casting is the model's — it saw the notes, the photographs and the speaker
    table together, which is the only place that judgement can be made. All that
    happens here is `resolve`, which checks each name is real (an invented one raises
    inside the TTS child and narration is mandatory, §11.1), and the BRACKET GATE below.
    With `saved` (a saved narrator) each name resolves among those saved voices only,
    else among the presets only: a name of the other kind reads in the narrator's voice,
    so one engine speaks the film (D107).

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
    cast = [resolve_saved(name, saved) if saved is not None else resolve(language, name)
            for name in story.voices]
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
    about. No bracket at all marks no passage: the model fills in every `voice`, and
    without this the film was spoken by whatever it wrote there. None when a bracket
    exists but there is no previous cut to measure against (nothing is known about
    where it belongs, so nothing is gated).
    """
    notes = [note for note in notes or () if "[" in note.text]
    if not notes:
        return set()
    if film_seconds <= 0 or passages <= 0:
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
