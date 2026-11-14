"""Bake one saved voice into a LoRA adapter and judge it blind (PLAN §2.17 I, §6.5; D64, D65).

    uv run --no-sync python -m scripts.voice_bake_check --voice me --list
    uv run --no-sync python -m scripts.voice_bake_check --voice me --confirm-texts [--text 9=WORDS]
    uv run --no-sync python -m scripts.voice_bake_check --voice me --measure [--repeats 5] [--label NAME]
    uv run --no-sync python -m scripts.voice_bake_check --voice me --sweep [--label NAME]
    uv run --no-sync python -m scripts.voice_bake_check --voice me --heldout [--label NAME]
    uv run --no-sync python -m scripts.voice_bake_check --voice me --longer-bake [--label NAME]
    uv run --no-sync python -m scripts.voice_bake_check --voice me --longer-bake --attach BAKE_ID --out DIR
    uv run --no-sync python -m scripts.voice_bake_check --compare OLD/report.json NEW/report.json

It reads the voice's takes from the user's database READ-ONLY, bakes all of them through
the models service, keeps the adapter only in its output folder, and speaks the
voice_check sentences three ways — A the adapter + the prompt take, B the prompt take
alone (the one-take films before voice models), C the adapter alone (as films speak a
saved voice) — as `<id>-X/Y/Z.wav` in a random order per sentence, with `key.json`
holding the order. The prompt take of A and B is the newest take of the language (D82):
they only help to diagnose.

`--measure` needs no bake: it speaks through the STORED voice model, N times per text and
variant. Every file of either run is scored against the voice's own takes of the same
language (design §11.1). The scores go in report.json only: the console prints none, so
the listening stays blind. `--compare` prints mean, worst and spread per arm and per text of
two reports, with the deltas and every frozen input that differs: the reports may come
from different models or architectures, because `results` holds nothing of VoxCPM2. Run
it after you listen.

`--sweep` needs no bake either: it speaks one new sentence through the stored voice model,
arm C only, on a grid of `cfg_value` x `inference_timesteps`, as `S-NN.wav` with `key.json`
holding the cell of each file. It is a test: films keep `CONTROLS`, and its meter score is a
diagnostic that ranks nothing (`docs/models.md`).

`--heldout` needs no bake either: it speaks five new texts through the stored voice model, arm
C only, at the baseline and the candidate settings, as `H-NN.wav`. `key.json` and `frozen.json`
go to disk before the first call, and a wall budget ends the run. It is a test too: films keep
`CONTROLS`, and only the frozen pass rule and the user's ears decide (`docs/models.md`).

`--longer-bake` bakes one 1000-step voice model from v2's takes and recipe, under a watchdog
that requests a cancel at 12 h, low memory, slow steps or a failure. The adapter stays in the
output folder only. After a complete bake it speaks the held-out texts through the new model and
the stored v2 at production `CONTROLS`, as `L-NN.wav`, for a blind listen (`docs/models.md`).
`--attach BAKE_ID` follows a longer bake that already runs, with no new start, when the output
folder and the bake's status match this run's `frozen.json`.

A thin client (CLAUDE.md § "One model process"): no torch, no model. The bake and every
synthesis are HTTP calls to the models service the user's `make up` runs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sqlite3
import sys
import time
import wave
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from audio.voxcpm_voice import CONTROLS, _mode
from config import Settings
from feed.saved_voices import (
    SavedVoice,
    VoiceExample,
    VoiceModel,
    _fingerprint,
    example_for,
    list_voices,
    model_reference,
    read_example_audio,
    read_voice_model,
    reference_for,
    stored_model,
)
from feed.voice_likeness import Meter, Reference, build_reference, language_takes
from models import catalog
from scripts.voice_check import MODEL_KEY, SENTENCES, _data_dir, is_void, wav_metrics
from storage.local import LocalStorage

TOKEN = "saved:check"  # pragma: allowlist secret
POLL_SECONDS = 2.0
POLL_TIMEOUT_SECONDS = 60.0
BAKE_ID_FILE = "bake.json"
_ENDED = ("done", "failed", "cancelled")
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130
VARIANTS = {
    "A": "the adapter + the prompt take",
    "B": "today's films on the LoRA-built child",
    "C": "the adapter alone (use_clip=False)",
}
FILE_LABELS = ("X", "Y", "Z")
ADAPTER_FILE = "adapter.safetensors"
MIN_FREE_MB = 1024
MAX_SECONDS_PER_STEP = 6.0
DEFAULT_REPEATS = 5
COMPARED = ("mean", "worst", "spread")
# A long clip can pass /voice/embed's 4 MiB, and windows show drift across a piece
# boundary. A stated choice: inside the 5-30 s of the takes the own range comes from.
WINDOW_SECONDS = 10.0
# Over the 500 characters of one synthesis piece, so a run crosses a boundary. No kit text
# and no test sentence holds these words, so they are unseen. Never change them: every
# result is compared against the same text.
LONG_PASSAGES: dict[str, list[tuple[str, str]]] = {
    "ru": [
        ("RU-LONG", (
            "Осенью мы каждый год ездили к бабушке в маленький город на реке. Поезд уходил "
            "вечером, и мы просыпались уже среди жёлтых полей. На станции нас встречал дед "
            "на старой синей машине, которая заводилась только со второго раза. Дома пахло "
            "яблоками и печеньем, а на столе всегда стоял большой чайник. После обеда мы шли "
            "на берег смотреть, как рыбаки вытаскивают лодки. Бабушка рассказывала нам про "
            "свою молодость, про школу, где она учила детей рисовать, и про соседа, который "
            "однажды привёз из города настоящего попугая. Вечером над рекой поднимался "
            "холодный туман, мы закрывали окна и садились играть в карты. Никто не смотрел "
            "на часы, и каждый день казался длиннее предыдущего.")),
    ],
}
HELDOUT_LONG_ID = "RU-HELD-LONG"
# The held-out long text is windowed too, but it is not in LONG_PASSAGES: that feeds --measure.
WINDOWED = frozenset(text_id for passages in LONG_PASSAGES.values()
                     for text_id, _ in passages) | {HELDOUT_LONG_ID}
# The one sentence of the sweep. Not a check sentence, a kit text or a sentence of the earlier
# pilots, and one synthesis piece. Never change it: every clip of a run says the same words.
SWEEP_TEXT_ID = "RU-SWEEP"
SWEEP_LANGUAGE = "ru"
SWEEP_TEXT = ("Вечером соседка принесла нам тёплый пирог с капустой, и мы вместе пили чай "
              "на маленькой кухне.")
SWEEP_REPEATS = 4
# The user's permission names this version: a newer model is another experiment.
SWEEP_MODEL_VERSION = 2
SWEEP_CELLS = tuple({"cfg_value": cfg_value, "inference_timesteps": inference_timesteps}
                    for cfg_value in (1.5, 2.0, 3.0) for inference_timesteps in (10, 20))
SWEEP_PASS_RULE = (
    "Fixed before listening. Per clip: sounds like me yes/no, likeness 1-5, naturalness 1-5, "
    "wrong words. It needs a complete run: all 24 valid clips and 24 ballots, else incomplete "
    "and no winner. A candidate passes with >= 3 of 4 yes and more yes than 2.0/10, mean "
    "likeness >= 2.0/10's + 0.5, mean naturalness >= 3.5, and zero wrong words in its 4 clips. "
    "The highest mean likeness wins; a tie is no winner. With no pass production stays "
    "unchanged. A pass only earns a held-out permission question."
)
# Texts no take, kit text or earlier run holds. Never change them, the rule or the budget: they
# were fixed before any held-out audio existed.
HELDOUT_TEXTS = {
    "RU-HELD-1": ("В четверг поезд отправляется в восемь часов, поэтому мы встретимся у кассы за "
                  "двадцать минут до отъезда."),
    "RU-HELD-2": ("Ты помнишь, как мы заблудились в горах и до самой ночи искали дорогу обратно "
                  "к лагерю?"),
    "RU-HELD-3": ("Какой же это был счастливый день: дети смеялись, собака носилась по пляжу, а "
                  "солнце никак не садилось!"),
    "RU-HELD-4": ("На этой фотографии мама держит меня на руках у входа в зоопарк, а папа "
                  "пытается поймать шляпу."),
    HELDOUT_LONG_ID: (
        "В воскресенье мы решили разобрать старый шкаф в прихожей. На верхней полке лежала "
        "коробка с открытками, которые родители привозили из поездок. Я прочитал несколько "
        "коротких записок, а сестра стала вспоминать названия городов. Потом мы нашли маленький "
        "фонарик и проверили, горит ли в нём лампочка. За окном начался дождь, поэтому прогулку "
        "пришлось отложить. Мы включили свет, поставили коробку на стол и позвали отца. Он долго "
        "рассматривал одну фотографию, затем улыбнулся и рассказал, как однажды перепутал "
        "чемоданы на вокзале. К вечеру шкаф опустел, но выбросить мы решили только сломанный "
        "зонт."),
}
HELDOUT_SHORT_REPEATS = 3
HELDOUT_LONG_REPEATS = 2
HELDOUT_CELLS = ({"cfg_value": 2.0, "inference_timesteps": 10},
                 {"cfg_value": 3.0, "inference_timesteps": 20})
# The stored model the sweep ran on: a model baked since is another experiment.
HELDOUT_MODEL_SHA256 = "dbc656783f02f535498ca0120ea2fcae317ccbb9d7cd346f431960df3b258a5a"
HELDOUT_WALL_SECONDS = 1500
HELDOUT_WALL_BUDGET = (
    "No call starts after 1500 s. The client stops waiting for a call at 1500 s, and any timeout "
    "ends the run. This is not a compute cutoff: the one call in flight may go on running on the "
    "service."
)
HELDOUT_PASS_RULE = (
    "Fixed before any held-out audio exists and never revised after listening. Candidate is "
    "3.0/20; baseline is 2.0/10; both are sent through controls. Per clip: sounds like me yes/no, "
    "likeness 1-5, naturalness 1-5, wrong words. Accepted means yes AND likeness >= 3. Complete "
    "means all 28 planned valid clips and 28 complete blind ballots; otherwise incomplete and no "
    "pass. Every clause is required. a. Across the 12 short clips per cell: candidate mean "
    "likeness >= baseline mean + 0.5, candidate mean naturalness >= 3.5, candidate accepted >= 9 "
    "of 12 and >= baseline accepted count (a tie is allowed). b. On at least 3 of the 4 short "
    "texts, candidate mean likeness >= baseline mean. c. Both candidate long clips are accepted, "
    "and each has naturalness >= 3. d. Zero wrong words across all 14 candidate clips. The meter "
    "never decides. A pass earns only the question of changing CONTROLS, with the estimated 1.6x "
    "generation time per audio second stated. Otherwise production stays unchanged."
)
# The user's permission names this bake: one run, 1000 steps, v2's takes and recipe.
LONGER_BAKE_STEPS = 1000
LONGER_BAKE_MAX_SECONDS = 12 * 3600
LONGER_BAKE_MAX_SECONDS_PER_STEP = 45.0
LONGER_BAKE_SLOW_AFTER_STEP = 50
LONGER_BAKE_ROLES = ({"role": "candidate"}, {"role": "baseline"})
LONGER_BAKE_PASS_RULE = (
    "Fixed before any audio of the new model exists and never revised after listening. "
    "Candidate is the new 1000-step voice model; baseline is the stored voice model v2. Both "
    "speak at the production CONTROLS: no call sends controls. Per clip: sounds like me yes/no, "
    "likeness 1-5, naturalness 1-5, wrong words. Accepted means yes AND likeness >= 3. Complete "
    "means all 28 planned valid clips and 28 complete blind ballots; otherwise incomplete and no "
    "pass. Every clause is required. a. Across the 12 short clips per model: candidate mean "
    "likeness >= baseline mean + 0.5, candidate mean naturalness >= 3.5, candidate accepted >= 9 "
    "of 12 and >= baseline accepted count (a tie is allowed). b. On at least 3 of the 4 short "
    "texts, candidate mean likeness >= baseline mean. c. Both candidate long clips are accepted, "
    "and each has naturalness >= 3. d. Zero wrong words across all 14 candidate clips. The meter "
    "and the training loss never decide. A pass earns only a separate question of adopting the "
    "new model for this voice; changing the default or replacing v2 needs its own approval. A "
    "fail keeps v2 and ends the experiment."
)
GUIDANCE = "the package authors suggest 5-10 minutes for one speaker (their guidance, not a promise)"
LISTEN_FIRST = ("Listen to and score every X/Y/Z file of a sentence BEFORE you open key.json "
                "or report.json.")
SWEEP_LISTEN_FIRST = "Listen to and score every S file BEFORE you open key.json or report.json."
HELDOUT_LISTEN_FIRST = ("Listen to and score every H file BEFORE you open key.json, frozen.json "
                        "or report.json.")
LONGER_LISTEN_FIRST = ("Listen to and score every L file BEFORE you open key.json, frozen.json "
                       "or report.json.")
THRESHOLDS = (
    "PASS needs (§6.5 step 8): every automatic check true; min_free_mb >= 1024 and macOS "
    "ended no process; seconds_per_step <= 6.0; C (the adapter alone, as films will use it) "
    "at mean similarity >= B's mean + 0.5, mean naturalness >= 3.5, <= 1 wrong word per "
    f"sentence. A and B only help to diagnose.\n{LISTEN_FIRST}"
)


class UsageError(ValueError):
    """The run cannot start as asked; the message names the take or the argument."""


@dataclass(frozen=True)
class Take:
    example: VoiceExample
    words: str | None
    label: str             # where the words come from (D104): see `takes_of`


@dataclass(frozen=True)
class VoiceStore:
    """The user's saved voices, opened read-only."""

    conn: sqlite3.Connection
    storage: LocalStorage
    owner_id: int


@dataclass
class BakeRun:
    status: dict
    interrupted: bool
    free_mb_before: float | None
    min_free_mb: float | None
    adapter: bytes | None = None
    stop: dict | None = None    # `{trigger, elapsed_s}` of the first cancel request
    attached: bool = False
    poll_failures: int = 0


@dataclass(frozen=True)
class Attach:
    """A bake that already runs: its first status and the local clock time its request was sent."""

    status: dict
    sent: float


@dataclass(frozen=True)
class CheckRun:
    store: VoiceStore
    voice: SavedVoice
    takes: list[Take]
    out_dir: Path
    rng: random.Random = field(default_factory=random.Random)
    clock: Callable[[], float] = time.monotonic
    label: str | None = None  # names this run in `--compare`


@dataclass(frozen=True)
class Scale:
    """What every file of a run is scored with: one evaluator and one snapshot of the
    takes per language."""

    meter: Meter
    references: dict[str, Reference]


def find_voice(conn: sqlite3.Connection, owner_id: int, name: str) -> SavedVoice:
    voices = list_voices(conn, owner_id)
    voice = next((v for v in voices if v.name == name), None)
    if voice is None:
        names = ", ".join(repr(v.name) for v in voices) or "none"
        raise UsageError(f"no saved voice is named {name!r}; the voices are: {names}")
    if voice.deleting:
        raise UsageError(f"the voice {name!r} is being deleted")
    return voice


def takes_of(voice: SavedVoice, texts: dict[int, str]) -> list[Take]:
    """Every take that is not being deleted, with the words this run trains on and where
    they come from (D104): "given with --text", "heard by <model>@<rev7>" (the app's bake
    heard them), "shown text, not heard" (a take stored before ASR), "no usable words:
    <reason>" or "NO WORDS"."""
    usable = [example for example in voice.examples if not example.deleting]
    unknown = sorted(set(texts) - {example.id for example in usable})
    if unknown:
        raise UsageError(f"--text names no take of {voice.name!r}: "
                         f"{', '.join(str(take_id) for take_id in unknown)}")
    takes = []
    for example in usable:
        if example.id in texts:
            takes.append(Take(example, texts[example.id], "given with --text"))
        elif example.words_state == "heard":
            model, _, revision = example.transcript_model.rpartition("@")
            takes.append(Take(example, example.transcript, f"heard by {model}@{revision[:7]}"))
        elif example.words_state == "unusable":
            takes.append(Take(example, None, f"no usable words: {example.transcript_error}"))
        elif example.transcript and example.transcript.strip():
            takes.append(Take(example, example.transcript, "shown text, not heard"))
        else:
            takes.append(Take(example, None, "NO WORDS"))
    return takes


def seconds_by_language(takes: list[Take]) -> dict[str, float]:
    seconds: dict[str, float] = {}
    for take in takes:
        language = take.example.language
        seconds[language] = round(seconds.get(language, 0.0) + take.example.seconds, 3)
    return seconds


def format_takes(voice: SavedVoice, takes: list[Take]) -> str:
    lines = [f"voice {voice.name!r} (id {voice.id}): {len(takes)} takes", ""]
    for take in takes:
        example = take.example
        lines.append(f"take {example.id} · {example.language} · {example.seconds:.1f} s · "
                     f"{example.source} · {take.label}")
        lines.append(f"    {take.words or '-'}")
    lines += ["", _minutes_line(takes)]
    return "\n".join(lines)


def _minutes_line(takes: list[Take]) -> str:
    per_language = ", ".join(f"{language} {seconds / 60:.1f} min"
                             for language, seconds in sorted(seconds_by_language(takes).items()))
    total = sum(take.example.seconds for take in takes) / 60
    return f"recorded: {total:.1f} min ({per_language or 'none'}); {GUIDANCE}"


def check_words(takes: list[Take]) -> None:
    """A run trains on every take, so every take needs its words; none is left out."""
    if not takes:
        raise UsageError("the voice has no take to bake")
    missing = [take.example for take in takes if take.words is None]
    if missing:
        named = "; ".join(f"take {e.id} ({e.language}, {e.seconds:.1f} s, {e.source})"
                          for e in missing)
        raise UsageError(f"no words for {named}. Give them with --text ID=WORDS")


def progress_line(status: dict) -> str:
    loss = status.get("loss_last")
    shown = "-" if loss is None else f"{loss:.4f}"
    return (f"{status['stage']} · {int(status['fraction'] * 100)}% · "
            f"step {status['step']} of {status['steps']} · loss {shown}")


def _reason(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail")
        except ValueError:
            detail = exc.response.text
        return f"{exc.response.status_code} {detail}"
    text = str(exc).strip()
    return text.splitlines()[0] if text else repr(exc)


def watchdog_trigger(status: dict, elapsed: float) -> str | None:
    """Why the longer bake must stop now, or None. `min_free_mb` is the service's lowest
    reading during the bake; `seconds_per_step` is the trainer's mean since its first step."""
    if status["state"] == "failed":
        return "failed"
    if elapsed >= LONGER_BAKE_MAX_SECONDS:
        return "12 h passed"
    free = status.get("min_free_mb")
    if free is not None and free < MIN_FREE_MB:
        return f"free memory under {MIN_FREE_MB} MiB"
    per_step = status.get("seconds_per_step")
    if (status["step"] > LONGER_BAKE_SLOW_AFTER_STEP and per_step is not None
            and per_step > LONGER_BAKE_MAX_SECONDS_PER_STEP):
        return f"over {LONGER_BAKE_MAX_SECONDS_PER_STEP:g} s per step"
    return None


def run_bake(client, clips: list[dict], tag: dict, config: dict | None, *,
             sleep: Callable[[float], None] = time.sleep, watchdog: bool = False,
             clock: Callable[[], float] = time.monotonic, attach: Attach | None = None,
             id_file: Path | None = None) -> BakeRun:
    """Start the bake (or follow `attach`, with no start) and follow it to an end state. A
    first Ctrl-C cancels it and keeps waiting for ANY end (t31): a bake that reached `done` or
    `failed` first still ends the wait, and no adapter is fetched. A second Ctrl-C leaves. With
    `watchdog`, a `watchdog_trigger` requests the cancel once, in the same way. A cancel is
    cooperative: the trainer stops at its next step boundary. A poll or a cancel that times out
    or loses its connection is retried on the same bake at the next poll. The watchdog's clock
    starts at the send time of the first status minus its `elapsed_s` (the service reports no
    start time) and is never reset. The service's bake folder is deleted only after an end
    state."""
    if attach is None:
        free_before = client.models_state()["free_mb"]
        sent = clock()
        status = client.bake_start(clips, tag=tag, config=config)
        if id_file is not None:
            id_file.write_text(json.dumps({"id": status["id"]}) + "\n")
    else:
        status, sent = attach.status, attach.sent
        free_before = status.get("free_mb_before")
    bake_id = status["id"]
    started = sent - status["elapsed_s"]
    deadline = started + LONGER_BAKE_MAX_SECONDS
    interrupted = False
    stop = None
    cancel_sent = False
    failures = 0
    adapter = None

    def request_cancel() -> None:
        nonlocal status, cancel_sent
        try:
            status = client.bake_cancel(bake_id)
            cancel_sent = True
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            print(f"\n  ! the cancel of bake {bake_id} failed ({_reason(exc)}); retrying",
                  flush=True)

    try:
        while status["state"] == "running":
            try:
                if watchdog and stop is None and clock() >= deadline:
                    stop = {"trigger": "12 h passed", "elapsed_s": round(clock() - started, 3)}
                    print("\n  12 h passed: cancelling the bake…", flush=True)
                if stop is not None and not cancel_sent:
                    request_cancel()
                sleep(POLL_SECONDS)
                if not watchdog:
                    timeout = {}
                elif stop is not None:
                    timeout = {"timeout": POLL_TIMEOUT_SECONDS}
                elif deadline - clock() <= 0:
                    continue
                else:
                    timeout = {"timeout": min(POLL_TIMEOUT_SECONDS, deadline - clock())}
                try:
                    status = client.bake_status(bake_id, **timeout)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    failures += 1
                    print(f"\n  ! no status of bake {bake_id} ({_reason(exc)}); retrying",
                          flush=True)
                    continue
                print(f"\r{progress_line(status):<100}", end="", flush=True)
                if watchdog and stop is None:
                    elapsed = clock() - started
                    trigger = watchdog_trigger(status, elapsed)
                    if trigger is not None:
                        stop = {"trigger": trigger, "elapsed_s": round(elapsed, 3)}
                        print(f"\n  {trigger}: cancelling the bake…", flush=True)
                        request_cancel()
            except KeyboardInterrupt:
                if interrupted:
                    raise
                interrupted = True
                stop = stop or {"trigger": "Ctrl-C", "elapsed_s": round(clock() - started, 3)}
                print("\n  cancelling the bake…", flush=True)
                if not cancel_sent:
                    request_cancel()
        print()
        if status["state"] == "done" and not interrupted and stop is None:
            adapter = client.bake_adapter(bake_id)
    finally:
        if status["state"] in _ENDED:
            try:
                client.bake_delete(bake_id)
            except Exception as exc:  # noqa: BLE001 - best effort: the run's result stands
                print(f"  ! the bake folder {bake_id} was not deleted: {_reason(exc)}",
                      file=sys.stderr)
        else:
            print(f"  ! the bake {bake_id} still runs on the service; its folder was kept",
                  file=sys.stderr)
    lowest = [free for free in (free_before, status.get("min_free_mb")) if free is not None]
    return BakeRun(status, interrupted, free_before, min(lowest, default=None), adapter, stop,
                   attach is not None, failures)


def attach_to(client, run: CheckRun, bake_id: str, frozen: dict, lora: str) -> Attach:
    """The running bake `bake_id`, once the output folder and the bake's status prove it is
    this run's bake; `UsageError` names the first mismatch. Nothing is written or cancelled."""
    files = sorted(path.name for path in run.out_dir.iterdir()) if run.out_dir.exists() else []
    if "frozen.json" not in files or set(files) - {"frozen.json", BAKE_ID_FILE}:
        raise UsageError(f"--attach needs the output folder {run.out_dir} to hold only this "
                         f"run's frozen.json and {BAKE_ID_FILE}; it holds {files}")
    try:
        on_disk = json.loads((run.out_dir / "frozen.json").read_text())
        named = (json.loads((run.out_dir / BAKE_ID_FILE).read_text()).get("id")
                 if BAKE_ID_FILE in files else bake_id)
    except ValueError as exc:
        raise UsageError(f"cannot read the output folder {run.out_dir}: {exc}") from exc
    if on_disk != json.loads(json.dumps(frozen)):
        raise UsageError(f"{run.out_dir / 'frozen.json'} is not this run's: the takes, v2 or "
                         "the limits differ")
    if named != bake_id:
        raise UsageError(f"{BAKE_ID_FILE} names bake {named}, not {bake_id}")
    sent = run.clock()
    status = client.bake_status(bake_id, timeout=POLL_TIMEOUT_SECONDS)
    expected = {"id": bake_id, "tag": {"check": run.voice.name},
                "model": frozen["voice_model"]["model_key"], "revision": frozen["base_revision"],
                "lora": lora, "config": frozen["recipe"], "clips": len(frozen["takes"]),
                "dataset_sha256": frozen["dataset_sha256"]}
    for name, value in expected.items():
        if status.get(name) != value:
            raise UsageError(f"bake {bake_id} is not this run's bake: its {name} is "
                             f"{status.get(name)!r}, frozen.json needs {value!r}")
    return Attach(status, sent)


def speak_row(client, run: CheckRun, row: dict, text: str, ref: dict,
              timeout: float | None = None) -> None:
    """One `/speak` call into `row`, its file into the output folder. A failed call is a
    row with `error`, not a crash. With `timeout` the call runs on a worker thread, because
    httpx bounds each phase and not the total: a row that waited too long is `timed_out`, and
    its late reply is dropped. A leftover worker holds the process exit until httpx gives up."""
    revision = catalog.get("voice", MODEL_KEY).source.revision
    started = run.clock()
    try:
        if timeout is None:
            reply = client.speak(text, row["language"], TOKEN, None, None, references={TOKEN: ref})  # pragma: allowlist secret
        else:
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                reply = pool.submit(client.speak, text, row["language"], TOKEN, None, None,
                                    references={TOKEN: ref}, timeout=timeout).result(timeout)  # pragma: allowlist secret
            finally:
                pool.shutdown(wait=False)
    except Exception as exc:  # noqa: BLE001 - every reason is reported
        row["wall_s"] = round(run.clock() - started, 3)
        row["error"] = _reason(exc)
        if timeout is not None and isinstance(exc, TimeoutError | httpx.TimeoutException):
            row["timed_out"] = True
        return
    wall = run.clock() - started
    metrics = wav_metrics(reply["audio"])
    seconds = metrics["duration_s"]
    adapter_state = reply["adapters"].get(TOKEN) or {}
    conditioning = reply["conditioning"].get(TOKEN) or {}
    row.update(metrics)
    row.update({
        "sample_rate": reply["sample_rate"],
        "model": reply["model"],
        "revision": reply["revision"],
        "adapter_state": adapter_state.get("state"),
        "adapter_sha256": adapter_state.get("sha256"),
        "conditioning_state": conditioning.get("state"),
        "wall_s": round(wall, 3),
        "rtf": round(wall / seconds, 3) if seconds else None,
        "void": is_void(reply, revision),
    })
    if "controls" in ref:
        row["controls_echo"] = reply["controls"].get(TOKEN)
    (run.out_dir / row["file"]).write_bytes(reply["audio"])


def speak_variants(client, run: CheckRun, adapter: bytes) -> tuple[list[dict], dict, dict]:
    """A, B and C for every sentence of every language of the takes: `(rows, key,
    prompt_takes)`. The prompt take of A and B is the newest take of each language (D82)."""
    sha256 = hashlib.sha256(adapter).hexdigest()
    rows: list[dict] = []
    key: dict[str, dict[str, str]] = {}
    prompt_takes: dict[str, dict] = {}
    for language in sorted({take.example.language for take in run.takes}):
        example = example_for(run.store.conn, run.store.owner_id, run.voice.id, language)
        prompt_takes[language] = {"example_id": example.id if example else None, "rule": "newest"}
        base = (reference_for(run.store.conn, run.store.storage, example)
                if example is not None else None)
        for sentence_id, text in SENTENCES[language]:
            order = list(VARIANTS)
            run.rng.shuffle(order)
            key[sentence_id] = dict(zip(FILE_LABELS, order))
            for label, variant in key[sentence_id].items():
                row: dict = {"id": sentence_id, "language": language, "variant": variant,
                             "file": f"{sentence_id}-{label}.wav"}
                rows.append(row)
                if base is None:
                    row["error"] = f"no prompt take in {language}"
                    continue
                ref = base
                if variant != "B":
                    ref = base | {"adapter": {"blob": adapter, "sha256": sha256},
                                  "use_clip": variant == "A"}
                speak_row(client, run, row, text, ref)
    return rows, key, prompt_takes


def stored_models(run: CheckRun) -> tuple[dict[str, VoiceModel], dict[str, str]]:
    """Per language of the takes the voice model films speak through, and the languages
    it cannot speak with the reason."""
    models: dict[str, VoiceModel] = {}
    left_out: dict[str, str] = {}
    for language in sorted({take.example.language for take in run.takes}):
        found = stored_model(run.store.conn, run.store.owner_id, run.voice.id, language)
        if found.problem is None:
            models[language] = found.model
        else:
            left_out[language] = found.problem
    if not models:
        raise UsageError(f"{run.voice.name!r} has no stored voice model to measure: "
                         f"{left_out or 'it has no take'}")
    return models, left_out


def stored_arms(run: CheckRun, language: str, model: VoiceModel) -> tuple[dict[str, dict], dict]:
    """`(variant -> /speak reference, prompt take)` of the stored voice model. C is what
    a film sends; A and B need a prompt take, the newest of the language (D82)."""
    store = run.store
    stored = model_reference(model, read_voice_model(store.storage, model), language)
    example = example_for(store.conn, store.owner_id, run.voice.id, language)
    prompt_take = {"example_id": example.id if example else None, "rule": "newest"}
    if example is None:
        return {"C": stored}, prompt_take
    base = reference_for(store.conn, store.storage, example)
    return {"A": base | {"adapter": stored["adapter"], "use_clip": True}, "B": base,
            "C": stored}, prompt_take


def speak_repeats(client, run: CheckRun, arms: dict[str, dict[str, dict]],
                  repeats: int) -> tuple[list[dict], dict, bool]:
    """A, B and C for every text, `repeats` times, a whole repeat at a time: `(rows, key,
    interrupted)`. A Ctrl-C ends the speaking and keeps what was made."""
    texts = [(language, *text) for language in arms
             for text in SENTENCES[language] + LONG_PASSAGES.get(language, [])]
    total = repeats * len(texts) * len(VARIANTS)
    rows: list[dict] = []
    key: dict[str, dict[str, str]] = {}
    try:
        for repeat in range(1, repeats + 1):
            for language, text_id, text in texts:
                order = list(VARIANTS)
                run.rng.shuffle(order)
                key[f"{text_id}-{repeat}"] = dict(zip(FILE_LABELS, order))
                for label, variant in key[f"{text_id}-{repeat}"].items():
                    row: dict = {"id": text_id, "language": language, "variant": variant,
                                 "repeat": repeat, "file": f"{text_id}-{repeat}-{label}.wav"}
                    print(f"\rcall {len(rows) + 1} of {total} · {row['file']:<24}", end="",
                          flush=True)
                    if variant in arms[language]:
                        speak_row(client, run, row, text, arms[language][variant])
                    else:
                        row["error"] = f"no prompt take in {language}"
                    rows.append(row)
    except KeyboardInterrupt:
        print("\n  stopped; scoring the files made so far", flush=True)
        return rows, key, True
    print()
    return rows, key, False


def sweep_plan(rng: random.Random) -> list[dict]:
    """Every call of a sweep: one shuffled block of all cells per repeat, so a Ctrl-C stays
    balanced. The numbers come from a second shuffle, so a file name holds neither the cell
    nor the place in the run."""
    calls: list[tuple[int, dict]] = []
    for repeat in range(1, SWEEP_REPEATS + 1):
        block = list(SWEEP_CELLS)
        rng.shuffle(block)
        calls += [(repeat, cell) for cell in block]
    numbers = list(range(1, len(calls) + 1))
    rng.shuffle(numbers)
    return [{"id": SWEEP_TEXT_ID, "language": SWEEP_LANGUAGE, "variant": "C", "repeat": repeat,
             "file": f"S-{number:02d}.wav"} | cell
            for number, (repeat, cell) in zip(numbers, calls, strict=True)]


def heldout_plan(rng: random.Random, arms: tuple[dict, ...] = HELDOUT_CELLS,
                 prefix: str = "H") -> list[dict]:
    """Every call of a held-out run: one shuffled block per repeat of all text and arm pairs
    still due, so a stop stays balanced. An arm is a cell, or a model of the longer bake. The
    numbers come from a second shuffle, so a file name holds neither the arm, the text nor
    the place in the run."""
    calls: list[tuple[int, str, dict]] = []
    for repeat in range(1, HELDOUT_SHORT_REPEATS + 1):
        block = [(text_id, arm) for text_id in HELDOUT_TEXTS for arm in arms
                 if text_id != HELDOUT_LONG_ID or repeat <= HELDOUT_LONG_REPEATS]
        rng.shuffle(block)
        calls += [(repeat, text_id, arm) for text_id, arm in block]
    numbers = list(range(1, len(calls) + 1))
    rng.shuffle(numbers)
    return [{"id": text_id, "language": SWEEP_LANGUAGE, "variant": "C", "repeat": repeat,
             "file": f"{prefix}-{number:02d}.wav"} | arm
            for number, (repeat, text_id, arm) in zip(numbers, calls, strict=True)]


def _cell(row: dict) -> dict:
    return {"cfg_value": row["cfg_value"], "inference_timesteps": row["inference_timesteps"]}


def speak_sweep(client, run: CheckRun, reference: Callable[[dict], dict], plan: list[dict],
                texts: dict[str, str], deadline: float | None = None
                ) -> tuple[list[dict], str | None, bool, bool]:
    """`plan`, each row through `reference(row)` saying `texts[row["id"]]`: `(rows, file of a
    wrong echo, interrupted, stopped by the deadline)`. A failed call is a row and the run goes
    on. A reference with `controls` must be echoed: a wrong echo stops the run at once, because
    a stale service ignores `controls`, and every further clip would be spoken at the fixed
    values. A Ctrl-C ends the speaking and keeps what was made.
    `deadline` on `run.clock` gives each call the time that remains: no call starts past it,
    and any timeout ends the run, with time left or not."""
    rows: list[dict] = []
    try:
        for planned in plan:
            remaining = None if deadline is None else deadline - run.clock()
            if remaining is not None and remaining <= 0:
                print()
                return rows, None, False, True
            row = dict(planned)
            print(f"\rcall {len(rows) + 1} of {len(plan)} · {row['file']:<24}", end="", flush=True)
            ref = reference(row)
            speak_row(client, run, row, texts[row["id"]], ref, remaining)
            rows.append(row)
            sent = ref.get("controls")
            if "error" not in row and sent is not None and row["controls_echo"] != sent:
                row["error"] = f"the reply echoed {row['controls_echo']}, not {sent}"
                print()
                return rows, row["file"], False, False
            late = deadline is not None and run.clock() >= deadline
            if late or row.get("timed_out"):
                print()
                return rows, None, False, late
    except KeyboardInterrupt:
        print("\n  stopped; scoring the files made so far", flush=True)
        return rows, None, True, False
    print()
    return rows, None, False, False


def build_scale(client, run: CheckRun, languages: list[str]) -> Scale:
    """Takes go through the service every run: this script holds the database read-only,
    so it never reads or writes `voice_take_embeddings`."""
    meter = Meter(client)
    references = {
        language: build_reference(meter, language_takes(run.voice, language),
                                  lambda take: read_example_audio(run.store.storage, take))
        for language in languages}
    return Scale(meter, references)


def fixed_windows(audio: bytes) -> list[tuple[float, float, bytes]]:
    """`(start_s, end_s, wav)` of `WINDOW_SECONDS` each. The last window ends with the clip
    and may overlap the one before: every window has one length, because a short clip may
    score lower for its length alone, and no audio goes unscored."""
    with wave.open(io.BytesIO(audio), "rb") as handle:
        params = handle.getparams()
        pcm = handle.readframes(params.nframes)
    length = int(WINDOW_SECONDS * params.framerate)
    if params.nframes <= length:
        return [(0.0, params.nframes / params.framerate, audio)]
    starts = list(range(0, params.nframes - length + 1, length))
    if starts[-1] != params.nframes - length:
        starts.append(params.nframes - length)
    frame_bytes = params.sampwidth * params.nchannels
    windows = []
    for start in starts:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setparams(params)
            handle.writeframes(pcm[start * frame_bytes:(start + length) * frame_bytes])
        windows.append((round(start / params.framerate, 3),
                        round((start + length) / params.framerate, 3), buffer.getvalue()))
    return windows


def score_windows(scale: Scale, row: dict, audio: bytes) -> None:
    """Each window keeps its own score or its failure. The row's `cosine` is the mean of
    its windows only when every window scored: a mean of the rest hides the refused part."""
    reference = scale.references[row["language"]]
    row["windows"] = []
    for start, end, clip in fixed_windows(audio):
        window: dict = {"start_s": start, "end_s": end}
        row["windows"].append(window)
        try:
            window["cosine"] = round(reference.similarity(scale.meter.embed(clip)), 6)
        except Exception as exc:  # noqa: BLE001 - a failure is kept in its window
            window["score_error"] = _reason(exc)
    scored = [window["cosine"] for window in row["windows"] if "cosine" in window]
    total = len(row["windows"])
    if len(scored) == total:
        row["cosine"] = round(sum(scored) / total, 6)
    else:
        row["score_error"] = f"{total - len(scored)} of {total} windows could not be scored"


def score_rows(scale: Scale, rows: list[dict], out_dir: Path) -> None:
    """Every file that was made, scored on its real WAV. A file the meter cannot score
    keeps its reason in `score_error` and never a number."""
    for row in rows:
        if "error" in row:
            continue
        audio = (out_dir / row["file"]).read_bytes()
        if row["id"] in WINDOWED:
            score_windows(scale, row, audio)
            continue
        try:
            row["cosine"] = round(
                scale.references[row["language"]].similarity(scale.meter.embed(audio)), 6)
        except Exception as exc:  # noqa: BLE001 - a failure is kept in its row
            row["score_error"] = _reason(exc)


def _stats(values: list[float], failures: int) -> dict:
    """`spread` is highest minus lowest. No threshold: the scale is `own` of `frozen`."""
    if not values:
        return {"runs": 0, "failures": failures, "mean": None, "worst": None, "spread": None}
    return {"runs": len(values), "failures": failures,
            "mean": round(sum(values) / len(values), 6), "worst": min(values),
            "spread": round(max(values) - min(values), 6)}


def results_of(rows: list[dict], label: str) -> list[dict]:
    """The report's model-neutral half, one entry per file: a report of another model or
    architecture holds the same fields, so `--compare` reads only these and `frozen`."""
    return [{"label": label, "arm": row["variant"], "model": row.get("model"),
             "revision": row.get("revision"), "text_id": row["id"],
             "repeat": row.get("repeat", 1), "cosine": row.get("cosine"),
             "seconds": row.get("duration_s"), "windows": row.get("windows", []),
             "error": row.get("error") or row.get("score_error")} for row in rows]


def _result_stats(results: list[dict]) -> dict:
    scored = [result["cosine"] for result in results if result["cosine"] is not None]
    return _stats(scored, len(results) - len(scored))


def _window_stats(results: list[dict]) -> list[dict]:
    """Per window position over the repeats, so drift along a long text shows."""
    windows = [result["windows"] for result in results]
    stats = []
    for index in range(max(map(len, windows), default=0)):
        found = [each[index] for each in windows if index < len(each)]
        scored = [window["cosine"] for window in found if "cosine" in window]
        stats.append(_stats(scored, len(found) - len(scored)))
    return stats


def likeness_summary(results: list[dict]) -> dict:
    """Mean, worst and spread per arm, and per text and arm. Every failure counts."""
    arms = sorted({result["arm"] for result in results})
    texts = list(dict.fromkeys(result["text_id"] for result in results))

    def of(text_id: str | None, arm: str) -> list[dict]:
        return [result for result in results if result["arm"] == arm
                and (text_id is None or result["text_id"] == text_id)]

    return {
        "by_arm": {arm: _result_stats(of(None, arm)) for arm in arms},
        "by_text": {text_id: {arm: _result_stats(of(text_id, arm)) for arm in arms}
                    for text_id in texts},
        "windows": {text_id: {arm: _window_stats(of(text_id, arm)) for arm in arms}
                    for text_id in texts
                    if any(result["windows"] for result in results if result["text_id"] == text_id)},
    }


def frozen_inputs(run: CheckRun, adapter_sha256: str, texts: dict[str, str], scale: Scale) -> dict:
    """What must stay the same for two reports to compare one change."""
    modes = {}
    for language in scale.references:
        example = example_for(run.store.conn, run.store.owner_id, run.voice.id, language)
        prompt = _mode(example.transcript) if example is not None else "no prompt take"
        modes[language] = {"A": f"adapter + {prompt}", "B": prompt, "C": "adapter alone"}
    return {
        "adapter_sha256": adapter_sha256,
        "base_revision": catalog.get("voice", MODEL_KEY).source.revision,
        "mode": modes,
        "texts": texts,
        "controls": CONTROLS,
        "reference": {language: reference.provenance()
                      for language, reference in scale.references.items()},
    }


def run_measure(client, run: CheckRun, repeats: int) -> tuple[dict, int]:
    """No bake: the stored voice model, the three variants and the report; `(report, exit
    code)`. The takes are embedded first, so a meter that cannot run stops before any speech."""
    models, left_out = stored_models(run)
    scale = build_scale(client, run, sorted(models))
    arms = {language: stored_arms(run, language, model) for language, model in models.items()}
    run.out_dir.mkdir(parents=True, exist_ok=True)
    rows, key, interrupted = speak_repeats(
        client, run, {language: found[0] for language, found in arms.items()}, repeats)
    (run.out_dir / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    score_rows(scale, rows, run.out_dir)
    model = next(iter(models.values()))
    label = run.label or f"{model.model_key} v{model.version}"
    results = results_of(rows, label)
    texts = {text_id: text for language in models
             for text_id, text in SENTENCES[language] + LONG_PASSAGES.get(language, [])}
    checks = speech_checks(rows) | {
        "every A and C reply names the stored voice model": all(
            row["adapter_sha256"] == model.sha256 for row in rows
            if "error" not in row and row["variant"] != "B"),
        "every file scored": bool(rows) and all("cosine" in row for row in rows),
    }
    report = {
        "label": label,
        "voice": {"id": run.voice.id, "name": run.voice.name},
        "takes": {"count": len(run.takes), "ids": [take.example.id for take in run.takes],
                  "seconds_by_language": seconds_by_language(run.takes)},
        "voice_model": {"version": model.version, "sha256": model.sha256},
        "languages_left_out": left_out,
        "repeats": repeats,
        "cancelled_by_user": interrupted,
        "prompt_takes": {language: found[1] for language, found in arms.items()},
        "frozen": frozen_inputs(run, model.sha256, texts, scale),
        "variants": VARIANTS,
        "calls": rows,
        "mean_rtf": mean_rtf(rows),
        "results": results,
        "likeness": likeness_summary(results),
        "checks": checks,
    }
    (run.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if interrupted:
        return report, EXIT_INTERRUPTED
    return report, 0 if all(checks.values()) else 1


def _letters(text: str) -> str:
    return "".join(char for char in text.casefold() if char.isalnum())


def _valid_clip(row: dict, sha256: str) -> bool:
    return ("error" not in row and row["adapter_state"] in ("loaded", "kept")
            and row["adapter_sha256"] == sha256 and row["duration_s"] > 0
            and not row["silent"] and not row["nan"] and not row["void"])


def run_sweep(client, run: CheckRun) -> tuple[dict, int]:
    """The grid on the stored voice model and the report; `(report, exit code)`. A wrong
    model or a known sentence stops before any call, and the takes are embedded before any
    speech, as in `run_measure`. The meter is only a diagnostic here: no check reads it."""
    models, left_out = stored_models(run)
    model = models.get(SWEEP_LANGUAGE)
    if model is None:
        raise UsageError(f"{run.voice.name!r} cannot speak {SWEEP_LANGUAGE}: "
                         f"{left_out.get(SWEEP_LANGUAGE, 'it has no take in it')}")
    if model.version != SWEEP_MODEL_VERSION:
        raise UsageError(f"the sweep is for voice model v{SWEEP_MODEL_VERSION}; "
                         f"{run.voice.name!r} has v{model.version}")
    said = [take.example.id for take in run.takes
            if take.words is not None and _letters(take.words) == _letters(SWEEP_TEXT)]
    if said:
        raise UsageError(f"the sweep sentence must be new, and take {said[0]} says it")
    scale = build_scale(client, run, [SWEEP_LANGUAGE])
    stored = stored_arms(run, SWEEP_LANGUAGE, model)[0]["C"]
    run.out_dir.mkdir(parents=True, exist_ok=True)
    plan = sweep_plan(run.rng)
    rows, wrong_echo, interrupted, _ = speak_sweep(
        client, run, lambda row: stored | {"controls": _cell(row)}, plan,
        {SWEEP_TEXT_ID: SWEEP_TEXT})
    key = {row["file"]: _cell(row) | {"repeat": row["repeat"]} for row in rows}
    (run.out_dir / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    score_rows(scale, rows, run.out_dir)
    valid = [row for row in rows if _valid_clip(row, model.sha256)]
    complete = len(valid) == len(plan)
    report = {
        "label": run.label or f"{model.model_key} v{model.version} sweep",
        "voice": {"id": run.voice.id, "name": run.voice.name},
        "takes": {"count": len(run.takes), "ids": [take.example.id for take in run.takes],
                  "seconds_by_language": seconds_by_language(run.takes)},
        "voice_model": {"version": model.version, "sha256": model.sha256},
        "repeats": SWEEP_REPEATS,
        "cancelled_by_user": interrupted,
        "stopped_by_echo": wrong_echo,
        "complete": complete,
        "frozen": frozen_inputs(run, model.sha256, {SWEEP_TEXT_ID: SWEEP_TEXT}, scale) | {
            "mode": {SWEEP_LANGUAGE: {"C": "adapter alone"}},
            "controls": [CONTROLS | cell for cell in SWEEP_CELLS],
            "voice_model": {"model_key": model.model_key, "version": model.version,
                            "sha256": model.sha256},
            "planned_calls": len(plan),
            "pass_rule": SWEEP_PASS_RULE,
        },
        "cells": [cell | {"valid_clips": sum(_cell(row) == cell for row in valid)}
                  for cell in SWEEP_CELLS],
        "calls": rows,
        "checks": speech_checks(rows) | {f"all {len(plan)} clips valid": complete},
    }
    (run.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if interrupted:
        return report, EXIT_INTERRUPTED
    return report, 0 if complete else 1


def stored_v2(run: CheckRun, purpose: str) -> VoiceModel:
    """The stored Russian voice model the held-out texts are frozen for: v2 with the sha256
    `HELDOUT_MODEL_SHA256`. A model baked since is another experiment."""
    models, left_out = stored_models(run)
    model = models.get(SWEEP_LANGUAGE)
    if model is None:
        raise UsageError(f"{run.voice.name!r} cannot speak {SWEEP_LANGUAGE}: "
                         f"{left_out.get(SWEEP_LANGUAGE, 'it has no take in it')}")
    if model.version != SWEEP_MODEL_VERSION:
        raise UsageError(f"the {purpose} is for voice model v{SWEEP_MODEL_VERSION}; "
                         f"{run.voice.name!r} has v{model.version}")
    if model.sha256 != HELDOUT_MODEL_SHA256:
        raise UsageError(f"the {purpose} is for voice model {HELDOUT_MODEL_SHA256[:12]}; "
                         f"{run.voice.name!r} has {model.sha256[:12]}")
    return model


def run_heldout(client, run: CheckRun) -> tuple[dict, int]:
    """The frozen held-out texts on the stored voice model and the report; `(report, exit
    code)`. Every refusal comes before any call. `key.json` and `frozen.json` go to disk before
    the first speech, so a killed run leaves every file named. The budget covers the speaking
    only. The meter is a diagnostic here: no check reads it."""
    model = stored_v2(run, "held-out run")
    said = [(take.example.id, text_id) for take in run.takes if take.words is not None
            for text_id, text in HELDOUT_TEXTS.items() if _letters(text) in _letters(take.words)]
    if said:
        raise UsageError(f"the held-out texts must be new, and take {said[0][0]} says "
                         f"{said[0][1]}")
    if run.out_dir.exists() and any(run.out_dir.iterdir()):
        raise UsageError(f"the output folder {run.out_dir} is not empty")
    scale = build_scale(client, run, [SWEEP_LANGUAGE])
    stored = stored_arms(run, SWEEP_LANGUAGE, model)[0]["C"]
    plan = heldout_plan(run.rng)
    frozen = frozen_inputs(run, model.sha256, HELDOUT_TEXTS, scale) | {
        "mode": {SWEEP_LANGUAGE: {"C": "adapter alone"}},
        "controls": [CONTROLS | cell for cell in HELDOUT_CELLS],
        "voice_model": {"model_key": model.model_key, "version": model.version,
                        "sha256": model.sha256},
        "repeats": {"short": HELDOUT_SHORT_REPEATS, "long": HELDOUT_LONG_REPEATS},
        "planned_calls": len(plan),
        "planned_rows": plan,
        "pass_rule": HELDOUT_PASS_RULE,
        "wall_budget": HELDOUT_WALL_BUDGET,
    }
    run.out_dir.mkdir(parents=True, exist_ok=True)
    key = {row["file"]: _cell(row) | {"id": row["id"], "repeat": row["repeat"]} for row in plan}
    (run.out_dir / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    (run.out_dir / "frozen.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    started = datetime.now(UTC)
    begun = run.clock()
    rows, wrong_echo, interrupted, stopped_by_deadline = speak_sweep(
        client, run, lambda row: stored | {"controls": _cell(row)}, plan, HELDOUT_TEXTS,
        begun + HELDOUT_WALL_SECONDS)
    elapsed = run.clock() - begun
    ended = datetime.now(UTC)
    score_rows(scale, rows, run.out_dir)
    valid = [row for row in rows if _valid_clip(row, model.sha256)]
    complete = len(valid) == len(plan)
    report = {
        "label": run.label or f"{model.model_key} v{model.version} held-out",
        "voice": {"id": run.voice.id, "name": run.voice.name},
        "takes": {"count": len(run.takes), "ids": [take.example.id for take in run.takes],
                  "seconds_by_language": seconds_by_language(run.takes)},
        "voice_model": {"version": model.version, "sha256": model.sha256},
        "repeats": frozen["repeats"],
        "cancelled_by_user": interrupted,
        "stopped_by_echo": wrong_echo,
        "stopped_by_deadline": stopped_by_deadline,
        "timed_out": next((row["file"] for row in rows if row.get("timed_out")), None),
        "started": started.isoformat(),
        "ended": ended.isoformat(),
        "elapsed_s": round(elapsed, 3),
        "complete": complete,
        "frozen": frozen,
        "cells": [cell | {"valid_clips": sum(_cell(row) == cell for row in valid)}
                  for cell in HELDOUT_CELLS],
        "calls": rows,
        "checks": speech_checks(rows) | {f"all {len(plan)} clips valid": complete},
    }
    (run.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if interrupted:
        return report, EXIT_INTERRUPTED
    return report, 0 if complete else 1


def mean_rtf(rows: list[dict]) -> dict[str, dict]:
    means = {}
    for variant, label in VARIANTS.items():
        values = [row["rtf"] for row in rows
                  if row["variant"] == variant and row.get("rtf") is not None]
        means[variant] = {"label": label,
                          "rtf": round(sum(values) / len(values), 3) if values else None}
    return means


def speech_checks(rows: list[dict]) -> dict[str, bool]:
    ok = [row for row in rows if "error" not in row]
    baked = [row for row in ok if row["variant"] != "B"]
    revision = catalog.get("voice", MODEL_KEY).source.revision
    return {
        "every call answered": bool(rows) and len(ok) == len(rows),
        "adapter loaded or kept in every A and C reply": bool(baked) and all(
            row["adapter_state"] in ("loaded", "kept") for row in baked),
        "audio, not silent, no NaN": bool(ok) and all(
            row["duration_s"] > 0 and not row["silent"] and not row["nan"] for row in ok),
        f"no VOID reply ({MODEL_KEY} @ {revision})": bool(ok) and all(
            not row["void"] for row in ok),
    }


def automatic_checks(status: dict, rows: list[dict]) -> dict[str, bool]:
    """The machine-checkable half of D65. The other half is the user's ears."""
    device = str(status.get("device") or "").split(":")[0]
    return {"bake done on mps": status["state"] == "done" and device == "mps"} | speech_checks(rows)


def build_report(run: CheckRun, bake: BakeRun, speech: tuple[list[dict], dict, dict]) -> dict:
    rows, _key, prompt_takes = speech
    status = bake.status
    adapter = None
    if bake.adapter is not None:
        adapter = {"file": ADAPTER_FILE, **(status.get("adapter") or {})}
    by_language = seconds_by_language(run.takes)
    total = sum(by_language.values())
    seconds_per_step = status.get("seconds_per_step")
    return {
        "voice": {"id": run.voice.id, "name": run.voice.name},
        "takes": {"count": len(run.takes), "ids": [take.example.id for take in run.takes],
                  "words": {str(take.example.id): take.label for take in run.takes},
                  "seconds_by_language": by_language, "total_seconds": round(total, 3),
                  "total_minutes": round(total / 60, 2), "guidance": GUIDANCE},
        "bake": status,
        "cancelled_by_user": bake.interrupted,
        "free_mb_before": bake.free_mb_before,
        "min_free_mb": bake.min_free_mb,
        "adapter": adapter,
        "prompt_takes": prompt_takes,
        "variants": VARIANTS,
        "calls": rows,
        "mean_rtf": mean_rtf(rows),
        "checks": automatic_checks(status, rows),
        "measured_thresholds": {
            f"min_free_mb >= {MIN_FREE_MB}": bake.min_free_mb >= MIN_FREE_MB,
            f"seconds_per_step <= {MAX_SECONDS_PER_STEP}":
                seconds_per_step is not None and seconds_per_step <= MAX_SECONDS_PER_STEP,
        },
        "thresholds": THRESHOLDS,
    }


def score_bake(client, run: CheckRun, adapter: bytes, rows: list[dict]) -> dict:
    """The report's `frozen` and `likeness` of a bake's files. A meter that cannot run
    leaves `likeness.error`: it never costs the result of hours of training."""
    languages = sorted({row["language"] for row in rows})
    try:
        scale = build_scale(client, run, languages)
    except Exception as exc:  # noqa: BLE001 - the reason goes in the report
        return {"likeness": {"error": _reason(exc)}}
    score_rows(scale, rows, run.out_dir)
    texts = {text_id: text for language in languages for text_id, text in SENTENCES[language]}
    label = run.label or f"{MODEL_KEY} bake"
    results = results_of(rows, label)
    return {"label": label, "results": results, "likeness": likeness_summary(results),
            "frozen": frozen_inputs(run, hashlib.sha256(adapter).hexdigest(), texts, scale)}


def run_check(client, run: CheckRun, config: dict | None, *,
              sleep: Callable[[float], None] = time.sleep) -> tuple[dict, int]:
    """The bake, the three variants and the report; `(report, exit code)`."""
    clips = [{"audio": read_example_audio(run.store.storage, take.example),
              "sha256": take.example.content_hash, "text": take.words,
              "language": take.example.language} for take in run.takes]
    bake = run_bake(client, clips, {"check": run.voice.name}, config, sleep=sleep)
    run.out_dir.mkdir(parents=True, exist_ok=True)
    speech: tuple[list[dict], dict, dict] = ([], {}, {})
    if bake.adapter is not None:
        (run.out_dir / ADAPTER_FILE).write_bytes(bake.adapter)
        speech = speak_variants(client, run, bake.adapter)
        (run.out_dir / "key.json").write_text(json.dumps(speech[1], indent=2) + "\n")
    report = build_report(run, bake, speech)
    if bake.adapter is not None:
        report |= score_bake(client, run, bake.adapter, speech[0])
    (run.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if bake.interrupted:
        return report, EXIT_INTERRUPTED
    if bake.status["state"] != "done":
        return report, 1
    return report, 0 if all(report["checks"].values()) else 1


def run_longer_bake(client, run: CheckRun, *, sleep: Callable[[float], None] = time.sleep,
                    attach: str | None = None) -> tuple[dict, int]:
    """One bake of `LONGER_BAKE_STEPS` steps from v2's takes and recipe under the watchdog,
    then the held-out texts through the new model and the stored v2, and the report;
    `(report, exit code)`. Every refusal comes before the bake. `frozen.json` goes to disk
    before the bake, and `key.json` with the rest of `frozen.json` before the first speech.
    The new adapter stays in the output folder only. The meter is a diagnostic here. With
    `attach`, it follows that running bake instead of a new one (`attach_to`)."""
    model = stored_v2(run, "longer bake")
    learned = {row["example_id"]: row["fingerprint"] for row in run.store.conn.execute(
        "SELECT example_id, fingerprint FROM voice_model_takes WHERE voice_id = ?",
        (run.voice.id,))}
    takes = [take for take in run.takes if take.example.id in learned]
    gone = sorted(learned.keys() - {take.example.id for take in takes})
    if gone:
        raise UsageError(f"take {gone[0]}, which v2 learned, is gone or being deleted")
    fingerprints = {take.example.id: _fingerprint(take.example.content_hash, take.example.language,
                                                  take.words) for take in takes}
    changed = [take_id for take_id, fingerprint in fingerprints.items()
               if fingerprint != learned[take_id]]
    if changed:
        raise UsageError(f"take {changed[0]} differs from what v2 learned: its audio, language "
                         "or words changed")
    v2_config = json.loads(model.config_json)
    config = {"steps": LONGER_BAKE_STEPS}
    frozen = {
        "voice_model": {"model_key": model.model_key, "version": model.version,
                        "sha256": model.sha256, "revision": model.revision,
                        "config": v2_config, "dataset_sha256": model.dataset_hash},
        "base_revision": catalog.get("voice", MODEL_KEY).source.revision,
        "config_sent": config,
        "recipe": v2_config | config,
        "steps": LONGER_BAKE_STEPS,
        "seed": v2_config.get("seed"),
        "takes": [{"example_id": take.example.id, "content_hash": take.example.content_hash,
                   "language": take.example.language, "seconds": take.example.seconds,
                   "fingerprint": fingerprints[take.example.id]} for take in takes],
        "takes_left_out": sorted(take.example.id for take in run.takes
                                 if take.example.id not in learned),
        "dataset_sha256": model.dataset_hash,
        "watchdog": {"max_seconds": LONGER_BAKE_MAX_SECONDS, "min_free_mb": MIN_FREE_MB,
                     "max_seconds_per_step": LONGER_BAKE_MAX_SECONDS_PER_STEP,
                     "slow_after_step": LONGER_BAKE_SLOW_AFTER_STEP},
    }
    id_file = run.out_dir / BAKE_ID_FILE
    if attach is None:
        if run.out_dir.exists() and any(run.out_dir.iterdir()):
            raise UsageError(f"the output folder {run.out_dir} is not empty")
        run.out_dir.mkdir(parents=True, exist_ok=True)
        (run.out_dir / "frozen.json").write_text(
            json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
        clips = [{"audio": read_example_audio(run.store.storage, take.example),
                  "sha256": take.example.content_hash, "text": take.words,
                  "language": take.example.language} for take in takes]
        bake = run_bake(client, clips, {"check": run.voice.name}, config, sleep=sleep,
                        watchdog=True, clock=run.clock, id_file=id_file)
    else:
        attached = attach_to(client, run, attach, frozen, model.lora_json)
        if not id_file.exists():
            id_file.write_text(json.dumps({"id": attach}) + "\n")
        bake = run_bake(client, [], {}, None, sleep=sleep, watchdog=True, clock=run.clock,
                        attach=attached)
    status = bake.status
    seconds_per_step = status.get("seconds_per_step")
    finished = bake.adapter is not None and status["step"] == LONGER_BAKE_STEPS
    checks = {f"bake done with all {LONGER_BAKE_STEPS} steps": finished}
    report = {
        "label": run.label or f"{model.model_key} {LONGER_BAKE_STEPS}-step bake vs v{model.version}",
        "voice": {"id": run.voice.id, "name": run.voice.name},
        "takes": {"count": len(takes), "ids": [take.example.id for take in takes],
                  "left_out": frozen["takes_left_out"],
                  "seconds_by_language": seconds_by_language(takes)},
        "voice_model": {"version": model.version, "sha256": model.sha256},
        "bake": status,
        "steps_done": status["step"],
        "seconds_per_step": seconds_per_step,
        "loss_first": status.get("loss_first"),
        "loss_last": status.get("loss_last"),
        "peak_device_bytes": status.get("peak_device_bytes"),
        "free_mb_before": bake.free_mb_before,
        "min_free_mb": bake.min_free_mb,
        "stop": bake.stop,
        "attached": bake.attached,
        "bake_id": status["id"],
        "poll_failures": bake.poll_failures,
        "measured_thresholds": {
            f"seconds_per_step <= {MAX_SECONDS_PER_STEP}":
                seconds_per_step is not None and seconds_per_step <= MAX_SECONDS_PER_STEP},
        "adapter": None,
        "cancelled_by_user": bake.interrupted,
        "stopped_by_echo": None,
        "stopped_by_deadline": False,
        "timed_out": None,
        "complete": False,
        "frozen": frozen,
        "calls": [],
        "checks": checks,
    }
    if bake.adapter is not None:
        (run.out_dir / ADAPTER_FILE).write_bytes(bake.adapter)
        report["adapter"] = {"file": ADAPTER_FILE, **(status.get("adapter") or {})}
    if not finished:
        (run.out_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return report, EXIT_INTERRUPTED if bake.interrupted else 1

    adapter_sha256 = hashlib.sha256(bake.adapter).hexdigest()
    references = {
        "candidate": {"language": SWEEP_LANGUAGE,
                      "adapter": {"blob": bake.adapter, "sha256": adapter_sha256},
                      "use_clip": False},
        "baseline": model_reference(model, read_voice_model(run.store.storage, model),
                                    SWEEP_LANGUAGE),
    }
    shas = {"candidate": adapter_sha256, "baseline": model.sha256}
    scale = None
    try:
        scale = build_scale(client, run, [SWEEP_LANGUAGE])
    except Exception as exc:  # noqa: BLE001 - the meter never costs a bake of hours
        report["likeness"] = {"error": _reason(exc)}
    plan = heldout_plan(run.rng, LONGER_BAKE_ROLES, "L")
    frozen |= {
        "texts": HELDOUT_TEXTS,
        "mode": {SWEEP_LANGUAGE: {"C": "adapter alone"}},
        "controls": CONTROLS,
        "adapter_sha256": shas,
        "repeats": {"short": HELDOUT_SHORT_REPEATS, "long": HELDOUT_LONG_REPEATS},
        "planned_calls": len(plan),
        "planned_rows": plan,
        "pass_rule": LONGER_BAKE_PASS_RULE,
        "wall_budget": HELDOUT_WALL_BUDGET,
    }
    if scale is not None:
        frozen["reference"] = {language: reference.provenance()
                               for language, reference in scale.references.items()}
    key = {row["file"]: {"role": row["role"], "id": row["id"], "repeat": row["repeat"]}
           for row in plan}
    (run.out_dir / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    (run.out_dir / "frozen.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    started = datetime.now(UTC)
    begun = run.clock()
    rows, wrong_echo, interrupted, stopped_by_deadline = speak_sweep(
        client, run, lambda row: references[row["role"]], plan, HELDOUT_TEXTS,
        begun + HELDOUT_WALL_SECONDS)
    elapsed = run.clock() - begun
    ended = datetime.now(UTC)
    if scale is not None:
        score_rows(scale, rows, run.out_dir)
    valid = [row for row in rows if _valid_clip(row, shas[row["role"]])]
    complete = len(valid) == len(plan)
    report |= {
        "cancelled_by_user": interrupted,
        "stopped_by_echo": wrong_echo,
        "stopped_by_deadline": stopped_by_deadline,
        "timed_out": next((row["file"] for row in rows if row.get("timed_out")), None),
        "started": started.isoformat(),
        "ended": ended.isoformat(),
        "elapsed_s": round(elapsed, 3),
        "complete": complete,
        "roles": [role | {"valid_clips": sum(row["role"] == role["role"] for row in valid)}
                  for role in LONGER_BAKE_ROLES],
        "calls": rows,
        "checks": checks | {
            "the bake trained the takes of v2": status.get("dataset_sha256") == model.dataset_hash,
            f"the recipe is v2's with {LONGER_BAKE_STEPS} steps": status.get("config") == frozen["recipe"],
        } | speech_checks(rows) | {f"all {len(plan)} clips valid": complete},
    }
    (run.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if interrupted:
        return report, EXIT_INTERRUPTED
    return report, 0 if all(report["checks"].values()) else 1


def _file_lines(rows: list[dict]) -> list[str]:
    # RU-HELD-LONG is wider than the column every other id fits.
    width = max([10] + [len(row["id"]) + 2 for row in rows])
    lines = [f"{'sentence':<{width}}{'file':<20}{'dur s':>7}{'dBFS':>7}  identity"]
    for row in rows:
        if "error" in row:
            lines.append(f"{row['id']:<{width}}{row['file']:<20}ERROR")
            continue
        lines.append(
            f"{row['id']:<{width}}{row['file']:<20}{row['duration_s']:>7.2f}"
            f"{'-' if row['rms_dbfs'] is None else f'{row['rms_dbfs']:.1f}':>7}"
            f"  {'VOID' if row['void'] else 'ok'}")
    return lines


def format_measure_summary(report: dict, out_dir: Path) -> str:
    """Blind, as `format_summary`: it prints no score and no variant."""
    lines = ["stopped by Ctrl-C: the report holds the files made so far"] if (
        report["cancelled_by_user"]) else []
    lines += _file_lines(report["calls"])
    lines += [""] + [f"not measured: {language} ({problem})"
                     for language, problem in report["languages_left_out"].items()]
    lines += [f"  [{'x' if passed else ' '}] {name}" for name, passed in report["checks"].items()]
    lines += ["", LISTEN_FIRST, f"Output: {out_dir}"]
    return "\n".join(lines)


def format_sweep_summary(report: dict, out_dir: Path,
                         listen_first: str = SWEEP_LISTEN_FIRST) -> str:
    """Blind, as `format_summary`: it prints no score and no cell."""
    lines = ["stopped by Ctrl-C: the report holds the files made so far"] if (
        report["cancelled_by_user"]) else []
    if report["stopped_by_echo"]:
        lines.append(f"stopped at {report['stopped_by_echo']}: the reply did not echo the "
                     "settings it was sent, so the service may be stale")
    lines += _file_lines(report["calls"])
    lines += [""] + [f"  [{'x' if passed else ' '}] {name}"
                     for name, passed in report["checks"].items()]
    lines += ["", listen_first, f"Output: {out_dir}"]
    return "\n".join(lines)


def format_heldout_summary(report: dict, out_dir: Path,
                           listen_first: str = HELDOUT_LISTEN_FIRST) -> str:
    """Blind, as `format_summary`: it prints no score and no cell."""
    lines = []
    if report["stopped_by_deadline"]:
        lines.append("stopped by the wall budget: the report holds the files made so far")
    if report["timed_out"]:
        lines.append(f"stopped at {report['timed_out']}: the client stopped waiting for it, and "
                     "that call may still run on the service")
    return "\n".join([*lines, format_sweep_summary(report, out_dir, listen_first)])


def format_longer_summary(report: dict, out_dir: Path) -> str:
    """Blind, as `format_summary`: it prints no score and no model per file."""
    bake = report["bake"]
    lines = [
        (f"bake: {bake['state']} · steps {report['steps_done']} of {bake['steps']} · "
         f"seconds_per_step {report['seconds_per_step']} · loss {report['loss_first']} → "
         f"{report['loss_last']}"),
        (f"free_mb_before {report['free_mb_before']} · min_free_mb {report['min_free_mb']} · "
         f"peak_device_bytes {report['peak_device_bytes']}"),
    ]
    if report["stop"] is not None:
        lines.append(f"cancel requested after {report['stop']['elapsed_s']:.0f} s: "
                     f"{report['stop']['trigger']} (the bake stops at a step boundary)")
    if bake["state"] == "failed":
        lines.append(f"the bake failed: {bake.get('error')}")
    lines += [f"  [{'x' if passed else ' '}] {name}"
              for name, passed in report["measured_thresholds"].items()]
    if not report["checks"][f"bake done with all {LONGER_BAKE_STEPS} steps"]:
        lines += ["", f"no complete {LONGER_BAKE_STEPS}-step voice model: nothing was spoken"]
        lines += [f"  [{'x' if passed else ' '}] {name}" for name, passed in report["checks"].items()]
        return "\n".join([*lines, f"Output: {out_dir}"])
    return "\n".join([*lines, "", format_heldout_summary(report, out_dir, LONGER_LISTEN_FIRST)])


def _flat(value, prefix: str = "") -> dict:
    if not isinstance(value, dict):
        return {prefix: value}
    return {path: leaf for key, inner in value.items()
            for path, leaf in _flat(inner, f"{prefix}.{key}" if prefix else str(key)).items()}


def _named(report: dict) -> str:
    made_by = sorted({f"{result['model']} @ {str(result['revision'])[:7]}"
                      for result in report.get("results", []) if result["model"] is not None})
    return f"{report.get('label') or '-'} ({', '.join(made_by) or 'no call answered'})"


def _compare_line(scope: str, arm: str, old: dict, new: dict) -> str:
    cells = [f"{scope:<10}{arm:<5}"]
    for stats in (old, new):
        cells += [f"{'-' if stats.get(name) is None else format(stats[name], '.4f'):>9}"
                  for name in COMPARED]
        cells.append(f"{stats.get('failures', 0):>6}")
    failed = old.get("failures") or new.get("failures")
    for name in COMPARED:
        known = not failed and old.get(name) is not None and new.get(name) is not None
        cells.append(f"{format(new[name] - old[name], '+.4f') if known else '-':>9}")
    return "".join(cells)


def format_compare(old: dict, new: dict) -> str:
    """Mean, worst and spread per arm and per text of two reports, and their deltas. It
    reads only `results` and `frozen`, so the two may come from different models or
    architectures. A delta is evidence, not a verdict. A scope with a failure on either
    side prints no delta: its values cover other files, and a failed hard file raises a mean."""
    before, after = (likeness_summary(report.get("results", [])) for report in (old, new))
    names = "".join(f"{name:>9}" for name in COMPARED)
    lines = [f"old: {_named(old)}", f"new: {_named(new)}", "",
             f"{'':<15}{'old':<33}{'new':<33}delta",
             f"{'text':<10}{'arm':<5}{names}{'fail':>6}{names}{'fail':>6}{names}"]
    arms = sorted(before["by_arm"].keys() | after["by_arm"].keys())
    lines += [_compare_line("all", arm, before["by_arm"].get(arm, {}), after["by_arm"].get(arm, {}))
              for arm in arms]
    for text_id in dict.fromkeys([*before["by_text"], *after["by_text"]]):
        lines += [_compare_line(text_id, arm, before["by_text"].get(text_id, {}).get(arm, {}),
                                after["by_text"].get(text_id, {}).get(arm, {})) for arm in arms]
    frozen = _flat(old.get("frozen", {})), _flat(new.get("frozen", {}))
    changed = sorted(name for name in frozen[0].keys() | frozen[1].keys()
                     if frozen[0].get(name) != frozen[1].get(name))
    lines += ["", f"frozen inputs that differ: {', '.join(changed) or 'none'}"]
    return "\n".join(lines)


def format_summary(report: dict, out_dir: Path) -> str:
    bake = report["bake"]
    lines = []
    if report["cancelled_by_user"]:
        lines.append(f"cancelled at step {bake['step']}" if bake["state"] == "cancelled"
                     else f"the bake ended {bake['state']} before the cancel; "
                          "no adapter is kept")
    if bake["state"] == "failed":
        lines.append(f"the bake failed: {bake.get('error')}")
    lines += _file_lines(report["calls"])
    adapter = report["adapter"] or {}
    stage_seconds = ", ".join(f"{stage} {seconds:.0f} s"
                              for stage, seconds in (bake.get("stage_seconds") or {}).items())
    lines += [
        "",
        (f"bake: {bake['state']} on {bake.get('device') or '-'} · steps {bake['step']} of "
         f"{bake['steps']} · seconds_per_step {bake.get('seconds_per_step')} · loss "
         f"{bake.get('loss_first')} → {bake.get('loss_last')}"),
        f"stage seconds: {stage_seconds or '-'}",
        (f"free_mb_before {report['free_mb_before']} · min_free_mb {report['min_free_mb']} · "
         f"peak_device_bytes {bake.get('peak_device_bytes')} · adapter "
         f"{adapter.get('bytes', '-')} bytes"),
    ]
    lines += [f"mean RTF {variant} ({mean['label']}): {mean['rtf'] if mean['rtf'] is not None else '-'}"
              for variant, mean in report["mean_rtf"].items()]
    takes = report["takes"]
    lines += ["", f"recorded: {takes['total_minutes']:.1f} min in {takes['count']} takes; {GUIDANCE}",
              ""]
    lines += [f"  [{'x' if passed else ' '}] {name}" for name, passed in report["checks"].items()]
    lines += [f"  [{'x' if passed else ' '}] {name}"
              for name, passed in report["measured_thresholds"].items()]
    lines += ["", THRESHOLDS, f"Output: {out_dir}"]
    return "\n".join(lines)


def _text_arg(value: str) -> tuple[int, str]:
    take_id, sep, words = value.partition("=")
    if not sep or not take_id.strip().isdigit() or not words.strip():
        raise argparse.ArgumentTypeError(
            f"{value!r} is not ID=WORDS; quote the words: --text \"9=the words you said\"")
    return int(take_id), words.strip()


def main(argv: list[str] | None = None, *, client=None, settings: Settings | None = None,
         sleep: Callable[[float], None] = time.sleep, rng: random.Random | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bake a saved voice into a LoRA adapter through the running models "
                    "service and write blind X/Y/Z files: A adapter + prompt take, B prompt "
                    "take alone, C adapter alone. The database is opened read-only.")
    parser.add_argument("--voice", help="the saved voice's name")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true",
                      help="print every take with its words, then stop")
    mode.add_argument("--confirm-texts", action="store_true",
                      help="you read --list; bake every take with its words")
    mode.add_argument("--measure", action="store_true",
                      help="no bake: speak through the stored voice model and score every file "
                           "in report.json")
    mode.add_argument("--sweep", action="store_true",
                      help="no bake: speak one new sentence through the stored voice model on "
                           "the grid of cfg_value and inference_timesteps, under neutral names")
    mode.add_argument("--heldout", action="store_true",
                      help="no bake: speak the frozen held-out texts through the stored voice "
                           "model at the baseline and candidate settings, under neutral names")
    mode.add_argument("--longer-bake", action="store_true",
                      help=f"bake {LONGER_BAKE_STEPS} steps from v2's takes and recipe under a "
                           "watchdog, then speak the held-out texts through the new model and "
                           "v2, under neutral names")
    mode.add_argument("--compare", nargs=2, type=Path, metavar=("OLD", "NEW"),
                      help="print mean, worst and spread per arm and per text of two "
                           "report.json files, the deltas and the frozen inputs that differ; "
                           "run it after you listen")
    parser.add_argument("--text", type=_text_arg, action="extend", nargs="+", default=[],
                        metavar="ID=WORDS", help="the words of one take, for this run only")
    parser.add_argument("--steps", type=int, default=None, help="training steps")
    parser.add_argument("--learning-rate", type=float, default=None, help="learning rate")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="--measure: calls per text and variant")
    parser.add_argument("--label", default=None,
                        help="this run's name in report.json and --compare")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default <data_dir>/voice-bake-check/<UTC stamp>/)")
    parser.add_argument("--attach", default=None, metavar="BAKE_ID",
                        help="--longer-bake: follow this running bake instead of starting one; "
                             "--out must hold only its run's frozen.json")
    args = parser.parse_args(argv)
    if args.compare is None and args.voice is None:
        parser.error("--voice is required")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.longer_bake and args.steps not in (None, LONGER_BAKE_STEPS):
        parser.error(f"--longer-bake trains {LONGER_BAKE_STEPS} steps, not {args.steps}")
    if args.longer_bake and args.learning_rate is not None:
        parser.error("--longer-bake keeps v2's learning rate")
    if args.attach is not None and not args.longer_bake:
        parser.error("--attach follows a --longer-bake only")
    if args.compare is not None:
        try:
            print(format_compare(*(json.loads(path.read_text()) for path in args.compare)))
        except (OSError, ValueError) as exc:
            print(f"  ! cannot read the reports: {exc}", file=sys.stderr)
            return EXIT_USAGE
        return 0

    settings = settings or Settings()
    settings = settings.model_copy(update={"data_dir": _data_dir(settings)})
    try:
        conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        print(f"  ! cannot open {settings.db_path} read-only: {exc}", file=sys.stderr)
        return 1
    conn.row_factory = sqlite3.Row
    try:
        store = VoiceStore(conn, LocalStorage(settings.voices_dir), settings.owner_id)
        try:
            voice = find_voice(conn, settings.owner_id, args.voice)
            takes = takes_of(voice, dict(args.text))
            if args.list:
                print(format_takes(voice, takes))
                return 0
            if not (args.measure or args.sweep or args.heldout or args.longer_bake):
                check_words(takes)
        except UsageError as exc:
            print(f"  ! {exc}", file=sys.stderr)
            return EXIT_USAGE
        except sqlite3.Error as exc:
            print(f"  ! cannot read {settings.db_path}: {exc}", file=sys.stderr)
            return 1
        config = {name: value for name, value in
                  (("steps", args.steps), ("learning_rate", args.learning_rate))
                  if value is not None} or None
        out_dir = args.out or (settings.data_dir / "voice-bake-check"
                               / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
        run = CheckRun(store, voice, takes, out_dir, rng or random.Random(), label=args.label)
        client = client or settings.build_models_client()
        try:
            if args.measure:
                report, code = run_measure(client, run, args.repeats)
                print(format_measure_summary(report, out_dir))
                return code
            if args.sweep:
                report, code = run_sweep(client, run)
                print(format_sweep_summary(report, out_dir))
                return code
            if args.heldout:
                report, code = run_heldout(client, run)
                print(format_heldout_summary(report, out_dir))
                return code
            if args.longer_bake:
                report, code = run_longer_bake(client, run, sleep=sleep, attach=args.attach)
                print(format_longer_summary(report, out_dir))
                return code
            report, code = run_check(client, run, config, sleep=sleep)
        except UsageError as exc:
            print(f"  ! {exc}", file=sys.stderr)
            return EXIT_USAGE
        except KeyboardInterrupt:
            print("\n  ! stopped waiting; no report was written", file=sys.stderr)
            return EXIT_INTERRUPTED
        except Exception as exc:  # noqa: BLE001 - say what failed, then stop
            print(f"  ! the check failed ({settings.models_base_url}): {_reason(exc)}",
                  file=sys.stderr)
            return 1
        print(format_summary(report, out_dir))
        return code
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
