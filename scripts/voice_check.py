"""Validate voice copy in the user's own stack (design §4.1, PLAN §2.13, §6.1).

One command turns the findings §10 protocol into files to listen to:

    uv run --no-sync python -m scripts.voice_check --wav ~/voice-en.wav --language en --cross

It reads one example clip, unloads the voice unit, speaks the fixed sentences with no
stored conditioning (pass 1, "fresh"), unloads again and speaks them once more with the
blob pass 1 built (pass 2, "restored"). Each pair lands as `<id>-A.wav` / `<id>-B.wav` in a
random order, so the user can judge "same voice" before opening `key.json`.

A thin client (CLAUDE.md § "One model process"): no torch, no model. Every synthesis is a
`POST /speak` to the models service the user's `make up` runs. It keeps its own small
clip check because it lands before the saved-voice store exists.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from config import Settings
from models import catalog
from storage.wav import decode_wav

MODEL_KEY = "voxcpm2"
TOKEN = "check"  # pragma: allowlist secret
SAMPLE_RATE = 48000
MAX_BYTES = 4 * 1024 * 1024
MIN_SECONDS, MAX_SECONDS = 5.0, 30.0
SILENCE_RMS = 0.00316  # -50 dBFS

# The fixed test sentences of findings §10. Never change them: every result is compared
# against the same text.
SENTENCES: dict[str, list[tuple[str, str]]] = {
    "en": [
        ("EN-1", "We drove to the lake early on Saturday, before the fog had lifted."),
        ("EN-2", "Julia laughed when the dog jumped into the water for the third time."),
        ("EN-3", "It was the fourteenth of July, 2023, and nobody wanted to go home."),
        ("EN-4", "Was it really that cold, or do we only remember it that way?"),
        ("EN-5", ("The road wound through the hills for hours. Every turn showed another valley, "
                  "greener than the last. By evening we reached the old house, tired and happy, "
                  "and sat on the porch until the stars came out. Nobody said much. "
                  "Nobody needed to.")),
    ],
    "ru": [
        ("RU-1", "В субботу рано утром мы поехали к озеру, пока туман ещё не рассеялся."),
        ("RU-2", "Юля засмеялась, когда собака в третий раз прыгнула в воду."),
        ("RU-3", "Это было четырнадцатое июля 2023 года, и никто не хотел уезжать домой."),
        ("RU-4", "Неужели тогда было так холодно, или нам это только кажется?"),
        ("RU-5", ("Дорога часами петляла между холмами. За каждым поворотом открывалась новая "
                  "долина, зеленее прежней. К вечеру мы добрались до старого дома, усталые и "
                  "счастливые, и сидели на крыльце, пока не показались звёзды. Никто почти "
                  "ничего не говорил. Слова были не нужны.")),
    ],
}

# Help text only: what the user reads into the microphone (PLAN §2.7).
RECORDING_PROMPTS = {
    "en": "The quick morning light crept across the kitchen table. I poured a cup of coffee, "
          "opened the window, and listened to the city slowly waking up below.",
    "ru": "Утренний свет медленно скользил по кухонному столу. За окном понемногу просыпался "
          "город, а в комнате пахло свежим кофе и тёплым хлебом.",
}

THRESHOLDS = (
    "PASS needs (§6.1 step 8): same-language mean similarity >= 4 and naturalness >= 4 "
    "in en and in ru; <= 1 wrong word per sentence; A and B judged the same voice; "
    "RTF <= 1.0. Cross-language results are reported only.\n"
    "Listen to every A/B pair BEFORE you open key.json."
)


class ClipError(ValueError):
    """The clip breaks the example contract (§2.7). The message names what was found."""


@dataclass(frozen=True)
class Clip:
    data: bytes
    sha256: str
    seconds: float
    rms: float


@dataclass(frozen=True)
class Call:
    """One `/speak` call of a pass."""

    id: str
    language: str
    text: str
    segments: list[str] | None
    cross: bool


def check_clip(data: bytes) -> Clip:
    """The clip contract of `add_example` (§2.7): ≤ 4 MiB, an uncompressed 16-bit mono
    48 kHz WAV with every frame present, 5-30 s long and not silent."""
    if len(data) > MAX_BYTES:
        raise ClipError(f"The clip is {len(data)} bytes; the limit is {MAX_BYTES} bytes.")
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            comptype = handle.getcomptype()
            width = handle.getsampwidth()
            channels = handle.getnchannels()
            rate = handle.getframerate()
            nframes = handle.getnframes()
            payload = handle.readframes(nframes)
    except (wave.Error, EOFError) as exc:
        raise ClipError("This file is not a WAV file.") from exc
    if comptype != "NONE" or width != 2 or channels != 1 or rate != SAMPLE_RATE:
        found = "compressed" if comptype != "NONE" else f"{width * 8}-bit"
        layout = "mono" if channels == 1 else ("stereo" if channels == 2 else f"{channels}-channel")
        raise ClipError(
            f"The clip must be mono 16-bit PCM at {SAMPLE_RATE} Hz; "
            f"this file is {found} {layout} at {rate} Hz."
        )
    if nframes <= 0:
        raise ClipError("The clip has no audio frames.")
    if len(payload) != nframes * 2:
        raise ClipError(
            f"The clip is truncated: its header declares {nframes} frames, "
            f"but only {len(payload) // 2} are present."
        )
    seconds = nframes / rate
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise ClipError(
            f"The clip is {seconds:.2f} s long; it must be {MIN_SECONDS:g}-{MAX_SECONDS:g} s."
        )
    samples, _rate = decode_wav(data)
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms < SILENCE_RMS:
        raise ClipError(f"The clip is silent: RMS {_dbfs(rms):.1f} dBFS is below -50 dBFS.")
    return Clip(data=data, sha256=hashlib.sha256(data).hexdigest(), seconds=seconds, rms=rms)


def split_sentences(text: str) -> list[str]:
    """The passages of EN-5 / RU-5, one per sentence, for the 5-segment call."""
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def plan_calls(language: str, cross: bool) -> list[Call]:
    """The calls of ONE pass, in order: the language's sentences, its long sentence as
    5 segments in one call, then (with `--cross`) the other language's sentences."""
    calls = [Call(sid, language, text, None, False) for sid, text in SENTENCES[language]]
    long_id, long_text = SENTENCES[language][-1]
    calls.append(Call(f"{long_id}-segments", language, long_text,
                      split_sentences(long_text), False))
    if cross:
        other = "ru" if language == "en" else "en"
        calls += [Call(sid, other, text, None, True) for sid, text in SENTENCES[other]]
    return calls


def wav_metrics(data: bytes) -> dict:
    """Duration, header sample rate, byte count, RMS and NaN of one reply WAV."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        rate = handle.getframerate()
        seconds = handle.getnframes() / rate if rate else 0.0
    samples, _rate = decode_wav(data)
    nan = bool(np.isnan(samples).any())
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    return {"duration_s": round(seconds, 3), "wav_sample_rate": rate, "bytes": len(data),
            "rms": round(rms, 6), "rms_dbfs": round(_dbfs(rms), 1) if rms > 0 else None, "nan": nan,
            "silent": rms < SILENCE_RMS}


def segments_monotonic(segments: list[dict], expected: int | None) -> bool:
    """Every passage has `start <= end` and starts no earlier than the one before ends;
    with `expected`, exactly that many passages came back."""
    if expected is not None and len(segments) != expected:
        return False
    previous_end = 0.0
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        if start > end or start < previous_end - 1e-3:
            return False
        previous_end = end
    return True


def is_void(reply: dict, revision: str) -> bool:
    """Findings §10 "Declared identity": a reply from any other model or Hub commit
    proves nothing about VoxCPM2 and is neither a pass nor a fail."""
    return reply.get("model") != MODEL_KEY or reply.get("revision") != revision


def reference(clip: Clip, language: str, conditioning: list[dict]) -> dict:
    return {TOKEN: {"audio": clip.data, "sha256": clip.sha256, "language": language,
                    "transcript": None, "conditioning": conditioning}}


def _dbfs(rms: float) -> float:
    return 20 * math.log10(rms) if rms > 0 else float("-inf")


def _speak(client, call: Call, refs: dict, revision: str,
           clock: Callable[[], float]) -> tuple[dict, dict | None]:
    """`(row, reply)` for one call. A failed call is a row with `error`, not a crash:
    one bad sentence must not hide the results of the others."""
    row: dict = {"id": call.id, "language": call.language, "cross": call.cross,
                 "chars": len(call.text), "segments_sent": len(call.segments or [])}
    started = clock()
    try:
        reply = client.speak(call.text, call.language, TOKEN, call.segments, None,
                             references=refs)
    except Exception as exc:  # noqa: BLE001 - every reason is reported, none swallowed
        row["wall_s"] = round(clock() - started, 3)
        row["error"] = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
        return row, None
    wall = clock() - started
    metrics = wav_metrics(reply["audio"])
    state = (reply.get("conditioning") or {}).get(TOKEN) or {}
    seconds = metrics["duration_s"]
    row.update(metrics)
    row.update({
        "sample_rate": reply.get("sample_rate"),
        "model": reply.get("model"),
        "revision": reply.get("revision"),
        "conditioning_state": state.get("state"),
        "conditioning_reason": state.get("reason"),
        "wall_s": round(wall, 3),
        "rtf": round(wall / seconds, 3) if seconds else None,
        "chars_per_s": round(len(call.text) / seconds, 2) if seconds else None,
        "segments_monotonic": segments_monotonic(
            reply.get("segments") or [], len(call.segments) if call.segments else None),
        "void": is_void(reply, revision),
    })
    return row, reply


def run_check(
    client,
    clip: Clip,
    language: str,
    out_dir: Path,
    *,
    cross: bool = False,
    revision: str | None = None,
    rng: random.Random | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Both passes against `client`; writes the WAVs, `key.json` and `report.json` into
    `out_dir` and returns the report."""
    revision = revision or catalog.get("voice", MODEL_KEY).source.revision
    rng = rng or random.Random()
    calls = plan_calls(language, cross)
    out_dir.mkdir(parents=True, exist_ok=True)

    client.model_unload("voice")
    free_before = client.models_state()["free_mb"]
    free_after: float | None = None
    rows: list[dict] = []
    audio: dict[str, dict[str, bytes]] = {}
    blob: dict | None = None

    # Pass 1 — fresh: no stored conditioning, so the child builds it from the clip.
    for index, call in enumerate(calls):
        row, reply = _speak(client, call, reference(clip, language, []), revision, clock)
        row["pass"] = "fresh"
        rows.append(row)
        if index == 0:
            free_after = client.models_state()["free_mb"]
        if reply is None:
            continue
        audio.setdefault(call.id, {})["fresh"] = reply["audio"]
        state = (reply.get("conditioning") or {}).get(TOKEN) or {}
        if blob is None and state.get("blob"):
            blob = {"model": state["model"], "revision": state["revision"],
                    "blob": state["blob"]}

    # Pass 2 — restored: a fresh child, handed the blob pass 1 built.
    client.model_unload("voice")
    stored = [blob] if blob else []
    for call in calls:
        row, reply = _speak(client, call, reference(clip, language, stored), revision, clock)
        row["pass"] = "restored"
        rows.append(row)
        if reply is not None:
            audio.setdefault(call.id, {})["restored"] = reply["audio"]

    key: dict[str, dict[str, str]] = {}
    for call in calls:
        takes = audio.get(call.id, {})
        order = ["fresh", "restored"]
        rng.shuffle(order)
        key[call.id] = {"A": order[0], "B": order[1]}
        for label, which in zip(("A", "B"), order):
            if which in takes:
                (out_dir / f"{call.id}-{label}.wav").write_bytes(takes[which])
    for row in rows:
        label = "A" if key[row["id"]]["A"] == row["pass"] else "B"
        row["file"] = f"{row['id']}-{label}.wav" if "error" not in row else None

    drop = (free_before - free_after) if free_after is not None else None
    report = {
        "clip": {"sha256": clip.sha256, "seconds": round(clip.seconds, 3),
                 "rms_dbfs": round(_dbfs(clip.rms), 1), "language": language},
        "expected": {"model": MODEL_KEY, "revision": revision, "sample_rate": SAMPLE_RATE},
        "free_mb_before": free_before,
        "free_mb_after_first_call": free_after,
        "resident_cost_mb": round(drop, 1) if drop is not None else None,
        "calls": rows,
        "checks": automatic_checks(rows),
        "thresholds": THRESHOLDS,
    }
    (out_dir / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def automatic_checks(rows: list[dict]) -> dict[str, bool]:
    """The machine-checkable half of §6.1 step 4. The other half is the user's ears."""
    ok = [row for row in rows if "error" not in row]
    return {
        "every call answered": len(ok) == len(rows),
        "no VOID reply": all(not row["void"] for row in ok),
        f"sample rate {SAMPLE_RATE}": all(row["sample_rate"] == SAMPLE_RATE for row in ok),
        "audio, not silent, no NaN": all(
            row["duration_s"] > 0 and not row["silent"] and not row["nan"] for row in ok),
        "segments monotonic": all(row["segments_monotonic"] for row in ok),
        "pass 1 built": all(row["conditioning_state"] == "built"
                            for row in ok if row["pass"] == "fresh"),
        "pass 2 restored": all(row["conditioning_state"] == "restored"
                               for row in ok if row["pass"] == "restored"),
    }


def format_summary(report: dict, out_dir: Path) -> str:
    header = (f"{'call':<16}{'pass':<10}{'file':<22}{'dur s':>7}{'rate':>7}{'dBFS':>7}"
              f"{'state':>10}{'wall s':>8}{'RTF':>7}  identity")
    lines = [header, "-" * 104]
    for row in report["calls"]:
        if "error" in row:
            lines.append(f"{row['id']:<16}{row['pass']:<10}ERROR {row['error']}")
            continue
        lines.append(
            f"{row['id']:<16}{row['pass']:<10}{row['file']:<22}{row['duration_s']:>7.2f}"
            f"{row['sample_rate']:>7}{'-' if row['rms_dbfs'] is None else f'{row['rms_dbfs']:.1f}':>7}"
            f"{row['conditioning_state'] or '-':>10}"
            f"{row['wall_s']:>8.2f}{row['rtf'] if row['rtf'] is not None else '-':>7}"
            f"  {'VOID' if row['void'] else 'ok'}"
        )
    cost = report["resident_cost_mb"]
    lines += [
        "",
        "resident cost (free_mb drop across the first call): "
        + (f"{cost:.0f} MB" if cost is not None else "unknown"),
        "",
    ]
    lines += [f"  [{'x' if passed else ' '}] {name}" for name, passed in report["checks"].items()]
    lines += ["", THRESHOLDS, f"Output: {out_dir}"]
    return "\n".join(lines)


def _data_dir(settings: Settings) -> Path:
    """The data dir of the user's stack. A plain shell on a mac has no
    `IVMS777_DATA_DIR`, and `Settings` then means the container path `/data`; `make up`
    uses `~/.ivms777` there, so this does too."""
    if "IVMS777_DATA_DIR" not in os.environ and settings.profile == "mac":
        return Path.home() / ".ivms777"
    return settings.data_dir


def main(argv: list[str] | None = None, *, client=None, settings: Settings | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate voice copy (VoxCPM2) in the running models service: "
                    "fresh and restored passes, blind A/B files, report.json.",
        epilog=(
            "Record 12-15 s in a quiet room, reading once:\n"
            f"  en: {RECORDING_PROMPTS['en']}\n"
            f"  ru: {RECORDING_PROMPTS['ru']}\n"
            "Convert it: afconvert -f WAVE -d LEI16@48000 -c 1 in.m4a ~/voice-en.wav"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--wav", type=Path, required=True,
                        help="mono 16-bit PCM WAV at 48000 Hz, 5-30 s")
    parser.add_argument("--language", choices=sorted(SENTENCES), required=True,
                        help="the language spoken in the clip")
    parser.add_argument("--cross", action="store_true",
                        help="also speak the other language's sentences (reported only)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default <data_dir>/voice-check/<UTC stamp>/)")
    args = parser.parse_args(argv)

    try:
        clip = check_clip(args.wav.read_bytes())
    except (OSError, ClipError) as exc:
        print(f"  ! {args.wav}: {exc}", file=sys.stderr)
        return 1
    settings = settings or Settings()
    out_dir = args.out or (_data_dir(settings) / "voice-check"
                           / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    client = client or settings.build_models_client()
    try:
        report = run_check(client, clip, args.language, out_dir, cross=args.cross)
    except Exception as exc:  # noqa: BLE001 - unload/state failed: say where, then stop
        print(f"  ! the models service at {settings.models_base_url} failed: {exc}",
              file=sys.stderr)
        return 1
    print(format_summary(report, out_dir))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
