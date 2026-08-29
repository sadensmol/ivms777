"""Speaking a film's passages without chopping the delivery to pieces (design §11.3).

A film's narration is handed to the voice as one chunk per PASSAGE, because each shot
must begin exactly when its own words do (§11.1). Synthesising each of those separately
is the obvious implementation and it sounds wrong: every call starts a fresh prosody
contour, so an eight-passage film reads as eight different people, even when every
passage is the same speaker.

So passages are grouped into RUNS that share a voice, each run is spoken in ONE call,
and the per-passage timings are recovered by splitting the run's audio in proportion to
the text. A voice genuinely changes only where the viewer asked it to.

The proportional split is an approximation — a long word takes longer than a short one —
but it is bounded inside a single continuous utterance and worth far more than the
alternative: exact boundaries around a delivery that lurches every few seconds.

PURE except for the `speak_one` callable it is handed, so the grouping and the timing
arithmetic are unit-tested without loading a model.
"""

import numpy as np


def runs(chunks: list[str], voices: list[str | None], default: str) -> list[tuple]:
    """Consecutive passages that share a voice, as `(voice, [(index, text), …])`.

    Grouping is what keeps a delivery continuous; the indices are what let each
    passage's timing be recovered afterwards.
    """
    grouped: list[tuple] = []
    for index, text in enumerate(chunks):
        voice = (voices[index] if index < len(voices) and voices[index] else default)
        if grouped and grouped[-1][0] == voice:
            grouped[-1][1].append((index, text))
        else:
            grouped.append((voice, [(index, text)]))
    return grouped


def speak_passages(
    chunks: list[str],
    voices: list[str | None],
    default: str,
    speak_one,
    sample_rate: int,
    gap_seconds: float = 0.0,
) -> tuple[np.ndarray, list[dict]]:
    """`(samples, segments)` for a whole narration.

    `speak_one(text, voice) -> np.ndarray` does the synthesis; everything else here is
    grouping and arithmetic. `segments` carries one `{text, start, end}` per PASSAGE, in
    order, which is what the film's shots are timed from.
    """
    gap = np.zeros(int(sample_rate * gap_seconds), dtype=np.float32)
    clips: list[np.ndarray] = []
    timings: list[dict | None] = [None] * len(chunks)
    frames = 0
    for voice, passages in runs(chunks, voices, default):
        # ONE call for the whole run: the join between two passages of the same voice
        # should be a breath, not a splice.
        spoken = np.asarray(speak_one(" ".join(text for _index, text in passages), voice),
                            dtype=np.float32)
        if clips and gap.size:
            clips.append(gap)
            frames += gap.size
        lengths = [max(len(text), 1) for _index, text in passages]
        total = sum(lengths)
        offset = 0
        for (index, text), length in zip(passages, lengths):
            # Where this passage sits inside the run, by share of the text.
            span = round(spoken.size * length / total)
            start = frames + offset
            end = start + span
            timings[index] = {
                "text": text,
                "start": round(start / sample_rate, 3),
                "end": round(end / sample_rate, 3),
            }
            offset += span
        # The last passage of a run owns any rounding remainder, so no gap opens up.
        if passages:
            timings[passages[-1][0]]["end"] = round((frames + spoken.size) / sample_rate, 3)
        clips.append(spoken)
        frames += spoken.size
    samples = np.concatenate(clips) if clips else np.zeros(1, dtype=np.float32)
    return samples, [t for t in timings if t is not None]
