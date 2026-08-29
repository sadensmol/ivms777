"""Speaking a film's passages as continuous runs (design §11.3).

Pure: `speak_passages` is handed the synthesiser, so grouping and timing arithmetic are
tested without loading a voice model.
"""

from itertools import pairwise

import numpy as np

from audio.segments import runs, speak_passages

CHUNKS = ["one.", "two.", "three.", "four.", "five."]


def fake_speak(text, voice):
    """100 samples per character, so timings are checkable by hand."""
    return np.zeros(len(text) * 100, dtype=np.float32)


def test_passages_sharing_a_voice_become_one_call():
    # Synthesising each passage separately restarts the prosody, so an eight-passage
    # film reads as eight different narrators even where the speaker never changes.
    grouped = runs(CHUNKS, [None, None, "aidar", None, None], "xenia")
    assert [voice for voice, _ in grouped] == ["xenia", "aidar", "xenia"]
    assert [[i for i, _ in passages] for _, passages in grouped] == [[0, 1], [2], [3, 4]]


def test_one_voice_throughout_is_a_single_call():
    calls = []
    speak_passages(CHUNKS, [], "xenia",
                   lambda t, v: (calls.append(v), fake_speak(t, v))[1], 1000)
    assert len(calls) == 1


def test_every_passage_still_gets_its_own_timing():
    _samples, timings = speak_passages(CHUNKS, [], "xenia", fake_speak, 1000)
    assert len(timings) == len(CHUNKS)
    assert [t["text"] for t in timings] == CHUNKS
    assert timings[0]["start"] == 0.0
    # Contiguous: a gap between passages would show as black between shots.
    for earlier, later in pairwise(timings):
        assert earlier["end"] == later["start"]


def test_the_timings_span_the_whole_clip():
    samples, timings = speak_passages(CHUNKS, [], "xenia", fake_speak, 1000)
    assert timings[-1]["end"] == round(samples.size / 1000, 3)


def test_a_run_boundary_is_where_the_voice_actually_changes():
    _samples, timings = speak_passages(
        CHUNKS, [None, None, "aidar", None, None], "xenia", fake_speak, 1000
    )
    assert len(timings) == 5
    assert timings[2]["start"] < timings[2]["end"] <= timings[3]["start"]


def test_no_passages_is_not_a_crash():
    samples, timings = speak_passages([], [], "xenia", fake_speak, 1000)
    assert timings == [] and samples.size >= 1
