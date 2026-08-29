"""Splitting a narration script for Silero (design §11.1).

`audio/silero_voice.py` imports torch, so ONLY the pure splitter is imported here —
never the class (CLAUDE.md § "One model process").
"""

from audio.silero_voice import MAX_SYMBOLS, split_text


def test_a_short_script_is_one_piece():
    assert split_text("Осенний день. Пасмурный, но тихий.") == [
        "Осенний день. Пасмурный, но тихий."
    ]


def test_every_piece_stays_under_the_limit():
    text = "Мы ехали по Темкину. " * 60
    assert all(len(piece) <= MAX_SYMBOLS for piece in split_text(text))


def test_nothing_is_lost():
    text = "Первое. Второе! Третье? Четвёртое… " * 30
    assert "".join(split_text(text)).replace(" ", "") == text.replace(" ", "")


def test_it_cuts_at_sentence_ends():
    text = ("А" * 200 + ". ") + ("Б" * 200 + ". ") + ("В" * 200 + ".")
    assert split_text(text, limit=250) == ["А" * 200 + ".", "Б" * 200 + ".", "В" * 200 + "."]


def test_one_sentence_longer_than_the_limit_is_still_split():
    # Mid-word is audible; a dropped paragraph is worse. Silero raises rather than
    # truncating, so an over-long sentence must not come back as one piece.
    text = " ".join(["слово"] * 400)
    pieces = split_text(text)
    assert len(pieces) > 1
    assert all(len(piece) <= MAX_SYMBOLS for piece in pieces)


def test_empty_text_is_no_pieces():
    assert split_text("   ") == []
