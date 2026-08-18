"""`CitationFilter` — a fabricated `[photo:ID]` must never reach the user (§10)."""

from chat.citations import CitationFilter


def _stream(text: str, allowed: set[int], *, chunk: int = 1) -> str:
    """Feed `text` through the filter in `chunk`-sized pieces, as SSE does."""
    f = CitationFilter(allowed)
    out = "".join(f.feed(text[i : i + chunk]) for i in range(0, len(text), chunk))
    return out + f.flush()


def test_a_citation_the_model_was_given_passes_through():
    assert _stream("Here it is [photo:177].", {177}) == "Here it is [photo:177]."


def test_a_fabricated_citation_is_removed():
    # The real failure: gathered data was 'count: 1 photo(s) matching "dog"' — a
    # QUANTITY, no photo blocks — and the model emitted [photo:1].
    assert _stream("Here it is [photo:1].", set()) == "Here it is ."


def test_it_works_when_the_citation_arrives_one_character_at_a_time():
    # llama-server streams '[photo', ':', '1', '7', '7', ']' as separate deltas, so
    # checking only the finished answer would let a bad citation render first.
    assert _stream("a [photo:1] b [photo:9] c", {9}, chunk=1) == "a  b [photo:9] c"


def test_it_works_for_any_chunking():
    text = "x [photo:5] y [photo:6] z"
    for chunk in (1, 2, 3, 5, 8, 100):
        assert _stream(text, {5}, chunk=chunk) == "x [photo:5] y  z"


def test_dropped_ids_are_reported():
    f = CitationFilter({7})
    out = f.feed("a [photo:1] b [photo:7]") + f.flush()
    assert out == "a  b [photo:7]"
    assert f.dropped == [1]


def test_bracket_text_that_is_not_a_citation_is_untouched():
    assert _stream("a [note] and [photo] and [x:1]", set()) == "a [note] and [photo] and [x:1]"


def test_an_unterminated_citation_is_released_on_flush():
    # Never swallow real text: a stream that ends mid-token must still show it.
    assert _stream("trailing [photo:12", {12}) == "trailing [photo:12"


def test_two_citations_back_to_back():
    assert _stream("[photo:1][photo:2]", {2}) == "[photo:2]"
