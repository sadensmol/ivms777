"""Drop `[photo:ID]` citations the model was never given (design §10).

The answer prompt already forbids citing a photo that is not in the gathered
context, in capitals, with a worked example. `gemma4-E2B` ignores it: asked to
"find me all photos with dog" the agentic loop gathered exactly

    count: 1 photo(s) matching "dog"

— a COUNT, no photo blocks — and the model emitted `[photo:1]`, reading the
quantity as an id, then invented a caption for it. Photo 1 was an unrelated
portrait that retrieval never returned.

**A 2-billion-parameter model cannot be relied on to honour a negative
constraint, so this is enforced in code.** A citation is a claim about the user's
own library; showing them a photo the retrieval never surfaced is worse than
showing none, because it looks like evidence.

The filter runs over the STREAM, because the answer arrives a character at a time
(`[photo`, `:`, `1`, `7`, `7`, `]`) — checking only the finished text would let a
bad citation render before it could be removed. Text that might still become a
citation is held back until it is decided, so nothing bad is ever displayed.
"""

import re

_CITE = re.compile(r"\[photo:(\d+)\]")
# The longest text that could still turn into a citation: a prefix of "[photo:"
# followed by digits. Anything else is released immediately.
_PARTIAL = re.compile(r"\[(?:p(?:h(?:o(?:t(?:o(?::\d*)?)?)?)?)?)?$")


class CitationFilter:
    """Streaming filter: emits text with unknown `[photo:ID]` citations removed.

    `allowed` is the set of ids actually present in the context the model was
    given. Feed deltas in order, then `flush()` once the stream ends.
    """

    def __init__(self, allowed: set[int]) -> None:
        self._allowed = allowed
        self._held = ""
        self.dropped: list[int] = []

    def feed(self, delta: str) -> str:
        self._held += delta
        out = []
        while True:
            match = _CITE.search(self._held)
            if match:
                before = self._held[: match.start()]
                if _PARTIAL.search(before):
                    break  # an earlier '[' is still undecided; wait for more text
                out.append(before)
                if int(match.group(1)) in self._allowed:
                    out.append(match.group(0))
                else:
                    self.dropped.append(int(match.group(1)))
                self._held = self._held[match.end() :]
                continue
            partial = _PARTIAL.search(self._held)
            if partial:
                out.append(self._held[: partial.start()])
                self._held = self._held[partial.start() :]
            else:
                out.append(self._held)
                self._held = ""
            break
        return "".join(out)

    def flush(self) -> str:
        """Release whatever is still held — an unterminated `[photo:` is just text."""
        rest, self._held = self._held, ""
        return rest
