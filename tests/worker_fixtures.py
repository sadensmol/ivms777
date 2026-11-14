"""Objects hosted inside a `TorchWorker` child in tests (importable by `spawn`)."""

import os


class Doubler:
    def __init__(self, offset: int = 0) -> None:
        self._offset = offset
        self._warmed = False

    def warm(self) -> None:
        self._warmed = True

    def double(self, values: list[int]) -> list[int]:
        return [v * 2 + self._offset for v in values]

    def warmed(self) -> bool:
        return self._warmed

    def pid(self) -> int:
        return os.getpid()

    def boom(self) -> None:
        raise ValueError("nope")

    def die(self) -> None:
        os._exit(1)  # simulates llama-server-style SIGABRT: no reply ever comes


class DiesWhileLoading:
    """A model whose load kills the process — a native crash, or a library calling
    `os._exit` — so no load result is ever sent."""

    def __init__(self, code: int = 3) -> None:
        import sys

        sys.stderr.write("fatal: spaCy model missing\n")
        sys.stderr.flush()
        os._exit(code)
