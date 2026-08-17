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
