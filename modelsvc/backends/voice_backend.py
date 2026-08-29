"""Voice sub-backend (design §5.1, §11.1).

Torch-free by construction, exactly like `SiglipBackend`: Kokoro lives in a
`TorchWorker` child, so this module imports nothing from `audio/` and the models
PARENT process never builds a CUDA context for it.

`provider`, not a snapshot: switching the `voice` slot builds a NEW child (§4.1), so
the live worker is asked for on every call.
"""


class VoiceBackend:
    def __init__(self, provider) -> None:
        self._provider = provider

    def speak(self, text: str, voice: str | None = None,
              segments: list[str] | None = None,
              voices: list[str | None] | None = None) -> dict:
        return self._provider().call("speak", text, voice, segments, voices)
