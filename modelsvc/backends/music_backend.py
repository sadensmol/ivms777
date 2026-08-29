"""Music sub-backend (design §5.1, §11.1).

Torch-free, a provider not a snapshot — see `voice_backend.py` for both reasons.
"""


class MusicBackend:
    def __init__(self, provider) -> None:
        self._provider = provider

    def music(
        self, prompt: str = "", seconds: float = 30.0, seed: int | None = None,
        sections: list[dict] | None = None, temperature: float | None = None,
        guidance: float | None = None,
    ) -> dict:
        # POSITIONAL: `TorchWorker.call` forwards `*args` only, so the child's signature
        # and this list are one contract.
        return self._provider().call(
            "music", prompt, seconds, seed, sections, temperature, guidance
        )
