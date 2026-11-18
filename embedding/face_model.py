"""The face model in a `TorchWorker` child (design §4.1, §5.1 `/faces/detect`, §8.1
`faces`).

Models-service-only. It turns ONE upright image into its faces: a box in 0..1 of the
image, the detection score and an L2-normalised vector per face. InsightFace
`buffalo_l` (SCRFD-10G detection, ArcFace R50 recognition), pinned in
`models/catalog.py:FACE_MODELS`.

Module scope imports no torch, no onnx and no onnx2torch, exactly like
`audio/speaker_embed.py`: importing this module must not build a torch context in the
importing process (CLAUDE.md § "One model process").

Licence: the InsightFace weights are non-commercial, personal use only.
"""

from collections.abc import Callable
from contextlib import nullcontext


class FaceModel:
    def __init__(self, repo: str, device: str, revision: str) -> None:
        self._setup(None, None, revision, None, nullcontext)
        self._repo = repo
        self._device = device

    @classmethod
    def loaded(cls, detector: Callable, recognizer: Callable, revision: str, *,
               tensor: Callable) -> "FaceModel":
        """A child around an already built detector and recogniser — the test seam.

        `detector` takes the detector's input tensor and returns its raw outputs;
        `recognizer` takes a batch of aligned 112×112 faces and returns their vectors;
        `tensor` replaces `torch.from_numpy`. No inference mode is used, so a test drives
        `detect` with fakes and never imports torch.
        """
        model = cls.__new__(cls)
        model._setup(detector, recognizer, revision, tensor, nullcontext)
        return model

    def _setup(self, detector: Callable | None, recognizer: Callable | None, revision: str,
               tensor: Callable | None, inference: Callable) -> None:
        self._detector = detector
        self._recognizer = recognizer
        self.revision = revision
        self._tensor = tensor
        self._inference = inference

    def detect(self, image: bytes) -> dict:
        """The faces of one upright image: `{faces: [{box, score, vector}], revision,
        elapsed_s}`."""
        return {"faces": [], "revision": self.revision, "elapsed_s": 0.0}
