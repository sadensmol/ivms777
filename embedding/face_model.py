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

import io
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from models.catalog import FACE_MODELS, ModelEntry

STRIDES = (8, 16, 32)
ANCHORS_PER_CELL = 2
NMS_IOU = 0.4
DETECT_MEAN, DETECT_STD = 127.5, 128.0
RECOGNIZE_MEAN, RECOGNIZE_STD = 127.5, 127.5
ALIGNED_EDGE = 112
# The five ArcFace reference points (eyes, nose, mouth corners) of a 112x112 face, from
# InsightFace `face_align.arcface_dst`: the recogniser was trained on faces aligned to them.
ARCFACE_POINTS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                           [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def _entry(revision: str) -> ModelEntry:
    for entry in FACE_MODELS:
        if entry.source.revision == revision:
            return entry
    raise RuntimeError(f"no face model of revision {revision!r} is declared in the catalog")


class FaceModel:
    def __init__(self, repo: str, device: str, revision: str) -> None:
        # GPU only (`docs/design.md` §5): there is no CPU fallback.
        if device != "mps" and not str(device).startswith("cuda"):
            raise RuntimeError(f"the face model runs on mps or cuda only, not {device!r}")

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        # The PINNED snapshot on disk only: a child never downloads inside a test.
        try:
            path = snapshot_download(repo, revision=revision, local_files_only=True)
        except LocalEntryNotFoundError as exc:
            raise RuntimeError(
                f"face weights for {revision} are not on disk: run `make models`"
            ) from exc
        # The snapshot folder is the commit; anything else is another revision.
        if Path(path).name != revision:
            raise RuntimeError(
                f"face snapshot {Path(path).name!r} is not the pinned revision {revision!r}"
            )

        import torch
        from onnx2torch import convert

        detector = convert(str(Path(path) / "detection" / "model.onnx")).to(device).eval()
        recognizer = convert(str(Path(path) / "recognition" / "model.onnx")).to(device).eval()
        self._setup(detector, recognizer, revision,
                    lambda array: torch.from_numpy(array).to(device), torch.inference_mode)

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

    def _setup(self, detector: Callable, recognizer: Callable, revision: str,
               tensor: Callable, inference: Callable) -> None:
        self._detector = detector
        self._recognizer = recognizer
        self.revision = revision
        self._entry = _entry(revision)
        self._tensor = tensor
        self._inference = inference

    def detect(self, image: bytes) -> dict:
        """The faces of one upright image: `{faces: [{box, score, vector}], revision,
        elapsed_s}`, or `{refused, revision}` with the reason for an image it cannot read.
        A refusal is an answer, not an error: a child's exception reaches the parent only
        as text."""
        started = time.monotonic()
        try:
            picture = Image.open(io.BytesIO(image)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            return {"refused": str(exc), "revision": self.revision}
        pixels = np.asarray(picture)
        boxes, scores, landmarks = self._find(picture)
        vectors = self._describe(pixels, landmarks) if len(boxes) else np.zeros((0, 0))
        height, width = pixels.shape[:2]
        faces = []
        for box, score, vector in zip(boxes, scores, vectors, strict=True):
            x1, x2 = np.clip(box[[0, 2]], 0, width)
            y1, y2 = np.clip(box[[1, 3]], 0, height)
            faces.append({"box": [float(x1 / width), float(y1 / height),
                                  float((x2 - x1) / width), float((y2 - y1) / height)],
                          "score": float(score), "vector": vector.tolist()})
        return {"faces": faces, "revision": self.revision,
                "elapsed_s": round(time.monotonic() - started, 3)}

    def _find(self, picture: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SCRFD on the picture letterboxed into a square of the declared input edge:
        boxes `[x1, y1, x2, y2]` and five landmarks per face in px of the picture, kept
        over the score floor and the minimum size, after non-maximum suppression."""
        edge = self._entry.input_edge
        scale = edge / max(picture.width, picture.height)
        resized = picture.resize((max(1, int(picture.width * scale)),
                                  max(1, int(picture.height * scale))), Image.BILINEAR)
        canvas = np.zeros((edge, edge, 3), dtype=np.float32)
        canvas[:resized.height, :resized.width] = np.asarray(resized, dtype=np.float32)
        blob = ((canvas - DETECT_MEAN) / DETECT_STD).transpose(2, 0, 1)[np.newaxis]
        with self._inference():
            outputs = [np.asarray(output.detach().cpu().numpy())
                       for output in self._detector(self._tensor(np.ascontiguousarray(blob)))]
        outputs = [output[0] if output.ndim == 3 else output for output in outputs]

        floor = self._entry.score_floor
        all_scores, all_boxes, all_points = [], [], []
        for index, stride in enumerate(STRIDES):
            scores = outputs[index].reshape(-1)
            distances = outputs[index + len(STRIDES)].reshape(-1, 4) * stride
            points = outputs[index + 2 * len(STRIDES)].reshape(-1, 10) * stride
            cells = edge // stride
            grid = np.stack(np.mgrid[:cells, :cells][::-1], axis=-1).reshape(-1, 2) * stride
            centers = np.repeat(grid, ANCHORS_PER_CELL, axis=0).astype(np.float32)
            keep = scores >= floor
            centers = centers[keep]
            all_scores.append(scores[keep])
            all_boxes.append(np.concatenate([centers - distances[keep][:, :2],
                                             centers + distances[keep][:, 2:]], axis=1))
            all_points.append(points[keep].reshape(-1, 5, 2) + centers[:, np.newaxis, :])
        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes) / scale
        points = np.concatenate(all_points) / scale

        order = _suppress(boxes, scores)
        boxes, scores, points = boxes[order], scores[order], points[order]
        sides = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        big = sides >= self._entry.min_face_px
        return boxes[big], scores[big], points[big]

    def _describe(self, pixels: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """ArcFace vectors, L2-normalised, of the faces aligned to 112x112 by their five
        landmarks."""
        picture = Image.fromarray(pixels)
        aligned = np.stack([_align(picture, points) for points in landmarks])
        batch = ((aligned - RECOGNIZE_MEAN) / RECOGNIZE_STD).transpose(0, 3, 1, 2)
        with self._inference():
            raw = self._recognizer(self._tensor(np.ascontiguousarray(batch, dtype=np.float32)))
        vectors = np.asarray(raw.detach().cpu().numpy(), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or not (norms > 0).all():
            raise RuntimeError("a face vector is not finite or has a zero norm")
        return (vectors / norms).astype(np.float32)


def _suppress(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    """The indexes kept by greedy non-maximum suppression, highest score first."""
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    pending = list(np.argsort(-scores))
    kept: list[int] = []
    while pending:
        best = pending.pop(0)
        kept.append(int(best))
        rest = np.array(pending, dtype=int)
        if not len(rest):
            break
        width = np.clip(np.minimum(boxes[best, 2], boxes[rest, 2])
                        - np.maximum(boxes[best, 0], boxes[rest, 0]), 0, None)
        height = np.clip(np.minimum(boxes[best, 3], boxes[rest, 3])
                         - np.maximum(boxes[best, 1], boxes[rest, 1]), 0, None)
        overlap = width * height / (areas[best] + areas[rest] - width * height)
        pending = [int(index) for index in rest[overlap <= NMS_IOU]]
    return kept


def _align(picture: Image.Image, points: np.ndarray) -> np.ndarray:
    """The face cut to 112x112 by the similarity transform that maps its five landmarks
    closest onto `ARCFACE_POINTS` (Umeyama, least squares)."""
    source = points.astype(np.float64)
    source_mean, target_mean = source.mean(axis=0), ARCFACE_POINTS.mean(axis=0)
    source_centered, target_centered = source - source_mean, ARCFACE_POINTS - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt
    variance = (source_centered ** 2).sum() / len(source)
    factor = (singular * np.diag(sign)).sum() / variance
    forward = np.eye(3)
    forward[:2, :2] = factor * rotation
    forward[:2, 2] = target_mean - factor * rotation @ source_mean
    # PIL maps each OUTPUT pixel back to the input, so it takes the inverse transform.
    inverse = np.linalg.inv(forward)[:2].reshape(-1)
    face = picture.transform((ALIGNED_EDGE, ALIGNED_EDGE), Image.AFFINE, tuple(inverse),
                             resample=Image.BILINEAR)
    return np.asarray(face, dtype=np.float32)
