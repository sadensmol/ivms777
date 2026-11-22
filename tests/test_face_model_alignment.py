"""The face child's suppression and alignment (`embedding/face_model.py`, design §4.1,
AC2): overlapping detections of one face give one face, and the recogniser sees each face
aligned by its five landmarks onto the ArcFace reference points, not a plain crop.

Fakes only, in the format of SCRFD-10G (nine outputs: scores, box distances and landmarks
of strides 8, 16, 32, two anchors per cell). No torch, no onnx, no model load."""

import io
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from embedding.face_model import FaceModel

REVISION = "d09715916a0778919a770c343533641e250b8699"
STRIDES = (8, 16, 32)
ANCHORS = 2
EDGE = 640
STRIDE = 32
SIDE = 160
# The ArcFace reference points of a 112x112 face (InsightFace `arcface_dst`), written out
# here so the test does not read them from the code it checks.
REFERENCE = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)
LANDMARK_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255))
DOT_RADIUS = 5
COLOR_TOLERANCE = 40


class _Tensor(np.ndarray):
    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return np.asarray(self)


def _tensor(array) -> _Tensor:
    return np.asarray(array).view(_Tensor)


class _Detector:
    """Stride-32 detections: each `(cell_x, cell_y, anchor, score, landmarks)` puts a
    SIDE-px box centred on that anchor, with landmarks as px offsets from the centre."""

    def __init__(self, detections) -> None:
        self.detections = detections

    def __call__(self, x):
        scores, boxes, points = [], [], []
        for stride in STRIDES:
            cells = EDGE // stride
            count = cells * cells * ANCHORS
            score = np.zeros((count, 1), np.float32)
            box = np.zeros((count, 4), np.float32)
            kps = np.zeros((count, 10), np.float32)
            if stride == STRIDE:
                for cell_x, cell_y, anchor, value, landmarks in self.detections:
                    index = (cell_y * cells + cell_x) * ANCHORS + anchor
                    score[index, 0] = value
                    box[index] = SIDE / 2 / stride
                    kps[index] = [v / stride for point in landmarks for v in point]
            scores.append(_tensor(score))
            boxes.append(_tensor(box))
            points.append(_tensor(kps))
        return [*scores, *boxes, *points]


class _Recognizer:
    def __init__(self) -> None:
        self.batches: list[np.ndarray] = []

    def __call__(self, batch):
        self.batches.append(np.asarray(batch).copy())
        return _tensor(
            np.tile(np.asarray([1.0] + [0.0] * 511, np.float32), (np.asarray(batch).shape[0], 1))
        )


SQUARE_LANDMARKS = ((-30, -20), (30, -20), (0, 10), (-25, 40), (25, 40))


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _gray() -> bytes:
    return _png(Image.new("RGB", (EDGE, EDGE), "gray"))


def _faces(detections, image: bytes) -> tuple[list[dict], _Recognizer]:
    recognizer = _Recognizer()
    child = FaceModel.loaded(_Detector(detections), recognizer, REVISION, tensor=_tensor)
    return child.detect(image)["faces"], recognizer


def test_two_anchors_on_one_face_give_one_face_with_the_higher_score():
    faces, _ = _faces(
        [(10, 7, 0, 0.8, SQUARE_LANDMARKS), (10, 7, 1, 0.95, SQUARE_LANDMARKS)], _gray()
    )

    assert [round(face["score"], 2) for face in faces] == [0.95]


def test_a_detection_overlapping_a_stronger_one_past_the_limit_is_dropped():
    # Neighbour cells 32 px apart: the 160 px boxes share 128/192 of their union.
    faces, _ = _faces(
        [(10, 7, 0, 0.9, SQUARE_LANDMARKS), (11, 7, 0, 0.85, SQUARE_LANDMARKS)], _gray()
    )

    assert [round(face["score"], 2) for face in faces] == [0.9]


def test_two_detections_that_barely_overlap_are_two_faces():
    # Cells 128 px apart: the 160 px boxes share 32/288 of their union.
    faces, _ = _faces(
        [(10, 7, 0, 0.9, SQUARE_LANDMARKS), (14, 7, 0, 0.85, SQUARE_LANDMARKS)], _gray()
    )

    assert sorted(round(face["score"], 2) for face in faces) == [0.85, 0.9]


def test_the_recogniser_sees_the_landmarks_moved_onto_the_reference_points():
    # The five landmarks are the reference points tilted by 30 degrees and doubled in size,
    # so only an alignment by the landmarks brings each coloured dot to its reference point.
    center = (320.0, 224.0)
    angle = math.radians(30)
    middle = (56.0, 72.0)
    landmarks = []
    for x, y in REFERENCE:
        dx, dy = (x - middle[0]) * 2, (y - middle[1]) * 2
        landmarks.append(
            (
                dx * math.cos(angle) - dy * math.sin(angle),
                dx * math.sin(angle) + dy * math.cos(angle),
            )
        )
    image = Image.new("RGB", (EDGE, EDGE), "gray")
    draw = ImageDraw.Draw(image)
    for (dx, dy), color in zip(landmarks, LANDMARK_COLORS, strict=True):
        x, y = center[0] + dx, center[1] + dy
        draw.ellipse((x - DOT_RADIUS, y - DOT_RADIUS, x + DOT_RADIUS, y + DOT_RADIUS), fill=color)

    faces, recognizer = _faces([(10, 7, 0, 0.9, landmarks)], _png(image))

    assert len(faces) == 1
    face = recognizer.batches[0][0].transpose(1, 2, 0) * 127.5 + 127.5
    assert face.shape == (112, 112, 3)
    for (x, y), color in zip(REFERENCE, LANDMARK_COLORS, strict=True):
        assert face[round(y), round(x)] == pytest.approx(color, abs=COLOR_TOLERANCE)
