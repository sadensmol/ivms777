"""The face child (`embedding/face_model.py`, design §4.1, §5.1 `/faces/detect`).

Fakes only: no torch, no onnx, no onnx2torch, no model load. The fake detector answers in
the format of the real SCRFD-10G graph (`detection/model.onnx` of `buffalo_l`): nine
outputs, the scores of strides 8, 16 and 32, then their box distances, then their five
landmarks, two anchors per grid cell, unbatched. The fake recogniser takes the aligned
112x112 faces and answers one raw (not normalised) 512-value vector per face.
"""

import io

import numpy as np
import pytest
from PIL import Image

from embedding.face_model import FaceModel

REVISION = "d09715916a0778919a770c343533641e250b8699"
STRIDES = (8, 16, 32)
ANCHORS = 2
RAW_VECTOR = [3.0, 4.0] + [0.0] * 510
LANDMARKS = ((-30, -20), (30, -20), (0, 10), (-25, 40), (25, 40))


class _Tensor(np.ndarray):
    """A numpy array that answers the torch calls a child makes on a tensor."""

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def float(self):
        return self.astype(np.float32)

    def contiguous(self):
        return self

    def permute(self, *dims):
        return self.transpose(*dims)

    def unsqueeze(self, dim):
        return np.expand_dims(self, dim)


def _tensor(array) -> _Tensor:
    return np.asarray(array).view(_Tensor)


def _face(center, side, score, stride=32) -> dict:
    return {"center": center, "side": side, "score": score, "stride": stride}


class _Detector:
    """SCRFD outputs for the given faces: each face sits on the anchor at its center."""

    def __init__(self, faces) -> None:
        self.faces = faces
        self.inputs: list[tuple[int, ...]] = []

    def __call__(self, x):
        shape = np.asarray(x).shape
        self.inputs.append(shape)
        height, width = shape[-2], shape[-1]
        scores, boxes, points = [], [], []
        for stride in STRIDES:
            cols = width // stride
            count = (height // stride) * cols * ANCHORS
            score = np.zeros((count, 1), np.float32)
            box = np.zeros((count, 4), np.float32)
            kps = np.zeros((count, 10), np.float32)
            for face in self.faces:
                if face["stride"] != stride:
                    continue
                cx, cy = face["center"]
                index = ((cy // stride) * cols + cx // stride) * ANCHORS
                score[index, 0] = face["score"]
                box[index] = face["side"] / 2 / stride
                kps[index] = [v / stride for point in LANDMARKS for v in point]
            scores.append(_tensor(score))
            boxes.append(_tensor(box))
            points.append(_tensor(kps))
        return [*scores, *boxes, *points]


class _Recognizer:
    def __init__(self) -> None:
        self.inputs: list[tuple[int, ...]] = []

    def __call__(self, batch):
        shape = np.asarray(batch).shape
        self.inputs.append(shape)
        return _tensor(np.tile(np.asarray(RAW_VECTOR, np.float32), (shape[0], 1)))


def _png(width: int = 640, height: int = 640) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "gray").save(buffer, format="PNG")
    return buffer.getvalue()


def _child(faces) -> tuple[FaceModel, _Detector, _Recognizer]:
    detector, recognizer = _Detector(faces), _Recognizer()
    return FaceModel.loaded(detector, recognizer, REVISION, tensor=_tensor), detector, recognizer


def test_detect_answers_the_box_in_0_1_the_score_and_the_normalised_vector():
    child, _, recognizer = _child([_face((320, 224), 160, 0.9)])

    result = child.detect(_png())

    assert result["revision"] == REVISION
    assert len(result["faces"]) == 1
    face = result["faces"][0]
    assert face["box"] == pytest.approx([240 / 640, 144 / 640, 160 / 640, 160 / 640], abs=0.01)
    assert face["score"] == pytest.approx(0.9, abs=0.001)
    assert len(face["vector"]) == 512
    assert face["vector"][:2] == pytest.approx([0.6, 0.8], abs=1e-4)
    assert float(np.linalg.norm(face["vector"])) == pytest.approx(1.0, abs=1e-4)
    assert recognizer.inputs and recognizer.inputs[0][-2:] == (112, 112)


def test_the_box_is_a_fraction_of_a_landscape_image_on_each_axis():
    child, _, _ = _child([_face((320, 160), 80, 0.95)])

    faces = child.detect(_png(640, 320))["faces"]

    assert len(faces) == 1
    assert faces[0]["box"] == pytest.approx([280 / 640, 120 / 320, 80 / 640, 80 / 320], abs=0.01)


def test_faces_under_the_score_floor_or_the_minimum_size_are_dropped():
    child, _, _ = _child([
        _face((320, 224), 160, 0.9),
        _face((128, 512), 160, 0.5),              # under the 0.7 floor
        _face((544, 544), 16, 0.95, stride=8),    # under the 32 px minimum side
    ])

    faces = child.detect(_png())["faces"]

    assert [round(face["score"], 2) for face in faces] == [0.9]


def test_an_image_the_child_cannot_read_is_refused_with_the_reason():
    child, detector, _ = _child([_face((320, 224), 160, 0.9)])

    result = child.detect(b"not an image")

    assert result.get("refused")
    assert result["revision"] == REVISION
    assert "faces" not in result
    assert detector.inputs == []


def test_the_child_refuses_a_cpu_device():
    refusal = None
    try:
        FaceModel("immich-app/buffalo_l", "cpu", REVISION)
    except RuntimeError as error:
        refusal = str(error)
    assert refusal is not None and "mps or cuda" in refusal

