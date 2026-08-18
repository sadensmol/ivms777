"""`RemoteEmbedder` — the `embedding.base.Embedder` shim over `ModelsClient`
(design §5.1, plan 15 task 2). Round-tripped against `create_models_app`
+ `FakeBackend`, exactly like `tests/test_models_client.py`, so it never
touches a real model.

Also guards the hard rule (CLAUDE.md § "One model process"): `config` and
`inference.remote_embedder` must stay torch-free. That can't be checked by
inspecting `sys.modules` in-process — other test modules in the suite (e.g.
`tests/test_siglip_release.py`) import `embedding.siglip` (torch) at
collection time, so torch is already resident by the time any test body
runs. A fresh subprocess is the only reliable way to prove these two modules
don't pull it in themselves.
"""

import subprocess
import sys
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from inference.models_client import ModelsClient
from inference.remote_embedder import RemoteEmbedder
from modelsvc.app import create_models_app
from modelsvc.backends.fake import FakeBackend

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _client(profile: str = "mac") -> ModelsClient:
    app = create_models_app(FakeBackend(profile=profile))
    transport = TestClient(app)._transport
    return ModelsClient("http://modelsvc", transport=transport)


def test_embed_images_round_trips_through_the_fake_backend():
    embedder = RemoteEmbedder(_client(), "fake")
    vectors = embedder.embed_images([Image.new("RGB", (8, 8))])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0


def _captured_payload(images: list[Image.Image], client: ModelsClient | None = None) -> list[bytes]:
    """Return the PNG bytes `embed_images` actually puts on the wire."""
    client = client or _client()
    sent: list[bytes] = []

    def capture(payload, **kwargs):
        sent.extend(payload)
        return [[0.0]] * len(payload)

    client.embed_image = capture
    RemoteEmbedder(client, "fake").embed_images(images)
    return sent


def test_images_are_downscaled_to_the_models_384px_input_before_encoding():
    # SigLIP 2 so400m/patch14-384 squashes every input to 384x384 (its own
    # preprocessor_config.json: do_resize, size 384x384, resample BILINEAR), so
    # sending the full-resolution original is pure waste: on the Jetson the PNG
    # encode alone cost 5.7 s and a 15 MB body per photo, against ~10 ms of GPU
    # (§8.1). Verified out-of-band against the real processor: pre-resizing with
    # the same filter leaves `pixel_values` bit-identical, so the vectors are
    # unchanged.
    sent = _captured_payload([Image.new("RGB", (3024, 4032))])
    assert len(sent) == 1
    with Image.open(BytesIO(sent[0])) as encoded:
        assert encoded.size == (384, 384)


def test_downscaling_is_applied_to_every_image_in_a_batch():
    sent = _captured_payload(
        [Image.new("RGB", (4032, 3024)), Image.new("RGB", (800, 533))]
    )
    assert len(sent) == 2
    for png in sent:
        with Image.open(BytesIO(png)) as encoded:
            assert encoded.size == (384, 384)


def test_embed_texts_round_trips_through_the_fake_backend():
    embedder = RemoteEmbedder(_client(), "fake")
    vectors = embedder.embed_texts(["cat"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0


def test_calibration_matches_fake_embedder_values():
    embedder = RemoteEmbedder(_client(), "fake")
    assert embedder.logit_scale == 10.0
    assert embedder.logit_bias == -5.0


def test_the_spec_is_fetched_once_and_cached():
    client = _client()
    calls = []
    original = client.embed_spec

    def counting_spec(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    client.embed_spec = counting_spec
    embedder = RemoteEmbedder(client, "fake")
    assert embedder.logit_scale == 10.0
    assert embedder.logit_bias == -5.0  # second attr access, still one HTTP call
    embedder.embed_images([Image.new("RGB", (8, 8))])  # and the resize reuses it
    assert len(calls) == 1


def test_the_resize_follows_the_selected_model_not_a_constant():
    # Switching to the 512px checkpoint must change what goes on the wire, with no
    # code change and no constant to edit (design §4.1).
    client = _client()
    client.set_slots({"image_embed": "siglip2-so400m-512"})
    sent = _captured_payload([Image.new("RGB", (3024, 4032))], client)
    with Image.open(BytesIO(sent[0])) as encoded:
        assert encoded.size == (512, 512)


def test_native_mode_keeps_the_aspect_ratio():
    # NaFlex / native-resolution encoders consume the real aspect ratio; squashing
    # one of those would throw away signal (design §4.1).
    client = _client()
    client.embed_spec = lambda **kw: {
        "logit_scale": 1.0,
        "logit_bias": 0.0,
        "preprocess": {"input_px": 512, "resample": "bicubic", "mode": "native"},
        "generation": 0,
    }
    sent = _captured_payload([Image.new("RGB", (3000, 2000))], client)
    with Image.open(BytesIO(sent[0])) as encoded:
        assert encoded.size == (512, 341)


def test_refresh_drops_the_cached_spec():
    client = _client()
    embedder = RemoteEmbedder(client, "fake")
    assert embedder.generation == 0
    client.set_slots({"image_embed": "siglip2-so400m-512"})
    assert embedder.generation == 0  # cached: a stale spec until told otherwise
    embedder.refresh()
    assert embedder.generation == 1


def _assert_module_import_is_torch_free(module: str) -> None:
    code = (
        f"import {module}; import sys; "
        "assert 'torch' not in sys.modules, 'torch imported'; "
        "assert 'embedding.siglip' not in sys.modules, 'embedding.siglip imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_importing_config_does_not_import_torch():
    _assert_module_import_is_torch_free("config")


def test_importing_remote_embedder_does_not_import_torch():
    _assert_module_import_is_torch_free("inference.remote_embedder")


def test_importing_modelsvc_backends_does_not_import_torch():
    # build_backend pulls in SiglipBackend/CompositeBackend transitively; those
    # must stay lazy about embedding.siglip so importing the package alone
    # (never calling the real SigLIP path) still doesn't load torch.
    _assert_module_import_is_torch_free("modelsvc.backends")
