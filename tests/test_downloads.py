"""On-demand model downloads inside the `models` service (design §4.1, docs/models.md).

Fetchers are injected, so nothing here touches the network. `spawn` is injected too:
the tests run the "background" work themselves, which makes progress deterministic.
"""

import threading
from pathlib import Path

import pytest

from models import catalog
from modelsvc.downloads import Downloads, _blob_bytes, _hf_cache_dir, _hf_fetch

GGUF = catalog.get("planner", "gemma4-E2B")  # weights only
GGUF_VISION = catalog.get("caption", "gemma4-E2B")  # weights + projector
HF = catalog.get("image_embed", "siglip2-so400m-384")


class _Spawner:
    """Collects the "background" callables so a test can run them on demand."""

    def __init__(self):
        self.pending = []

    def __call__(self, fn):
        self.pending.append(fn)

    def run_all(self):
        while self.pending:
            self.pending.pop(0)()


def _downloads(tmp_path, *, http=None, hf=None, hf_present=None, hf_bytes=None, spawn=None):
    return Downloads(
        tmp_path,
        http_fetch=http or (lambda url, dest, progress: dest.write_bytes(b"x")),
        hf_fetch=hf or (lambda repo, progress: None),
        hf_present=hf_present or (lambda repo: False),
        # never probe the developer's REAL HF cache from a test
        hf_bytes=hf_bytes or (lambda repo: 0),
        spawn=spawn or (lambda fn: fn()),
    )


def test_a_gguf_already_on_disk_is_ready_without_fetching(tmp_path):
    # Presence is a PATH PROBE, not a flag: a manually placed GGUF counts.
    (tmp_path / GGUF.source.file).write_bytes(b"weights")
    calls = []
    d = _downloads(tmp_path, http=lambda *a, **kw: calls.append(a))
    assert d.status(GGUF)["state"] == "ready"
    d.start(GGUF)
    assert calls == []


def test_start_fetches_and_reports_progress_then_ready(tmp_path):
    seen = []

    def http(url, dest, progress):
        progress(0, 100)
        seen.append(url)
        progress(60, 100)
        dest.write_bytes(b"weights")
        progress(100, 100)

    spawn = _Spawner()
    d = _downloads(tmp_path, http=http, spawn=spawn)
    d.start(GGUF)
    assert d.status(GGUF)["state"] == "downloading"
    spawn.run_all()
    status = d.status(GGUF)
    assert status["state"] == "ready"
    assert (tmp_path / GGUF.source.file).exists()
    assert seen == [
        f"https://huggingface.co/{GGUF.source.repo}/resolve/main/{GGUF.source.file}"
    ]


def test_progress_is_readable_mid_download(tmp_path):
    progressed = {}

    def http(url, dest, progress):
        progress(25, 100)
        progressed["mid"] = d.status(GGUF)
        dest.write_bytes(b"w")

    spawn = _Spawner()
    d = _downloads(tmp_path, http=http, spawn=spawn)
    d.start(GGUF)
    spawn.run_all()
    assert progressed["mid"]["bytes"] == 25
    assert progressed["mid"]["total"] == 100
    assert progressed["mid"]["state"] == "downloading"


def test_a_vision_entry_fetches_weights_and_projector(tmp_path):
    urls = []

    def http(url, dest, progress):
        urls.append(url)
        dest.write_bytes(b"x")

    d = _downloads(tmp_path, http=http)
    d.start(GGUF_VISION)
    assert len(urls) == 2
    assert any(GGUF_VISION.source.mmproj_file in u for u in urls)
    assert d.status(GGUF_VISION)["state"] == "ready"


def test_a_vision_entry_with_only_the_weights_is_not_ready(tmp_path):
    (tmp_path / GGUF_VISION.source.file).write_bytes(b"weights")
    d = _downloads(tmp_path)
    assert d.status(GGUF_VISION)["state"] == "absent"


def test_an_absent_entry_reports_the_bytes_ALREADY_on_disk(tmp_path):
    # The caption and planner entries share one GGUF and differ only by the vision
    # projector, so the caption slot reads "absent" while the planner slot reads
    # "on disk" for the SAME displayed model. Without this the popup then offers
    # the full multi-GB download when only the projector is actually missing.
    (tmp_path / GGUF_VISION.source.file).write_bytes(b"w" * 700)
    d = _downloads(tmp_path)
    status = d.status(GGUF_VISION)
    assert status["state"] == "absent"
    assert status["bytes"] == 700


def test_a_failed_fetch_reports_the_error_and_can_be_retried(tmp_path):
    attempts = {"n": 0}

    def http(url, dest, progress):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("connection reset")
        dest.write_bytes(b"weights")

    d = _downloads(tmp_path, http=http)
    d.start(GGUF)
    status = d.status(GGUF)
    assert status["state"] == "error"
    assert "connection reset" in status["error"]

    d.start(GGUF)  # retry
    assert d.status(GGUF)["state"] == "ready"


def test_a_partial_file_is_never_mistaken_for_a_finished_one(tmp_path):
    # A killed download must not leave a truncated file the path probe calls ready.
    def http(url, dest, progress):
        (dest.parent / (dest.name + ".part")).write_bytes(b"half")
        raise OSError("killed")

    d = _downloads(tmp_path, http=http)
    d.start(GGUF)
    assert d.status(GGUF)["state"] == "error"
    assert not (tmp_path / GGUF.source.file).exists()


def test_starting_twice_while_in_flight_fetches_once(tmp_path):
    spawn = _Spawner()
    calls = []
    d = _downloads(
        tmp_path, http=lambda url, dest, progress: calls.append(url), spawn=spawn
    )
    d.start(GGUF)
    d.start(GGUF)
    assert len(spawn.pending) == 1
    spawn.run_all()
    assert len(calls) == 1


def test_hf_entries_probe_the_cache_and_snapshot_once(tmp_path):
    fetched = []
    d = _downloads(
        tmp_path, hf=lambda repo, progress: fetched.append(repo), hf_present=lambda r: False
    )
    assert d.status(HF)["state"] == "absent"
    d.start(HF)
    assert fetched == [HF.source.repo]


def test_an_hf_entry_already_in_the_cache_is_ready(tmp_path):
    d = _downloads(tmp_path, hf_present=lambda repo: repo == HF.source.repo)
    assert d.status(HF)["state"] == "ready"


def test_status_of_an_untouched_entry_has_the_full_shape(tmp_path):
    d = _downloads(tmp_path)
    assert d.status(GGUF) == {
        "state": "absent",
        "bytes": 0,
        "total": GGUF.size_mb * 1024 * 1024,
        "error": None,
    }


def test_two_slots_sharing_one_file_share_its_presence(tmp_path):
    # `gemma4-E2B` is the same GGUF under `planner` and `caption`; downloading the
    # planner's weights must not make the UI ask for them again.
    d = _downloads(tmp_path)
    d.start(GGUF)
    assert d.status(GGUF)["state"] == "ready"
    # the caption entry still needs its projector, but the weights are there
    assert d.status(GGUF_VISION)["state"] == "absent"
    d.start(GGUF_VISION)
    assert d.status(GGUF_VISION)["state"] == "ready"


# --- HF snapshot progress ----------------------------------------------------
# `snapshot_download` takes no progress callback, so the bytes are read off the
# cache directory it is filling. Without this the bar sits at 0% for a 4 GB
# download and the user cannot tell it from a stall.


def test_blob_bytes_counts_partial_blobs_and_ignores_the_snapshot_symlinks(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "abc").write_bytes(b"x" * 10)
    (blobs / "def.incomplete").write_bytes(b"y" * 5)  # the file being streamed
    snap = tmp_path / "snapshots" / "main"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blobs / "abc")  # would double-count
    assert _blob_bytes(tmp_path) == 15


def test_blob_bytes_of_a_cache_dir_that_does_not_exist_yet_is_zero(tmp_path):
    assert _blob_bytes(tmp_path / "nope") == 0


def test_the_cache_dir_is_resolved_against_the_installed_huggingface_hub():
    # Exercises the REAL resolver, not an injected one: it is the only part of the
    # progress path that depends on `huggingface_hub`'s layout, so a moved import
    # or a renamed cache folder must fail here rather than at download time.
    from huggingface_hub.constants import HF_HUB_CACHE

    directory = _hf_cache_dir("google/siglip2-so400m-patch16-512")
    assert directory.name == "models--google--siglip2-so400m-patch16-512"
    assert directory.parent == Path(HF_HUB_CACHE)


def test_hf_fetch_reports_bytes_while_the_snapshot_is_running(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    seen: list[tuple[int, int]] = []
    sampled = threading.Event()

    def snapshot(repo):
        (blobs / "part.incomplete").write_bytes(b"z" * 400)
        assert sampled.wait(5), "the watcher never sampled the cache dir"

    def progress(done, total):
        seen.append((done, total))
        if done == 400:
            sampled.set()

    _hf_fetch(
        "some/repo", progress,
        snapshot=snapshot, cache_dir=lambda repo: tmp_path,
        total_bytes=lambda repo: 1000, interval=0.01,
    )
    assert (400, 1000) in seen, seen  # mid-flight, not only at the end
    assert seen[-1] == (400, 1000)


def test_hf_fetch_still_finishes_when_the_size_lookup_fails(tmp_path):
    # An offline/gated metadata call must not fail the download — the bar just
    # falls back to the catalog's estimated total (`Downloads.status`).
    (tmp_path / "blobs").mkdir()

    def exploding_total(repo):
        raise OSError("no network")

    seen: list[tuple[int, int]] = []
    _hf_fetch(
        "some/repo", lambda done, total: seen.append((done, total)),
        snapshot=lambda repo: (tmp_path / "blobs" / "b").write_bytes(b"q" * 7),
        cache_dir=lambda repo: tmp_path, total_bytes=exploding_total, interval=0.01,
    )
    assert seen[-1] == (7, 7)


def test_unknown_source_type_is_a_programming_error(tmp_path):
    # A new source kind must be taught to the downloader explicitly, not silently
    # reported as "absent forever".
    import dataclasses

    d = _downloads(tmp_path)
    with pytest.raises(TypeError):
        d.start(dataclasses.replace(GGUF, source="s3://somewhere"))
