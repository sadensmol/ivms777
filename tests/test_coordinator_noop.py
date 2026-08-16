"""`models.coordinator.NoopCoordinator` (design §8.1, plan 15 task 7) — the
torch-free stub that replaced the cross-process `ModelCoordinator`. app/worker
hold no models anymore, so `require(workload)` must be a genuine nullcontext:
no raise, no side effect, and `captioner` stays `None`."""
import subprocess
import sys

from models.coordinator import NoopCoordinator


def test_require_is_a_working_nullcontext_and_captioner_is_none():
    coord = NoopCoordinator()
    assert coord.captioner is None
    with coord.require("CHAT"):
        pass  # must not raise


def test_make_coordinator_returns_a_noop_coordinator(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    from web.deps import build_context

    ctx = build_context(settings)
    coord = ctx.make_coordinator(client=None, holder="app")
    assert isinstance(coord, NoopCoordinator)


def test_app_worker_ingest_import_paths_never_load_torch():
    # R2 (plan 15 task 7): the coordinator's `from embedding.siglip import ...`
    # was the LAST torch import reachable from app/worker/ingest. After this
    # task nothing under those import paths pulls torch/transformers in.
    #
    # Run in a FRESH subprocess, not in-process: pytest collects other test
    # modules (e.g. tests/test_siglip_release.py, tests/test_residency.py) that
    # legitimately import `embedding.siglip`/torch for the `modelsvc` side of
    # the codebase, which would pollute `sys.modules` for the rest of this
    # process and make an in-process check pass or fail by test order alone.
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import ingest.pipeline, ingest.cli, web.deps\n"
            "import sys\n"
            "assert 'torch' not in sys.modules\n",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
