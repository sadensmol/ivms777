"""Fetching model weights on demand, inside the `models` service (design §4.1).

The service owns the model files, so it owns the downloads: `app` only asks for one
and polls the progress back through `GET /models/catalog`. A download holds **no**
scheduler slot and loads **no** model — it is bytes to disk, nothing more, so it
never blocks inference.

Two shapes:

- **GGUF** — a streaming HTTP GET into `data_dir/models`, the same directory
  `llama-server` is pointed at and `scripts/llama-server-entrypoint.sh` /
  `make llama-mac` populate. Written to `<name>.part` and renamed on success, so a
  killed download never leaves a truncated file that the presence probe would call
  "ready".
- **Hugging Face** — `snapshot_download` into the HF cache, the same place
  `transformers` looks. Imported lazily inside the fetcher: the `models` PARENT
  process must stay torch-free (§8.1), and `huggingface_hub` is cheap but there is
  no reason to import it before a download is asked for. It exposes no progress
  callback, so a watcher thread reads the growing `blobs/` directory instead —
  a multi-GB pull that shows 0% until it finishes looks exactly like a stall.

"Downloaded" is always a **path/cache probe**, never a flag — a manually placed
GGUF counts, and the state survives a service restart with no bookkeeping.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from models.catalog import GgufSource, HfSource, ModelEntry

Progress = Callable[[int, int], None]


def _hf_url(repo: str, file: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{file}"


def _http_fetch(url: str, dest: Path, progress: Progress) -> None:
    import httpx

    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        done = 0
        with part.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
                done += len(chunk)
                progress(done, total)
    part.rename(dest)  # atomic: the file appears only when it is complete


def _hf_snapshot(repo: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo)


def _hf_cache_dir(repo: str) -> Path:
    # `models--<org>--<name>` is the documented cache layout. Spelled out rather
    # than imported: `repo_folder_name` has moved between private modules across
    # huggingface_hub releases, and an ImportError here would break downloads.
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"


def _hf_total_bytes(repo: str) -> int:
    """The repo's real size, so the bar has a true denominator instead of the
    catalog's rounded estimate."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    return sum(sibling.size or 0 for sibling in (info.siblings or ()))


def _blob_bytes(cache_dir: Path) -> int:
    """Bytes fetched so far for one repo.

    Only `blobs/` is counted, and only real files: `snapshots/` holds SYMLINKS to
    the same blobs, so walking the whole tree would double every byte. The
    in-flight `<hash>.incomplete` file lives in `blobs/` too, which is what makes
    this readable mid-download.
    """
    blobs = cache_dir / "blobs"
    if not blobs.is_dir():
        return 0
    return sum(
        f.stat().st_size for f in blobs.iterdir() if f.is_file() and not f.is_symlink()
    )


def _hf_fetch(
    repo: str,
    progress: Progress,
    *,
    snapshot: Callable[[str], None] = _hf_snapshot,
    cache_dir: Callable[[str], Path] = _hf_cache_dir,
    total_bytes: Callable[[str], int] = _hf_total_bytes,
    interval: float = 1.0,
) -> None:
    """`snapshot_download` takes no progress callback and its internal tqdm classes
    are private, so progress is READ OFF THE CACHE DIRECTORY it is filling by a
    watcher thread. A 4 GB pull otherwise shows 0% until it finishes, which is
    indistinguishable from a stall (§4.1)."""
    directory = cache_dir(repo)
    try:
        total = total_bytes(repo)
    except Exception:  # noqa: BLE001 - a metadata miss must not fail the download
        total = 0
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(interval):
            progress(_blob_bytes(directory), total)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        snapshot(repo)
    finally:
        stop.set()
        watcher.join(timeout=5)
    # The final read is the authority: the watcher may have missed the last chunk,
    # and `total` is 0 when the metadata call failed.
    done = _blob_bytes(directory)
    progress(done, total or done)


def _hf_cache_bytes(repo: str) -> int:
    return _blob_bytes(_hf_cache_dir(repo))


def _hf_present(repo: str) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo, local_files_only=True)
    except Exception:  # noqa: BLE001 - any miss means "not cached", which is not an error
        return False
    return True


def _spawn(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, daemon=True).start()


class Downloads:
    def __init__(
        self,
        model_dir: Path,
        *,
        http_fetch: Callable[[str, Path, Progress], None] = _http_fetch,
        hf_fetch: Callable[[str, Progress], None] = _hf_fetch,
        hf_present: Callable[[str], bool] = _hf_present,
        hf_bytes: Callable[[str], int] = _hf_cache_bytes,
        spawn: Callable[[Callable[[], None]], None] = _spawn,
    ) -> None:
        self._dir = Path(model_dir)
        self._http_fetch = http_fetch
        self._hf_fetch = hf_fetch
        self._hf_present = hf_present
        self._hf_bytes = hf_bytes
        self._spawn = spawn
        self._lock = threading.Lock()
        # source id -> {"bytes", "total", "error", "running"}
        self._state: dict[str, dict] = {}

    # --- identity ------------------------------------------------------------
    @staticmethod
    def _source_id(entry: ModelEntry) -> str:
        source = entry.source
        if isinstance(source, GgufSource):
            return f"gguf:{source.repo}/{source.file}+{source.mmproj_file or ''}"
        if isinstance(source, HfSource):
            return f"hf:{source.repo}"
        raise TypeError(f"unknown source: {source!r}")

    def _files(self, entry: ModelEntry) -> list[tuple[str, Path]]:
        """`(url, destination)` for every file a GGUF entry needs."""
        source = entry.source
        assert isinstance(source, GgufSource)
        files = [(_hf_url(source.repo, source.file), self._dir / source.file)]
        if source.mmproj_file:
            repo = source.mmproj_repo or source.repo
            files.append((_hf_url(repo, source.mmproj_file), self._dir / source.mmproj_file))
        return files

    # --- state ---------------------------------------------------------------
    def is_present(self, entry: ModelEntry) -> bool:
        source = entry.source
        if isinstance(source, GgufSource):
            return all(dest.exists() for _, dest in self._files(entry))
        if isinstance(source, HfSource):
            return self._hf_present(source.repo)
        raise TypeError(f"unknown source: {source!r}")

    def present_bytes(self, entry: ModelEntry) -> int:
        """Bytes of this entry ALREADY on disk, complete or not.

        The `caption` and `planner` entries for one model share their weights file
        and differ only by the vision projector, so the same model reads "on disk"
        in one slot and "absent" in the other. Reporting what is already there lets
        the popup offer the REMAINDER instead of the whole multi-GB download (§13).
        """
        source = entry.source
        if isinstance(source, GgufSource):
            return sum(dest.stat().st_size for _, dest in self._files(entry) if dest.exists())
        if isinstance(source, HfSource):
            return self._hf_bytes(source.repo)
        raise TypeError(f"unknown source: {source!r}")

    def status(self, entry: ModelEntry) -> dict:
        total_guess = entry.size_mb * 1024 * 1024
        with self._lock:
            state = dict(self._state.get(self._source_id(entry), {}))
        if state.get("running"):
            return {
                "state": "downloading",
                "bytes": state.get("bytes", 0),
                "total": state.get("total") or total_guess,
                "error": None,
            }
        if self.is_present(entry):
            total = state.get("total") or total_guess
            return {"state": "ready", "bytes": total, "total": total, "error": None}
        if state.get("error"):
            return {
                "state": "error",
                "bytes": state.get("bytes", 0),
                "total": state.get("total") or total_guess,
                "error": state["error"],
            }
        return {
            "state": "absent",
            "bytes": self.present_bytes(entry),
            "total": total_guess,
            "error": None,
        }

    # --- fetching ------------------------------------------------------------
    def start(self, entry: ModelEntry) -> None:
        """Idempotent: already on disk, or already in flight, is a no-op."""
        source_id = self._source_id(entry)  # TypeError on an unknown source
        if self.is_present(entry):
            return
        with self._lock:
            if self._state.get(source_id, {}).get("running"):
                return
            self._state[source_id] = {"bytes": 0, "total": 0, "error": None, "running": True}
        self._spawn(lambda: self._run(entry, source_id))

    def _run(self, entry: ModelEntry, source_id: str) -> None:
        def progress(done: int, total: int) -> None:
            with self._lock:
                state = self._state.setdefault(source_id, {})
                state["bytes"] = done
                if total:
                    state["total"] = total

        try:
            source = entry.source
            if isinstance(source, GgufSource):
                for url, dest in self._files(entry):
                    if dest.exists():
                        continue
                    self._http_fetch(url, dest, progress)
            else:
                self._hf_fetch(source.repo, progress)
            error = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            state = self._state.setdefault(source_id, {})
            state["running"] = False
            state["error"] = error
