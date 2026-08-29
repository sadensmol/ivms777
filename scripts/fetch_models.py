"""Pre-fetch the selected slots' model weights (design §4.1).

`make up` already fetches the GGUF (`make llama-mac`) with a progress bar. This does
the same for everything else, so a first run downloads visibly and UP FRONT instead of
stalling inside the first request that needs a model — where a missing or gated repo
looks like a 503 rather than a download. That silent path is exactly how a
licence-gated music model turned into "the film just has no music".

Deliberately NEVER fails the build. A gated repo the user has not accepted, or no
network at all, must still leave a startable app: that slot is simply silent, and the
reason is printed once, here, where it is readable.
"""

from __future__ import annotations

import argparse
import sys

from config import Settings
from models import catalog, slots
from models.catalog import HfSource, UrlSource


def wanted(settings: Settings, conn=None) -> list[tuple[str, str]]:
    """`(slot, what-to-fetch)` for every selected slot this script can pre-fetch.

    GGUF slots are excluded — `make llama-mac` owns those, and it already streams them
    with a progress bar.
    """
    pairs: list[tuple[str, str]] = []
    for slot, key in slots.resolve_keys(conn, settings).items():
        if slot in catalog.AUTOMATIC_SLOTS:
            # An AUTOMATIC slot is chosen at RUNTIME, not from the stored selection —
            # `voice` follows the film's language (§11.1). Fetching only the currently
            # selected one guarantees a missing file the moment a Russian film asks
            # for Silero, so every entry the slot can route to is pre-fetched.
            entries = catalog.entries_for(slot, settings.profile)
        else:
            entries = [catalog.get(slot, key)]
        for entry in entries:
            if isinstance(entry.source, HfSource):
                pairs.append((slot, entry.source.repo))
            elif isinstance(entry.source, UrlSource):
                pairs.append((slot, entry.source.url))
    return pairs


def _fetch(target: str, settings: Settings) -> None:
    if target.startswith(("http://", "https://")):
        _fetch_url(target, settings)
        return
    from huggingface_hub import snapshot_download

    snapshot_download(target)


def _fetch_url(url: str, settings: Settings) -> None:
    """A plain file on someone else's host (Silero). Written to the same
    `data_dir/models` directory the models service reads, via a `.part` rename so a
    killed download never leaves a truncated file that looks complete."""
    import httpx

    directory = settings.data_dir / "models"
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / url.rsplit("/", 1)[-1]
    if dest.exists():
        return
    part = dest.with_name(dest.name + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        with part.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    part.rename(dest)


def main(argv: list[str] | None = None, *, fetch=None) -> int:
    parser = argparse.ArgumentParser(description="Pre-fetch selected model weights.")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args(argv)
    settings = Settings(profile=args.profile) if args.profile else Settings()
    fetch = fetch or (lambda target: _fetch(target, settings))

    pairs = wanted(settings)
    print(f"  pre-fetching {len(pairs)} model(s) for profile {settings.profile}…")
    skipped: list[tuple[str, str]] = []
    for slot, target in pairs:
        print(f"  · {slot}: {target}")
        try:
            fetch(target)
        except Exception as exc:  # noqa: BLE001 - never fail the target
            skipped.append((target, str(exc).strip().splitlines()[0]))
    for target, reason in skipped:
        print(f"  ! skipped {target} — {reason}")
    if skipped:
        print("  the app still starts; those slots stay silent until the weights land.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
