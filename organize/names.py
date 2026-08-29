"""Album title -> one safe path segment (docs/sync-cli.md, "Folder names").

The CLI creates these folders inside the user's own photo folder, so the rules
are the strictest common denominator: no separators, nothing Windows rejects,
nothing that ends in a dot or a space. Deterministic, so a re-run of the same
plan produces the same tree.
"""

MAX_SEGMENT = 120
_ILLEGAL = set('/\\<>:"|?*')


def folder_name(title: str, *, fallback: str) -> str:
    cleaned = "".join(
        "-" if ch in _ILLEGAL else (" " if ord(ch) < 32 else ch) for ch in title
    )
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.strip(" .")[:MAX_SEGMENT].strip(" .")
    # Return fallback if result is empty or contains only dashes/spaces (§12)
    return cleaned if cleaned and any(c not in "- " for c in cleaned) else fallback


def unique_folder_names(pairs: list[tuple[str, str]]) -> list[str]:
    """One folder per (title, fallback), de-duplicated in order.

    The comparison is case-insensitive because macOS and Windows filesystems are:
    two albums differing only in case would otherwise collide on disk.
    """
    taken: set[str] = set()
    names: list[str] = []
    for title, fallback in pairs:
        base = folder_name(title, fallback=fallback)
        name, n = base, 1
        while name.casefold() in taken:
            n += 1
            name = f"{base}_{n}"
        taken.add(name.casefold())
        names.append(name)
    return names
