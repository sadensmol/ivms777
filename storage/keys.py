from pathlib import PurePosixPath

# Two levels of fan-out keep any one directory under a few thousand entries at
# 100k+ photos, which matters on filesystems that degrade with wide directories.
FANOUT = 2


def content_key(content_hash: str, suffix: str) -> str:
    """Storage key for an original, derived only from its bytes.

    The suffix is cosmetic — it makes the store browsable and lets a webserver
    guess a content type. Identity is the hash alone.
    """
    clean = suffix.lower()
    if not clean.startswith("."):
        clean = f".{clean}" if clean else ""
    return f"{content_hash[:FANOUT]}/{content_hash[FANOUT:FANOUT * 2]}/{content_hash}{clean}"


def suffix_of(rel_path: str) -> str:
    return PurePosixPath(rel_path).suffix.lower()
