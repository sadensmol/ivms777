import shutil
from collections.abc import Iterator
from pathlib import Path

from storage.base import StorageStat

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff"})


class LocalStorage:
    def __init__(self, root: Path, extensions: frozenset[str] | None = None) -> None:
        self.root = root.resolve()
        self.extensions = extensions

    def _resolve(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"key escapes storage root: {key!r}")
        return candidate

    def iter_keys(self) -> Iterator[str]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if self.extensions is not None and path.suffix.lower() not in self.extensions:
                continue
            yield path.relative_to(self.root).as_posix()

    def read(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def write(self, key: str, data: bytes) -> None:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def delete(self, key: str) -> None:
        """Remove a stored object. A no-op if it is already gone, so a partial
        prior delete still converges (used by folder deletion, §3.2c)."""
        self._resolve(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def stat(self, key: str) -> StorageStat:
        info = self._resolve(key).stat()
        return StorageStat(size=info.st_size, mtime=info.st_mtime)

    def local_path(self, key: str) -> Path | None:
        return self._resolve(key)

    def free_bytes(self) -> int:
        """Space left where originals land. Checked before accepting an upload."""
        self.root.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(self.root).free
