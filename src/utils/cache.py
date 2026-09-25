"""Tiny JSON disk cache. Key -> one file. Survives restarts."""
import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class JsonCache:
    def __init__(self, root: Path, namespace: str):
        self.dir = Path(root) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: Any) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f)
        tmp.replace(self._path(key))   # atomic

    def __contains__(self, key: str) -> bool:
        return self._path(key).exists()
