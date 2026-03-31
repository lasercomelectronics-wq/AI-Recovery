from __future__ import annotations


class DedupFilter:
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_duplicate(self, digest: str) -> bool:
        return digest in self._seen

    def add_hash(self, digest: str) -> None:
        self._seen.add(digest)
