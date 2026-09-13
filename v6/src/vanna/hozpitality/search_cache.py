"""Small bounded per-process cache for fast repeated searches."""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Optional


class TTLSearchCache:
    def __init__(self, max_items: int = 512, ttl_seconds: int = 45):
        self.max_items = max(16, max_items)
        self.ttl_seconds = max(1, ttl_seconds)
        self._data: OrderedDict[str, tuple[float, object]] = OrderedDict()

    @staticmethod
    def key(*parts: str) -> str:
        raw = "\x1f".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str):
        item = self._data.get(key)
        if not item:
            return None
        expires, value = item
        if expires <= time.monotonic():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: object) -> None:
        self._data[key] = (time.monotonic() + self.ttl_seconds, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()
