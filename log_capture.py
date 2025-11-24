"""Lightweight log/print capture with in-memory ring buffer."""

from __future__ import annotations

import builtins
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List


@dataclass
class LogEntry:
    seq: int
    ts: float
    text: str


class LogCapture:
    def __init__(self, max_entries: int = 2000) -> None:
        self.max_entries = max_entries
        self._buffer: Deque[LogEntry] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._seq = 0
        self._enabled = False
        self._orig_print = builtins.print

    def enable(self) -> None:
        with self._lock:
            if self._enabled:
                return
            self._enabled = True
            builtins.print = self._patched_print  # type: ignore[assignment]

    def disable(self) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._enabled = False
            builtins.print = self._orig_print  # type: ignore[assignment]

    def status(self) -> bool:
        with self._lock:
            return self._enabled

    def _append(self, text: str) -> None:
        with self._lock:
            self._seq += 1
            self._buffer.append(LogEntry(seq=self._seq, ts=time.time(), text=text))

    def _patched_print(self, *args, **kwargs):  # type: ignore[override]
        text = " ".join(str(a) for a in args)
        if kwargs.get("sep") is not None or kwargs.get("end") not in (None, "\n"):
            # Fallback to original formatting to avoid drift
            try:
                formatted = self._orig_print(*args, **kwargs)
            finally:
                pass
        else:
            formatted = None
        try:
            self._append(text)
        except Exception:
            pass
        # Always call original print
        return self._orig_print(*args, **kwargs) if formatted is None else formatted

    def get_logs(self, after: int = 0, limit: int = 200) -> List[LogEntry]:
        with self._lock:
            data = [e for e in self._buffer if e.seq > after]
            if limit:
                data = data[-limit:]
            return list(data)


CAPTURE = LogCapture()


def enable_capture():
    CAPTURE.enable()


def disable_capture():
    CAPTURE.disable()


def capture_status() -> bool:
    return CAPTURE.status()


def get_logs(after: int = 0, limit: int = 200) -> List[LogEntry]:
    return CAPTURE.get_logs(after=after, limit=limit)
