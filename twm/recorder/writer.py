"""EpisodeWriter — the only thread that touches HDF5 during recording.

Design
  * The queue is bounded in BYTES (queue + in-flight batch), sized from
    seconds-of-ticks, so memory is predictable (3 s ≈ 750 MB, not 2.5 GB).
  * It never drops. `submit` raises WriterOverloaded when the next tick
    would not fit; `check` reports sustained occupancy above a threshold
    before that happens, plus low disk and write faults. The capture loop
    turns any of those into a fail-fast end of the episode.
  * Ticks are written in batches (one resize per dataset) and the file is
    H5Fflush-ed every `flush_interval_s` so a crash loses seconds, not the
    episode (pushT/2026-06-18/episode_004.h5, 79 GB, was lost that way).
"""
from __future__ import annotations

import collections
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional, Sequence, Tuple

from twm.recorder.frames import Tick
from twm.recorder.schema import append_ticks

log = logging.getLogger("twm.recorder")


class WriterOverloaded(RuntimeError):
    """The next tick does not fit in the queue. Nothing was dropped."""


class WriterFault(RuntimeError):
    """A write failed (disk full, I/O error) or the writer is stopped."""


@dataclass(frozen=True)
class WriterStats:
    queue_items: int
    queue_bytes: int
    capacity_bytes: int
    peak_fraction: float
    bytes_written: int
    write_seconds: float
    last_batch_ms: float
    flushes: int
    file_mb_s: float
    disk_free_gb: Optional[float]
    overloaded_since: Optional[float]
    fault: Optional[str]

    @property
    def fraction(self) -> float:
        return self.queue_bytes / self.capacity_bytes if self.capacity_bytes else 0.0

    @property
    def mean_mb_s(self) -> float:
        """Raw (uncompressed) throughput of the sink."""
        return self.bytes_written / self.write_seconds / 1e6 if self.write_seconds else 0.0


def queue_capacity_bytes(queue_seconds: float, fps: float, tick_nbytes: int) -> int:
    return int(queue_seconds * fps * tick_nbytes)


class EpisodeWriter:
    def __init__(self, capacity_bytes: int, batch_size: int = 10,
                 flush_interval_s: float = 10.0, overload_fraction: float = 0.5,
                 overload_sustained_s: float = 3.0,
                 min_free_gb: Optional[float] = None,
                 sink: Callable[[Any, Sequence[Tick]], None] = append_ticks,
                 clock: Callable[[], float] = time.monotonic,
                 disk_usage: Callable[[str], Any] = shutil.disk_usage):
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = int(capacity_bytes)
        self.batch_size = max(1, int(batch_size))
        self.flush_interval_s = flush_interval_s
        self.overload_fraction = overload_fraction
        self.overload_sustained_s = overload_sustained_s
        self.min_free_gb = min_free_gb
        self._sink = sink
        self._clock = clock
        self._disk_usage = disk_usage

        self._cv = threading.Condition()
        self._items: Deque[Tuple[Any, Tick]] = collections.deque()
        self._queue_bytes = 0          # queued + in-flight
        self._in_flight = 0
        self._peak_fraction = 0.0
        self._bytes_written = 0
        self._write_seconds = 0.0
        self._last_batch_ms = 0.0
        self._flushes = 0
        self._last_flush_t = clock()
        self._last_file_bytes = 0
        self._file_mb_s = 0.0
        self._disk_free_gb: Optional[float] = None
        self._overloaded_since: Optional[float] = None
        self._fault: Optional[str] = None
        self._stop_requested = False
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="EpisodeWriter",
                                        daemon=True)
        self._thread.start()

    # ── producer side ────────────────────────────────────────────────────────
    def submit(self, f, tick: Tick) -> None:
        """Enqueue one tick or raise. Never blocks, never drops."""
        nbytes = tick.nbytes()
        with self._cv:
            if self._fault is not None:
                raise WriterFault(self._fault)
            if self._stop_requested:
                raise WriterFault("writer stopped")
            if self._queue_bytes + nbytes > self.capacity_bytes:
                raise WriterOverloaded(
                    f"writer queue full ({len(self._items) + self._in_flight} ticks, "
                    f"{self._queue_bytes / 1e6:.0f} of {self.capacity_bytes / 1e6:.0f} MB)")
            self._items.append((f, tick))
            self._queue_bytes += nbytes
            self._peak_fraction = max(self._peak_fraction,
                                      self._queue_bytes / self.capacity_bytes)
            self._cv.notify_all()

    def check(self) -> Optional[Tuple[str, str]]:
        """Return (kind, detail) if the episode must end now, else None.

        kinds: "writer_fault", "disk_low", "overload".
        """
        with self._cv:
            if self._fault is not None:
                return "writer_fault", self._fault
            if (self.min_free_gb is not None and self._disk_free_gb is not None
                    and self._disk_free_gb < self.min_free_gb):
                return "disk_low", (f"{self._disk_free_gb:.1f} GB free on the "
                                    f"recording disk (< {self.min_free_gb:g} GB)")
            fraction = self._queue_bytes / self.capacity_bytes
            now = self._clock()
            if fraction > self.overload_fraction:
                if self._overloaded_since is None:
                    self._overloaded_since = now
                elif now - self._overloaded_since >= self.overload_sustained_s:
                    return "overload", (
                        f"writer queue above {self.overload_fraction:.0%} for "
                        f"{now - self._overloaded_since:.1f}s ({fraction:.0%} now)")
            else:
                self._overloaded_since = None
        return None

    def stats(self) -> WriterStats:
        with self._cv:
            return WriterStats(
                queue_items=len(self._items) + self._in_flight,
                queue_bytes=self._queue_bytes,
                capacity_bytes=self.capacity_bytes,
                peak_fraction=self._peak_fraction,
                bytes_written=self._bytes_written,
                write_seconds=self._write_seconds,
                last_batch_ms=self._last_batch_ms,
                flushes=self._flushes,
                file_mb_s=self._file_mb_s,
                disk_free_gb=self._disk_free_gb,
                overloaded_since=self._overloaded_since,
                fault=self._fault,
            )

    def reset_episode_stats(self) -> None:
        with self._cv:
            self._peak_fraction = 0.0
            self._bytes_written = 0
            self._write_seconds = 0.0
            self._last_batch_ms = 0.0
            self._overloaded_since = None
            self._last_file_bytes = 0
            self._file_mb_s = 0.0

    def drain(self, timeout: Optional[float] = None) -> None:
        """Block until every submitted tick is written. Raises WriterFault."""
        with self._cv:
            ok = self._cv.wait_for(
                lambda: (self._fault is not None
                         or (not self._items and self._in_flight == 0)),
                timeout=timeout)
            if self._fault is not None:
                raise WriterFault(self._fault)
            if not ok:
                raise WriterFault(f"drain timed out after {timeout}s with "
                                  f"{len(self._items)} ticks queued")

    def stop(self) -> None:
        """Write what is queued, then stop the thread. Idempotent."""
        with self._cv:
            if self._stopped:
                return
            self._stop_requested = True
            self._cv.notify_all()
        self._thread.join()
        with self._cv:
            self._stopped = True

    # ── writer thread ────────────────────────────────────────────────────────
    def _next_batch(self):
        with self._cv:
            self._cv.wait_for(lambda: self._items or self._stop_requested)
            if not self._items:
                return None, []
            f, first = self._items.popleft()
            batch = [first]
            while (self._items and len(batch) < self.batch_size
                   and self._items[0][0] is f):
                batch.append(self._items.popleft()[1])
            self._in_flight = len(batch)
            return f, batch

    def _run(self):
        while True:
            try:
                f, batch = self._next_batch()
                if not batch:
                    return
                nbytes = sum(t.nbytes() for t in batch)
                t0 = self._clock()
                try:
                    self._sink(f, batch)
                except Exception as exc:
                    self._record_fault(f"{type(exc).__name__}: {exc}")
                    continue
                dt = self._clock() - t0
                with self._cv:
                    self._queue_bytes -= nbytes
                    self._in_flight = 0
                    self._bytes_written += nbytes
                    self._write_seconds += dt
                    self._last_batch_ms = dt * 1e3
                    self._cv.notify_all()
                self._maybe_flush(f)
            except BaseException as exc:
                # Nothing above this point may kill the thread silently: a
                # dead thread with `_fault is None` means drain() blocks
                # forever and check() keeps reporting healthy. Anything
                # that escapes _next_batch(), nbytes()/clock() bookkeeping,
                # or _maybe_flush's own setup lines ends the episode here,
                # same as a sink failure.
                self._record_fault(f"writer thread died: {type(exc).__name__}: {exc}")
                return

    def _record_fault(self, message: str):
        with self._cv:
            discarded_ticks = len(self._items) + self._in_flight
            discarded_mb = self._queue_bytes / 1e6
            self._fault = message
            self._items.clear()
            self._queue_bytes = 0
            self._in_flight = 0
            self._cv.notify_all()
        log.error("writer fault: %s — episode must end, discarding %d queued "
                 "ticks (%.1f MB)", message, discarded_ticks, discarded_mb)

    def _maybe_flush(self, f):
        now = self._clock()
        if now - self._last_flush_t < self.flush_interval_s:
            return
        elapsed = now - self._last_flush_t
        self._last_flush_t = now
        try:
            # H5Fflush on a multi-GB file and the disk_usage() syscall can
            # take tens of ms. They must run OUTSIDE the writer lock: this
            # is the writer thread's only lock, and submit() on the 30 Hz
            # capture thread takes the same lock — holding it across real
            # I/O would stall capture, once per flush interval, with no
            # signal anyone could act on.
            flush = getattr(f, "flush", None)
            if flush is not None:
                flush()
            filename = getattr(f, "filename", None)
            file_bytes = os.path.getsize(filename) if filename else 0
            free_gb = self._disk_usage(os.path.dirname(filename) or ".").free / 1e9 \
                if filename else None
            with self._cv:
                self._flushes += 1
                if self._last_file_bytes and elapsed > 0:
                    self._file_mb_s = (file_bytes - self._last_file_bytes) / elapsed / 1e6
                self._last_file_bytes = file_bytes
                self._disk_free_gb = free_gb
        except Exception as exc:  # a failed flush must not kill the recording
            log.warning("flush failed (%s); a crash now would lose the file", exc)
