# structures.py - ring buffer + memory pool.
# No queue.Queue, no deque. Head/tail pointers, one lock, two conditions.

from __future__ import annotations

import threading
import time
from typing import Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------
# Custom exceptions
# --------------------------------------------------------------------------
class QueueOverflowException(Exception):
    """Raised when an item is pushed into a buffer that is already full."""


class QueueUnderflowException(Exception):
    """Raised when an item is requested from a buffer that is empty."""


class QueueClosedException(Exception):
    """Raised when a producer touches a buffer that has been closed."""


class PoolExhaustedException(Exception):
    """Raised when every block in a MemoryPool is currently checked out."""


class InvalidBlockException(Exception):
    """Raised on out-of-range block indices, double frees, or stale views."""


# --------------------------------------------------------------------------
# Thread-safe bounded circular buffer
# --------------------------------------------------------------------------
class ThreadSafeCircularBuffer(Generic[T]):
    """
    Bounded FIFO ring buffer.

    Layout:
        _slots : fixed-size list, allocated once
        _head  : index of the next slot to READ  (oldest element)
        _tail  : index of the next slot to WRITE (one past the newest)
        _count : number of live elements (needed to tell full from empty,
                 because head == tail is true in both situations)

    Both pointers advance modulo capacity. All state is guarded by a single
    threading.Lock. Two Conditions (not_empty / not_full) are bound to that
    same lock so blocked producers and consumers can be woken independently.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an int, got %r" % (type(capacity),))
        if capacity < 1:
            raise ValueError("capacity must be >= 1, got %d" % capacity)

        self._capacity: int = capacity
        self._slots: List[Optional[T]] = [None] * capacity
        self._head: int = 0
        self._tail: int = 0
        self._count: int = 0
        self._closed: bool = False

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._overflow_rejections: int = 0
        self._underflow_rejections: int = 0
        self._high_watermark: int = 0

    # ---- helpers -----------------------------------------------------
    @staticmethod
    def _validate_timeout(timeout: Optional[float]) -> None:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be None or a non-negative number")

    def _enqueue_locked(self, item: T) -> None:
        """Write at tail. Caller must hold the lock and have checked space."""
        if self._count >= self._capacity:
            raise QueueOverflowException(
                "internal invariant broken: enqueue on full buffer (cap=%d)"
                % self._capacity
            )
        self._slots[self._tail] = item
        self._tail = (self._tail + 1) % self._capacity
        self._count += 1
        self._total_enqueued += 1
        if self._count > self._high_watermark:
            self._high_watermark = self._count

    def _dequeue_locked(self) -> T:
        """Read at head. Caller must hold the lock and have checked content."""
        if self._count <= 0:
            raise QueueUnderflowException(
                "internal invariant broken: dequeue on empty buffer"
            )
        item = self._slots[self._head]
        self._slots[self._head] = None  # drop the reference for the GC
        self._head = (self._head + 1) % self._capacity
        self._count -= 1
        self._total_dequeued += 1
        return item  # type: ignore[return-value]

    # ---- producer side ----------------------------------------------
    def put(self, item: T, block: bool = True, timeout: Optional[float] = None) -> None:
        """
        Append an item.

        block=False  -> raise QueueOverflowException immediately if full.
        block=True   -> wait for space; if `timeout` (seconds) elapses first,
                        raise QueueOverflowException.
        """
        self._validate_timeout(timeout)
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._not_full:
            if self._closed:
                raise QueueClosedException("put() called on a closed buffer")

            while self._count >= self._capacity:
                if not block:
                    self._overflow_rejections += 1
                    raise QueueOverflowException(
                        "buffer full (capacity=%d)" % self._capacity
                    )
                remaining: Optional[float] = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        self._overflow_rejections += 1
                        raise QueueOverflowException(
                            "timed out waiting for space (capacity=%d)" % self._capacity
                        )
                self._not_full.wait(remaining)
                if self._closed:
                    raise QueueClosedException("buffer closed while waiting to put()")

            self._enqueue_locked(item)
            self._not_empty.notify()

    def put_nowait(self, item: T) -> None:
        self.put(item, block=False)

    # ---- consumer side ----------------------------------------------
    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        """
        Remove and return the oldest item.

        block=False -> raise QueueUnderflowException immediately if empty.
        block=True  -> wait for data; on timeout raise QueueUnderflowException.
        A closed buffer can still be drained; once empty it raises
        QueueUnderflowException instead of blocking forever.
        """
        self._validate_timeout(timeout)
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._not_empty:
            while self._count == 0:
                if self._closed:
                    self._underflow_rejections += 1
                    raise QueueUnderflowException("buffer is closed and fully drained")
                if not block:
                    self._underflow_rejections += 1
                    raise QueueUnderflowException("buffer empty")
                remaining: Optional[float] = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        self._underflow_rejections += 1
                        raise QueueUnderflowException("timed out waiting for data")
                self._not_empty.wait(remaining)

            item = self._dequeue_locked()
            self._not_full.notify()
            return item

    def get_nowait(self) -> T:
        return self.get(block=False)

    # ---- lifecycle & introspection ----------------------------------
    def close(self) -> None:
        """Reject further puts and wake every waiting thread."""
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def clear(self) -> int:
        """Drop all queued items and return how many were discarded."""
        with self._lock:
            discarded = self._count
            for i in range(self._capacity):
                self._slots[i] = None
            self._head = 0
            self._tail = 0
            self._count = 0
            self._not_full.notify_all()
            return discarded

    @property
    def capacity(self) -> int:
        return self._capacity

    def size(self) -> int:
        with self._lock:
            return self._count

    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0

    def is_full(self) -> bool:
        with self._lock:
            return self._count >= self._capacity

    def pointers(self) -> Dict[str, int]:
        """Snapshot of head/tail/count, useful for diagnostics and tests."""
        with self._lock:
            return {"head": self._head, "tail": self._tail, "count": self._count}

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "capacity": self._capacity,
                "size": self._count,
                "head": self._head,
                "tail": self._tail,
                "total_enqueued": self._total_enqueued,
                "total_dequeued": self._total_dequeued,
                "overflow_rejections": self._overflow_rejections,
                "underflow_rejections": self._underflow_rejections,
                "high_watermark": self._high_watermark,
            }


# --------------------------------------------------------------------------
# Pre-allocated memory pool
# --------------------------------------------------------------------------
class MemoryPool:
    """
    Fixed pool of equally sized bytearray blocks.

    The blocks (and a memoryview over each) are created once in __init__.
    Free block indices live in their own small ring (head/tail/count), so
    acquire() and release() are O(1) and allocate nothing.

    Callers receive an integer block index. view(index) returns a writable
    memoryview over the block; slicing that view does not copy data.
    """

    def __init__(self, block_count: int, block_size: int) -> None:
        if isinstance(block_count, bool) or not isinstance(block_count, int):
            raise TypeError("block_count must be an int")
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise TypeError("block_size must be an int")
        if block_count < 1:
            raise ValueError("block_count must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")

        self._block_count: int = block_count
        self._block_size: int = block_size
        self._blocks: List[bytearray] = [bytearray(block_size) for _ in range(block_count)]
        self._views: List[memoryview] = [memoryview(b) for b in self._blocks]
        self._in_use: List[bool] = [False] * block_count

        # Free-index ring: initially holds 0..n-1 in order.
        self._free_ring: List[int] = list(range(block_count))
        self._free_head: int = 0
        self._free_tail: int = 0  # (n % n) == 0: next write wraps to slot 0
        self._free_count: int = block_count

        self._lock = threading.Lock()
        self._available = threading.Condition(self._lock)

        self._acquisitions: int = 0
        self._releases: int = 0
        self._exhaustions: int = 0
        self._peak_in_use: int = 0

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def block_count(self) -> int:
        return self._block_count

    def acquire(self, block: bool = False, timeout: Optional[float] = None) -> int:
        """Check out a free block and return its index."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be None or non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._available:
            while self._free_count == 0:
                if not block:
                    self._exhaustions += 1
                    raise PoolExhaustedException(
                        "all %d blocks are in use" % self._block_count
                    )
                remaining: Optional[float] = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        self._exhaustions += 1
                        raise PoolExhaustedException("timed out waiting for a free block")
                self._available.wait(remaining)

            index = self._free_ring[self._free_head]
            self._free_head = (self._free_head + 1) % self._block_count
            self._free_count -= 1
            self._in_use[index] = True
            self._acquisitions += 1
            in_use_now = self._block_count - self._free_count
            if in_use_now > self._peak_in_use:
                self._peak_in_use = in_use_now
            return index

    def release(self, index: int) -> None:
        """Return a block to the pool. Double frees are rejected."""
        with self._available:
            self._check_index_locked(index)
            if not self._in_use[index]:
                raise InvalidBlockException("block %d released twice" % index)
            self._in_use[index] = False
            self._free_ring[self._free_tail] = index
            self._free_tail = (self._free_tail + 1) % self._block_count
            self._free_count += 1
            self._releases += 1
            self._available.notify()

    def view(self, index: int) -> memoryview:
        """Writable memoryview over a currently checked-out block."""
        with self._lock:
            self._check_index_locked(index)
            if not self._in_use[index]:
                raise InvalidBlockException("block %d is not checked out" % index)
            return self._views[index]

    def free_count(self) -> int:
        with self._lock:
            return self._free_count

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "block_count": self._block_count,
                "block_size": self._block_size,
                "free": self._free_count,
                "in_use": self._block_count - self._free_count,
                "acquisitions": self._acquisitions,
                "releases": self._releases,
                "exhaustions": self._exhaustions,
                "peak_in_use": self._peak_in_use,
            }

    def _check_index_locked(self, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise InvalidBlockException("block index must be an int")
        if index < 0 or index >= self._block_count:
            raise InvalidBlockException(
                "block index %d out of range [0, %d)" % (index, self._block_count)
            )