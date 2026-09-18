# simulator.py - fake 720p frame source.
# Packet layout: magic, seq, timestamp, payload size, then raw RGB.
# Producer thread paces itself off perf_counter_ns() for 60 fps.

from __future__ import annotations

import random
import struct
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from structures import (
    MemoryPool,
    PoolExhaustedException,
    QueueClosedException,
    QueueOverflowException,
    ThreadSafeCircularBuffer,
)
# --------------------------------------------------------------------------
# Wire format constants
# --------------------------------------------------------------------------
MAGIC_NUMBER: bytes = b"SNAP"
HEADER_STRUCT = struct.Struct("<4sIQI")
HEADER_SIZE: int = HEADER_STRUCT.size  # 20 bytes, no padding with "<"

FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
BYTES_PER_PIXEL: int = 3
PAYLOAD_SIZE: int = FRAME_WIDTH * FRAME_HEIGHT * BYTES_PER_PIXEL
BLOCK_SIZE: int = HEADER_SIZE + PAYLOAD_SIZE

TARGET_FPS: int = 60
NANOSECONDS_PER_SECOND: int = 1_000_000_000

# Scheduler tuning: sleep while far from the deadline, then spin close to it.
COARSE_SLEEP_THRESHOLD_NS: int = 3_000_000
SPIN_GUARD_NS: int = 2_000_000


class InvalidFrameError(ValueError):
    """Raised when a packet fails header validation."""


@dataclass(frozen=True)
class FrameHeader:
    magic: bytes
    sequence: int
    timestamp_ns: int
    payload_size: int


@dataclass(frozen=True)
class FramePacket:
    """Lightweight handle placed on the ring buffer (the pixels stay in the pool)."""

    pool_index: int
    sequence: int
    timestamp_ns: int


def parse_frame_header(frame: memoryview) -> FrameHeader:
    """Decode and validate the 20-byte header at the start of `frame`."""
    if len(frame) < HEADER_SIZE:
        raise InvalidFrameError(
            "frame shorter than header: %d < %d bytes" % (len(frame), HEADER_SIZE)
        )
    magic, sequence, timestamp_ns, payload_size = HEADER_STRUCT.unpack_from(frame, 0)
    if magic != MAGIC_NUMBER:
        raise InvalidFrameError("bad magic number: %r" % (magic,))
    if payload_size > len(frame) - HEADER_SIZE:
        raise InvalidFrameError(
            "payload size %d exceeds available %d bytes"
            % (payload_size, len(frame) - HEADER_SIZE)
        )
    return FrameHeader(magic, sequence, timestamp_ns, payload_size)


# --------------------------------------------------------------------------
# Frame synthesizer (shared by the server thread and the benchmark)
# --------------------------------------------------------------------------
class FrameSynthesizer:
    """
    Builds packets into caller-supplied memory blocks without allocating.

    Pixel noise comes from a single pre-generated noise bank. Each frame copies
    a window of that bank starting at a sequence-dependent offset, so frames
    differ but no per-frame random number generation is needed. A bright
    50x1-pixel marker then sweeps across the frame as a stand-in for a
    crosshair / HUD element in a competitive shooter or MOBA.
    """

    NOISE_MARGIN: int = 1 << 16
    MARKER_PIXELS: int = 50
    GAME_TITLES: Tuple[str, str] = ("VALORANT", "LEAGUE OF LEGENDS")

    def __init__(self, seed: int = 0x5EED) -> None:
        rng = random.Random(seed)
        self._noise_bank: bytes = rng.randbytes(PAYLOAD_SIZE + self.NOISE_MARGIN)
        self._noise_view = memoryview(self._noise_bank)
        self._marker: bytes = bytes([235]) * (self.MARKER_PIXELS * BYTES_PER_PIXEL)

    def game_title(self, sequence: int) -> str:
        """Alternate the nominal title every 600 frames (10 s at 60 FPS)."""
        return self.GAME_TITLES[(sequence // 600) % len(self.GAME_TITLES)]

    def synthesize(self, block: memoryview, sequence: int, timestamp_ns: int) -> int:
        """Fill `block` with one frame. Returns the number of bytes written."""
        if len(block) < BLOCK_SIZE:
            raise ValueError(
                "block too small: %d < %d bytes" % (len(block), BLOCK_SIZE)
            )

        HEADER_STRUCT.pack_into(
            block, 0, MAGIC_NUMBER, sequence & 0xFFFFFFFF, timestamp_ns, PAYLOAD_SIZE
        )

        offset = (sequence * 4099) % self.NOISE_MARGIN
        block[HEADER_SIZE:BLOCK_SIZE] = self._noise_view[offset : offset + PAYLOAD_SIZE]

        row = (sequence * 5) % FRAME_HEIGHT
        col = (sequence * 11) % (FRAME_WIDTH - self.MARKER_PIXELS)
        start = HEADER_SIZE + (row * FRAME_WIDTH + col) * BYTES_PER_PIXEL
        block[start : start + len(self._marker)] = self._marker
        return BLOCK_SIZE


# --------------------------------------------------------------------------
# Background stream server
# --------------------------------------------------------------------------
class MockStreamServer(threading.Thread):
    """Daemon thread that emits synthetic frames at a fixed frame rate."""

    def __init__(
        self,
        buffer: ThreadSafeCircularBuffer[FramePacket],
        pool: MemoryPool,
        target_fps: int = TARGET_FPS,
        max_frames: Optional[int] = None,
        synthesizer: Optional[FrameSynthesizer] = None,
    ) -> None:
        super().__init__(name="MockStreamServer", daemon=True)
        if target_fps < 1:
            raise ValueError("target_fps must be >= 1")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be None or >= 1")
        if pool.block_size < BLOCK_SIZE:
            raise ValueError(
                "pool block size %d is smaller than a frame (%d)"
                % (pool.block_size, BLOCK_SIZE)
            )

        self._buffer = buffer
        self._pool = pool
        self._fps = target_fps
        self._max_frames = max_frames
        self._synth = synthesizer if synthesizer is not None else FrameSynthesizer()

        self._stop_event = threading.Event()
        self._finished_event = threading.Event()
        self._stats_lock = threading.Lock()

        self._generated: int = 0
        self._dropped_pool: int = 0
        self._dropped_buffer: int = 0
        self._lateness_sum_ns: int = 0
        self._lateness_max_ns: int = 0

    # ---- control -----------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()

    @property
    def finished(self) -> threading.Event:
        return self._finished_event

    # ---- thread body -------------------------------------------------
    def run(self) -> None:
        try:
            start_ns = time.perf_counter_ns()
            sequence = 0
            while not self._stop_event.is_set():
                deadline_ns = start_ns + (sequence * NANOSECONDS_PER_SECOND) // self._fps
                self._wait_until(deadline_ns)
                if self._stop_event.is_set():
                    break

                emit_ns = time.perf_counter_ns()
                self._record_lateness(emit_ns - deadline_ns)
                self._emit(sequence, emit_ns)
                sequence += 1

                if self._max_frames is not None and sequence >= self._max_frames:
                    break
        finally:
            self._finished_event.set()

    def _wait_until(self, deadline_ns: int) -> None:
        """Coarse sleep, then yield-spin for the last couple of milliseconds."""
        while True:
            if self._stop_event.is_set():
                return
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns <= 0:
                return
            if remaining_ns > COARSE_SLEEP_THRESHOLD_NS:
                time.sleep((remaining_ns - SPIN_GUARD_NS) / NANOSECONDS_PER_SECOND)
            else:
                # sleep(0) releases the GIL so the consumer thread is not starved.
                time.sleep(0)

    def _emit(self, sequence: int, timestamp_ns: int) -> None:
        try:
            index = self._pool.acquire(block=False)
        except PoolExhaustedException:
            with self._stats_lock:
                self._dropped_pool += 1
            return

        try:
            self._synth.synthesize(self._pool.view(index), sequence, timestamp_ns)
            packet = FramePacket(index, sequence, timestamp_ns)
            self._buffer.put(packet, block=False)
        except QueueOverflowException:
            self._pool.release(index)
            with self._stats_lock:
                self._dropped_buffer += 1
            return
        except QueueClosedException:
            self._pool.release(index)
            self._stop_event.set()
            return
        except Exception:
            self._pool.release(index)
            raise

        with self._stats_lock:
            self._generated += 1

    def _record_lateness(self, lateness_ns: int) -> None:
        if lateness_ns < 0:
            lateness_ns = 0
        with self._stats_lock:
            self._lateness_sum_ns += lateness_ns
            if lateness_ns > self._lateness_max_ns:
                self._lateness_max_ns = lateness_ns

    # ---- reporting ---------------------------------------------------
    def stats(self) -> Dict[str, float]:
        with self._stats_lock:
            attempts = self._generated + self._dropped_pool + self._dropped_buffer
            mean_late_us = (self._lateness_sum_ns / attempts / 1000.0) if attempts else 0.0
            return {
                "generated": self._generated,
                "dropped_pool_exhausted": self._dropped_pool,
                "dropped_buffer_full": self._dropped_buffer,
                "mean_emit_lateness_us": mean_late_us,
                "max_emit_lateness_us": self._lateness_max_ns / 1000.0,
            }