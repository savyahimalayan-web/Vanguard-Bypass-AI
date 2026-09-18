# pipeline.py - NPU path + CPU fallback.
# QNN EP setup is real on ARM64. Hexagon latency is a stage model
# (DMA in, HTP run, DMA out). CPU path does real bilinear sampling.

from __future__ import annotations

import logging
import os
import platform
import random
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from simulator import (
    BYTES_PER_PIXEL,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HEADER_SIZE,
    FramePacket,
    InvalidFrameError,
    parse_frame_header,
)
from structures import (
    MemoryPool,
    QueueUnderflowException,
    ThreadSafeCircularBuffer,
)

TARGET_WIDTH: int = 2560   # 1440p
TARGET_HEIGHT: int = 1440
INFERENCE_WINDOW_MS: float = 2.0


class NPUUnavailableError(RuntimeError):
    """Raised when the Qualcomm NPU path cannot be initialised on this host."""


class ExecutionMode(Enum):
    HEXAGON_HTP = "HEXAGON_HTP"
    CPU_FALLBACK = "CPU_FALLBACK"


# --------------------------------------------------------------------------
# QNN Execution Provider configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class QNNConfig:
    """Everything needed to build an ORT session on the Hexagon HTP backend."""

    backend_path: str = "QnnHtp.dll"
    performance_mode: str = "HIGH_PERFORMANCE"
    model_path: str = os.path.join("models", "xlsr_int8_qdq.onnx")
    precision: str = "INT8"
    vtcm_mb: int = 8
    graph_finalization_mode: int = 3
    rpc_control_latency_us: int = 100
    profiling_level: str = "off"
    inference_window_ms: float = INFERENCE_WINDOW_MS

    _ALLOWED_MODES = (
        "BURST",
        "BALANCED",
        "HIGH_PERFORMANCE",
        "SUSTAINED_HIGH_PERFORMANCE",
        "LOW_BALANCED",
        "POWER_SAVER",
    )

    def validate(self) -> None:
        if not self.backend_path.lower().endswith((".dll", ".so")):
            raise ValueError("backend_path must be a shared library: %r" % self.backend_path)
        if self.performance_mode not in self._ALLOWED_MODES:
            raise ValueError("unsupported performance_mode %r" % self.performance_mode)
        if self.precision not in ("INT8", "FP16"):
            raise ValueError("precision must be INT8 or FP16, got %r" % self.precision)
        if self.inference_window_ms <= 0:
            raise ValueError("inference_window_ms must be positive")

    def to_provider_options(self) -> Dict[str, str]:
        """Provider option dictionary passed to QNNExecutionProvider (all str values)."""
        return {
            "backend_path": self.backend_path,
            "htp_performance_mode": self.performance_mode.lower(),
            "htp_graph_finalization_optimization_mode": str(self.graph_finalization_mode),
            "enable_htp_fp16_precision": "1" if self.precision == "FP16" else "0",
            "vtcm_mb": str(self.vtcm_mb),
            "rpc_control_latency": str(self.rpc_control_latency_us),
            "profiling_level": self.profiling_level,
        }


# --------------------------------------------------------------------------
# Result / statistics containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameResult:
    sequence: int
    timestamp_ns: int
    source_resolution: str
    target_resolution: str
    latency_ns: int
    within_window: bool
    mode: ExecutionMode
    backend_label: str
    checksum: int
    stage_ns: Tuple[int, int, int]

    @property
    def latency_us(self) -> float:
        return self.latency_ns / 1000.0


@dataclass
class PipelineStats:
    frames_processed: int = 0
    frames_corrupt: int = 0
    window_violations: int = 0
    latency_sum_ns: int = 0
    latency_min_ns: int = 0
    latency_max_ns: int = 0

    @property
    def mean_latency_us(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.latency_sum_ns / self.frames_processed / 1000.0


# --------------------------------------------------------------------------
# Hexagon latency profile (modeled)
# --------------------------------------------------------------------------
class HexagonLatencyModel:
    # fake per-frame NPU latency. three stages + jitter. capped under 2 ms.

    DMA_IN_MS: float = 0.22
    HTP_EXEC_MS: float = 1.02
    DMA_OUT_MS: float = 0.26
    SIGMA_MS: float = 0.045
    SPIKE_PROBABILITY: float = 0.01
    SPIKE_MS: float = 0.15
    FLOOR_MS: float = 0.05
    WINDOW_FRACTION_CAP: float = 0.975

    def __init__(self, window_ms: float, seed: int = 0xC0FFEE) -> None:
        self._window_ms = window_ms
        self._rng = random.Random(seed)

    def sample_ns(self) -> Tuple[int, int, int]:
        """Return (dma_in_ns, htp_exec_ns, dma_out_ns)."""
        dma_in = max(self.FLOOR_MS, self.DMA_IN_MS + self._rng.gauss(0.0, self.SIGMA_MS))
        htp = max(self.FLOOR_MS, self.HTP_EXEC_MS + self._rng.gauss(0.0, self.SIGMA_MS))
        dma_out = max(self.FLOOR_MS, self.DMA_OUT_MS + self._rng.gauss(0.0, self.SIGMA_MS))
        if self._rng.random() < self.SPIKE_PROBABILITY:
            htp += self.SPIKE_MS

        total = dma_in + htp + dma_out
        cap = self._window_ms * self.WINDOW_FRACTION_CAP
        if total > cap:
            scale = cap / total
            dma_in *= scale
            htp *= scale
            dma_out *= scale

        return (int(dma_in * 1e6), int(htp * 1e6), int(dma_out * 1e6))


# --------------------------------------------------------------------------
# Coordinate mapping: 720p input domain -> 1440p reconstruction domain
# --------------------------------------------------------------------------
class UpscaleKernel:

# Precomputed bilinear sampling table: maps each 1440p lattice cell to four 720p neighbors

# with fixed-point weights; apply() blends them using integer math, with lattice density controlling workload.


    def __init__(
        self,
        src_w: int,
        src_h: int,
        dst_w: int,
        dst_h: int,
        lattice_w: int,
        lattice_h: int,
        bytes_per_pixel: int = BYTES_PER_PIXEL,
        channel: int = 1,
    ) -> None:
        for name, value in (
            ("src_w", src_w), ("src_h", src_h), ("dst_w", dst_w), ("dst_h", dst_h),
            ("lattice_w", lattice_w), ("lattice_h", lattice_h),
        ):
            if value < 1:
                raise ValueError("%s must be >= 1, got %d" % (name, value))
        if lattice_w > dst_w or lattice_h > dst_h:
            raise ValueError("lattice cannot be larger than the target plane")
        if not 0 <= channel < bytes_per_pixel:
            raise ValueError("channel out of range")

        self._src_w, self._src_h = src_w, src_h
        self._dst_w, self._dst_h = dst_w, dst_h
        self._lattice_w, self._lattice_h = lattice_w, lattice_h
        self._min_payload = src_w * src_h * bytes_per_pixel

        self._entries: List[Tuple[int, int, int, int, int, int]] = []
        for ly in range(lattice_h):
            ty = ((2 * ly + 1) * dst_h) // (2 * lattice_h)
            sy = self._clamp((ty + 0.5) * src_h / dst_h - 0.5, 0.0, src_h - 1.0)
            y0 = int(sy)
            y1 = min(y0 + 1, src_h - 1)
            wy = int((sy - y0) * 256)
            for lx in range(lattice_w):
                tx = ((2 * lx + 1) * dst_w) // (2 * lattice_w)
                sx = self._clamp((tx + 0.5) * src_w / dst_w - 0.5, 0.0, src_w - 1.0)
                x0 = int(sx)
                x1 = min(x0 + 1, src_w - 1)
                wx = int((sx - x0) * 256)
                o00 = (y0 * src_w + x0) * bytes_per_pixel + channel
                o01 = (y0 * src_w + x1) * bytes_per_pixel + channel
                o10 = (y1 * src_w + x0) * bytes_per_pixel + channel
                o11 = (y1 * src_w + x1) * bytes_per_pixel + channel
                self._entries.append((o00, o01, o10, o11, wx, wy))

        self._output = bytearray(len(self._entries))

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        if value < low:
            return low
        if value > high:
            return high
        return value

    @property
    def sample_count(self) -> int:
        return len(self._entries)

    def source_to_target(self, x: int, y: int) -> Tuple[int, int]:
        """Map a 720p pixel to the top-left pixel of its 1440p footprint."""
        if not (0 <= x < self._src_w and 0 <= y < self._src_h):
            raise ValueError("source coordinate (%d, %d) outside input plane" % (x, y))
        return (x * self._dst_w // self._src_w, y * self._dst_h // self._src_h)

    def apply(self, payload: memoryview) -> int:
        """Run the bilinear lattice pass and return a rolling checksum."""
        if len(payload) < self._min_payload:
            raise ValueError(
                "payload too small: %d < %d bytes" % (len(payload), self._min_payload)
            )
        out = self._output
        checksum = 0
        index = 0
        for o00, o01, o10, o11, wx, wy in self._entries:
            inv_x = 256 - wx
            top = (payload[o00] * inv_x + payload[o01] * wx) >> 8
            bottom = (payload[o10] * inv_x + payload[o11] * wx) >> 8
            value = (top * (256 - wy) + bottom * wy) >> 8
            out[index] = value
            checksum = (checksum * 31 + value) & 0xFFFFFFFF
            index += 1
        return checksum


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
class QualcommNPUPipeline:
    # Orchestrates upscaling via Hexagon/NPU or CPU fallback, with options to simulate either path.

    # Otherwise probes for Windows-on-ARM + QNN EP and falls back to CPU with a warning if detection fails.

    NPU_LATTICE: Tuple[int, int] = (16, 9)
    CPU_LATTICE: Tuple[int, int] = (160, 90)

    def __init__(
        self,
        config: Optional[QNNConfig] = None,
        simulate_hexagon_profile: bool = False,
        force_cpu_fallback: bool = False,
        logger: Optional[logging.Logger] = None,
        seed: int = 0xC0FFEE,
    ) -> None:
        self._config = config if config is not None else QNNConfig()
        self._config.validate()
        self._logger = logger if logger is not None else logging.getLogger("snapstream.pipeline")

        self._session: Any = None
        self._mode: ExecutionMode = ExecutionMode.CPU_FALLBACK
        self._backend_label: str = "CPU FALLBACK (uninitialised)"
        self._window_ns: int = int(self._config.inference_window_ms * 1e6)
        self._stats = PipelineStats()

        self._latency_model = HexagonLatencyModel(self._config.inference_window_ms, seed)
        self._npu_kernel = UpscaleKernel(
            FRAME_WIDTH, FRAME_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT, *self.NPU_LATTICE
        )
        self._cpu_kernel = UpscaleKernel(
            FRAME_WIDTH, FRAME_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT, *self.CPU_LATTICE
        )

        self._initialise_backend(simulate_hexagon_profile, force_cpu_fallback)

    # ---- properties --------------------------------------------------
    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    @property
    def backend_label(self) -> str:
        return self._backend_label

    @property
    def config(self) -> QNNConfig:
        return self._config

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    # ---- initialisation ---------------------------------------------
    @staticmethod
    def _is_windows_on_arm() -> bool:
        """True only for a native ARM64 Windows interpreter (x64 emulation reports AMD64)."""
        return sys.platform == "win32" and platform.machine().upper() in ("ARM64", "AARCH64")

    def _initialise_backend(self, simulate: bool, force_cpu: bool) -> None:
        try:
            if force_cpu:
                raise NPUUnavailableError("CPU fallback explicitly requested")
            if simulate:
                self._logger.info(
                    "Hexagon profile simulation requested; skipping hardware probe."
                )
                self._activate_hexagon(
                    "HEXAGON HTP (SIMULATED PROFILE, no hardware)", session=None
                )
                return

            session = self._create_qnn_session()
            if session is None:
                label = "HEXAGON HTP (QNN EP validated, model absent, profile-modeled)"
            else:
                label = "HEXAGON HTP via QNN EP (latency profile-modeled)"
            self._activate_hexagon(label, session=session)
        except NPUUnavailableError as exc:
            self._handle_npu_failure(str(exc))
        except Exception as exc:  # any provider/runtime error must not crash the app
            self._handle_npu_failure("%s: %s" % (type(exc).__name__, exc))

    def _create_qnn_session(self) -> Any:
        """
        Build the ONNX Runtime session on the QNN Execution Provider.

        Returns the InferenceSession, or None when the provider is present but
        the model file is missing. Raises NPUUnavailableError otherwise.
        """

# TODO: real XLSR model. right now this just builds the session and
# never actually runs inference on it.

        if not self._is_windows_on_arm():
            raise NPUUnavailableError(
                "host is %s/%s, not Windows on ARM64" % (sys.platform, platform.machine())
            )
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:
            raise NPUUnavailableError("onnxruntime is not installed") from exc

        if "QNNExecutionProvider" not in ort.get_available_providers():
            raise NPUUnavailableError("QNNExecutionProvider not in this onnxruntime build")

        if not os.path.isfile(self._config.model_path):
            self._logger.warning(
                "QNN EP available but model %r not found; running profile-modeled only.",
                self._config.model_path,
            )
            return None

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")

        provider_options = self._config.to_provider_options()
        self._logger.info("Creating QNN EP session with options: %s", provider_options)
        return ort.InferenceSession(
            self._config.model_path,
            sess_options=options,
            providers=[
                ("QNNExecutionProvider", provider_options),
                "CPUExecutionProvider",
            ],
        )

    def _activate_hexagon(self, label: str, session: Any) -> None:
        self._session = session
        self._mode = ExecutionMode.HEXAGON_HTP
        self._backend_label = label
        self._logger.info(
            "Backend active: %s | backend=%s mode=%s precision=%s window=%.2f ms",
            label,
            self._config.backend_path,
            self._config.performance_mode,
            self._config.precision,
            self._config.inference_window_ms,
        )

    def _handle_npu_failure(self, reason: str) -> None:
        self._emit_fallback_banner(reason)
        self._activate_cpu_fallback()

    def _emit_fallback_banner(self, reason: str) -> None:
        self._logger.warning(
        "NPU unavailable (%s). Falling back to CPU engine. "
        "Latency below is measured on this CPU, not NPU-modeled.",
        reason,
)

    def _activate_cpu_fallback(self) -> None:
        self._session = None
        self._mode = ExecutionMode.CPU_FALLBACK
        self._backend_label = "CPU FALLBACK (dense bilinear lattice)"

    # ---- per-frame execution ----------------------------------------
    @staticmethod
    def _spin_until(target_ns: int) -> None:
        """Yield-spin so other threads keep running while we honour the profile."""
        while time.perf_counter_ns() < target_ns:
            time.sleep(0)

    def _execute_hexagon(
        self, payload: memoryview, started_ns: int
    ) -> Tuple[int, Tuple[int, int, int]]:
        stages = self._latency_model.sample_ns()
        checksum = self._npu_kernel.apply(payload)
        self._spin_until(started_ns + stages[0] + stages[1] + stages[2])
        return checksum, stages

    def _execute_cpu(self, payload: memoryview) -> int:
        return self._cpu_kernel.apply(payload)

    def process_frame(self, frame: memoryview) -> FrameResult:
        """Validate one packet (header + payload) and run the active backend."""
        header = parse_frame_header(frame)
        expected = FRAME_WIDTH * FRAME_HEIGHT * BYTES_PER_PIXEL
        if header.payload_size != expected:
            raise InvalidFrameError(
                "unexpected payload size %d (expected %d)" % (header.payload_size, expected)
            )
        payload = frame[HEADER_SIZE : HEADER_SIZE + header.payload_size]

        started_ns = time.perf_counter_ns()
        if self._mode is ExecutionMode.HEXAGON_HTP:
            checksum, stages = self._execute_hexagon(payload, started_ns)
        else:
            checksum = self._execute_cpu(payload)
            stages = (0, 0, 0)
        latency_ns = time.perf_counter_ns() - started_ns
        if self._mode is ExecutionMode.CPU_FALLBACK:
            stages = (0, latency_ns, 0)

        within = latency_ns <= self._window_ns
        self._record(latency_ns, within)

        return FrameResult(
            sequence=header.sequence,
            timestamp_ns=header.timestamp_ns,
            source_resolution="%dx%d" % (FRAME_WIDTH, FRAME_HEIGHT),
            target_resolution="%dx%d" % (TARGET_WIDTH, TARGET_HEIGHT),
            latency_ns=latency_ns,
            within_window=within,
            mode=self._mode,
            backend_label=self._backend_label,
            checksum=checksum,
            stage_ns=stages,
        )

    def _record(self, latency_ns: int, within_window: bool) -> None:
        s = self._stats
        s.frames_processed += 1
        s.latency_sum_ns += latency_ns
        if s.frames_processed == 1 or latency_ns < s.latency_min_ns:
            s.latency_min_ns = latency_ns
        if latency_ns > s.latency_max_ns:
            s.latency_max_ns = latency_ns
        if not within_window:
            s.window_violations += 1

    # ---- consumer loop ----------------------------------------------
    def consume(
        self,
        buffer: ThreadSafeCircularBuffer[FramePacket],
        pool: MemoryPool,
        stop_event: threading.Event,
        on_result: Callable[[FrameResult], None],
        producer_finished: Optional[threading.Event] = None,
        poll_timeout_s: float = 0.25,
    ) -> int:
        """
        Pull packets until stop_event is set (or the producer finishes and the
        buffer drains). Every block is returned to the pool no matter what.
        Returns the number of frames processed successfully.
        """
        processed = 0
        while not stop_event.is_set():
            try:
                packet = buffer.get(timeout=poll_timeout_s)
            except QueueUnderflowException:
                if (
                    producer_finished is not None
                    and producer_finished.is_set()
                    and buffer.is_empty()
                ):
                    break
                continue

            try:
                frame = pool.view(packet.pool_index)
                result = self.process_frame(frame)
            except InvalidFrameError as exc:
                self._stats.frames_corrupt += 1
                self._logger.error("Dropped corrupt frame %d: %s", packet.sequence, exc)
                continue
            finally:
                pool.release(packet.pool_index)

            processed += 1
            on_result(result)
        return processed