# main.py - CLI, live dashboard, benchmark runner.
# Power numbers are guesses, not measurements. CPU latency is real,
# NPU latency is a profile model.

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence

from pipeline import (
    INFERENCE_WINDOW_MS,
    ExecutionMode,
    FrameResult,
    QualcommNPUPipeline,
)
from simulator import (
    BLOCK_SIZE,
    TARGET_FPS,
    FramePacket,
    FrameSynthesizer,
    MockStreamServer,
)
from structures import MemoryPool, ThreadSafeCircularBuffer

POOL_BLOCKS: int = 12
BUFFER_CAPACITY: int = 8
FRAME_BUDGET_MS: float = 1000.0 / TARGET_FPS
DASHBOARD_WIDTH: int = 78


# --------------------------------------------------------------------------
# Terminal helpers
# --------------------------------------------------------------------------
def enable_ansi_on_windows() -> None:
    """Turn on virtual-terminal processing so ANSI escapes render in conhost."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        os.system("")  # legacy trick that also enables VT mode


class Palette:
    """ANSI colour codes; all empty strings when colour is disabled."""

    def __init__(self, enabled: bool) -> None:
        def code(value: str) -> str:
            return value if enabled else ""

        self.reset = code("\x1b[0m")
        self.bold = code("\x1b[1m")
        self.dim = code("\x1b[2m")
        self.red = code("\x1b[31m")
        self.green = code("\x1b[32m")
        self.yellow = code("\x1b[33m")
        self.cyan = code("\x1b[36m")
        self.magenta = code("\x1b[35m")
        self.enabled = enabled


class RollingWindow:
    # fixed-size ring of floats, for mean/variance.

    def __init__(self, size: int) -> None:
        if size < 2:
            raise ValueError("size must be >= 2")
        self._size = size
        self._data: List[float] = [0.0] * size
        self._index = 0
        self._count = 0

    def push(self, value: float) -> None:
        self._data[self._index] = value
        self._index = (self._index + 1) % self._size
        if self._count < self._size:
            self._count += 1

    def count(self) -> int:
        return self._count

    def _values(self) -> List[float]:
        return self._data[: self._count]

    def mean(self) -> float:
        if self._count == 0:
            return 0.0
        return sum(self._values()) / self._count

    def variance(self) -> float:
        if self._count < 2:
            return 0.0
        mean = self.mean()
        return sum((v - mean) ** 2 for v in self._values()) / self._count


# --------------------------------------------------------------------------
# Modeled power draw
# --------------------------------------------------------------------------
class PowerModel:
    # rough duty-cycle power guess. replace with real numbers later.

    NPU_IDLE_W: float = 0.35
    NPU_ACTIVE_W: float = 1.45
    GPU_IDLE_W: float = 3.20
    GPU_ACTIVE_W: float = 11.50
    GPU_UPSCALE_MS: float = 3.8
    CPU_IDLE_W: float = 2.50
    CPU_ACTIVE_W: float = 9.00

    @staticmethod
    def _duty(compute_ms: float) -> float:
        return max(0.0, min(1.0, compute_ms / FRAME_BUDGET_MS))

    def device_watts(self, mode: ExecutionMode, latency_ms: float) -> float:
        duty = self._duty(latency_ms)
        if mode is ExecutionMode.HEXAGON_HTP:
            return self.NPU_IDLE_W + (self.NPU_ACTIVE_W - self.NPU_IDLE_W) * duty
        return self.CPU_IDLE_W + (self.CPU_ACTIVE_W - self.CPU_IDLE_W) * duty

    def gpu_watts(self) -> float:
        duty = self._duty(self.GPU_UPSCALE_MS)
        return self.GPU_IDLE_W + (self.GPU_ACTIVE_W - self.GPU_IDLE_W) * duty


# --------------------------------------------------------------------------
# Live dashboard
# --------------------------------------------------------------------------
class TelemetryDashboard:
    """Redraws a fixed-height ASCII panel in place,took a long time to design, once per processed frame."""

    PANEL_HEIGHT: int = 15

    def __init__(self, palette: Palette, power: Optional[PowerModel] = None) -> None:
        self._pal = palette
        self._power = power if power is not None else PowerModel()
        self._intervals_ms = RollingWindow(120)
        self._last_update_ns: Optional[int] = None
        self._drawn_once = False
        self._out = sys.stdout

    # ---- layout helpers ----------------------------------------------
    @staticmethod
    def _border() -> str:
        return "+" + "-" * (DASHBOARD_WIDTH - 2) + "+"

    @staticmethod
    def _gauge(fraction: float, width: int = 20) -> str:
        filled = int(round(max(0.0, min(1.0, fraction)) * width))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _row(self, label: str, plain: str, colored: Optional[str] = None) -> str:
        label_field = label.ljust(22)
        inner = DASHBOARD_WIDTH - 4
        pad = inner - len(label_field) - 2 - len(plain)
        shown = colored if colored is not None else plain
        return "| " + label_field + ": " + shown + " " * max(0, pad) + " |"

    def _center(self, text: str, colored: Optional[str] = None) -> str:
        inner = DASHBOARD_WIDTH - 4
        left = max(0, (inner - len(text)) // 2)
        right = max(0, inner - len(text) - left)
        shown = colored if colored is not None else text
        return "| " + " " * left + shown + " " * right + " |"

    # ---- public API ---------------------------------------------------
    def hide_cursor(self) -> None:
        if self._pal.enabled:
            self._out.write("\x1b[?25l")
            self._out.flush()

    def show_cursor(self) -> None:
        if self._pal.enabled:
            self._out.write("\x1b[?25h")
            self._out.flush()

    def update(self, result: FrameResult, queue_depth: int, dropped: int) -> None:
        now_ns = time.perf_counter_ns()
        if self._last_update_ns is not None:
            self._intervals_ms.push((now_ns - self._last_update_ns) / 1e6)
        self._last_update_ns = now_ns

        mean_interval = self._intervals_ms.mean()
        fps = 1000.0 / mean_interval if mean_interval > 0 else 0.0
        jitter_var = self._intervals_ms.variance()
        jitter_sigma = jitter_var ** 0.5

        p = self._pal
        latency_us = result.latency_us
        window_us = INFERENCE_WINDOW_MS * 1000.0
        window_ok = result.within_window
        lat_color = p.green if window_ok else p.red
        fps_color = p.green if fps >= 58.0 else (p.yellow if fps >= 45.0 else p.red)

        device_w = self._power.device_watts(result.mode, latency_us / 1000.0)
        gpu_w = self._power.gpu_watts()
        ratio = gpu_w / device_w if device_w > 0 else 0.0
        device_name = "Hexagon NPU" if result.mode is ExecutionMode.HEXAGON_HTP else "CPU"
        latency_kind = "modeled" if result.mode is ExecutionMode.HEXAGON_HTP else "measured"

        lat_plain = "%8.1f us %s %3.0f%% of %.0f us  %s" % (
            latency_us,
            self._gauge(latency_us / window_us, 12),
            100.0 * latency_us / window_us,
            window_us,
            "OK  " if window_ok else "OVER",
        )
        lat_colored = lat_plain.replace(
            "OK  " if window_ok else "OVER",
            p.bold + lat_color + ("OK  " if window_ok else "OVER") + p.reset,
        )

        fps_plain = "%6.2f fps (target %d)" % (fps, TARGET_FPS)
        jit_plain = "%.4f ms^2   (sigma %.3f ms)" % (jitter_var, jitter_sigma)
        pow_plain = "%.2f W (%s) vs %.2f W (GPU)  ->  %.1fx" % (
            device_w, device_name, gpu_w, ratio,
        )
        title_plain = "SNAPSTREAM AI  |  720p -> 1440p  |  Snapdragon Telemetry"
        title_colored = p.bold + p.cyan + title_plain + p.reset
        note_plain = "latency: %s | power: modeled estimate" % latency_kind

        lines = [
            self._border(),
            self._center(title_plain, title_colored),
            self._border(),
            self._row("Backend", result.backend_label[: DASHBOARD_WIDTH - 28]),
            self._row("Frame Sequence", "%08d" % result.sequence),
            self._row("Input Grid (720p)", result.source_resolution),
            self._row("Recon. Target (1440p)", result.target_resolution),
            self._row("Core Latency", lat_plain, lat_colored),
            self._row("Playback FPS", fps_plain, fps_color + fps_plain + p.reset),
            self._row("Frame Jitter Variance", jit_plain),
            self._row("Power (NPU vs GPU)", pow_plain, p.magenta + pow_plain + p.reset),
            self._row("Queue / Dropped", "depth %d/%d   dropped %d" % (
                queue_depth, BUFFER_CAPACITY, dropped)),
            self._center(note_plain, p.dim + note_plain + p.reset),
            self._border(),
            self._center("Ctrl+C to stop", p.dim + "Ctrl+C to stop" + p.reset),
        ]
        self._draw(lines)

    def _draw(self, lines: Sequence[str]) -> None:
        buf: List[str] = []
        if self._drawn_once and self._pal.enabled:
            buf.append("\x1b[%dA" % len(lines))
        for line in lines:
            buf.append("\r" + line + ("\x1b[K" if self._pal.enabled else "") + "\n")
        self._out.write("".join(buf))
        self._out.flush()
        self._drawn_once = True


# --------------------------------------------------------------------------
# Benchmark reporting
# --------------------------------------------------------------------------
def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be within [0, 100]")
    position = (len(sorted_values) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


class BenchmarkRun:
    """Latency samples plus derived statistics for one backend."""

    def __init__(self, label: str, mode: ExecutionMode) -> None:
        self.label = label
        self.mode = mode
        self.latencies_us: List[float] = []
        self.violations = 0

    def add(self, result: FrameResult) -> None:
        self.latencies_us.append(result.latency_us)
        if not result.within_window:
            self.violations += 1

    def summary(self) -> Dict[str, float]:
        ordered = sorted(self.latencies_us)
        count = len(ordered)
        mean = sum(ordered) / count if count else 0.0
        return {
            "frames": float(count),
            "mean": mean,
            "p50": percentile(ordered, 50.0),
            "p95": percentile(ordered, 95.0),
            "p99": percentile(ordered, 99.0),
            "max": ordered[-1] if ordered else 0.0,
            "hit_rate": 100.0 * (count - self.violations) / count if count else 0.0,
        }


def print_benchmark_report(runs: Sequence[BenchmarkRun], pal: Palette, power: PowerModel) -> None:
    line = "=" * 100
    print()
    print(pal.bold + line + pal.reset)
    print(pal.bold + "SNAPSTREAM AI  NPU BENCHMARK REPORT  (720p -> 1440p, %.1f ms inference window)"
          % INFERENCE_WINDOW_MS + pal.reset)
    print(line)
    header = "%-26s %7s %9s %9s %9s %9s %9s %8s %9s" % (
        "Backend", "Frames", "Mean us", "P50 us", "P95 us", "P99 us", "Max us", "Window%", "Est. W",
    )
    print(header)
    print("-" * 100)
    for run in runs:
        s = run.summary()
        watts = power.device_watts(run.mode, s["mean"] / 1000.0)
        color = pal.green if s["hit_rate"] >= 99.0 else pal.red
        row = "%-26s %7d %9.1f %9.1f %9.1f %9.1f %9.1f %7.1f%% %8.2f" % (
            run.label, int(s["frames"]), s["mean"], s["p50"], s["p95"], s["p99"],
            s["max"], s["hit_rate"], watts,
        )
        print(color + row + pal.reset)
    print("-" * 100)
    print("Reference GPU upscale (modeled): %.2f W" % power.gpu_watts())
    print(pal.yellow + "NOTE: Hexagon latency and ALL wattage figures are modeled estimates. "
          "CPU latency is measured on this host." + pal.reset)
    print(line)


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
def configure_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("snapstream")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console)

    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not open log file %r: %s", log_file, exc)
    return logger


class SnapStreamApp:
    def __init__(self, palette: Palette, logger: logging.Logger) -> None:
        self._pal = palette
        self._logger = logger
        self._power = PowerModel()

    # ---- mode 1: live ---------------------------------------------------
    def run_live(self, max_frames: Optional[int]) -> int:
        pool = MemoryPool(POOL_BLOCKS, BLOCK_SIZE)
        buffer: ThreadSafeCircularBuffer[FramePacket] = ThreadSafeCircularBuffer(BUFFER_CAPACITY)
        pipeline = QualcommNPUPipeline(logger=self._logger.getChild("pipeline"))
        server = MockStreamServer(buffer, pool, max_frames=max_frames)
        dashboard = TelemetryDashboard(self._pal, self._power)
        stop_event = threading.Event()
        # TODO: real socket transport instead of MockStreamServer


        def on_result(result: FrameResult) -> None:
            stats = server.stats()
            dropped = int(stats["dropped_pool_exhausted"] + stats["dropped_buffer_full"])
            dashboard.update(result, buffer.size(), dropped)

        gc.collect()
        gc.freeze()  # everything allocated so far is permanent; nothing left to collect
        print("\nStarting live local stream (%s)...\n" % pipeline.backend_label)
        dashboard.hide_cursor()
        server.start()
        try:
            pipeline.consume(
                buffer, pool, stop_event, on_result, producer_finished=server.finished
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            server.stop()
            server.join(timeout=2.0)
            buffer.close()
            stop_event.set()
            dashboard.show_cursor()
            gc.unfreeze()

        self._print_live_summary(server, buffer, pool, pipeline)
        return 0

    def _print_live_summary(
        self,
        server: MockStreamServer,
        buffer: ThreadSafeCircularBuffer[FramePacket],
        pool: MemoryPool,
        pipeline: QualcommNPUPipeline,
    ) -> None:
        ss = server.stats()
        bs = buffer.stats()
        ps = pool.stats()
        st = pipeline.stats
        print("\n--- Session summary ---")
        print("Backend           : %s" % pipeline.backend_label)
        print("Frames generated  : %d  (dropped: pool=%d, buffer=%d)" % (
            ss["generated"], ss["dropped_pool_exhausted"], ss["dropped_buffer_full"]))
        print("Frames processed  : %d  (corrupt=%d)" % (st.frames_processed, st.frames_corrupt))
        print("Mean latency      : %.1f us  (min %.1f, max %.1f)" % (
            st.mean_latency_us, st.latency_min_ns / 1000.0, st.latency_max_ns / 1000.0))
        print("Window violations : %d" % st.window_violations)
        print("Emit lateness     : mean %.1f us, max %.1f us" % (
            ss["mean_emit_lateness_us"], ss["max_emit_lateness_us"]))
        print("Buffer high-water : %d / %d" % (bs["high_watermark"], bs["capacity"]))
        print("Pool peak in use  : %d / %d" % (ps["peak_in_use"], ps["block_count"]))

    # ---- mode 2: benchmark ----------------------------------------------
    def run_benchmark(self, frames: int) -> int:
        if frames < 1:
            raise ValueError("frames must be >= 1")
        synth = FrameSynthesizer()
        pool = MemoryPool(2, BLOCK_SIZE)

        contenders = [
            ("HEXAGON HTP (SIMULATED)", QualcommNPUPipeline(
                simulate_hexagon_profile=True, logger=self._logger.getChild("bench.npu"))),
            ("CPU FALLBACK (MEASURED)", QualcommNPUPipeline(
                force_cpu_fallback=True, logger=self._logger.getChild("bench.cpu"))),
        ]

        gc.collect()
        gc.freeze()
        runs: List[BenchmarkRun] = []
        try:
            for label, pipeline in contenders:
                run = BenchmarkRun(label, pipeline.mode)
                print("\nRunning %s for %d frames..." % (label, frames))
                for sequence in range(frames):
                    index = pool.acquire()
                    try:
                        synth.synthesize(pool.view(index), sequence, time.perf_counter_ns())
                        run.add(pipeline.process_frame(pool.view(index)))
                    finally:
                        pool.release(index)
                    if (sequence + 1) % 10 == 0 or sequence + 1 == frames:
                        done = (sequence + 1) / frames
                        bar = "#" * int(done * 30)
                        sys.stdout.write("\r  [%-30s] %3d%%" % (bar, int(done * 100)))
                        sys.stdout.flush()
                sys.stdout.write("\n")
                runs.append(run)
        finally:
            gc.unfreeze()

        print_benchmark_report(runs, self._pal, self._power)
        return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def print_banner(pal: Palette) -> None:
    print(pal.bold + pal.cyan)
    print("  ____                  ____  _                                  _    ___ ")
    print(" / ___| _ __   __ _ _ __/ ___|| |_ _ __ ___  __ _ _ __ ___      / \\  |_ _|")
    print(" \\___ \\| '_ \\ / _` | '_ \\___ \\| __| '__/ _ \\/ _` | '_ ` _ \\    / _ \\  | | ")
    print("  ___) | | | | (_| | |_) |__) | |_| | |  __/ (_| | | | | | |  / ___ \\ | | ")
    print(" |____/|_| |_|\\__,_| .__/____/ \\__|_|  \\___|\\__,_|_| |_| |_| /_/   \\_\\___|")
    print("                   |_|" + pal.reset)
    print("  Real-time 720p -> 1440p upscaling pipeline for Snapdragon X laptops")
    print("  Built on the Hexagon NPU. Thanks Qualcomm.\n")


def prompt_for_mode() -> str:
    print("  [1] Live Local Stream Mode")
    print("  [2] NPU Benchmark Performance Simulator")
    print("  [q] Quit\n")
    while True:
        try:
            choice = input("Select mode > ").strip().lower()
        except EOFError:
            return "1"
        if choice in ("1", "2", "q"):
            return choice
        print("Please enter 1, 2 or q.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SnapStream AI - Snapdragon NPU upscaling demo")
    parser.add_argument("--mode", choices=["live", "benchmark", "1", "2"], default=None)
    parser.add_argument("--frames", type=int, default=None,
                        help="live: stop after N frames; benchmark: frames per backend")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colours")
    parser.add_argument("--log-file", default="snapstream.log")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    enable_ansi_on_windows()
    use_color = (not args.no_color) and sys.stdout.isatty()
    pal = Palette(use_color)
    logger = configure_logging(args.log_file)

    # A short switch interval keeps the 60 FPS producer responsive while the
    # consumer thread is busy.
    sys.setswitchinterval(0.0005)

    try:
        print_banner(pal)
        if args.mode is None:
            choice = prompt_for_mode()
        else:
            choice = {"live": "1", "1": "1", "benchmark": "2", "2": "2"}[args.mode]

        if choice == "q":
            print("Goodbye.")
            return 0

        app = SnapStreamApp(pal, logger)
        if choice == "1":
            return app.run_live(args.frames)
        return app.run_benchmark(args.frames if args.frames is not None else 300)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        logger.exception("Fatal error")
        print("\nFatal error: %s: %s (see %s)" % (type(exc).__name__, exc, args.log_file))
        return 1
    finally:
        if use_color:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())