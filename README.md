# SnapStream AI

> **On-device NPU super-resolution pipeline for low-latency ARM gaming**
> Built for the Snapdragon® AI Lab Build & Present Challenge. Targets Snapdragon-powered Windows on ARM PCs.

---

## The Problem

Snapdragon Windows on ARM laptops offer excellent efficiency and battery life, but competitive PC games are out of reach. Titles like **Valorant** and **League of Legends** rely on kernel-level anti-cheat (Riot Vanguard), which does not work under Windows on ARM's x86 translation layer, so the games will not install or run.

## The Solution

SnapStream AI sidesteps the anti-cheat problem by running the game natively on an x86 host and using the Snapdragon laptop as a thin client:

1. The x86 host renders the game at a low resolution (**720p**), keeping the stream light and the network latency low.
2. The Snapdragon laptop receives the stream and upscales each frame to **1440p** on the **Hexagon NPU**, using an **INT8-quantized XLSR (Extreme Lightweight Super-Resolution)** model.
3. The design target is an NPU inference budget of **2.0 ms per frame**, leaving the CPU and GPU free.

> The 2.0 ms figure covers upscaling inference only. End-to-end latency also includes encode, network transit and decode.

---

## What Works and What's Simulated

This repo is a prototype, and it doesn't depend on one specific machine. I marked each part so it's clear what actually runs and what is just a model.

| Component | Status |
|---|---|
| Thread-safe circular buffer, memory pool, custom exceptions | Working |
| Binary frame protocol (magic, sequence, timestamp, payload size) | Working |
| 60 FPS producer pacing with `time.perf_counter_ns()` | Working |
| ONNX Runtime + `QNNExecutionProvider` session configuration (`QnnHtp.dll`, high-performance mode, INT8) | Working; only created on Windows on ARM64 with `onnxruntime-qnn` and a model file |
| CPU fallback engine and its latency | Working; latency is measured on the host |
| Hexagon NPU per-frame latency | Simulated; stage-based model (DMA-in, HTP execute, DMA-out, jitter) |
| Power draw (NPU / GPU / CPU) | Modeled estimates; illustrative constants, not measurements |
| Upscaling arithmetic | Stand-in; bilinear sampling on a coordinate lattice, not a full-frame 1440p reconstruction |

The prototype does not actually run XLSR inference on tensors yet. The next thing to do is plug a real model into `QualcommNPUPipeline`.

---

## Architecture

Pure Python (standard library only), object-oriented code, clear data structures.

| File | Role |
|---|---|
| `structures.py` | `ThreadSafeCircularBuffer` (explicit head/tail pointers, `threading.Lock` + `Condition`, `QueueOverflowException` / `QueueUnderflowException`) and `MemoryPool` (pre-allocated frame buffers, so steady-state runs make no per-frame large allocations) |
| `simulator.py` | `MockStreamServer` daemon thread emitting synthetic 720p packets at drift-free 60 FPS; simulates the x86 host |
| `pipeline.py` | `QualcommNPUPipeline`: QNN EP configuration, Hexagon latency model, 720p → 1440p coordinate mapping, automatic CPU fallback with a big warning message |
| `main.py` | Interactive menu, ANSI telemetry dashboard, benchmark runner |

---

## Sample Benchmark Output

Real output from mode `2` (200 frames per backend, run on an x86 Linux machine). NPU latency and all wattage figures are modeled; CPU latency is measured.

```text
Backend                     Frames   Mean us    P50 us    P95 us    P99 us    Max us  Window%    Est. W
----------------------------------------------------------------------------------------------------
HEXAGON HTP (SIMULATED)        200    1541.9    1539.3    1682.3    1751.8    1765.9   100.0%     0.45
CPU FALLBACK (MEASURED)        200    5138.7    5032.5    5571.2    8698.9    9360.9     0.0%     4.50
----------------------------------------------------------------------------------------------------
Reference GPU upscale (modeled): 5.09 W
```

The live dashboard (mode `1`) redraws once per frame and shows: frame sequence, input grid (720p), reconstructed target (1440p), core latency in microseconds against the 2.0 ms window, playback FPS, frame jitter variance, and modeled NPU-vs-GPU power.

---

## Installation & Usage

**Requirements:** Python 3.9 or newer. No third-party packages are needed to run the demo.
On real Snapdragon hardware, additionally install `onnxruntime-qnn` and place the model at `models/xlsr_int8_qdq.onnx`.

```bash
git clone https://github.com/savyahimalayan-web/SnapStream-AI.git
cd SnapStream-AI
python main.py
```

Choose from the menu:

- `[1]` Live Local Stream Mode: 60 FPS synthetic stream with the live dashboard (Ctrl+C to stop)
- `[2]` NPU Benchmark Performance Simulator: runs the Hexagon profile and the CPU fallback side by side

Or skip the menu:

```bash
python main.py --mode live --frames 600
python main.py --mode benchmark --frames 300
python main.py --mode benchmark --no-color   # plain output
```

On non-ARM machines the app prints a large **"QUALCOMM NPU UNAVAILABLE"** warning and switches to the CPU fallback engine. This is expected behavior, not an error. Benchmark mode always includes the simulated Hexagon profile, so it works on any machine. Logs are written to `snapstream.log`.

---

## Roadmap

- Get the real INT8 XLSR ONNX model into `pipeline.py` and run it through `QNNExecutionProvider` (export from Qualcomm AI Hub).
- Replace the hardcoded power estimates in `pipeline.py` with real Snapdragon Profiler numbers.
- Replace `MockStreamServer` in `simulator.py` with real socket transport and hardware video decode.
- Test the whole pipeline on a Snapdragon X Elite laptop with `onnxruntime-qnn`.