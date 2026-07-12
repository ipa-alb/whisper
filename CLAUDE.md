# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A voice-controlled local AI assistant: record speech → transcribe with faster-whisper (GPU) → send to a local LLM (Ollama, `qwen3:30b`) → the LLM answers or calls Python tools. One of those tools (`look`) takes a camera snapshot and runs YOLO object detection, so the assistant can be asked what it sees. Everything runs on-device; nothing binds to a public interface, and audio/prompts/images never leave the machine.

**[PROJECT.md](PROJECT.md) is the source of truth** for scope, component choices, and roadmap. [INSTALL.md](INSTALL.md) covers environment setup and per-stage smoke tests. Keep both (and the README status table) in sync when phases advance or components change.

## Commands

```bash
source .venv/bin/activate            # always work inside the venv

python3 assistant.py                 # full pipeline: voice → LLM → tools
python3 assistant.py --text "..."    # same pipeline, typed input — use this to test without mic/GPU-whisper
python3 listen.py                    # Phase 1: transcription only
python3 mic_check.py                 # diagnostic: live level meter for all input devices
```

There is no test suite, linter, or build step. Verification is manual: `--text` mode exercises the LLM + tool layer without audio; the smoke tests in INSTALL.md Step 6 verify GPU/Whisper, microphone, and Ollama independently. Ollama must be running (`localhost:11434`) with `qwen3:30b` pulled. On the current machine Ollama is a user-space tarball install with no systemd service — if `localhost:11434` doesn't answer, start it with `~/.local/bin/ollama serve &`.

Audio comes from the system **default** input source. If transcription returns "(nothing recognized)" while someone is speaking, suspect the default source before suspecting Whisper — `mic_check.py` shows which device actually hears the voice.

## Architecture

- **[listen.py](listen.py)** — Phase 1 PoC, standalone transcription loop. Recording starts on launch; Enter ends an utterance (this replaced the original hotkey design).
- **[assistant.py](assistant.py)** — the main pipeline. Loads all tools, runs the record → transcribe → `ollama.chat` loop. Tool calls are executed in rounds (max `MAX_TOOL_ROUNDS`) until the LLM returns plain text. Tool exceptions are caught and returned to the LLM as `error: ...` strings — tool bugs must never kill the session.
- **[tools/](tools/)** — plugin directory, auto-discovered at startup.
- **[mic_check.py](mic_check.py)** — standalone diagnostic, not part of the pipeline.

### LLM output quirk

`qwen3:30b` sometimes emits its chain-of-thought inline despite `think=False`; `handle_utterance` in assistant.py strips everything up to a leaked `</think>` tag before returning the reply. Keep that strip when changing models — it's a no-op for models that behave.

### Tool plugin convention

Every `tools/*.py` file (names starting with `_` are skipped) must expose exactly two module-level things:

- `TOOL` — an Ollama/OpenAI-style function schema dict (`{"type": "function", "function": {...}}`)
- `run(**kwargs)` — the callable; parameters match the schema. Return a string (results and errors alike — return `"error: ..."`, don't raise).

Conventions the existing tools follow, keep them for new ones:

- **Path safety:** any file access resolves via `os.path.realpath` and verifies the result stays inside the workspace root.
- **Output caps:** truncate long output before returning it to the LLM (see `MAX_CHARS` / `MAX_OUTPUT`).
- **Confirmation split:** predefined tools run freely; LLM-*generated* code goes through `run_python`, which prints the code and requires an explicit `y` before executing. Never weaken this confirmation path.

### CUDA preload quirk

`listen.py` and `assistant.py` begin by `ctypes.CDLL`-loading every `nvidia/*/lib/*.so*` from the venv's site-packages **before** importing faster-whisper — the pip-installed cuBLAS/cuDNN libs aren't on the system loader path. Any new entry point that uses Whisper on GPU needs the same preload block.

### Camera / vision tool (`tools/see_camera.py` → `look`)

Deliberately dependency-light so it stays "lightweight YOLO": **no OpenCV, no PyTorch, no ultralytics at runtime.** `models/yolov8n.onnx` (stock COCO, 80 classes) runs under `onnxruntime` on **CPU** (one snapshot is fast enough there — the GPU stays reserved for Whisper/LLM); NMS and box-decode are plain numpy in the tool. Detections come back as `name (conf, left|center|right[, distance m])`.

- **Depth / distance:** the camera is an **Intel RealSense D455** (`8086:0b5c`). `_capture_realsense()` (pyrealsense2) grabs colour + depth *aligned to the colour lens* via `rs.align`, so distance is read straight from the depth map at each YOLO box centre (median of a small patch, ignoring zero/invalid pixels; depth scale ≈ 1 mm/unit). This is the whole reason detection runs on the colour node — depth and colour are two physically-offset lenses, and alignment is what makes "read depth at the box centre" correct.
- **Fallback:** if pyrealsense2/the RealSense isn't usable, `_capture_ffmpeg()` grabs a plain colour frame instead — same detection, **no distance**. Node layout: `/dev/video0` is the `Z16` depth stream, the RGB node is `/dev/video4` (`YUYV`); `_detect_camera()` auto-picks the first colour node (`MJPG`/`YUYV`), skipping depth/IR. Override with `SEE_CAMERA_DEVICE=/dev/videoN`.
- **Model:** `models/yolov8n.onnx` is git-ignored (not committed). It was produced once by exporting stock `yolov8n.pt` with `ultralytics` in a throwaway venv — the project venv never gets torch. To regenerate it, see the vision section in INSTALL.md.
- Runs entirely on CPU, so — unlike the Whisper entry points — it needs **no** CUDA preload block.

### GPU compatibility

Package versions and model sizes (`large-v3`, `qwen3:30b`) are tied to the current RTX 5090 (32 GB VRAM, Blackwell/sm_120 — needs `ctranslate2 >= 4.6` for its CUDA 12.8 kernels). The "GPU compatibility" section at the top of INSTALL.md records the verified versions — consult it before touching CUDA packages or model choices.
