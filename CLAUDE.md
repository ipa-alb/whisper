# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A voice-controlled **local** AI assistant: record voice → transcribe on the GPU
with faster-whisper → send the text to a local LLM (Ollama / `qwen3:8b`) → the
LLM executes Python tools in this workspace. Everything runs on-device; nothing
binds to a network interface and audio never leaves the machine.

This directory is its **own git repository**, independent of the parent
`unitree_g1_ros2_driver_wireless` ROS2 workspace it sits inside — it is not a
ROS2 package and shares nothing with the G1 driver code. Treat it as a
standalone Python proof-of-concept.

`PROJECT.md` is the source of truth for scope, component choices, and the phased
roadmap. `INSTALL.md` is the setup guide with per-stage smoke tests. Read
`PROJECT.md` before making design decisions.

## Pipeline

```
record-on-launch → faster-whisper (GPU, en) → qwen3:8b via Ollama → tools/*.py
```

`assistant.py` is the full pipeline; `listen.py` is the Phase-1 transcription-only
PoC (mic → transcript, no LLM). Recording begins at launch — speak, press
**Enter** to end the utterance and send it, `Ctrl+C` to quit.

## Commands

```bash
source .venv/bin/activate            # always work inside the venv

python3 assistant.py                 # full pipeline: voice → LLM → tools
python3 assistant.py --text "..."    # one-shot typed input — no mic/Whisper, for testing tool logic
python3 listen.py                    # transcription only (Phase 1)
```

There is no test suite, linter config, or build step. `--text` is the fastest way
to exercise the LLM/tool path without a microphone or GPU-loaded Whisper.

Environment setup (only when the venv is missing) is in `INSTALL.md`:
system deps (`ffmpeg`, `portaudio19-dev`) → venv → `faster-whisper` +
`nvidia-cublas-cu12 nvidia-cudnn-cu12` → `sounddevice numpy pynput ollama` →
`ollama pull qwen3:8b`.

## Architecture Notes

**cuDNN/cuBLAS preload.** `assistant.py` and `listen.py` both begin with a loop
that `ctypes.CDLL`-loads every `.so` under the venv's `nvidia/*/lib` directories
*before* importing `faster_whisper`. The pip-installed CUDA libs are not on the
system loader path, so faster-whisper fails to init CUDA without this. Keep this
block first in any new entry-point script.

**Tool plugin contract.** `load_tools()` in `assistant.py` imports every
`tools/*.py` (skipping `_`-prefixed files) and expects each module to export:
- `TOOL` — an Ollama/OpenAI-style function schema (`{"type": "function", "function": {...}}`)
- `run(**kwargs)` — the callable; its return value (stringified) is fed back to the LLM as the tool result

`TOOL["function"]["name"]` is the registry key and must match the schema. To add
a capability, drop a new file in `tools/` following this contract — no
registration elsewhere. Tools return error strings (e.g. `"error: ..."`) rather
than raising; `handle_utterance` also catches exceptions so a tool bug can't kill
the session.

**Execution model (the core safety design).** Predefined `tools/*.py` run freely.
The one exception is `run_python`, which executes LLM-generated ad-hoc Python: it
prints the code and requires an interactive `y` confirmation before running (in a
subprocess, 30s timeout, cwd = workspace). This confirm-before-run split is a
deliberate project principle — preserve it. Path-taking tools (`read_file`,
`list_files`) guard against escaping the workspace root via realpath checks; keep
that guard in any new file-touching tool.

**LLM loop.** `handle_utterance` runs up to `MAX_TOOL_ROUNDS` (5) chat rounds,
appending each tool result and re-querying until the model stops calling tools or
the cap is hit. `think=False` is set on `ollama.chat`. `messages` persists across
utterances within a session (conversational memory).

## Hardware / Migration

Written and verified on an **RTX 4080 Laptop (12 GB VRAM, Ada, CUDA 12 wheels)**.
Model choices (`small.en` Whisper, `qwen3:8b`) and the `nvidia-*-cu12` packages
are tied to this card. An **RTX 5090 (Blackwell, sm_120)** migration is planned
and needs CUDA 12.8+ — see the *GPU compatibility* section at the top of
`INSTALL.md` before switching cards, then scale models up (`large-v3`, 14–32B LLM)
and re-run the Step 6 smoke tests.

## Conventions

- Constants live at the top of each script (`SAMPLE_RATE=16000`, `WHISPER_MODEL`,
  `LLM_MODEL`, `MAX_TOOL_ROUNDS`). Change models there.
- Whisper is pinned to `language="en"` with `vad_filter=True` for speed/accuracy.
- `notes/`, `*.wav`, `error_logs.txt`, and `.venv/` are gitignored; `write_note`
  writes into the ignored `notes/` folder.
- Tools are written as clean, self-describing functions on purpose — a possible
  future direction is exposing them as an MCP server (see `PROJECT.md`).
