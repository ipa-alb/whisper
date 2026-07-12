# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A voice-controlled local AI assistant: record speech → transcribe with faster-whisper (GPU) → send to a local LLM (Ollama, `qwen3:30b`) → the LLM answers or calls Python tools. Everything runs on-device; nothing binds to a public interface, and audio/prompts never leave the machine.

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

### GPU compatibility

Package versions and model sizes (`large-v3`, `qwen3:30b`) are tied to the current RTX 5090 (32 GB VRAM, Blackwell/sm_120 — needs `ctranslate2 >= 4.6` for its CUDA 12.8 kernels). The "GPU compatibility" section at the top of INSTALL.md records the verified versions — consult it before touching CUDA packages or model choices.
