# Whisper Voice Pipeline

A voice-controlled local AI assistant: hold a hotkey, speak a command, and a
local LLM executes Python scripts in this workspace. Everything runs on-device —
no cloud, no network exposure.

```
Push-to-talk → faster-whisper (GPU) → local LLM (Ollama) → Python tool execution
```

## Documentation

- **[PROJECT.md](PROJECT.md)** — scope, architecture, component choices, roadmap.
  The source of truth for what this project is and why.
- **[INSTALL.md](INSTALL.md)** — step-by-step setup guide with per-stage smoke tests.

## Status

Proof-of-concept phase. Current roadmap position: **Phase 1 — Transcription PoC**
(see [PROJECT.md](PROJECT.md#roadmap)).

Migrated to the RTX 5090 (Dell Pro Max Tower) on 2026-07-11 and verified by
voice end-to-end on 2026-07-12 (see the GPU compatibility section in
[INSTALL.md](INSTALL.md)).

| Phase | Goal | Status |
| ----- | ---- | ------ |
| 1 | Record → transcript in terminal (`listen.py`) | ✅ done, verified by voice |
| 2 | Transcript → local LLM chat loop (`assistant.py`) | ✅ done |
| 3 | LLM tool calling + confirm-before-run for generated code | ✅ first version (4 tools) |
| 4 | Workspace instructions file for the toolset | pending |

## Quick start

```bash
source .venv/bin/activate
python3 assistant.py                 # full pipeline: voice → LLM → tools
python3 assistant.py --text "..."    # same pipeline with typed input (testing)
python3 listen.py                    # transcription only (Phase 1)
```

Recording starts on launch; speak, press Enter, and the transcript goes to the
LLM. Ask it to *do* something ("list the files", "write down a note that...")
and it calls the matching tool from [tools/](tools/); anything it wants to run
as ad-hoc Python is shown to you first and needs a `y` to execute.

See [INSTALL.md](INSTALL.md) if the environment isn't set up yet.

## Hardware

Runs on an **RTX 5090 (32 GB VRAM)** with Whisper `large-v3` and `qwen3:30b`.
Originally developed on an RTX 4080 Laptop (12 GB) with `small.en` + `qwen3:8b`;
the pipeline design is unchanged across cards — only CUDA packages and model
sizes differ. See the **GPU compatibility** section at the top of
[INSTALL.md](INSTALL.md) before switching cards.

## Principles

- **Local only** — audio and prompts never leave the machine; nothing binds to
  a public interface.
- **Confirmation for generated code** — predefined tools run freely, LLM-written
  code is shown and requires explicit approval before execution.
- **Testable stages** — every pipeline component (mic, Whisper, LLM, execution)
  can be verified independently.
