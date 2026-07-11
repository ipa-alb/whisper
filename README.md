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

| Phase | Goal | Status |
| ----- | ---- | ------ |
| 1 | Push-to-talk → transcript in terminal | in progress |
| 2 | Transcript → local LLM chat loop | pending |
| 3 | LLM tool calling + confirm-before-run for generated code | pending |
| 4 | Workspace instructions file for the toolset | pending |

## Quick start

```bash
source .venv/bin/activate
# (once Phase 1 lands:)
python3 listen.py
```

See [INSTALL.md](INSTALL.md) if the environment isn't set up yet.

## Hardware

Developed on an RTX 4080 Laptop GPU (12 GB VRAM); planned migration to an
RTX 5090 changes only the model sizes, not the pipeline.

## Principles

- **Local only** — audio and prompts never leave the machine; nothing binds to
  a public interface.
- **Confirmation for generated code** — predefined tools run freely, LLM-written
  code is shown and requires explicit approval before execution.
- **Testable stages** — every pipeline component (mic, Whisper, LLM, execution)
  can be verified independently.
