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

Environment fully installed and smoke-tested on 2026-07-11: Whisper `small.en`
loads on the GPU and transcribes, the microphone is reachable via PipeWire, and
Ollama answers with `qwen3:8b`. What's missing for Phase 1 is the push-to-talk
script itself.

| Phase | Goal | Status |
| ----- | ---- | ------ |
| 1 | Push-to-talk → transcript in terminal | environment ✅ verified, script pending |
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

Developed and verified on an **RTX 4080 Laptop GPU (12 GB VRAM)**. A migration
to an RTX 5090 is planned — the pipeline design stays the same, but the CUDA
packages and model sizes must be updated; see the
**GPU compatibility** section at the top of [INSTALL.md](INSTALL.md) before
switching cards.

## Principles

- **Local only** — audio and prompts never leave the machine; nothing binds to
  a public interface.
- **Confirmation for generated code** — predefined tools run freely, LLM-written
  code is shown and requires explicit approval before execution.
- **Testable stages** — every pipeline component (mic, Whisper, LLM, execution)
  can be verified independently.
