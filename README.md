# Whisper Voice Pipeline

A voice-controlled local AI assistant: speak a command, press Enter, and a
local LLM executes Python scripts in this workspace. Everything runs on-device —
no cloud, no network exposure.

```
Speak (Enter to send) → faster-whisper (GPU) → local LLM (Ollama) → Python tool execution
```

## Documentation

- **[PROJECT.md](PROJECT.md)** — scope, architecture, component choices, roadmap.
  The source of truth for what this project is and why.
- **[INSTALL.md](INSTALL.md)** — step-by-step setup guide with per-stage smoke tests.

## Status

Proof-of-concept phase. Phases 1–3 are done; next up is **Phase 4 — workspace
instructions file** (see [PROJECT.md](PROJECT.md#roadmap)).

Migrated to the RTX 5090 (Dell Pro Max Tower) on 2026-07-11 and verified by
voice end-to-end on 2026-07-12 (see the GPU compatibility section in
[INSTALL.md](INSTALL.md)).

| Phase | Goal | Status |
| ----- | ---- | ------ |
| 1 | Record → transcript in terminal (`listen.py`) | ✅ done, verified by voice |
| 2 | Transcript → local LLM chat loop (`assistant.py`) | ✅ done |
| 3 | LLM tool calling + confirm-before-run for generated code | ✅ first version (4 tools) |
| — | Vision tool: camera snapshot → YOLO detection + depth distance (`look`) | ✅ done, verified (person/bottle/chair + metres) |
| 4 | Workspace instructions file for the toolset | pending |

## Quick start

```bash
source .venv/bin/activate
python3 assistant.py                 # full pipeline: voice → LLM → tools
python3 assistant.py --text "..."    # same pipeline with typed input (testing)
python3 listen.py                    # transcription only (Phase 1)
python3 mic_check.py                 # live level meter — find the right mic
```

Ollama must be serving on `localhost:11434`. If it was installed from the
release tarball instead of the install script (no systemd service), start it
manually after a reboot: `~/.local/bin/ollama serve &`.

Recording starts on launch; speak, press Enter, and the transcript goes to the
LLM. Ask it to *do* something ("list the files", "write down a note that...",
"look at the camera — is there a person?") and it calls the matching tool from
[tools/](tools/); anything it wants to run as ad-hoc Python is shown to you
first and needs a `y` to execute.

The `look` tool takes one camera snapshot and runs a lightweight YOLO
(`yolov8n.onnx` on onnxruntime/CPU — no PyTorch) to report what it sees. On the
Intel RealSense D455 it also reports **how far away** each object is, by reading
the depth stream aligned to the colour lens (e.g. *"person, center, 0.8 m"*). It
needs `models/yolov8n.onnx`; see the vision section in [INSTALL.md](INSTALL.md)
to create it.

See [INSTALL.md](INSTALL.md) if the environment isn't set up yet.

If speech keeps coming back as "(nothing recognized)", run `mic_check.py` and
speak — whichever bar moves is the live microphone. The pipeline records from
the system **default** source, so switch it to that device (sound settings or
`pactl set-default-source ...`).

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
