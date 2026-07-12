# Voice-Controlled Local AI Pipeline

A push-to-talk speech interface that transcribes voice commands locally and feeds
them to a local LLM, which can execute Python scripts in a workspace.

## Scope (agreed 2026-07-11)

- **Runs entirely on the local machine** — currently RTX 5090 (32 GB VRAM);
  developed on an RTX 4080 Laptop (12 GB), migrated 2026-07-11. Proof of
  concept first, optimize later.
- **Interaction:** push-to-talk via global hotkey (X11). Press → speak → release →
  transcript goes to the LLM.
- **Language:** mostly English (Whisper pinned to `language=en` for speed/accuracy).
- **Output:** text only for now. TTS (e.g. Piper) is a possible later stage.
- **Execution model:** hybrid with confirmation —
  - *Predefined tools:* Python scripts in the workspace `tools/` folder, exposed to
    the LLM as callable tools. Run without confirmation.
  - *Generated code:* the LLM may write ad-hoc Python; it is shown to the user and
    only executed after explicit confirmation.
- **Future direction:** possibly expose the tool layer as an MCP server, so other
  clients (Claude, IDEs) can use the same tools. Tools are therefore defined as
  clean, self-describing functions from day one.

## Architecture

```
┌──────────────┐   audio    ┌────────────────┐   text    ┌──────────────┐
│ Push-to-talk │ ─────────► │ faster-whisper │ ────────► │  Local LLM   │
│  (hotkey +   │            │  (GPU, en)     │           │  via Ollama  │
│  mic capture)│            └────────────────┘           └──────┬───────┘
└──────────────┘                                                │ tool calls
                                                                ▼
                                                    ┌───────────────────────┐
                                                    │    Execution layer    │
                                                    │  tools/*.py  → free   │
                                                    │  generated → confirm  │
                                                    └───────────────────────┘
```

## Component choices

| Component      | Choice           | Why                                                            |
| -------------- | ---------------- | -------------------------------------------------------------- |
| Transcription  | faster-whisper   | ~4x faster than openai/whisper, far less VRAM, streaming-ready |
| Whisper model  | `large-v3` on the 5090 (was `small.en` on the 4080) | Best accuracy; still fast on 32 GB-class hardware |
| LLM runtime    | Ollama           | Simplest local serving, native tool-calling API, model swap is one command |
| LLM model      | `qwen3:30b` (MoE, 3B active) on the 5090 (was `qwen3:8b`) | Near-32B quality at small-model latency — voice needs fast responses |
| Audio capture  | sounddevice      | Simple PortAudio bindings, already in the original plan        |
| Hotkey         | pynput (X11)     | Global hotkey listener without desktop-environment plugins     |
| Object detection | `yolov8n.onnx` via onnxruntime (CPU) | Lightweight "what do you see" tool; no PyTorch/OpenCV at runtime — onnxruntime does inference |
| Depth / distance | Intel RealSense D455 + pyrealsense2 | Depth aligned to colour lens → per-object distance in metres; falls back to ffmpeg colour-only if absent |

VRAM budget on the 5090 (32 GB): Whisper large-v3 ≈ 3 GB + qwen3:30b quantized
≈ 19 GB → comfortable headroom. (On the 4080 it was small.en ≈ 1 GB + qwen3:8b
≈ 5-6 GB on 12 GB.)

## Roadmap

- [x] **Phase 1 — Transcription PoC:** mic capture + faster-whisper → print
      transcript to terminal (`listen.py`; hotkey replaced by record-on-launch,
      Enter ends the utterance).
- [x] **Phase 2 — LLM loop:** Ollama + `qwen3:8b`, transcript → response
      (`assistant.py`).
- [x] **Phase 3 — Tool execution:** `tools/` scripts with self-describing
      schemas (list_files, read_file, write_note), Ollama tool calling, and the
      confirm-before-run path for generated code (run_python).
- [x] **Vision tool (`look`):** single-snapshot YOLO object detection
      (`tools/see_camera.py`, `yolov8n.onnx` on onnxruntime/CPU) with **per-object
      distance** from the RealSense D455 depth stream (aligned to the colour lens).
      The user can ask "what do you see?", "is there a person?", or "how far away
      is it?" and the LLM calls it. Verified end-to-end: person/bottle/chair
      detected with correct positions and plausible distances (~0.3–0.8 m).
- [ ] **Phase 4 — Workspace instructions:** `.md` instruction file describing the
      available tools and conventions (deferred until tools exist).
- [x] **5090 migration (2026-07-11):** environment rebuilt on the RTX 5090,
      models bumped to `large-v3` + `qwen3:30b` — config change only, as planned.
- [ ] **Later:** MCP server exposure, TTS responses, wake word.

## Files

- `PROJECT.md` — this file, the source of truth for scope.
- `INSTALL.md` — setup guide for this pipeline (faster-whisper, sounddevice,
  pynput, Ollama). Replaced the original generic Whisper+Gradio tutorial.
