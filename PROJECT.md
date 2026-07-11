# Voice-Controlled Local AI Pipeline

A push-to-talk speech interface that transcribes voice commands locally and feeds
them to a local LLM, which can execute Python scripts in a workspace.

## Scope (agreed 2026-07-11)

- **Runs entirely on the local machine** — currently RTX 4080 Laptop (12 GB VRAM),
  planned migration to RTX 5090. Proof of concept first, optimize later.
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
| Whisper model  | `small.en` or `distil-small.en` to start | Fast, accurate enough for commands; easy to bump up |
| LLM runtime    | Ollama           | Simplest local serving, native tool-calling API, model swap is one command |
| LLM model      | 8B-class (e.g. Qwen3 8B) | Fits in ~5-6 GB quantized alongside Whisper on 12 GB; revisit on 5090 |
| Audio capture  | sounddevice      | Simple PortAudio bindings, already in the original plan        |
| Hotkey         | pynput (X11)     | Global hotkey listener without desktop-environment plugins     |

VRAM budget on the 4080 (12 GB): Whisper small ≈ 1 GB + 8B LLM quantized ≈ 5-6 GB
→ comfortable headroom. On the 5090 both can scale up (large-v3 + 14-32B model).

## Roadmap

- [x] **Phase 1 — Transcription PoC:** mic capture + faster-whisper → print
      transcript to terminal (`listen.py`; hotkey replaced by record-on-launch,
      Enter ends the utterance).
- [x] **Phase 2 — LLM loop:** Ollama + `qwen3:8b`, transcript → response
      (`assistant.py`).
- [x] **Phase 3 — Tool execution:** `tools/` scripts with self-describing
      schemas (list_files, read_file, write_note), Ollama tool calling, and the
      confirm-before-run path for generated code (run_python).
- [ ] **Phase 4 — Workspace instructions:** `.md` instruction file describing the
      available tools and conventions (deferred until tools exist).
- [ ] **Later:** MCP server exposure, TTS responses, wake word, 5090 migration
      (bigger models, config change only).

## Files

- `PROJECT.md` — this file, the source of truth for scope.
- `INSTALL.md` — setup guide for this pipeline (faster-whisper, sounddevice,
  pynput, Ollama). Replaced the original generic Whisper+Gradio tutorial.
