# Setup — Voice-Controlled Local AI Pipeline

Installation guide for the pipeline described in [PROJECT.md](PROJECT.md):
push-to-talk → faster-whisper (GPU) → local LLM (Ollama) → tool execution.

## GPU compatibility — read before installing

> **This guide was written and verified on an NVIDIA RTX 5090
> (32 GB VRAM, Blackwell architecture, sm_120, driver 595.71, CUDA 12 wheels).**
> All package choices below (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`,
> Whisper/LLM model sizes) are tied to this card.
>
> Migrated from the original RTX 4080 Laptop (12 GB) on 2026-07-11. Versions
> verified on the 5090: `ctranslate2 4.8.1`, `faster-whisper 1.2.1`,
> `nvidia-cudnn-cu12 9.24`, `nvidia-cublas-cu12 12.9`, Ollama 0.31.2.
> Blackwell needs CUDA 12.8+ kernels — `ctranslate2 >= 4.6` ships them; older
> wheels (and Ollama builds older than ~0.5.x) fail on sm_120.
>
> Model sizes are scaled for 32 GB VRAM: Whisper `large-v3` (~3 GB) and
> `qwen3:30b` (MoE, ~19 GB) instead of the 4080's `small.en` + `qwen3:8b`
> (see the VRAM notes in [PROJECT.md](PROJECT.md)).
>
> **If the GPU is switched again:** revisit every package/model choice in this
> file, re-run all smoke tests in Step 6 before trusting the pipeline, then
> update this section with the new card's data.

Target machine: local Ubuntu desktop with the GPU described above,
NVIDIA driver installed (verify with `nvidia-smi`).

## Step 1: System dependencies

```bash
sudo apt install -y ffmpeg portaudio19-dev python3 python3-venv python3-pip
```

- `ffmpeg` — audio decoding/resampling
- `portaudio19-dev` — required to build `sounddevice` (microphone access)

## Step 2: Python virtual environment

```bash
cd ~/workspaces/whisper
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## Step 3: Transcription stack (no PyTorch needed)

We use **faster-whisper** (CTranslate2-based) instead of the original
`openai/whisper`: ~4x faster, much lower VRAM, and no PyTorch dependency.

```bash
pip install faster-whisper
```

GPU execution needs cuBLAS and cuDNN 9 for CUDA 12 — install them as pip
packages so nothing touches the system CUDA setup:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## Step 4: Audio capture + hotkey

```bash
pip install sounddevice numpy pynput
```

- `sounddevice` — records from the microphone via PortAudio
- `pynput` — global push-to-talk hotkey listener (works on X11)

## Step 5: Local LLM runtime (Ollama)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:30b
```

The install script needs sudo (it sets up a systemd service). Without root, the
release tarball works too — extract to `~/.local` and run the server manually:

```bash
curl -fsSL https://github.com/ollama/ollama/releases/download/v0.31.2/ollama-linux-amd64.tar.zst \
  | tar --zstd -x -C ~/.local
~/.local/bin/ollama serve &   # must be running whenever the pipeline is used
```

Ollama runs as a local service on `localhost:11434`. The Python client:

```bash
pip install ollama
```

Model note: `qwen3:30b` quantized (~19 GB VRAM) fits comfortably next to
Whisper `large-v3` (~3 GB) on 32 GB. Swapping models is a single `ollama pull`
plus updating `LLM_MODEL` in `assistant.py` — no pipeline changes.

## Step 6: Smoke tests

Each stage can be verified independently before wiring them together.

**GPU + Whisper:**

The cuBLAS/cuDNN libraries installed via pip are not on the system loader path,
so they must be preloaded before faster-whisper initializes CUDA (the pipeline
code does this at startup too):

```bash
python3 - <<'EOF'
import os, glob, ctypes
for d in glob.glob(".venv/lib/python3*/site-packages/nvidia/*/lib"):
    for so in glob.glob(os.path.join(d, "*.so*")):
        try: ctypes.CDLL(so)
        except OSError: pass

from faster_whisper import WhisperModel
m = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Whisper model loaded on GPU OK")
EOF
```

**Microphone:**

```bash
python3 -c "
import sounddevice as sd
print(sd.query_devices())
"
```

**LLM:**

```bash
ollama run qwen3:30b "Say hello in one sentence."
```

## Principles (from the scope discussion)

- **Nothing is exposed to the network.** No web UI, no public share links.
  Everything binds to localhost or runs in-process.
- **Whisper model:** `large-v3` on the 5090 (`small.en` was the 12 GB-era
  choice; drop back to it or `distil-large-v3` on smaller cards).
- **Everything local:** audio never leaves the machine.

## What changed vs. the original guide

The first version of this file was a generic "Whisper + Gradio web UI on a GPU
server" tutorial. It was dropped because: Gradio's upload/record web page does
not fit a hands-free push-to-talk pipeline, `openai/whisper` + PyTorch is slow
and VRAM-heavy compared to faster-whisper, and `share=True` publicly exposed
the interface. See git history / PROJECT.md for the full rationale.
