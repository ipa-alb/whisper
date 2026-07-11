# Setup — Voice-Controlled Local AI Pipeline

Installation guide for the pipeline described in [PROJECT.md](PROJECT.md):
push-to-talk → faster-whisper (GPU) → local LLM (Ollama) → tool execution.

## GPU compatibility — read before installing

> **This guide was written and verified on an NVIDIA RTX 4080 Laptop GPU
> (12 GB VRAM, Ada Lovelace architecture, driver 595.71, CUDA 12 wheels).**
> All package choices below (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`,
> Whisper/LLM model sizes) are tied to this card.
>
> **If the GPU is switched — e.g. the planned RTX 5090 migration — this file
> must be revisited:**
>
> - **RTX 5090 (Blackwell, sm_120)** requires CUDA 12.8+ kernel support.
>   Upgrade `ctranslate2`, `faster-whisper`, the `nvidia-*-cu12` wheels, and
>   Ollama to their latest versions, and make sure the NVIDIA driver is 570+.
> - **Model sizes should be scaled up** with 32 GB VRAM: Whisper `large-v3`
>   instead of `small.en`, and a 14–32B LLM instead of `qwen3:8b`
>   (see the VRAM notes in [PROJECT.md](PROJECT.md)).
> - **Re-run all smoke tests in Step 6** after the swap before trusting the
>   pipeline.
>
> After a successful migration, update this section with the new card's data.

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
ollama pull qwen3:8b
```

Ollama runs as a local service on `localhost:11434`. The Python client:

```bash
pip install ollama
```

Model note: an 8B-class model quantized (~5–6 GB VRAM) fits comfortably next to
Whisper `small.en` (~1 GB) on 12 GB. After the 5090 migration, swap models with
a single `ollama pull` — no pipeline changes.

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
m = WhisperModel("small.en", device="cuda", compute_type="float16")
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
ollama run qwen3:8b "Say hello in one sentence."
```

## Principles (from the scope discussion)

- **Nothing is exposed to the network.** No web UI, no public share links.
  Everything binds to localhost or runs in-process.
- **Whisper model:** `small.en` to start (English-only, fast). Bump to
  `distil-large-v3` or `large-v3` later if accuracy demands it.
- **Everything local:** audio never leaves the machine.

## What changed vs. the original guide

The first version of this file was a generic "Whisper + Gradio web UI on a GPU
server" tutorial. It was dropped because: Gradio's upload/record web page does
not fit a hands-free push-to-talk pipeline, `openai/whisper` + PyTorch is slow
and VRAM-heavy compared to faster-whisper, and `share=True` publicly exposed
the interface. See git history / PROJECT.md for the full rationale.
