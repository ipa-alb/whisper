# Setup — Voice-Controlled Local AI Pipeline

Installation guide for the pipeline described in [PROJECT.md](PROJECT.md):
push-to-talk → faster-whisper (GPU) → local LLM (Ollama) → tool execution.

Target machine: local Ubuntu desktop, NVIDIA GPU (RTX 4080 Laptop, 12 GB VRAM),
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

```bash
python3 -c "
from faster_whisper import WhisperModel
m = WhisperModel('small.en', device='cuda', compute_type='float16')
print('Whisper model loaded on GPU OK')
"
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
