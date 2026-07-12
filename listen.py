#!/usr/bin/env python3
"""Phase 1 — transcription PoC.

Recording starts as soon as the script launches. Press Enter to stop the
current recording and transcribe it; recording resumes automatically.
Quit with Ctrl+C.

Run inside the venv:  python3 listen.py
"""

import ctypes
import glob
import os
import sys
import threading

# The pip-installed cuBLAS/cuDNN libraries are not on the system loader path;
# they must be loaded before faster-whisper initializes CUDA (see INSTALL.md).
for _d in glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "nvidia", "*", "lib")):
    for _so in glob.glob(os.path.join(_d, "*.so*")):
        try:
            ctypes.CDLL(_so)
        except OSError:
            pass

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # what Whisper expects
MODEL_NAME = "large-v3"

frames: list[np.ndarray] = []
frames_lock = threading.Lock()


def audio_callback(indata, frame_count, time_info, status):
    if status:
        print(f"[audio warning] {status}", file=sys.stderr)
    with frames_lock:
        frames.append(indata.copy())


def main():
    from faster_whisper import WhisperModel

    print(f"Loading Whisper {MODEL_NAME} on GPU...", flush=True)
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")

    mic = sd.query_devices(kind="input")
    print(f"Microphone: {mic['name']}")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback
    )

    with stream:
        while True:
            with frames_lock:
                frames.clear()
            print("\n🎤 recording — press Enter to transcribe (Ctrl+C to quit)...", flush=True)
            input()
            with frames_lock:
                if not frames:
                    print("📝 (no audio captured)")
                    continue
                audio = np.concatenate(frames)[:, 0]
            duration = len(audio) / SAMPLE_RATE
            print(f"⏳ transcribing {duration:.1f}s of audio...", flush=True)
            segments, _info = model.transcribe(audio, language="en", vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            print(f"📝 {text}" if text else "📝 (nothing recognized)")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
