#!/usr/bin/env python3
"""Phase 1 — push-to-talk transcription PoC.

Hold F9, speak, release: the recording is transcribed by faster-whisper on the
GPU and printed to the terminal. Quit with Ctrl+C in this terminal.

Run inside the venv:  python3 listen.py
"""

import ctypes
import glob
import os
import queue
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
from pynput import keyboard

SAMPLE_RATE = 16000  # what Whisper expects
PTT_KEY = keyboard.Key.f9
MODEL_NAME = "small.en"

recording = threading.Event()
frames: list[np.ndarray] = []
transcribe_queue: queue.Queue = queue.Queue()
# X11 auto-repeat fires press/release pairs while a key is held; a released
# key only counts if no new press arrives within this window.
release_timer: threading.Timer | None = None


def audio_callback(indata, frame_count, time_info, status):
    if status:
        print(f"[audio warning] {status}", file=sys.stderr)
    if recording.is_set():
        frames.append(indata.copy())


def stop_recording():
    recording.clear()
    if frames:
        transcribe_queue.put(np.concatenate(frames)[:, 0])


def on_press(key):
    global release_timer
    if key == PTT_KEY:
        if release_timer is not None:
            release_timer.cancel()
            release_timer = None
        if not recording.is_set():
            frames.clear()
            recording.set()
            print("\n🎤 recording — release F9 when done...", flush=True)


def on_release(key):
    global release_timer
    if key == PTT_KEY and recording.is_set():
        release_timer = threading.Timer(0.15, stop_recording)
        release_timer.start()


def main():
    from faster_whisper import WhisperModel

    print(f"Loading Whisper {MODEL_NAME} on GPU...", flush=True)
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")

    mic = sd.query_devices(kind="input")
    print(f"Microphone: {mic['name']}")
    print("Ready. Hold F9 to talk, release to transcribe. Ctrl+C here to quit.")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback
    )
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)

    with stream, listener:
        while True:
            audio = transcribe_queue.get()
            duration = len(audio) / SAMPLE_RATE
            print(f"⏳ transcribing {duration:.1f}s of audio...", flush=True)
            segments, _info = model.transcribe(audio, language="en", vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            print(f"📝 {text}" if text else "📝 (nothing recognized)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
