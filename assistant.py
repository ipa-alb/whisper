#!/usr/bin/env python3
"""Phases 2+3 — voice assistant pipeline.

speech → faster-whisper (GPU) → qwen3:8b (Ollama) → tool execution

Recording starts on launch. Press Enter to stop the current utterance; the
transcript goes to the LLM, which either answers in text or calls tools from
the tools/ folder. LLM-generated code (run_python) always asks for
confirmation. Quit with Ctrl+C.

Usage:
    python3 assistant.py                  # voice mode
    python3 assistant.py --text "..."     # one-shot typed input (no mic/whisper)
"""

import ctypes
import glob
import importlib.util
import json
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

import ollama

SAMPLE_RATE = 16000
WHISPER_MODEL = "small.en"
LLM_MODEL = "qwen3:8b"
MAX_TOOL_ROUNDS = 5
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(WORKSPACE, "tools")

SYSTEM_PROMPT = f"""You are a voice-controlled assistant operating inside the workspace directory {WORKSPACE}.
The user's speech is transcribed and sent to you, so expect occasional transcription errors and interpret the intent generously.

- If the user asks you to DO or OPERATE something, call the appropriate tool.
- Otherwise just answer briefly (1-3 sentences, plain text for a terminal — no markdown).
- Use run_python only when no predefined tool covers the request; the user must confirm that code before it runs.
- Never invent tool results; report errors honestly."""


def load_tools():
    """Import every tools/*.py module exposing TOOL (schema) and run (callable)."""
    registry, schemas = {}, []
    for path in sorted(glob.glob(os.path.join(TOOLS_DIR, "*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"tools.{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        registry[mod.TOOL["function"]["name"]] = mod.run
        schemas.append(mod.TOOL)
    return registry, schemas


def handle_utterance(text, messages, registry, schemas):
    """Send one user utterance through the LLM, executing tool calls as they come."""
    messages.append({"role": "user", "content": text})
    for _ in range(MAX_TOOL_ROUNDS):
        response = ollama.chat(model=LLM_MODEL, messages=messages, tools=schemas, think=False)
        msg = response.message
        messages.append(msg)
        if not msg.tool_calls:
            return (msg.content or "").strip()
        for call in msg.tool_calls:
            name = call.function.name
            args = dict(call.function.arguments or {})
            print(f"🔧 {name}({json.dumps(args, ensure_ascii=False)[:200]})")
            fn = registry.get(name)
            if fn is None:
                result = f"error: unknown tool '{name}'"
            else:
                try:
                    result = fn(**args)
                except Exception as e:  # tool bugs must not kill the session
                    result = f"error: {e}"
            messages.append({"role": "tool", "content": str(result)})
    return "(stopped: too many tool rounds in a row)"


def record_utterance(frames, frames_lock):
    """Recording already runs; wait for Enter, then return the captured audio."""
    import numpy as np

    with frames_lock:
        frames.clear()
    print("\n🎤 recording — press Enter to send (Ctrl+C to quit)...", flush=True)
    input()
    with frames_lock:
        if not frames:
            return None
        return np.concatenate(frames)[:, 0]


def main():
    registry, schemas = load_tools()
    print(f"Tools loaded: {', '.join(registry)}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if len(sys.argv) > 2 and sys.argv[1] == "--text":
        reply = handle_utterance(sys.argv[2], messages, registry, schemas)
        print(f"\n💬 {reply}")
        return

    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel

    print(f"Loading Whisper {WHISPER_MODEL} on GPU...", flush=True)
    whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print(f"Microphone: {sd.query_devices(kind='input')['name']}")

    frames, frames_lock = [], threading.Lock()

    def audio_callback(indata, frame_count, time_info, status):
        if status:
            print(f"[audio warning] {status}", file=sys.stderr)
        with frames_lock:
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback
    )
    with stream:
        while True:
            audio = record_utterance(frames, frames_lock)
            if audio is None:
                print("📝 (no audio captured)")
                continue
            segments, _ = whisper.transcribe(audio, language="en", vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            if not text:
                print("📝 (nothing recognized)")
                continue
            print(f"📝 you said: {text}")
            reply = handle_utterance(text, messages, registry, schemas)
            print(f"💬 {reply}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
