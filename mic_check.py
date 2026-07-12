#!/usr/bin/env python3
"""Diagnostic: live level meter for every input device.

Run it, speak normally for a few seconds, and watch which bar jumps.
That device is your working microphone. Ctrl+C to quit.
"""

import sys
import time

import numpy as np
import sounddevice as sd

BAR_WIDTH = 40
GAIN = 25  # visual amplification so quiet mics still show movement


def main():
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        if dev["name"] in ("pipewire", "pulse"):  # aggregates, skip
            continue
        devices.append((idx, dev))

    peaks = {idx: 0.0 for idx, _ in devices}
    streams = []
    for idx, dev in devices:
        def cb(indata, frames, t, status, idx=idx):
            peaks[idx] = float(np.abs(indata).max())

        try:
            s = sd.InputStream(
                device=idx,
                channels=1,
                samplerate=int(dev["default_samplerate"]),
                dtype="float32",
                callback=cb,
            )
            s.start()
            streams.append(s)
        except Exception as e:
            peaks[idx] = -1.0
            print(f"[{idx}] {dev['name']}: cannot open ({e})", file=sys.stderr)

    print("Speak now — watch which bar moves. Ctrl+C to quit.\n")
    try:
        while True:
            lines = []
            for idx, dev in devices:
                p = peaks[idx]
                if p < 0:
                    bar, val = "(unavailable)".ljust(BAR_WIDTH), "  --  "
                else:
                    filled = min(BAR_WIDTH, int(p * GAIN * BAR_WIDTH))
                    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
                    val = f"{p:.3f}"
                name = ("* " if idx == sd.default.device[0] else "  ") + dev["name"][:28]
                lines.append(f"[{idx:2d}] {name:<30} |{bar}| {val}")
            sys.stdout.write("\x1b[2K" + ("\x1b[F\x1b[2K" * (len(lines) - 1)) if lines else "")
            sys.stdout.write("\r" + "\n".join(lines))
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        for s in streams:
            s.stop(); s.close()


if __name__ == "__main__":
    main()
