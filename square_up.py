#!/usr/bin/env python3
"""Live orientation readout for squaring the G1 up to an ArUco marker.

Loops: snap the head cam, print the marker's lateral offset (X) and yaw with an
aim hint, until you Ctrl+C. Nudge the robot between readings and watch the
numbers head to zero.

  X  (lateral) = 0  -> marker is dead ahead (robot heading points straight at it)
  yaw          = 0  -> robot is perpendicular to the marker face

For a clean straight approach you mainly want X ~ 0 (so walking forward closes
the distance); yaw ~ 0 means you'll arrive square in front of the marker rather
than off to one side. Each reading = one ssh snap of the head cam (~3-4 s), so
the loop paces itself; adjust the robot, wait for the next line.

  python3 square_up.py
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "tools"))
os.environ["SEE_CAMERA_SOURCE"] = "g1"

import snap  # noqa: E402

X_TOL_CM = 3.0
YAW_TOL_DEG = 3.0


def main():
    print("Squaring up to the marker — Ctrl+C to stop.")
    print(f"Goal: |X| < {X_TOL_CM:.0f} cm and |yaw| < {YAW_TOL_DEG:.0f} deg.\n")
    try:
        while True:
            dets = snap.detect(*snap._capture_g1())
            if not dets:
                print("  no marker in view — turn the robot toward it")
                continue
            m = min(dets, key=lambda d: d["y_m"])
            x_cm, yaw, fwd = m["x_m"] * 100, m["yaw_deg"], m["y_m"] * 100

            if abs(x_cm) < X_TOL_CM:
                aim = "centered"
            else:
                aim = "marker LEFT — aim left" if x_cm < 0 else "marker RIGHT — aim right"
            face = "square" if abs(yaw) < YAW_TOL_DEG else f"angled {yaw:+.0f}deg"
            squared = abs(x_cm) < X_TOL_CM and abs(yaw) < YAW_TOL_DEG
            tag = "   <<< SQUARED" if squared else ""

            print(f"  X {x_cm:+6.1f} cm | yaw {yaw:+6.1f} deg ({face}) "
                  f"| fwd {fwd:6.1f} cm | {aim}{tag}")
    except KeyboardInterrupt:
        print("\n  stopped.")


if __name__ == "__main__":
    main()
