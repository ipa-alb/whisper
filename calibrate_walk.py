#!/usr/bin/env python3
"""Calibrate the G1's forward walk: seconds -> centimetres, using the robot's
head camera + an ArUco marker as the ruler (no tape measure).

Why an affine model instead of a single cm/s
---------------------------------------------
Every walk() ramps up from a standstill and ramps down to a stop, so a short
walk covers proportionally less ground than a long one -- a lone "cm/s" number
is wrong exactly where you care (the final creep). We instead fit

    distance(T) = slope * T + b          (b = ramp-up/-down loss, usually < 0)

from two durations, which inverts cleanly to give the walk time for a target
distance:  T = (L - b) / slope.

Procedure (open loop, supervised)
---------------------------------
Marker on a wall ahead, robot ~2-3 m back and facing it squarely, locomotion
already enabled (`python3 ../g1_agent/g1.py start`). For each (duration, trial):

    snap  -> forward distance d0
    walk forward `duration` s at `speed`   (g1.walk auto-stops)
    snap  -> forward distance d1
    travelled = d0 - d1

then you reposition the robot and press Enter for the next trial.

The measuring camera is the one that MOVES with the robot -- the head cam -- so
this always uses snap's G1 (ssh) source. "Distance" is the marker's FORWARD
component (optical-axis Z); lateral drift is printed so you can discard a trial
where the robot veered off straight.

    python3 calibrate_walk.py                    # slow tier (0.15 m/s), T=2s & 4s, 2 trials
    python3 calibrate_walk.py --speed 0.5        # fast tier
    python3 calibrate_walk.py --durations 2,3,4 --trials 3
"""

import argparse
import os
import pathlib
import statistics
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "tools"))            # snap.py
sys.path.insert(0, str(_HERE.parents[0] / "g1_agent"))  # g1.py (pure stdlib, shells out to ROS)

os.environ["SEE_CAMERA_SOURCE"] = "g1"              # camera lives on the robot's Orin

import snap  # noqa: E402
import g1  # noqa: E402

SAFETY_MARGIN_M = 0.5  # forward clearance that must remain AFTER a walk


def snap_forward():
    """One snapshot -> nearest marker's pose dict (by forward distance), or None."""
    color, depth, K, dist = snap._capture_g1()
    dets = snap.detect(color, depth, K, dist)
    if not dets:
        return None
    return min(dets, key=lambda d: d["y_m"])


def report(results, speed):
    means = {T: statistics.mean(v) for T, v in results.items() if v}
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    if not means:
        print("  No valid trials recorded.")
        return
    print(f"  {'duration':>8}  {'n':>2}  {'mean travelled':>16}  {'cm/s':>7}")
    for T in sorted(means):
        d = means[T]
        print(f"  {T:>6.1f}s  {len(results[T]):>2}  {d*100:>13.1f} cm  {d/T*100:>6.1f}")

    Ts = sorted(means)
    if len(Ts) < 2:
        print("\n  Need >=2 durations with data to fit the seconds->cm model.")
        return

    T1, T2 = Ts[0], Ts[-1]
    slope = (means[T2] - means[T1]) / (T2 - T1)   # effective steady m/s
    b = means[T1] - slope * T1                    # ramp offset (m)
    print("\n  Seconds -> distance model (commanded speed "
          f"{speed} m/s):")
    print(f"    distance_m = {slope:.4f} * T + ({b:+.4f})")
    print(f"    effective steady speed: {slope:.3f} m/s ({slope*100:.1f} cm/s)")
    print("\n  To walk a target distance L metres forward:")
    print(f"    T = (L - ({b:+.4f})) / {slope:.4f}")

    print("\n  Reaching \"5 cm in front of the marker\":")
    print("    1. Park the robot facing the marker, snap the forward distance D,")
    print("       tape-measure toe-to-marker M once.  offset = D - M  (camera sits")
    print("       back from and above the toes, so D > M).")
    print("    2. Desired forward travel  L = D_now - offset - 0.05")
    print("    3. Walk time  T from the formula above; command:")
    print(f"       python3 ../g1_agent/g1.py walk {speed} 0 0 <T>")
    print("    NB the head cam loses a 17.8 cm marker below ~0.4 m, so snap while")
    print("    still far, then walk the last segment blind (open loop).")


def main():
    ap = argparse.ArgumentParser(description="Calibrate G1 forward walk seconds->cm.")
    ap.add_argument("--speed", type=float, default=0.15,
                    help="forward vx in m/s (default 0.15 = slow tier; 0.5 = fast)")
    ap.add_argument("--durations", default="2,4",
                    help="comma-separated walk durations in s (default 2,4)")
    ap.add_argument("--trials", type=int, default=2, help="trials per duration (default 2)")
    args = ap.parse_args()

    speed = args.speed
    durations = [float(x) for x in args.durations.split(",")]
    trials = args.trials

    print("=" * 60)
    print("  G1 forward-walk calibration (camera ruler)")
    print(f"  speed={speed} m/s   durations={durations}s   trials={trials}")
    print("=" * 60)
    print("  SAFETY: the robot walks TOWARD the marker/wall. Keep the path")
    print("  clear, stay ready to `g1.py estop`, and start well back.\n")

    print("  Checking robot link...")
    if not g1.check():
        sys.exit(1)
    input("\n  Marker on wall ahead, robot ~2-3 m back and FACING it, and you have\n"
          "  run `python3 ../g1_agent/g1.py start`. Press Enter to begin (Ctrl+C aborts). ")

    results = {T: [] for T in durations}
    try:
        for T in durations:
            for t in range(trials):
                cmd = input(f"\n[T={T}s trial {t+1}/{trials}] robot facing marker, path clear. "
                            "Enter = snap+walk, q = quit: ").strip().lower()
                if cmd == "q":
                    raise KeyboardInterrupt

                m0 = snap_forward()
                if m0 is None:
                    print("  [!] no marker detected -- reposition, retry.")
                    continue
                expected = speed * T
                if m0["y_m"] < expected + SAFETY_MARGIN_M:
                    print(f"  [!] only {m0['y_m']:.2f} m ahead; need > "
                          f"{expected + SAFETY_MARGIN_M:.2f} m to keep clearance. Back up.")
                    continue
                print(f"  d0 forward = {m0['y_m']*100:6.1f} cm  (lateral {m0['x_m']*100:+.1f} cm)")

                g1.walk(speed, 0.0, 0.0, T)

                m1 = snap_forward()
                if m1 is None:
                    print("  [!] marker lost after walk -- trial discarded.")
                    continue
                travelled = m0["y_m"] - m1["y_m"]
                drift = m1["x_m"] - m0["x_m"]
                print(f"  d1 forward = {m1['y_m']*100:6.1f} cm  ->  travelled "
                      f"{travelled*100:6.1f} cm  ({travelled/T*100:.1f} cm/s)  "
                      f"lateral drift {drift*100:+.1f} cm")
                if travelled <= 0:
                    print("  [!] non-positive travel (locomotion not started, or veered) "
                          "-- not recorded.")
                    continue
                results[T].append(travelled)
    except KeyboardInterrupt:
        print("\n  Stopped.")

    report(results, speed)


if __name__ == "__main__":
    main()
