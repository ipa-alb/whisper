#!/usr/bin/env python3
"""fetch.py — find a person off to the right, orient on them, walk up slowly.

The human counterpart of go_to_marker.py, and deliberately LOOSE — "good enough to
walk up to someone", not the squared, calibrated table approach. Sequence:

  1. robot link check + enable locomotion  (g1.check / g1.start; skip with --no-start)
  2. turn ~90 deg to the RIGHT — the person is expected off to that side, out of the
     forward-facing belt D455's view, so the robot turns to point the camera at them
     (change/skip the angle with --turn <deg>)
  3. detect a person with the belt D455 (see_camera 'person' + depth). If none is in
     view, a small in-place scan tries to find one.
  4. orient toward them — turn to bring them near frame centre (no fine squaring)
  5. walk slowly toward them ONCE (a single continuous push planned from the one
     distance reading, stopping ~STOP_DIST short), then wait 2 s and open the hand
     to release whatever it's carrying (the hand-off).

Why the belt D455 (serial pinned): the request is "check with the 455", and it's the
same camera the marker approach measures from. Person detection reuses the `look`
tool's detector (see_camera), forced to the fast YOLO path — 'person' is a rock-solid
COCO class, and OWLv2 (the project default) is too slow for a multi-step servo.

Steering/geometry note: the belt cam is un-rotated by the Orin grabber, so a larger
pixel-x (cx) means the person is to the robot's RIGHT — the same convention snap.py
verified for the marker. Turning to null that offset points the robot at the person.

This is a VISION skill (like grab_hammer.py), so it runs under the whisper .venv
(needs onnxruntime via see_camera) and needs NO cv2 — its runtime never touches
ArUco. The turn/walk models below are read-only mirrors of the pure calibration in
approach_marker.py (the source of truth); fetch only READS the shared store
(approach_calib.json) and never writes person-derived samples back, so it can't
pollute the precise marker calibration.

  ~/workspace/unitree_g1_ros2_driver_wireless/whisper/.venv/bin/python fetch.py
  python fetch.py --turn 0        # person already ahead — skip the initial turn
  python fetch.py --turn -90      # look to the LEFT instead
  python fetch.py --dry-run       # plan/detect only, no motion
  python fetch.py --no-start      # locomotion already enabled

The hand-off is fixed to this rig: always stops ~1 m from the person and opens the
LEFT hand (left-arm pick, per grab_hammer.py) — neither is a command-line option.
"""

import argparse
import json
import math
import os
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "tools"))                 # see_camera.py
sys.path.insert(0, str(_HERE.parents[0] / "g1_agent"))   # g1.py

# Capture from the robot's Orin over ssh, pinned to the belt D455, and (unless the
# user overrides) use the fast YOLO detector — set BEFORE importing see_camera so it
# reads them at import time.
os.environ["SEE_CAMERA_SOURCE"] = "g1"
os.environ.setdefault("G1_CAM_SERIAL", "146222254681")   # belt-mounted D455 ("the 455")
os.environ.setdefault("SEE_CAMERA_DETECTOR", "yolo")     # person = fast, reliable COCO class

import g1  # noqa: E402                    # pure subprocess wrapper — no cv2
import see_camera  # noqa: E402            # person detection on the belt cam (onnxruntime)

# ── Turn/walk calibration (read-only mirror of approach_marker.py) ────────────
# fetch reuses the same self-improving models the marker approach tunes, but must
# stay cv2-free to run in the vision venv, so these pure helpers are copied here.
# Keep them in sync with approach_marker.py if that file's models change.
CALIB_PATH = _HERE / "approach_calib.json"
TURN_T = 1.0                       # calibrated turn duration (past the ~0.75 s deadband)
TURN_T_MAX, TURN_T_MIN = 3.0, 0.8  # single-command turn duration bounds
TURN_OMEGA_MIN, TURN_OMEGA_MAX = 0.15, 0.7
WALK_SPEED = 0.15                  # m/s, the calibrated "slow" tier

_SEED_TURN = [  # deg-per-omega seed (2026-07-17), so turn_k has data if the store is bare
    {"dir": "left", "omega": 0.15, "t": 1.0, "deg": 8.4},
    {"dir": "left", "omega": 0.40, "t": 1.0, "deg": 23.2},
    {"dir": "left", "omega": 0.60, "t": 1.0, "deg": 36.3},
    {"dir": "right", "omega": 0.15, "t": 1.0, "deg": 6.5},
    {"dir": "right", "omega": 0.40, "t": 1.0, "deg": 27.5},
    {"dir": "right", "omega": 0.60, "t": 1.0, "deg": 31.6},
]
_SEED_WALK = [{"T": 2.0, "cm": 17.6}, {"T": 4.0, "cm": 37.3}, {"T": 6.0, "cm": 49.4}]


def load_store():
    """Load the shared calibration store, or a seeded default if it's absent."""
    if CALIB_PATH.exists():
        with open(CALIB_PATH) as f:
            store = json.load(f)
        store.setdefault("turn", _SEED_TURN)
        store.setdefault("walk", _SEED_WALK)
        return store
    return {"turn": list(_SEED_TURN), "walk": list(_SEED_WALK)}


def _fit_line(pts, fallback):
    """Least-squares line cm = slope*T + b over (T, cm) samples."""
    Ts = {round(p["T"], 3) for p in pts}
    if len(pts) < 2 or len(Ts) < 2:
        return fallback
    n = len(pts)
    sx = sum(p["T"] for p in pts)
    sy = sum(p["cm"] for p in pts)
    sxx = sum(p["T"] ** 2 for p in pts)
    sxy = sum(p["T"] * p["cm"] for p in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return fallback
    slope = (n * sxy - sx * sy) / denom
    b = (sy - slope * sx) / n
    return (slope, b) if slope > 0 else fallback


def fit_walk(store):
    """Slow-tier walk model (cm = slope*T + b) over all walk samples."""
    return _fit_line(store.get("walk", []), (9.85, -2.06))


def turn_k(store, direction):
    """deg-per-omega at t~1 s, least-squares through the origin, per direction."""
    s = [p for p in store.get("turn", [])
         if p["dir"] == direction and 0.9 <= p["t"] <= 1.1]
    num = sum(p["omega"] * p["deg"] for p in s)
    den = sum(p["omega"] ** 2 for p in s)
    if den < 1e-9:
        return 57.0 if direction == "left" else 56.0
    return num / den


def plan_turn(store, signed_deg):
    """(omega_signed, t, achievable_deg) to rotate by signed_deg, or None if below
    the ~6.5 deg resolution floor. signed_deg>0 -> right turn (omega<0)."""
    direction = "right" if signed_deg > 0 else "left"
    k = turn_k(store, direction)
    deg = abs(signed_deg)
    omega_mag, t = deg / k, TURN_T
    if omega_mag > TURN_OMEGA_MAX:
        t = min(deg / (k * TURN_OMEGA_MAX) * TURN_T, TURN_T_MAX)
        omega_mag = TURN_OMEGA_MAX
    elif omega_mag < TURN_OMEGA_MIN:
        min_step = k * TURN_OMEGA_MIN * TURN_T_MIN / TURN_T   # ~6.5 deg
        if deg <= min_step / 2:
            return None
        omega_mag = TURN_OMEGA_MIN
        t = max(deg / (k * TURN_OMEGA_MIN) * TURN_T, TURN_T_MIN)
    omega = -omega_mag if signed_deg > 0 else omega_mag       # +omega increases bearing
    return omega, t, omega_mag * k * t / TURN_T


def turn_by(store, signed_deg, dry):
    """In-place turn by signed_deg (+ = right) as ONE command where possible; only
    rotations beyond ~one maxed-out command split back-to-back with no camera."""
    if abs(signed_deg) < 3.0:
        return
    direction = "right" if signed_deg > 0 else "left"
    k = turn_k(store, direction)
    per_max = k * TURN_OMEGA_MAX * TURN_T_MAX / TURN_T        # one command's ceiling
    n = max(1, math.ceil(abs(signed_deg) / per_max))
    deg_each = abs(signed_deg) / n
    omega_mag = min(TURN_OMEGA_MAX, max(TURN_OMEGA_MIN, deg_each / k))
    t = max(TURN_T_MIN, min(TURN_T_MAX, deg_each / (k * omega_mag) * TURN_T))
    omega = -omega_mag if signed_deg > 0 else omega_mag
    print(f"      TURN {signed_deg:+.0f} deg: {n}x omega={omega:+.2f} t={t:.1f}s")
    if dry:
        return
    for _ in range(n):
        g1.walk(0.0, 0.0, omega, t)


# --- tuning (deg / cm / m) ---------------------------------------------------
INITIAL_TURN_DEG = 90.0    # + = turn RIGHT to face the person (bearing sign convention)
STOP_DIST_M = 1.0          # ALWAYS stop 1 m from the person for the hand-off (fixed)
CAP_W = see_camera.CAP_W   # frame width the detector reports cx in (640)
# Belt D455 colour HFOV at 640x480 (~4:3 crop), approximate. Only sizes the orient
# turns, and the servo re-detects after each, so an imprecise FOV still converges.
HFOV_DEG = float(os.environ.get("FETCH_HFOV_DEG", "70"))
CENTER_TOL_PX = 80         # person within this of frame centre = oriented enough (loose)
ORIENT_MAX_TURNS = 5       # cap on orient micro-turns before giving up and walking
APPROACH_UNDERSHOOT = 0.9  # walk 90% of the gap to the stop distance (land just short)
BLIND_PUSH_CM = 40.0       # single forward push when the person has no depth reading
DETECT_RETRIES = 3         # belt D455 is flaky right after pipeline start — retry a few
WAIT_BEFORE_OPEN_S = 2.0   # pause at the person before releasing (the requested 2 s)
# Fixed by the setup: this is a LEFT-hand-only rig (grab_hammer picks left), and the
# hand-off always happens ~STOP_DIST_M from the person — neither is a CLI option.


def detect_person(retries=DETECT_RETRIES):
    """Best 'person' from the belt D455, or None.

    Prefers the NEAREST person that has a depth reading (that's the one we approach);
    falls back to the highest-confidence person when none carry depth. Returns
    {cx_px, dist_m, conf} — cx_px is horizontal centre in original-frame pixels
    (0..CAP_W), dist_m the torso depth in metres (None without depth)."""
    for attempt in range(1, retries + 1):
        try:
            dets, _ = see_camera.detect("person")
        except Exception as e:  # transient capture/model failure — retry
            print(f"    (vision attempt {attempt}/{retries} failed: {e})")
            time.sleep(0.5)
            continue
        people = [d for d in dets if d[0].lower() == "person"]
        if not people:
            continue
        with_depth = [d for d in people if d[3] is not None]
        pick = min(with_depth, key=lambda d: d[3]) if with_depth \
            else people[0]  # detect() sorts by conf, so [0] is the most confident
        _label, conf, cx, dist, _box = pick
        return {"cx_px": float(cx), "dist_m": dist, "conf": float(conf)}
    return None


def bearing_from_cx(cx):
    """Approx bearing to the person from their pixel-x. + = person to the RIGHT
    (feeds plan_turn directly: bearing>0 -> right turn)."""
    return ((cx - CAP_W / 2.0) / CAP_W) * HFOV_DEG


def scan_for_person(store, dry):
    """Person not in view — sweep in place to find one.

    A few small steps to the right first (the side we just turned toward), then a
    longer sweep back across the front. Returns the detection once seen, or None."""
    print("  no person in view — scanning to find one...")
    plan = [+18.0, +18.0, -18.0, -18.0, -18.0, -18.0]  # ~36 deg right, then ~72 back left
    for deg in plan:
        turn_by(store, deg, dry)
        if dry:
            return None
        det = detect_person(retries=1)  # fast seen-or-not; a blip-miss just scans on
        if det:
            dm = f"depth {det['dist_m']:.2f} m" if det["dist_m"] else "no depth"
            print(f"    found a person: cx {det['cx_px']:.0f} px, {dm}")
            return det
    return None


def orient_on_person(store, det, dry):
    """Turn to bring the person near frame centre. Returns the latest detection
    (re-measured after the last turn), or None if the person was lost."""
    for i in range(ORIENT_MAX_TURNS):
        off = det["cx_px"] - CAP_W / 2.0
        if abs(off) <= CENTER_TOL_PX:
            print(f"  oriented: person {off:+.0f} px from centre (within {CENTER_TOL_PX}).")
            return det
        bearing = bearing_from_cx(det["cx_px"])
        turn = plan_turn(store, bearing)
        if turn is None:
            print(f"  person {off:+.0f} px off — below the turn floor, good enough.")
            return det
        omega, t, _deg = turn
        print(f"  orient {i + 1}/{ORIENT_MAX_TURNS}: person {off:+.0f} px "
              f"(bearing {bearing:+.1f} deg) -> turn omega={omega:+.2f} t={t:.1f}s")
        if dry:
            return det
        g1.walk(0.0, 0.0, omega, t)
        det = detect_person()
        if det is None:
            print("  lost the person during orient.")
            return None
    print("  orient: hit the turn cap — walking from here.")
    return det


def push_forward(store, L_cm, dry):
    """Walk L_cm forward as ONE continuous push at the slow tier — split into
    back-to-back sub-walks only if it exceeds the per-command duration cap, with NO
    camera in between (this is the single, no-servo walk to the person)."""
    slope, b = fit_walk(store)  # slow-tier walk model: cm = slope*T + b
    T_total = max(0.0, (L_cm - b) / slope) if slope > 0 else 0.0
    if T_total < 0.05:
        print(f"  forward push {L_cm:.0f} cm is below the walk deadband — no walk.")
        return
    n = math.ceil(T_total / g1.WALK_MAX_DURATION)
    tag = f" ({n} back-to-back sub-walks)" if n > 1 else ""
    print(f"  walk {L_cm:.0f} cm forward in one push: T={T_total:.1f}s "
          f"@ {WALK_SPEED:.2f} m/s (slow){tag}")
    if dry:
        return
    remaining = T_total
    while remaining > 0.05:
        t = min(remaining, g1.WALK_MAX_DURATION)
        g1.walk(WALK_SPEED, 0.0, 0.0, t)
        remaining -= t


def approach_person(store, dry):
    """Find a person, orient once, walk up to STOP_DIST_M in ONE push, then stop,
    wait WAIT_BEFORE_OPEN_S, and open the left hand (release whatever it's carrying).
    Returns True on completion, False if no person was found."""
    stop_cm = STOP_DIST_M * 100.0

    det = detect_person()
    if det is None and not dry:
        det = scan_for_person(store, dry)
    if det is None:
        print("  ABORT: no person found.")
        return False

    # Orient toward the person (turns only — the single walk comes after).
    det = orient_on_person(store, det, dry)
    if det is None and not dry:
        det = scan_for_person(store, dry)
        if det is None:
            print("  ABORT: lost the person while orienting.")
            return False

    # Size the one walk from the single distance reading.
    dist_cm = det["dist_m"] * 100.0 if det and det["dist_m"] is not None else None
    if dist_cm is None:
        print(f"  person seen but no depth — cautious {BLIND_PUSH_CM:.0f} cm push.")
        L = BLIND_PUSH_CM
    else:
        L = max(0.0, (dist_cm - stop_cm) * APPROACH_UNDERSHOOT)
        print(f"  person at {dist_cm:.0f} cm; single walk to ~{stop_cm:.0f} cm "
              f"(cx {det['cx_px']:.0f} px, conf {det['conf']:.2f}).")

    print("\n-- walking up to the person (one push, no re-stepping) --")
    push_forward(store, L, dry)

    print(f"\n-- stop, wait {WAIT_BEFORE_OPEN_S:.0f} s, open the left hand (hand-off) --")
    if dry:
        print(f"  (dry) would wait {WAIT_BEFORE_OPEN_S:.0f} s, then g1.hand('left', 'OPEN')")
        return True
    time.sleep(WAIT_BEFORE_OPEN_S)
    g1.hand("left", "OPEN")
    print("  left hand opened — released.")
    return True


def fetch(turn_deg=INITIAL_TURN_DEG, dry=False, do_start=True):
    """Turn ~turn_deg (right = +), find a person on the belt D455, orient, walk up
    slowly to ~STOP_DIST_M in one push, then wait and open the left hand. Returns
    True on completion."""
    print("=" * 60)
    print("  FETCH — turn -> find a person -> orient -> walk up slowly")
    print(f"  turn {turn_deg:+.0f} deg, stop ~{STOP_DIST_M:.1f} m   "
          f"({'DRY-RUN' if dry else 'LIVE'})")
    print("=" * 60)

    if not dry:
        if not g1.check():
            return False
        if do_start:
            print("  enabling locomotion ...")
            g1.start()

    store = load_store()

    if abs(turn_deg) >= 3.0:
        side = "RIGHT" if turn_deg > 0 else "LEFT"
        print(f"\n-- turning ~{abs(turn_deg):.0f} deg to the {side} to face the person --")
        turn_by(store, turn_deg, dry)
    else:
        print("\n-- skipping the initial turn (person expected straight ahead) --")

    print("\n-- finding and approaching the person --")
    ok = approach_person(store, dry)
    print("  " + ("DONE." if ok else "ABORTED — see above."))
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Find a person off to the side, orient on them, and walk up slowly.")
    ap.add_argument("--turn", type=float, default=INITIAL_TURN_DEG,
                    help=f"initial in-place turn in deg, + = right (default {INITIAL_TURN_DEG:.0f}); "
                         "0 to skip when the person is already ahead")
    ap.add_argument("--dry-run", action="store_true", help="plan/detect only, no motion")
    ap.add_argument("--no-start", action="store_true",
                    help="skip locomotion enable (already in walk/loco mode)")
    args = ap.parse_args()

    ok = fetch(turn_deg=args.turn, dry=args.dry_run, do_start=not args.no_start)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
