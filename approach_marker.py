#!/usr/bin/env python3
"""Closed-loop approach to an ArUco marker, with a self-improving calibration store.

Three phases (all measured against the robot's head camera):
  1. APPROACH  -- walk forward in undershooting closed-loop steps until the
     marker is ~1 m ahead (marker stays well within camera range). Long steps
     (>= FAST_MIN_CM) use the FAST tier (0.5 m/s, its own self-calibrating
     model); short ones fall back to the slow tier. Turns never go fast.
  2. ORIENT    -- rotate to center the marker (bearing -> 0) so the robot points
     squarely at it. Steered by bearing = atan2(X, fwd), which is reliable;
     ArUco yaw is too noisy. Resolution floor ~8 deg (see turn calibration).
  3. FINAL     -- closed-loop down to the last reliably-visible distance
     (~45 cm), then ONE blind feed-forward creep to the target standoff
     (default 20 cm). The marker is undetectable below ~40 cm, so this last
     stretch is dead-reckoned from the walk model.

Self-improvement: every snap-walk-snap and every turn is a calibration sample.
Each run loads all prior samples from the store, refits the walk model
(cm = slope*T + b) and the turn model (deg = k*omega at t=1 s), uses them to plan
this run, then appends the new samples and saves. The camera->toe offset (needed
for the blind final creep) can't be self-observed, so it is learned from your
feedback: after a run, tape-measure the actual toe-to-marker distance and run
  python3 approach_marker.py --record-final <cm>
which updates the offset so the next run lands closer to the true target.

Runs on system python3 (needs cv2 via snap.py), NOT the whisper venv. Requires
the robot in walk mode and locomotion enabled (`python3 ../g1_agent/g1.py start`).

  python3 approach_marker.py                 # full approach, target 20 cm
  python3 approach_marker.py --standoff 25   # different final standoff
  python3 approach_marker.py --dry-run       # plan only, no motion
  python3 approach_marker.py --record-final 23.5   # feed back a measured result
"""

import argparse
import json
import math
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "tools"))                 # snap.py
sys.path.insert(0, str(_HERE.parents[0] / "g1_agent"))   # g1.py

os.environ["SEE_CAMERA_SOURCE"] = "g1"

import snap  # noqa: E402
import g1  # noqa: E402

CALIB_PATH = _HERE / "approach_calib.json"

# --- tuning (cm / deg / s) ---------------------------------------------------
PHASE1_STANDOFF = 100.0   # end APPROACH when marker is this far ahead
LAST_VISIBLE = 45.0       # closed-loop floor; below here the marker drops out
DEFAULT_FINAL = 20.0      # blind-creep target standoff (toe-to-marker)
STEP_UNDERSHOOT = 0.8     # take 80% of the remaining gap per closed-loop walk
STEP_CAP_CM = 60.0        # never plan a single walk longer than this
BLIND_UNDERSHOOT = 0.9    # extra safety margin on the one blind creep
DEFAULT_OFFSET = 30.0     # camera_fwd - toe distance; over-estimated => stops short
ORIENT_TOL_DEG = 8.0      # bearing within this = squared enough (resolution floor)
TURN_T = 1.0              # calibrated turn duration (past the 0.75 s deadband)
TURN_T_MAX = 3.0          # single-command turn cap: extend duration, don't chunk
TURN_T_MIN = 0.8          # shortest turn that still takes a step (deadband ~0.75 s)
TURN_OMEGA_MIN, TURN_OMEGA_MAX = 0.15, 0.7
MAX_ITERS = 12
ARRIVE_TOL_CM = 6.0       # within this of a target = arrived (don't micro-step)
MIN_WALK_T = 0.9          # forward-step deadband: shorter commands don't take a step
NO_PROGRESS_CM = 2.0      # a substantial walk moving less than this => locomotion off
WALK_SPEED = 0.15         # m/s, the calibrated "slow" tier
FAST_SPEED = 0.5          # m/s, the fast tier (g1.py VX_MAX) -- forward only, never turns
FAST_MIN_CM = 60.0        # planned steps shorter than this stay on the slow tier
FAST_STEP_CAP_CM = 150.0  # cap fast strides (heading drift grows with open-loop length)

# --- calibration store -------------------------------------------------------
def default_store():
    """Seed with the data measured on 2026-07-17 so run #1 is already tuned."""
    return {
        "walk": [  # {T seconds, cm travelled}
            {"T": 2.0, "cm": 17.6}, {"T": 4.0, "cm": 37.3}, {"T": 6.0, "cm": 49.4},
        ],
        "turn": [  # {dir, omega, t, deg} -- deg is |bearing change|
            {"dir": "left",  "omega": 0.15, "t": 1.0, "deg": 8.4},
            {"dir": "left",  "omega": 0.25, "t": 1.0, "deg": 14.4},
            {"dir": "left",  "omega": 0.40, "t": 1.0, "deg": 23.2},
            {"dir": "left",  "omega": 0.60, "t": 1.0, "deg": 36.3},
            {"dir": "left",  "omega": 0.30, "t": 1.0, "deg": 11.9},
            {"dir": "right", "omega": 0.15, "t": 1.0, "deg": 6.5},
            {"dir": "right", "omega": 0.25, "t": 1.0, "deg": 12.6},
            {"dir": "right", "omega": 0.40, "t": 1.0, "deg": 27.5},
            {"dir": "right", "omega": 0.60, "t": 1.0, "deg": 31.6},
            {"dir": "right", "omega": 0.30, "t": 1.0, "deg": 15.2},
        ],
        "walk_fast": [],           # {T, cm} samples at FAST_SPEED
        "offset_samples": [],      # camera_fwd - toe distance, cm
        "last_blind": None,        # {C_last, T_blind} from the most recent run
        "runs": 0,
    }


def load_store():
    if CALIB_PATH.exists():
        with open(CALIB_PATH) as f:
            store = json.load(f)
        store.setdefault("walk_fast", [])  # stores predating the fast tier
        return store
    return default_store()


def save_store(store):
    with open(CALIB_PATH, "w") as f:
        json.dump(store, f, indent=2)


# --- model fitting -----------------------------------------------------------
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
    if slope <= 0:  # nonsense fit -> fall back
        return fallback
    return slope, b


def fit_walk(store):
    """Slow-tier walk model over all walk samples."""
    return _fit_line(store["walk"], (9.85, -2.06))


def fit_walk_fast(store):
    """Fast-tier walk model. Before any data exists, assume the commanded
    speed is fully reached (50 cm/s) -- a deliberate over-estimate, so the
    first fast stride undershoots (safe) and its measurement seeds the fit."""
    pts = store.get("walk_fast", [])
    num = sum(p["T"] * p["cm"] for p in pts)
    den = sum(p["T"] ** 2 for p in pts)
    origin = (num / den, 0.0) if den > 1e-9 else (FAST_SPEED * 100.0, 0.0)
    return _fit_line(pts, origin)


def _invert(model, L_cm):
    slope, b = model
    T = (L_cm - b) / slope
    return max(0.0, min(T, g1.WALK_MAX_DURATION))


def invert_walk(store, L_cm):
    """Seconds needed to walk L_cm forward at the slow tier (clamped)."""
    return _invert(fit_walk(store), L_cm)


def invert_walk_fast(store, L_cm):
    """Seconds needed to walk L_cm forward at the fast tier (clamped)."""
    return _invert(fit_walk_fast(store), L_cm)


def turn_k(store, direction):
    """deg-per-omega at t~1 s, least-squares through the origin, per direction."""
    s = [p for p in store["turn"]
         if p["dir"] == direction and 0.9 <= p["t"] <= 1.1]
    num = sum(p["omega"] * p["deg"] for p in s)
    den = sum(p["omega"] ** 2 for p in s)
    if den < 1e-9:
        return 57.0 if direction == "left" else 56.0
    return num / den


def plan_turn(store, signed_deg):
    """Return (omega_signed, t, achievable_deg) to rotate by signed_deg, or None
    if below the ~8 deg resolution floor. bearing>0 needs a right turn (omega<0).

    Rotations beyond one calibrated second at omega_max extend the DURATION
    (deg ~ k*omega*t, linear-in-t extrapolation of the t=1 s fit) up to
    TURN_T_MAX -- ONE continuous command instead of several 1 s chunks.

    Fine end: below omega_min at 1 s (~8 deg) the duration SHRINKS toward the
    gait deadband (TURN_T_MIN), reaching ~6.5 deg. Even finer requests get the
    minimal step anyway IF the deliberate overshoot past zero still shrinks
    |error| (e.g. 4.9 deg -> 6.5 deg step -> land at -1.6); below half the
    minimal step, None (nothing the platform does can improve it)."""
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
            return None  # even the smallest step would leave a bigger |error|
        omega_mag = TURN_OMEGA_MIN
        t = max(deg / (k * TURN_OMEGA_MIN) * TURN_T, TURN_T_MIN)
    omega = -omega_mag if signed_deg > 0 else omega_mag  # +omega increases bearing
    return omega, t, omega_mag * k * t / TURN_T


# --- perception --------------------------------------------------------------
def snap_marker():
    """Nearest marker's pose, or None. bearing/x in the same sign convention."""
    dets = snap.detect(*snap._capture_g1())
    if not dets:
        return None
    m = min(dets, key=lambda d: d["y_m"])
    fwd, x = m["y_m"] * 100, m["x_m"] * 100
    return {
        "fwd": fwd, "x": x, "yaw": m["yaw_deg"],
        "bearing": math.degrees(math.atan2(m["x_m"], m["y_m"])),
    }


def median_bearing(n=3):
    """Median bearing over n snaps (beats the per-frame jitter); (bearing, fwd)."""
    vals = []
    fwd = None
    for _ in range(n):
        m = snap_marker()
        if m:
            vals.append(m["bearing"])
            fwd = m["fwd"]
    if not vals:
        return None, None
    vals.sort()
    return vals[len(vals) // 2], fwd


# --- phases ------------------------------------------------------------------
def walk_segment(store, target_cm, dry, allow_fast=False):
    """One undershooting closed-loop forward step toward target_cm. Returns
    ('ok'|'done'|'lost'|'stuck', fwd_after). With allow_fast, long steps use
    the fast tier; short ones always drop back to the calibrated slow tier."""
    m = snap_marker()
    if m is None:
        return "lost", None
    gap = m["fwd"] - target_cm
    if gap <= ARRIVE_TOL_CM:
        return "done", m["fwd"]
    L = gap * STEP_UNDERSHOOT
    fast = allow_fast and L >= FAST_MIN_CM
    if fast:
        L = min(L, FAST_STEP_CAP_CM)
        speed, T = FAST_SPEED, invert_walk_fast(store, L)
    else:
        L = min(L, STEP_CAP_CM)
        speed, T = WALK_SPEED, invert_walk(store, L)
    if T < MIN_WALK_T:
        # remaining gap is within one minimum step -- can't refine further
        return "done", m["fwd"]
    print(f"    fwd {m['fwd']:.1f} cm -> target {target_cm:.0f}; walk {L:.1f} cm "
          f"(T={T:.2f}s @ {speed:.2f} m/s{', FAST' if fast else ''})")
    if dry:
        return "ok", m["fwd"]
    g1.walk(speed, 0.0, 0.0, T)
    m2 = snap_marker()
    if m2 is None:
        return "lost", None
    moved = m["fwd"] - m2["fwd"]
    print(f"      moved {moved:.1f} cm (predicted {L:.1f})")
    if moved > 0:  # never record deadband/garbage samples
        store["walk_fast" if fast else "walk"].append(
            {"T": round(T, 3), "cm": round(moved, 1)})
    if moved < NO_PROGRESS_CM and L >= 10.0:
        return "stuck", m2["fwd"]
    return "ok", m2["fwd"]


def phase_approach(store, dry):
    print("\n[1/3] APPROACH -> 1 m (fast tier for long strides)")
    for _ in range(MAX_ITERS):
        status, fwd = walk_segment(store, PHASE1_STANDOFF, dry, allow_fast=True)
        if status == "done":
            print(f"    reached {fwd:.1f} cm")
            return True
        if status == "lost":
            print("    ABORT: lost the marker.")
            return False
        if status == "stuck":
            print("    ABORT: no forward progress -- is locomotion enabled? (g1.py start)")
            return False
        if dry:
            return True
    print("    ABORT: too many iterations.")
    return False


def phase_orient(store, dry):
    print("\n[2/3] ORIENT (center bearing)")
    prev_abs = None
    for _ in range(MAX_ITERS):
        bearing, fwd = median_bearing()
        if bearing is None:
            print("    ABORT: lost the marker.")
            return False
        print(f"    bearing {bearing:+.1f} deg (fwd {fwd:.1f} cm)")
        if abs(bearing) <= ORIENT_TOL_DEG:
            print("    squared (within resolution).")
            return True
        if prev_abs is not None and abs(bearing) >= prev_abs - 1.0:
            print("    can't improve further (turn resolution floor) -- accepting.")
            return True
        plan = plan_turn(store, bearing)
        if plan is None:
            print("    residual below the ~8 deg turn floor -- accepting.")
            return True
        omega, t, _ = plan
        print(f"    turn omega={omega:+.2f} t={t:.1f}s")
        if dry:
            return True
        g1.walk(0.0, 0.0, omega, t)
        b2, _ = median_bearing()
        if b2 is not None:
            direction = "right" if omega < 0 else "left"
            # non-1s samples are stored too; turn_k() filters to t~1 for its fit
            store["turn"].append({"dir": direction, "omega": round(abs(omega), 3),
                                  "t": round(t, 2), "deg": round(abs(b2 - bearing), 1)})
        prev_abs = abs(bearing)
    print("    reached iteration cap -- accepting current heading.")
    return True


def phase_final(store, final_standoff, dry):
    offset = (sum(store["offset_samples"]) / len(store["offset_samples"])
              if store["offset_samples"] else DEFAULT_OFFSET)
    camera_target = final_standoff + offset   # camera reading when the toe is at target
    cl_floor = max(camera_target, LAST_VISIBLE)  # closed-loop stop (stay visible)
    blind_needed = camera_target < LAST_VISIBLE
    print(f"\n[3/3] FINAL -> {final_standoff:.0f} cm toe (camera target {camera_target:.0f} cm, "
          f"offset {offset:.1f}; {'blind creep' if blind_needed else 'stays visible'})")

    for _ in range(MAX_ITERS):
        status, fwd = walk_segment(store, cl_floor, dry)
        if status == "done":
            break
        if status in ("lost", "stuck"):
            print(f"    ABORT during closed-loop: {status}.")
            return False
        if dry:
            break

    C_last = cl_floor if dry else None
    if not dry:
        m = snap_marker()
        if m is None:
            print("    ABORT: lost the marker before reaching the closed-loop floor.")
            return False
        C_last = m["fwd"]

    # Reached the target while still visible -> no blind motion.
    if C_last <= camera_target + 3:
        print(f"    arrived visible: camera {C_last:.1f} cm, est. toe {C_last - offset:.1f} cm.")
        store["last_blind"] = {"C_last": round(C_last, 1), "bt": 0.0}
        return True

    # Otherwise blind-creep the remaining camera distance down to camera_target.
    bt = (C_last - camera_target) * BLIND_UNDERSHOOT
    if bt > (C_last - 5):
        print("    SAFETY: planned blind creep too large -- capping.")
        bt = C_last - 5
    T_blind = invert_walk(store, bt)
    print(f"    at camera {C_last:.1f} cm; BLIND creep {bt:.1f} cm (T={T_blind:.2f}s) "
          f"-> predicted toe ~{final_standoff:.0f} cm")
    store["last_blind"] = {"C_last": round(C_last, 1), "bt": round(bt, 1)}
    if dry:
        return True
    g1.walk(WALK_SPEED, 0.0, 0.0, T_blind)
    print("    done (blind). Tape-measure the actual toe-to-marker distance and run:")
    print(f"      python3 {pathlib.Path(__file__).name} --record-final <cm>")
    return True


def record_final(store, actual_cm):
    """Learn the camera->toe offset from a measured final distance."""
    lb = store.get("last_blind")
    if not lb:
        print("No pending run on record -- run an approach first.")
        return
    # offset = camera reading at last snap - (intended) travel since - measured toe.
    offset_obs = lb["C_last"] - lb["bt"] - actual_cm
    store["offset_samples"].append(round(offset_obs, 1))
    store["last_blind"] = None
    mean = sum(store["offset_samples"]) / len(store["offset_samples"])
    print(f"  measured toe {actual_cm:.1f} cm -> offset sample {offset_obs:.1f} cm "
          f"(mean now {mean:.1f} over {len(store['offset_samples'])})")
    save_store(store)


def main():
    ap = argparse.ArgumentParser(description="Closed-loop marker approach with self-calibration.")
    ap.add_argument("--standoff", type=float, default=DEFAULT_FINAL,
                    help=f"final toe-to-marker target cm (default {DEFAULT_FINAL:.0f})")
    ap.add_argument("--dry-run", action="store_true", help="plan only, no robot motion")
    ap.add_argument("--record-final", type=float, metavar="CM",
                    help="feed back a measured final distance to learn the offset")
    args = ap.parse_args()

    store = load_store()

    if args.record_final is not None:
        record_final(store, args.record_final)
        return

    slope, b = fit_walk(store)
    fs, fb = fit_walk_fast(store)
    n_fast = len(store["walk_fast"])
    print("=" * 60)
    print(f"  Marker approach  (run #{store['runs'] + 1}, {'DRY-RUN' if args.dry_run else 'LIVE'})")
    print(f"  fast model: cm = {fs:.2f}*T {fb:+.2f}   "
          f"({n_fast} samples{'' if n_fast else ' -- seeded, undershoots'})")
    print(f"  walk model: cm = {slope:.2f}*T {b:+.2f}   "
          f"| offset {sum(store['offset_samples'])/len(store['offset_samples']) if store['offset_samples'] else DEFAULT_OFFSET:.1f} cm "
          f"({len(store['offset_samples'])} samples)")
    print("=" * 60)

    if not args.dry_run and not g1.check():
        sys.exit(1)

    ok = phase_approach(store, args.dry_run) \
        and phase_orient(store, args.dry_run) \
        and phase_final(store, args.standoff, args.dry_run)

    if not args.dry_run:
        store["runs"] += 1
        save_store(store)
        print(f"\n  calibration saved ({len(store['walk'])} walk, {len(store['turn'])} turn samples).")
    print("  " + ("done." if ok else "aborted -- see above."))


if __name__ == "__main__":
    main()
