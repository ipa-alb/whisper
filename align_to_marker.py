#!/usr/bin/env python3
"""Walk the G1 up to an ArUco marker on a table, using turn + forward only.

Goal: end up roughly 1 m in front of the marker, facing it, then (on confirmation)
up at the table. Being perfectly centered / perfectly on the marker normal is NOT
required — facing the marker is enough to face the table it sits on.

Why turn, not strafe: strafe was tried and fails — it slides the robot sideways
but its rotation coupling cancels the heading change (measured: ~14 cm lateral,
~3 deg random yaw), so it can't reliably reposition. Turn + forward is used
instead: turn to aim at the marker, then walk straight in.

The steering signal is the marker BEARING = atan2(x, fwd), which is reliable at
any range (unlike solvePnP yaw, trusted only close in). Turning to null the
bearing points the robot straight at the marker -- i.e. at the table it sits on.

Two-frame plan (NOT a measure->nudge->measure loop): from the FIRST frame we
compute the whole approach -- turn to face the marker, then one continuous
forward push (FAST tier for long pushes, with corrective top-ups) -- and walk it
out in one go, landing ~1 m in front. Then a SQUARE-UP step does exactly TWO
moves computed from one frame -- ORIENT square to the marker face, then STRAFE
sideways onto the table's normal -- run back-to-back with NO detection in
between (re-detecting mid-move kept losing the marker), judged by a single final
frame. (A pure in-place turn can't square: off the normal it trades yaw for
bearing 1:1, so it can only face OR be square, not both; strafe is the missing
lateral move.) Finally, on confirmation, plan+walk straight in to the table.
Self-calibrating: appends command->effect to approach_calib.json.

Runs on system python3 (needs cv2), robot in walk mode + `g1.py start`.
  python3 align_to_marker.py            # smallest speed
  python3 align_to_marker.py --dry-run
"""

import argparse
import math
import os
import pathlib
import statistics
import sys

import cv2
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "tools"))
sys.path.insert(0, str(_HERE.parents[0] / "g1_agent"))
os.environ["SEE_CAMERA_SOURCE"] = "g1"

import snap  # noqa: E402
import g1  # noqa: E402
import approach_marker as am  # noqa: E402  reuse store + walk/turn models

TARGET_FWD = 105.0    # cm; loose approach target (~1 m) — the "reading distance"
# Final table standoff. TABLE_FWD is a camera-to-marker target; empirically
# TABLE_FWD=60 with the 0.9 undershoot stopped the robot ~70 cm from the table, so
# each -1 cm here is ~-1 cm at the table. TABLE_FWD=5 + no undershoot aims ~10 cm.
# RAISE TABLE_FWD if the blind final push comes in too close / bumps the table.
TABLE_FWD = 5.0
TABLE_UNDERSHOOT = 1.0  # final leg lands ON target (0.9's short-bias can't reach ~10 cm)
APPROACH_UNDERSHOOT = 0.9  # size each go-there walk to land just short of its target
TOPUP_TOL = 30.0      # cm; after a push, top-up again while still this far past target
IPPE_AGREE = 5.0      # deg; def vs IPPE yaw must agree this close to trust yaw (readout only)
YAW_SQUARE_TOL = 3.5   # deg; the "90-deg lock" target (>= half the ~6.5 deg min step,
                       # so a micro-turn below this could only make |yaw| worse)
MAX_LOCK_ITERS = 3     # extra micro-turn+measure rounds after the square-up
STRAFE_VY = 0.2       # calibrated lateral velocity tier (|vy|); VY_MAX is 0.3
STRAFE_SPEED = 13.7   # cm/s sideways at |vy|=STRAFE_VY (fallback; refined from store)
STRAFE_CAP = 120.0    # cm; clamp the open-loop sideways move (tan blows up near 45 deg)
# vy>0 = robot LEFT — VERIFIED on the robot (commanded vy=+0.2, robot moved left).
STRAFE_LEFT_IS_POS_VY = True
# Which side of the marker's normal a +yaw reading means. +1: yaw>0 -> normal tilts
# right -> robot is left of it -> strafe RIGHT to square (physics-derived default).
# If the square-up STILL strafes/orients the wrong way live, flip this to -1 (that
# is the ONLY sign left to flip — vy, omega, and x are all verified correct).
YAW_NORMAL_SIGN = +1

_HALF = snap.MARKER_SIZE_M / 2
_OBJ = np.array([[-_HALF, _HALF, 0], [_HALF, _HALF, 0],
                 [_HALF, -_HALF, 0], [-_HALF, -_HALF, 0]], dtype=np.float32)


def _yaw(rvec):
    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2]
    return math.degrees(math.atan2(n[0], -n[2]))


def measure(n=2):
    """Median pose over n snaps. yaw_reliable = default and IPPE_SQUARE agree.
    Each snap is an ~4.5 s ssh round-trip, so n is kept minimal: 2 for
    plan/arrival frames (distance+bearing tolerate jitter), 3 where yaw is
    read (square-up), 1 inside the reacquire scan (seen-or-not)."""
    xs, ys, yd, yi = [], [], [], []
    for _ in range(n):
        color, depth, K, dist = snap._capture_g1()
        cs, ids, _ = snap.DETECTOR.detectMarkers(color)
        if ids is None:
            continue
        c = cs[0][0]
        _, rd, td = cv2.solvePnP(_OBJ, c, K, dist)
        _, ri, _ = cv2.solvePnP(_OBJ, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        tv = td.flatten()
        xs.append(tv[0] * 100)
        ys.append(tv[2] * 100)
        yd.append(_yaw(rd))
        yi.append(_yaw(ri))
    if not xs:
        return None
    x, fwd = statistics.median(xs), statistics.median(ys)
    yawd, yawi = statistics.median(yd), statistics.median(yi)
    return {"x": x, "fwd": fwd, "yaw": yawd,
            "yaw_reliable": abs(yawd - yawi) < IPPE_AGREE,
            "bearing": math.degrees(math.atan2(x, fwd))}


def reacquire(hint_bearing=None):
    """Marker left the frame (open-loop heading drift over a long go-there walk).
    Scan in place to re-find it, sweeping toward the side it was LAST seen FIRST
    (omega>0 = left, verified on the robot): a few small steps that way, then a
    long sweep back across center. hint_bearing<0 = last seen left -> sweep left
    first. Returns the pose once seen, or None if the full scan comes up empty."""
    toward_left = (hint_bearing is None) or (hint_bearing < 0)  # default: left first
    near, far = (0.3, -0.3) if toward_left else (-0.3, 0.3)
    side = "left" if toward_left else "right"
    print(f"  marker lost — scanning to re-acquire ({side} first, last bearing "
          f"{'n/a' if hint_bearing is None else f'{hint_bearing:+.0f}'})...")
    plan = [near] * 3 + [far] * 6  # ~48 deg toward last-seen side, then ~96 deg across
    for om in plan:
        g1.walk(0.0, 0.0, om, am.TURN_T)
        p = measure(1)  # scan wants a fast seen-or-not; a blip-miss just scans on
        if p:
            print(f"      re-acquired: fwd {p['fwd']:.0f} cm, bearing {p['bearing']:+.1f} deg")
            return p
    return None


def _measure_or_reacquire(dry, hint_bearing=None, n=2):
    """measure(n); on marker loss, scan to re-find it (skipped in dry-run),
    sweeping toward hint_bearing's side first."""
    p = measure(n)
    if p is None and not dry:
        p = reacquire(hint_bearing)
    return p


def _push_forward(store, L, dry, allow_fast=False):
    """Walk L cm forward as ONE continuous push — the whole distance computed once,
    executed as back-to-back sub-walks each <= WALK_MAX_DURATION with NO camera
    measuring in between (that mid-move pause was the annoyance). When allowed,
    pushes >= FAST_MIN_CM take the FAST tier (0.5 m/s, its own model); short
    pushes and the blind table creep stay on the calibrated slow tier. Returns
    (T_total, n_chunks, fast); n_chunks>1 means it was split, so it's not a
    clean single-command calibration sample."""
    fast = allow_fast and L >= am.FAST_MIN_CM
    slope, b = am.fit_walk_fast(store) if fast else am.fit_walk(store)
    speed = am.FAST_SPEED if fast else am.WALK_SPEED
    T_total = max(0.0, (L - b) / slope)
    chunks = max(1, math.ceil(T_total / g1.WALK_MAX_DURATION)) if T_total > 0.05 else 0
    tag = f" ({chunks} back-to-back sub-walks)" if chunks > 1 else ""
    print(f"      {L:.0f} cm in one push: T={T_total:.1f}s @ {speed:.2f} m/s"
          f"{' FAST' if fast else ''}{tag}")
    if dry or chunks == 0:
        return T_total, chunks, fast
    remaining = T_total
    while remaining > 0.05:
        t = min(remaining, g1.WALK_MAX_DURATION)
        g1.walk(speed, 0.0, 0.0, t)
        remaining -= t
    return T_total, chunks, fast


def plan_and_go(store, target_fwd, label, dry, undershoot=APPROACH_UNDERSHOOT,
                measure_after=True, allow_fast=False):
    """Single-frame motion plan: from ONE measurement, turn in place to face the
    marker (null the bearing) and then push forward to land ~target_fwd cm in
    front of it — executed turn-then-push with NO camera frames in between.

    This replaces the old measure->nudge->measure loop: the whole move is computed
    from the first frame, so the robot walks the plan out in one go. Returns the
    arrival pose (a fresh frame), or None on marker loss."""
    print(f"\n-- {label}: plan from one frame -> ~{target_fwd:.0f} cm, facing the marker --")
    p = _measure_or_reacquire(dry)
    if p is None:
        print("  ABORT: lost the marker.")
        return None

    rng = math.hypot(p["x"], p["fwd"])   # true straight-line distance to the marker
    print(f"  frame: fwd {p['fwd']:.0f} cm, x {p['x']:+.0f} cm, "
          f"range {rng:.0f} cm, bearing {p['bearing']:+.1f} deg")

    # 1) turn in place to face the marker (null the bearing) — one command,
    #    duration extended for big bearings instead of chunking
    turn = am.plan_turn(store, p["bearing"])
    if turn is not None:
        omega, t, deg = turn
        print(f"  TURN omega={omega:+.2f} t={t:.1f}s  "
              f"(bearing {p['bearing']:+.1f}->0, ~{deg:.0f} deg)")
        if not dry:
            g1.walk(0.0, 0.0, omega, t)
    else:
        print(f"  bearing {p['bearing']:+.1f} within the turn floor — no turn.")

    # 2) forward push(es); heading now points at the marker, so the straight-line
    #    range is what we close, not the (pre-turn) fwd component. A fresh fast
    #    model is seeded to undershoot (safe), so allow a couple of corrective
    #    top-up pushes — each remeasured, each a new calibration sample — until
    #    the remaining gap is within TOPUP_TOL.
    p2 = p
    for hop in range(3):
        rng = math.hypot(p2["x"], p2["fwd"])
        gap = rng - target_fwd
        if hop == 0 and gap * undershoot <= 2.0:
            print(f"  already at range {rng:.0f} cm (target {target_fwd:.0f}); no walk needed.")
            return p if dry else _measure_or_reacquire(dry)
        if hop > 0:
            if gap <= TOPUP_TOL:
                break
            print(f"  TOP-UP: still {gap:.0f} cm past target.")
        L = gap * undershoot
        T_total, chunks, fast = _push_forward(store, L, dry, allow_fast)
        if dry:
            return p

        # Blind arrival: when target_fwd is inside the marker's ~40 cm detection
        # floor (the final creep to the table), there's no frame to confirm —
        # measuring would only trigger a reacquire SCAN right next to the table.
        if not measure_after:
            print(f"  arrived (blind creep to ~{target_fwd:.0f} cm; no confirming frame — "
                  f"the marker is too close to detect here).")
            return p

        # 3) one confirmation frame; the turn was in place, so range closed ==
        #    forward travelled -> a clean seconds->cm sample for the tier that
        #    walked it (only if it was one command). If the push drifted the
        #    marker out of frame, scan toward the side it was last seen.
        prev_bearing = p2["bearing"]
        p2 = _measure_or_reacquire(dry, hint_bearing=prev_bearing)
        if p2 is None:
            print("  ABORT: lost the marker after the move.")
            return None
        moved = rng - math.hypot(p2["x"], p2["fwd"])
        print(f"  arrived: fwd {p2['fwd']:.0f} cm, bearing {p2['bearing']:+.1f} deg "
              f"(moved {moved:.0f} cm)")
        if chunks == 1 and moved > 0:
            store["walk_fast" if fast else "walk"].append(
                {"T": round(T_total, 3), "cm": round(moved, 1)})
    return p2


def confirm(prompt, dry, assume_yes):
    """y/N gate. Auto-yes under --yes; in dry-run shows the plan; on a non-tty
    (e.g. piped) defaults to NO so a run never hangs waiting on input."""
    if dry:
        print(f"  [{prompt}]  (dry-run — showing the plan)")
        return True
    if assume_yes:
        print(f"  {prompt} -> yes (--yes)")
        return True
    if not sys.stdin.isatty():
        print(f"  {prompt} -> no (no interactive terminal)")
        return False
    try:
        return input(f"  {prompt} [y/N]: ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _turn_by(store, signed_deg, dry):
    """In-place turn by signed_deg (+ = right, omega<0) as ONE command where
    possible: omega up to the calibrated max, then the DURATION extends
    (deg ~ k*omega*t, up to TURN_T_MAX) — no more 1 s chunking. Only rotations
    beyond one maxed-out command (~115 deg) split, back-to-back, no camera."""
    if abs(signed_deg) < 3.0:
        return
    direction = "right" if signed_deg > 0 else "left"
    k = am.turn_k(store, direction)
    per_max = k * am.TURN_OMEGA_MAX * am.TURN_T_MAX / am.TURN_T  # one command's ceiling
    n = max(1, math.ceil(abs(signed_deg) / per_max))
    deg_each = abs(signed_deg) / n
    omega_mag = min(am.TURN_OMEGA_MAX, max(am.TURN_OMEGA_MIN, deg_each / k))
    t = max(am.TURN_T_MIN, min(am.TURN_T_MAX, deg_each / (k * omega_mag) * am.TURN_T))
    omega = -omega_mag if signed_deg > 0 else omega_mag
    print(f"      TURN {signed_deg:+.0f} deg: {n}x omega={omega:+.2f} t={t:.1f}s")
    if dry:
        return
    for _ in range(n):
        g1.walk(0.0, 0.0, omega, t)


def _strafe_speed(store):
    """cm/s sideways at |vy|=STRAFE_VY, averaged over stored strafe samples."""
    sp = [abs(p["dx"]) / p["t"] for p in store.get("strafe", []) if p.get("t")]
    return sum(sp) / len(sp) if sp else STRAFE_SPEED


def _strafe_by(store, signed_cm, dry):
    """Sideways shuffle by signed_cm (+ = robot's RIGHT), as back-to-back commands
    with NO camera in between. Strafe is imprecise (~40% CoV) and yaw-coupled, so
    this is one open-loop move judged only by the final measurement."""
    if abs(signed_cm) < 3.0:
        return
    left = signed_cm < 0
    vy = STRAFE_VY if left == STRAFE_LEFT_IS_POS_VY else -STRAFE_VY
    T_total = abs(signed_cm) / _strafe_speed(store)
    print(f"      STRAFE {'LEFT' if left else 'RIGHT'} {abs(signed_cm):.0f} cm "
          f"(vy={vy:+.2f}, T={T_total:.1f}s)")
    if dry:
        return
    remaining = T_total
    while remaining > 0.05:
        t = min(remaining, g1.WALK_MAX_DURATION)
        g1.walk(0.0, vy, 0.0, t)
        remaining -= t


def square_onto_normal(store, dry, yaw_tol=YAW_SQUARE_TOL, pose=None):
    """Two-move square-up at the reading distance: from ONE measurement, ORIENT
    square to the marker face, then STRAFE sideways onto the table's normal. Both
    moves are computed up front and run back-to-back with NO detection in between
    (re-detecting mid-move kept losing the marker); a single final measurement
    reports the result. A pure in-place turn can't square -- off the normal it
    trades yaw for bearing 1:1 -- so strafe supplies the lateral move a turn cannot.

    Geometry (robot at origin facing +Y, marker M=(x,fwd), off-normal angle yaw):
    turn by -sign*yaw to face square, then strafe (perpendicular to the new heading)
    by  x*cos(yaw) + sign*fwd*sin(yaw)  to land on the normal, centered. This slide
    is bounded (<= |x|+fwd), unlike a tan-based one. `sign` = YAW_NORMAL_SIGN
    (+1: yaw>0 -> orient left, strafe right).

    `pose`: a just-taken measurement to plan from (e.g. the approach's arrival
    frame) — skips this function's own initial measure, saving ~3 ssh snaps of
    standing still right after the walk. Robot must not have moved since."""
    p = pose if pose is not None else \
        _measure_or_reacquire(dry, n=3)  # yaw is read here -> full median-of-3
    if p is None:
        print("  ABORT: lost the marker.")
        return None
    if not p["yaw_reliable"]:
        print(f"  yaw unreliable at {p['fwd']:.0f} cm — skipping square-up.")
        return p
    if abs(p["yaw"]) <= yaw_tol:
        print(f"  already square: yaw {p['yaw']:+.1f} deg (within {yaw_tol:.0f}), "
              f"bearing {p['bearing']:+.1f} deg.")
        return p

    x, fwd, yaw = p["x"], p["fwd"], p["yaw"]
    yr = math.radians(yaw)
    orient_deg = -YAW_NORMAL_SIGN * yaw                # 1) turn square to the marker face
    strafe_cm = x * math.cos(yr) + YAW_NORMAL_SIGN * fwd * math.sin(yr)  # 2) slide onto normal
    strafe_cm = max(-STRAFE_CAP, min(STRAFE_CAP, strafe_cm))            #    (perp to new heading)
    print(f"  off-normal yaw {yaw:+.1f} deg at {fwd:.0f} cm -> 2 moves "
          f"(no detect between): orient {orient_deg:+.0f} deg, then strafe {strafe_cm:+.0f} cm")
    _turn_by(store, orient_deg, dry)
    _strafe_by(store, strafe_cm, dry)
    if dry:
        return p

    q = _measure_or_reacquire(dry, n=3)  # final yaw readout
    if q is None:
        print("  final measure: marker not seen (likely just off-frame after the strafe).")
        return p

    # 90-DEG LOCK: iterate orient-only micro-turns (short-duration fine steps
    # down to ~6.5 deg; below that a deliberate small overshoot past zero) until
    # |yaw| <= yaw_tol or it stops improving. Strafe is NOT repeated — a
    # micro-turn barely moves the bearing, and re-strafing couples noise back in.
    for _ in range(MAX_LOCK_ITERS):
        if not q["yaw_reliable"] or abs(q["yaw"]) <= yaw_tol:
            break
        prev = abs(q["yaw"])
        print(f"  LOCK: yaw {q['yaw']:+.1f} deg -> micro-turn")
        _turn_by(store, -YAW_NORMAL_SIGN * q["yaw"], dry)
        if dry:
            break
        q2 = _measure_or_reacquire(dry, n=3)
        if q2 is None:
            break
        q = q2
        if abs(q["yaw"]) >= prev - 0.5:
            break  # turn floor / measurement noise — accept what we have

    tag = "" if abs(q["yaw"]) <= yaw_tol else \
        f"  (residual {q['yaw']:+.0f} deg — at the turn/noise floor)"
    print(f"  after square-up: fwd {q['fwd']:.0f} cm, bearing {q['bearing']:+.1f} deg, "
          f"yaw {q['yaw']:+.1f} deg.{tag}")
    return q


def full_align(store, dry, assume_yes=False):
    """Two-frame approach + strafe/orient square-up: from the first frame plan the
    whole move to ~1 m in front of the marker (turn to face + fast forward push,
    with top-ups) and walk it out; then square up in two moves (orient square to
    the face, then strafe onto the table's normal) so the robot is square AND
    facing it; then, on confirmation, blind-creep straight in to the table
    standoff. No incremental stepping."""
    p = plan_and_go(store, TARGET_FWD, "APPROACH", dry, allow_fast=True)
    if p is None:
        return False
    print("\n-- SQUARE onto the table normal --")
    p = square_onto_normal(store, dry, pose=p)  # reuse arrival frame: no re-measure
    if p is None:
        return False
    if not dry:
        rel = "" if p["yaw_reliable"] else " (yaw unreliable)"
        print(f"\n-- at reading distance: fwd {p['fwd']:.0f} cm, "
              f"bearing {p['bearing']:+.1f} deg, yaw {p['yaw']:+.1f} deg{rel} --")
    if confirm("Walk up to the table?", dry, assume_yes):
        if plan_and_go(store, TABLE_FWD, "TO TABLE", dry,
                       undershoot=TABLE_UNDERSHOOT, measure_after=False) is None:
            return False
    else:
        print("  stopping at the reading distance (not walking to the table).")
    return True


def main():
    ap = argparse.ArgumentParser(description="Square the G1 up to a marker, then optionally walk to the table.")
    ap.add_argument("--dry-run", action="store_true", help="plan the moves, no motion")
    ap.add_argument("--yes", action="store_true", help="auto-confirm the walk-to-table gate")
    args = ap.parse_args()

    store = am.load_store()
    store.pop("_flipped", None)  # reset per-run flip guard
    print("=" * 60)
    print(f"  Align + approach  ({'DRY-RUN' if args.dry_run else 'LIVE, smallest speed'})")
    print("  frame -> plan (aim turn + fast push) -> ~1 m -> square-up -> gate -> table")
    print("=" * 60)

    if not args.dry_run and not g1.check():
        sys.exit(1)

    ok = full_align(store, args.dry_run, assume_yes=args.yes)
    if not args.dry_run:
        store.pop("_flipped", None)
        am.save_store(store)
        print(f"  saved ({len(store['walk'])} walk, {len(store['turn'])} turn samples).")
    print("  " + ("done." if ok else "aborted."))


if __name__ == "__main__":
    main()
