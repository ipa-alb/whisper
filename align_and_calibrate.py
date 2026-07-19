#!/usr/bin/env python3
"""One command, end-to-end, no stops: bring the robot up, walk it to the ArUco
marker, and recompute the motion calibration + stats.

Chains the whole pipeline that was previously run by hand:
  1. robot link check          (g1.check)
  2. enable locomotion         (g1.start)   -- skip with --no-start
  3. approach the marker       (align_to_marker.full_align: one-frame plan ->
     aim turn + fast push to ~1 m -> square onto the table normal -> table,
     self-calibrating, with the re-acquire-on-loss scan)
  4. recompute + print stats   (calib_stats.print_stats)

Every approach walk and turn is appended to approach_calib.json, so each run
also refines the walk/turn models. Runs on system python3 (needs cv2 via
snap.py), robot in walk mode.

  python3 align_and_calibrate.py              # full pipeline, one pass
  python3 align_and_calibrate.py --dry-run    # plan only, no motion, then stats
  python3 align_and_calibrate.py --no-start   # locomotion already enabled
"""

import argparse
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "tools"))                 # snap.py
sys.path.insert(0, str(_HERE.parents[0] / "g1_agent"))   # g1.py
os.environ["SEE_CAMERA_SOURCE"] = "g1"

import g1  # noqa: E402
import approach_marker as am  # noqa: E402  store load/save + models
import align_to_marker as al  # noqa: E402  full_align() + reacquire()
import calib_stats as cs  # noqa: E402      stats


def main():
    ap = argparse.ArgumentParser(
        description="End-to-end: bring-up -> square to marker -> recalibrate + stats.")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan one move, no motion; still prints stats")
    ap.add_argument("--no-start", action="store_true",
                    help="skip locomotion enable (already in walk/loco mode)")
    ap.add_argument("--yes", action="store_true",
                    help="auto-confirm the walk-to-table gate (no prompt)")
    args = ap.parse_args()

    print("#" * 66)
    print("  ALIGN + CALIBRATE  (one-shot pipeline)")
    print("#" * 66)

    if not args.dry_run:
        print("\n[1/4] robot link check ...")
        if not g1.check():
            sys.exit(1)
        if not args.no_start:
            print("\n[2/4] enabling locomotion ...")
            g1.start()
            # no settle sleep: the first camera frame (~4.5 s ssh round-trip)
            # happens before any walk and covers the loco FSM settling time
        else:
            print("\n[2/4] locomotion enable skipped (--no-start).")
    else:
        print("\n[1-2/4] dry-run: skipping link check / locomotion enable.")

    print("\n[3/4] one-frame plan -> ~1 m -> frame -> gate -> table (self-calibrating) ...")
    store = am.load_store()
    store.pop("_flipped", None)          # reset per-run flip guard
    ok = al.full_align(store, args.dry_run, assume_yes=args.yes)
    if not args.dry_run:
        store.pop("_flipped", None)
        am.save_store(store)
        print(f"  calibration saved "
              f"({len(store['walk'])} walk, {len(store['turn'])} turn samples).")

    print("\n[4/4] recalibrated stats:\n")
    cs.print_stats(am.CALIB_PATH)

    print("\n  " + ("PIPELINE DONE." if ok else "PIPELINE ABORTED — see above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
