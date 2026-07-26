#!/usr/bin/env python3
"""go_to_marker.py — ONE command: drive the G1 up to an ArUco marker, end to end.

This is the single entry point for the marker approach. It runs the whole
sequence with no stops and no mode-picking:

  1. robot link check     (g1.check)
  2. enable locomotion    (g1.start; skip with --no-start)
  3. orient -> walk in -> square up -> walk to the table
     (align_to_marker.full_align: one-frame plan, aim turn + fast forward push to
      ~1 m in front, strafe/orient square onto the table normal, then a blind
      creep to the final standoff). Self-calibrating: every walk and turn is
      appended to approach_calib.json, so each run refines the motion models.

The heavy geometry/calibration logic lives in align_to_marker.py (tested); this
script just chains bring-up + the full pipeline into one obvious command. Uses
the belt-mounted D455 (pinned in tools/snap.py). Runs on system python3 (needs
cv2 via snap.py), robot in walk mode.

  python3 go_to_marker.py                 # full run, stops ~10 cm from the table
  python3 go_to_marker.py --standoff 20   # stop ~20 cm from the table instead
  python3 go_to_marker.py --dry-run       # plan only, no motion
  python3 go_to_marker.py --no-start      # locomotion already enabled
  python3 go_to_marker.py --stop-short    # stop ~1 m in front (skip the table leg)
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
import approach_marker as am  # noqa: E402  store load/save
import align_to_marker as al  # noqa: E402  full_align() — the tested pipeline


def main():
    ap = argparse.ArgumentParser(
        description="One command: orient, walk to, square up to, and reach an ArUco marker.")
    ap.add_argument("--standoff", type=float, default=al.TABLE_FWD,
                    help=f"final camera-to-marker standoff cm (default {al.TABLE_FWD:.0f}); "
                         "raise to stop further from the table")
    ap.add_argument("--dry-run", action="store_true", help="plan the moves, no motion")
    ap.add_argument("--no-start", action="store_true",
                    help="skip locomotion enable (already in walk/loco mode)")
    ap.add_argument("--stop-short", action="store_true",
                    help="stop ~1 m in front (skip the final walk to the table)")
    args = ap.parse_args()

    al.TABLE_FWD = args.standoff  # apply the requested final standoff to the pipeline

    print("=" * 60)
    print("  GO TO MARKER — orient -> approach -> square -> table")
    print(f"  final standoff ~{args.standoff:.0f} cm   "
          f"({'DRY-RUN' if args.dry_run else 'LIVE'})")
    print("=" * 60)

    if not args.dry_run:
        if not g1.check():
            sys.exit(1)
        if not args.no_start:
            print("  enabling locomotion ...")
            g1.start()
            # no settle sleep: the first camera frame (~4.5 s ssh round-trip)
            # runs before any walk and covers the loco FSM settling time.

    store = am.load_store()
    store.pop("_flipped", None)  # reset per-run flip guard
    # assume_yes: reaching the table is this script's whole point, so the
    # walk-to-table gate is auto-confirmed unless the user asked to stop short.
    ok = al.full_align(store, args.dry_run, assume_yes=not args.stop_short)
    if not args.dry_run:
        store.pop("_flipped", None)
        am.save_store(store)
        print(f"  calibration saved "
              f"({len(store['walk'])} walk, {len(store['turn'])} turn samples).")
    print("  " + ("DONE." if ok else "ABORTED — see above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
