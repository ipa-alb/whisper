#!/usr/bin/env python3
"""Dissertation stats for the G1 marker-approach calibration store.

Reads approach_calib.json and reports, with dispersion + fit quality:
  - forward walk model  cm = slope*T + b   (OLS, R^2, residual RMSE)
  - turn model per direction  deg = k*omega  (through origin, R^2)
  - strafe characterisation (dx per command, yaw coupling)
  - sample counts / provenance
Pure stdlib.  python3 calib_stats.py [path-to-json]
"""
import json, math, statistics, sys, pathlib

DEFAULT_PATH = pathlib.Path(__file__).resolve().parent / "approach_calib.json"


def ols(xs, ys):
    """cm = slope*T + b via ordinary least squares. Returns slope,b,r2,rmse,n."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom
    b = (sy - slope * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    rmse = math.sqrt(ss_res / n)
    return slope, b, r2, rmse, n


def origin_fit(xs, ys):
    """y = k*x through the origin. Returns k,r2,rmse,n."""
    n = len(xs)
    k = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    ybar = sum(ys) / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - k * x) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    rmse = math.sqrt(ss_res / n)
    return k, r2, rmse, n


def print_stats(path=DEFAULT_PATH):
    """Read a calibration store and print the walk/turn/strafe fit stats."""
    path = pathlib.Path(path)
    store = json.load(open(path))
    print("=" * 66)
    print(f"  G1 marker-approach calibration stats   ({path.name})")
    print(f"  completed runs on record: {store.get('runs', '?')}")
    print("=" * 66)

    # ---- forward walk ------------------------------------------------------
    w = store.get("walk", [])
    print(f"\nFORWARD WALK  (n={len(w)} snap-walk-snap samples, speed 0.15 m/s)")
    if len(w) >= 2:
        Ts = [p["T"] for p in w]
        cms = [p["cm"] for p in w]
        slope, b, r2, rmse, n = ols(Ts, cms)
        print(f"  model      cm = {slope:.3f}*T {b:+.3f}      (OLS, n={n})")
        print(f"  R^2 = {r2:.4f}   residual RMSE = {rmse:.2f} cm")
        print(f"  effective steady speed = {slope:.2f} cm/s")
        print(f"  ramp loss b = {b:+.2f} cm  (start/stop deceleration cost)")
        # per-command cm/s spread as a robustness figure
        rate = [p["cm"] / p["T"] for p in w]
        print(f"  raw cm/s: mean {statistics.mean(rate):.2f}, "
              f"sd {statistics.pstdev(rate):.2f}, "
              f"CoV {statistics.pstdev(rate)/statistics.mean(rate)*100:.1f}%  "
              f"[{min(rate):.2f}..{max(rate):.2f}]")

    # ---- turn, per direction -----------------------------------------------
    t = store.get("turn", [])
    print(f"\nTURN  (n={len(t)} samples, t=1 s, deg = |bearing change|)")
    for d in ("left", "right"):
        s = [p for p in t if p["dir"] == d and 0.9 <= p["t"] <= 1.1]
        if len(s) >= 2:
            om = [p["omega"] for p in s]
            dg = [p["deg"] for p in s]
            k, r2, rmse, n = origin_fit(om, dg)
            print(f"  {d:>5}:  deg = {k:.1f}*omega   R^2={r2:.3f}  "
                  f"RMSE={rmse:.1f} deg  (n={n})")
    # asymmetry
    kl = origin_fit([p["omega"] for p in t if p["dir"] == "left"],
                    [p["deg"] for p in t if p["dir"] == "left"])[0] if any(p["dir"] == "left" for p in t) else None
    kr = origin_fit([p["omega"] for p in t if p["dir"] == "right"],
                    [p["deg"] for p in t if p["dir"] == "right"])[0] if any(p["dir"] == "right" for p in t) else None
    if kl and kr:
        print(f"  L/R asymmetry: left gain is {(kl/kr-1)*100:+.1f}% vs right "
              f"({kl:.1f} vs {kr:.1f} deg per rad/s)")

    # ---- strafe ------------------------------------------------------------
    sf = store.get("strafe", [])
    if sf:
        print(f"\nSTRAFE  (n={len(sf)} samples, t=1 s) — why strafing can't square:")
        for vy in sorted({p["vy"] for p in sf}):
            g = [p for p in sf if p["vy"] == vy]
            dx = [p["dx"] for p in g]
            dyaw = [abs(p["dyaw"]) for p in g]
            print(f"  vy={vy:+.2f}:  dx mean {statistics.mean(dx):+.1f} cm "
                  f"(sd {statistics.pstdev(dx):.1f}, n={len(g)}),  "
                  f"|dyaw| mean {statistics.mean(dyaw):.1f} deg")

    print("\n" + "=" * 66)


if __name__ == "__main__":
    print_stats(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
