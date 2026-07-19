# G1 Marker-Approach Motion Calibration — Statistics

Camera-in-the-loop calibration of the Unitree G1's open-loop locomotion
primitives (forward walk, in-place turn, lateral strafe), measured against a
fixed ArUco marker using the robot's head RealSense as the ruler. The whole
pipeline runs from one command — bring-up check, enable locomotion, square to
the marker (self-calibrating), then recompute and print these stats:

```bash
cd whisper
python3 align_and_calibrate.py    # one-shot: bring-up -> align -> recalibrate + stats
python3 align_and_calibrate.py --dry-run   # plan only, no motion
python3 calib_stats.py            # just recompute stats from approach_calib.json
```

- **Source data:** `approach_calib.json` (self-accumulating store).
- **Reference:** single ArUco marker (`snap.MARKER_SIZE_M`), pose via `cv2.solvePnP`.
- **Measurement:** each sample is a `snap → move → snap` triple; the marker's
  forward (optical-axis Z) component gives distance, `atan2(X, Z)` gives bearing,
  and the marker-plane normal gives yaw.
- **Commanded speed (walk):** 0.15 m/s (the calibrated "slow" tier).
- **Turn / strafe duration:** fixed 1.0 s (past the ~0.75 s gait deadband).

## 1. Forward walk model

Rather than a single cm/s figure, distance is modelled affinely because every
`walk()` ramps up from and down to a standstill, so short walks cover
proportionally less ground than long ones:

$$d(T) = m\,T + b$$

fit by ordinary least squares and inverted to plan a move: `T = (d − b) / m`.

| Quantity | Value |
|---|---|
| Samples (n) | 17 |
| Model | **d = 9.80·T + 0.02** (cm, T in s) |
| Effective steady speed, m | **9.80 cm/s** |
| Ramp loss, b | +0.02 cm (start/stop cost) |
| Coefficient of determination, R² | **0.993** |
| Residual RMSE | 4.1 cm |
| Per-command rate spread | mean 9.84 cm/s, SD 0.71, CoV **7.2 %**, range 8.23–11.56 |

**Note on model evolution (self-calibration).** The store refits on every run,
so the model tightens as samples accumulate and span more of the duration range:

| After | n | Model | R² | RMSE |
|---|---|---|---|---|
| Seed (short walks only) | 9  | d = 9.17·T + 1.33 | 0.970 | 3.0 cm |
| Align run 1 | 11 | d = 10.20·T − 1.43 | 0.988 | 4.0 cm |
| Align run 2 | 13 | d = 9.68·T + 0.27 | 0.988 | 4.6 cm |
| Align run 3 | 15 | d = 9.77·T + 0.04 | 0.991 | 4.4 cm |
| Align run 4 | 17 | d = 9.80·T + 0.02 | **0.993** | 4.1 cm |

The seed fit — built from short walks only — under-estimated steady speed by
~10 %. Each alignment run contributed a *long* (13–15 s) go-there sample; those
long-baseline points anchored the steady-state slope, which **converged to
≈ 9.8 cm/s** with the intercept collapsing to ≈ 0 and R² rising to 0.991 by run 3.
The ~8 % run-to-run spread on the 15 s move (consistent with the per-command CoV)
is the residual noise the fit averages over. RMSE settled around 4.4 cm — larger
than the seed's 3.0 cm only because the longer moves carry larger absolute
residuals than the short seed walks.

## 2. In-place turn model

Turn rotation is modelled through the origin (`deg = k·ω` at t = 1 s), fit
separately per direction to expose any left/right asymmetry:

| Direction | Model | R² | RMSE | n |
|---|---|---|---|---|
| Left  | deg = 54.9·ω | 0.910 | 2.6° | 10 |
| Right | deg = 54.0·ω | 0.909 | 2.8° | 7 |

- **Gain:** ≈ 54 °/(rad·s) at 1 s.
- **L/R asymmetry:** +1.7 % (left vs right) — effectively symmetric.
- **Resolution floor:** the platform will not take a turn step below
  ω ≈ 0.15 rad·s, i.e. ~8° per command, which sets the achievable squaring
  precision (the alignment loop accepts residual yaw once it hits this floor).

## 3. Lateral strafe characterisation

Strafe is characterised (not used for squaring) to justify the turn-only
alignment strategy: strafing translates the robot but its rotation coupling
cancels any usable yaw change.

| Command (vy, 1 s) | Δx mean | Δx SD | \|Δyaw\| mean | n |
|---|---|---|---|---|
| −0.20 m/s | −13.2 cm | 6.6 | 2.3° | 12 |
| +0.20 m/s | +15.5 cm | 2.6 | 6.1° | 4 |

The large Δx dispersion (SD up to 6.6 cm) and near-random yaw coupling confirm
strafe cannot drive yaw → 0; alignment therefore uses **turn + forward only**.

## 4. Alignment run (worked example)

The recalibration data above came from a live `align_to_marker.py` pass. It
performs a "go-there" approach (largest single walk the platform allows,
clamped by `WALK_MAX_DURATION = 15 s ≈ 1.35 m at the slow tier`) to ~1 m, then
squares to the marker face by turning:

| Iter | fwd (cm) | yaw (°) | bearing (°) | action |
|---|---|---|---|---|
| 0 | 302 | — | — | FORWARD 177 cm (T=15 s) → 147 cm |
| 1 | 147 | — | — | FORWARD 38 cm (T=3.95 s) → 102 cm |
| 2 | 102 | +39.9 | −30.7 | TURN ω=+0.50, 1 s |
| 3 | 123 | +13.0 | −5.1 | TURN ω=+0.23, 1 s |
| 4 | 123 | **+2.0** | +8.7 | **SQUARED** (within ±6° floor) |

Two go-there moves reached ~1 m; two turns squared the robot to +2.0° yaw. Every
move (2 walks + 2 turns) was appended to the store, feeding the models above —
the closed loop is self-calibrating.

A second pass from 289 cm reached square in the approach alone — after
289 → 150 → 111 cm the yaw was already −5.6° (within the ±6° floor), so **no turn
step was required**. This is the intended outcome when the robot is placed
roughly facing the marker: the go-there approach both closes distance and, by
walking straight along the optical axis, tends to reduce residual yaw.

### 4.1 Open-loop heading drift (limitation of the go-there strategy)

The single long go-there walk trades marker-tracking robustness for fewer steps.
The G1's forward walk is not perfectly straight: over a 13–15 s open-loop command
it accumulates several degrees of **heading drift**, and on one pass this was
enough to curve the robot off course and lose the marker from the camera frame
entirely (it ended ~2.8 m from the marker, ~63 cm laterally displaced, pointing
into the room). This is the trade-off against the incremental closed-loop
approach, which re-acquires and re-centres the marker every short step and so
cannot drift out of view.

**Mitigation implemented:** on marker loss the aligner no longer aborts — it runs
an in-place **scan** (small turn steps one way, then a sweep back across centre)
to re-acquire the marker, then resumes. In the recovery event above the marker
was re-found after ~32° of scan and the subsequent pass completed normally. The
robustness cost of long open-loop moves is therefore bounded by a re-acquisition
scan rather than a failed run.

## 5. Reproducibility / provenance

- **Alignment passes run:** 4 completed (+1 aborted-then-recovered by scan).
- **Sample counts:** 17 forward-walk, 17 turn, 16 strafe.
- **Platform:** camera source `SEE_CAMERA_SOURCE=g1` (head RealSense on the
  Orin, mounted 180° rolled and un-rotated by the grabber).
- **Regenerate stats:** `python3 calib_stats.py [path-to-json]`.
