#!/usr/bin/env python3
"""Single-shot ArUco: 2D position (x, y) and yaw of a detected marker.

Two capture sources, switched with the same env var as see_camera.py's `look`:

  * SEE_CAMERA_SOURCE=g1  — the RealSense is on the G1 robot's Orin, not this PC.
    We ssh in, run the deployed grabber (_g1_capture_rgbd.py → python3.8 +
    pyrealsense2 on the Orin), pull back one compressed npz (colour + depth
    aligned to colour + colour-lens intrinsics) and run ArUco on it here.
    This is the mode to use now that the camera lives inside the robot.
  * unset (default)       — local USB RealSense on this PC, via pyrealsense2.

Either way the maths is identical: solvePnP on the colour frame with the colour
intrinsics gives the marker pose; depth (aligned to colour) is a cross-check.
"""

import math
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

# ArUco setup
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

MARKER_SIZE_M = 0.178  # measured physical marker size in meters

# Belt-mounted D455 — the camera the marker approach uses (rolled 180°, so the
# Orin grabber un-rotates it). With the head D435i also plugged into the Orin,
# the grabber would otherwise open whichever device enumerates first, so we pin
# this serial to guarantee the approach measures from the belt cam, not the head.
# Override with G1_CAM_SERIAL (e.g. the head D435i 327122073256 for `look`).
APPROACH_CAM_SERIAL = "146222254681"


def print_marker_report(marker_id, x_m, y_m, dist_m, depth_m, yaw_deg):
    """Print a top-down map of where the marker sits relative to the camera.

    Schematic (not to scale). The camera is FIXED at [cam], bottom-centre, looking
    'up' the page — it is the sensor, so it never moves. Only the marker moves: it
    is drawn to the right or left of the camera to match the sign of X. The
    right-angle corner sits directly below the marker, so the three legs are the
    real quantities: vertical = Y (forward), horizontal = X (sideways), and the
    hypotenuse = straight-line distance.
    """
    W, H = 60, 8
    grid = [[" "] * W for _ in range(H)]

    def put(r, c, s):
        for i, ch in enumerate(s):
            if 0 <= c + i < W and 0 <= r < H:
                grid[r][c + i] = ch

    cam_c = 29                       # camera pinned here on EVERY render
    to_right = x_m >= 0
    side = "right" if to_right else "left"
    mk_c = cam_c + 14 if to_right else cam_c - 14   # marker moves to its real side

    # Marker + label, and the vertical leg (Y) down to the right-angle corner
    put(0, mk_c, "*")
    put(0, mk_c + 2 if to_right else mk_c - 7, "marker")
    for r in range(1, 7):
        put(r, mk_c, "|")
    put(7, mk_c, "+")

    # Hypotenuse (distance) from the fixed camera up to the marker
    for r in range(1, 7):
        if to_right:
            put(r, (mk_c - 2) - 2 * (r - 1), "/")
        else:
            put(r, (mk_c + 2) + 2 * (r - 1), "\\")

    # Baseline (X leg), camera pinned at cam_c regardless of side
    if to_right:
        for c in range(cam_c + 3, mk_c):
            put(7, c, "-")
    else:
        for c in range(mk_c + 1, cam_c - 2):
            put(7, c, "-")
    put(7, cam_c - 2, "[cam]")

    # Value labels in the open margins: Y beside the vertical leg, distance out
    # past the hypotenuse on the far side.
    put(2, mk_c + 2 if to_right else mk_c - 13, "Y (forward)")
    put(3, mk_c + 2 if to_right else mk_c - 13, f"{y_m:+.3f} m")
    dist_c = 2 if to_right else W - 11
    put(3, dist_c, "distance")
    put(4, dist_c, f"{dist_m:.3f} m")

    print()
    print(f"  ---- Marker ID {marker_id} ----")
    print()
    for row in grid:
        print("  " + "".join(row).rstrip())
    print(f"        X ({side}) {x_m:+.3f} m   (0 = straight ahead)")
    print()
    print(f"        Yaw         : {yaw_deg:+.1f} deg   (0 = facing camera, + = CW)")
    print(f"        Depth check : {depth_m:.3f} m   (depth sensor; should match distance)")
    print("  ----------------------")


def rotation_matrix_to_yaw(rvec):
    """Extract yaw from marker normal projected onto the XZ (ground) plane.
    Returns 0 when marker faces the camera, + = rotated CW."""
    R, _ = cv2.Rodrigues(rvec)
    # marker Z-axis (normal) in camera frame = third column of R
    normal = R[:, 2]
    # project onto XZ plane (x=right, z=forward)
    # when marker faces camera, normal points toward camera: ~[0, 0, -1]
    # atan2(nx, -nz) gives 0 when normal = [0, 0, -1]
    yaw = math.atan2(normal[0], -normal[2])
    return math.degrees(yaw)


def _depth_at(depth_m, cx, cy):
    """Median non-zero depth (metres) in a small patch around a pixel, or 0.0.

    Depth has holes at edges/reflective surfaces (stored as 0); a single-pixel
    read is fragile, so we widen the patch until it holds enough valid samples.
    """
    if depth_m is None:
        return 0.0
    x, y = int(round(cx)), int(round(cy))
    h, w = depth_m.shape[:2]
    for r in (4, 8, 16):
        y0, y1 = max(0, y - r), min(h, y + r)
        x0, x1 = max(0, x - r), min(w, x + r)
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if valid.size >= 5:
            return float(np.median(valid))
    return 0.0


def _capture_g1():
    """Capture colour + depth + intrinsics from the G1 robot's Orin over ssh.

    Mirrors see_camera._capture_g1: run the deployed grabber on the Orin, scp back
    the npz. The grabber now also stores the colour-lens intrinsics, which ArUco
    solvePnP needs. Configure via env (defaults match see_camera):
      G1_ORIN            ssh target (default unitree@192.168.0.87)
      G1_CAPTURE_SCRIPT  remote grabber path (default /home/unitree/g1_capture_rgbd.py)

    Returns (color_bgr, depth_m, camera_matrix, dist_coeffs).
    """
    orin = os.environ.get("G1_ORIN", "unitree@192.168.0.87")
    remote_script = os.environ.get("G1_CAPTURE_SCRIPT", "/home/unitree/g1_capture_rgbd.py")
    remote_npz = "/tmp/g1_snap.npz"
    # Pin the head camera so a second RealSense (the belt D455) plugged into the
    # Orin can't be opened by enumeration order. ssh joins these args into one
    # remote shell command, so the env prefix applies to the grabber.
    serial = os.environ.get("G1_CAM_SERIAL", APPROACH_CAM_SERIAL)

    t0 = time.time()
    print("      [camera] snap (belt cam over ssh)...", flush=True)
    # The D455 intermittently times out on wait_for_frames() right after the
    # pipeline starts (RealSense "Frame didn't arrive"). A fresh grab almost
    # always succeeds, so retry a few times rather than aborting the whole
    # approach on one bad frame. Tune with G1_CAPTURE_RETRIES.
    attempts = max(1, int(os.environ.get("G1_CAPTURE_RETRIES", "3")))
    last_err = ""
    for attempt in range(1, attempts + 1):
        grab = subprocess.run(
            ["ssh", orin, f"G1_CAM_SERIAL={serial}", "python3", remote_script, remote_npz],
            capture_output=True, text=True, timeout=45,
        )
        if grab.returncode == 0:
            break
        last_err = (grab.stderr or grab.stdout).strip()[:300]
        if attempt < attempts:
            print(f"      [camera] grab {attempt}/{attempts} failed "
                  f"({last_err.splitlines()[-1][:80] if last_err else '?'}), retrying...",
                  flush=True)
            time.sleep(1.0)
    else:
        raise RuntimeError(f"orin capture failed after {attempts} attempts: {last_err}")

    with tempfile.TemporaryDirectory() as tmp:
        local_npz = os.path.join(tmp, "snap.npz")
        pull = subprocess.run(
            ["scp", "-q", f"{orin}:{remote_npz}", local_npz],
            capture_output=True, text=True, timeout=30,
        )
        if pull.returncode != 0:
            raise RuntimeError(f"scp of snapshot failed: {pull.stderr.strip()[:300]}")
        data = np.load(local_npz)
        if "camera_matrix" not in data:
            raise RuntimeError(
                "npz has no camera_matrix — the Orin's g1_capture_rgbd.py is an old "
                "copy without intrinsics; redeploy tools/_g1_capture_rgbd.py to it."
            )
        rgb = data["color"]
        depth_m = data["depth"].astype(np.float32)
        camera_matrix = data["camera_matrix"].astype(np.float64)
        dist_coeffs = data["dist_coeffs"].astype(np.float64)

    color_bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # RGB → BGR for cv2 draw/imwrite
    print(f"      [camera] done ({time.time() - t0:.1f}s)", flush=True)
    return color_bgr, depth_m, camera_matrix, dist_coeffs


def _make_local_capturer():
    """Open a local USB RealSense once; return (capture_fn, close_fn).

    capture_fn() -> (color_bgr, depth_m, camera_matrix, dist_coeffs) or None.
    pyrealsense2 is imported here so the G1 (ssh) path needs no local RealSense.
    """
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipe.start(cfg)

    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    intr = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float64)

    for _ in range(30):  # let auto-exposure settle
        pipe.wait_for_frames()

    def capture():
        frames = align.process(pipe.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
        color_bgr = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale
        return color_bgr, depth_m, camera_matrix, dist_coeffs

    return capture, pipe.stop


def detect(color_bgr, depth_m, camera_matrix, dist_coeffs):
    """Value-returning variant of process_frame: detect markers, return their
    poses as a list of dicts (empty if none). Prints nothing, saves nothing.

    Same pose maths as process_frame. Each dict:
      id, x_m (right+), y_m (forward), dist_m (straight-line), depth_m, yaw_deg.
    """
    corners, ids, _ = DETECTOR.detectMarkers(color_bgr)
    if ids is None or len(ids) == 0:
        return []

    half = MARKER_SIZE_M / 2
    obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)

    out = []
    for i, marker_id in enumerate(ids.flatten()):
        _, rvec, tvec = cv2.solvePnP(
            obj_points, corners[i][0], camera_matrix, dist_coeffs
        )
        rvec = rvec.flatten()
        tvec = tvec.flatten()
        x_m, y_m = float(tvec[0]), float(tvec[2])
        c = corners[i][0]  # corner order [TL, TR, BR, BL]
        cx, cy = c.mean(axis=0)
        # Keystone: ratio of the left vertical edge to the right vertical edge in
        # pixels. 1.0 = square-on; !=1 = viewed off the marker's normal. Unlike
        # solvePnP yaw this is flip-free (no planar pose ambiguity).
        left_edge = float(np.linalg.norm(c[0] - c[3]))
        right_edge = float(np.linalg.norm(c[1] - c[2]))
        keystone = left_edge / right_edge if right_edge > 1e-6 else 1.0
        out.append({
            "id": int(marker_id),
            "x_m": x_m,
            "y_m": y_m,
            "dist_m": math.hypot(x_m, y_m),
            "depth_m": _depth_at(depth_m, cx, cy),
            "yaw_deg": rotation_matrix_to_yaw(rvec),
            "keystone": keystone,
        })
    return out


def process_frame(color_bgr, depth_m, camera_matrix, dist_coeffs):
    """Detect ArUco markers, report pose for each, and save an annotated overlay."""
    overlay = color_bgr.copy()

    corners, ids, _ = DETECTOR.detectMarkers(color_bgr)
    if ids is None or len(ids) == 0:
        print("  [!] No ArUco marker detected.")
        cv2.imwrite("snap_result.png", overlay)
        print("  Saved → snap_result.png")
        return

    # marker corners in object frame (centered, z=0)
    half = MARKER_SIZE_M / 2
    obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)

    for i, marker_id in enumerate(ids.flatten()):
        _, rvec, tvec = cv2.solvePnP(
            obj_points, corners[i][0], camera_matrix, dist_coeffs
        )
        rvec = rvec.flatten()
        tvec = tvec.flatten()

        # RS/OpenCV camera frame: x=right, y=down, z=forward
        x_m = tvec[0]   # lateral
        y_m = tvec[2]   # forward distance
        yaw = rotation_matrix_to_yaw(rvec)

        # depth cross-check at marker center
        c = corners[i][0]
        center_px = c.mean(axis=0).astype(int)
        depth_dist = _depth_at(depth_m, center_px[0], center_px[1])

        print_marker_report(
            marker_id, x_m, y_m, math.hypot(x_m, y_m), depth_dist, yaw
        )

        # draw on overlay
        cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
        cv2.drawFrameAxes(overlay, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
        label = f"ID{marker_id} X:{x_m:+.2f} Y:{y_m:.2f} Yaw:{yaw:+.1f}"
        cv2.putText(
            overlay, label,
            (int(center_px[0]) - 100, int(center_px[1]) - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    cv2.imwrite("snap_result.png", overlay)
    print("  Saved → snap_result.png")


def main():
    source = "g1" if os.environ.get("SEE_CAMERA_SOURCE") == "g1" else "local"

    print("=" * 50)
    print("  ArUco 2D Positioning Test")
    print(f"  Source: {'G1 robot camera (over ssh)' if source == 'g1' else 'local RealSense'}")
    print(f"  Marker: DICT_5X5_50, size={MARKER_SIZE_M*1000:.0f}mm")
    print("  X = right, Y = forward, Yaw = CW from front")
    print("=" * 50)
    print("  Press ENTER to snap  |  q + ENTER to quit")
    print("=" * 50)

    capture, close = (None, lambda: None)
    if source == "local":
        capture, close = _make_local_capturer()

    try:
        while True:
            cmd = input("\n>>> ").strip().lower()
            if cmd == "q":
                break

            try:
                if source == "g1":
                    frame = _capture_g1()
                else:
                    frame = capture()
            except Exception as e:  # keep the loop alive on a transient failure
                print(f"  [!] Capture failed: {e}")
                continue

            if frame is None:
                print("  [!] Capture failed, try again.")
                continue

            process_frame(*frame)
    finally:
        close()


if __name__ == "__main__":
    main()
