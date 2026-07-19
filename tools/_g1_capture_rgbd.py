"""Orin-side RGBD grabber for the G1's RealSense — deployed to the robot, run over ssh.

The `_`-prefix keeps the assistant's tool loader from importing it on the PC (it
imports pyrealsense2, which only exists on the Orin host's python3.8). It is the
source-of-truth copy; `see_camera.py::_capture_g1()` deploys it to the Orin and
runs it there.

Grabs ONE colour frame + depth-frame aligned to the colour lens (same warmup,
rs.align and depth-scale as see_camera._capture_realsense) and writes a single
compressed npz the PC can np.load:
    color        : HxWx3 uint8,   RGB
    depth        : HxW    float32, metres (aligned to colour)
    camera_matrix: 3x3    float32, colour-lens intrinsics [[fx,0,ppx],[0,fy,ppy],[0,0,1]]
    dist_coeffs  : 5      float32, colour-lens distortion

The two extra keys are what lets snap.py run ArUco solvePnP on a frame it never
captured itself; they are additive, so see_camera._capture_g1 (which reads only
color/depth) is unaffected.

The RealSense on the G1's head is mounted UPSIDE-DOWN (rolled 180 deg about its
optical axis) — raw frames come in with the ceiling at the bottom. We rotate
colour and depth 180 deg here, together (they are aligned, so they must rotate
as one), and move the intrinsics' principal point to match, so every consumer
(snap.py's solvePnP left/right + yaw, see_camera's YOLO + left/right) works in an
upright frame. Set ROTATE_180 = False if the camera is ever remounted upright.

Usage (on the Orin):  python3 _g1_capture_rgbd.py /tmp/g1_snap.npz
"""

import sys

import numpy as np
import pyrealsense2 as rs

CAP_W, CAP_H = 640, 480
WARMUP = 10
ROTATE_180 = True  # head camera is physically mounted upside-down


def main(out_path):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, CAP_W, CAP_H, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, CAP_W, CAP_H, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    try:
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)  # warp depth into the colour frame
        # Colour-lens intrinsics — the right ones for a depth-aligned-to-colour
        # frame, so snap.py's solvePnP on the colour image is correct.
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        frames = None
        for _ in range(WARMUP):  # let auto-exposure settle
            frames = align.process(pipe.wait_for_frames())
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())
        depth_raw = np.asanyarray(frames.get_depth_frame().get_data())
    finally:
        pipe.stop()

    rgb = color_bgr[:, :, ::-1]                     # BGR → RGB
    depth_m = depth_raw.astype(np.float32) * scale  # units → metres
    ppx, ppy = intr.ppx, intr.ppy
    if ROTATE_180:
        # Un-rotate the upside-down sensor: reverse rows+cols of both aligned
        # frames, and reflect the principal point about the image centre so the
        # intrinsics stay valid ((W-1)-ppx, (H-1)-ppy).
        rgb = rgb[::-1, ::-1]
        depth_m = depth_m[::-1, ::-1]
        ppx = (CAP_W - 1) - intr.ppx
        ppy = (CAP_H - 1) - intr.ppy
    camera_matrix = np.array([
        [intr.fx, 0, ppx],
        [0, intr.fy, ppy],
        [0, 0, 1],
    ], dtype=np.float32)
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)
    np.savez_compressed(
        out_path,
        color=np.ascontiguousarray(rgb),
        depth=depth_m,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    print(f"saved {out_path}: color{rgb.shape} depth{depth_m.shape} "
          f"scale={scale:.6f} depth_median={float(np.median(depth_m[depth_m > 0])):.2f}m")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g1_snap.npz")
