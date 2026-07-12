"""Orin-side RGBD grabber for the G1's RealSense — deployed to the robot, run over ssh.

The `_`-prefix keeps the assistant's tool loader from importing it on the PC (it
imports pyrealsense2, which only exists on the Orin host's python3.8). It is the
source-of-truth copy; `see_camera.py::_capture_g1()` deploys it to the Orin and
runs it there.

Grabs ONE colour frame + depth-frame aligned to the colour lens (same warmup,
rs.align and depth-scale as see_camera._capture_realsense) and writes a single
compressed npz the PC can np.load:
    color : HxWx3 uint8, RGB
    depth : HxW    float32, metres (aligned to colour)

Usage (on the Orin):  python3 _g1_capture_rgbd.py /tmp/g1_snap.npz
"""

import sys

import numpy as np
import pyrealsense2 as rs

CAP_W, CAP_H = 640, 480
WARMUP = 10


def main(out_path):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, CAP_W, CAP_H, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, CAP_W, CAP_H, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    try:
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)  # warp depth into the colour frame
        frames = None
        for _ in range(WARMUP):  # let auto-exposure settle
            frames = align.process(pipe.wait_for_frames())
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())
        depth_raw = np.asanyarray(frames.get_depth_frame().get_data())
    finally:
        pipe.stop()

    rgb = color_bgr[:, :, ::-1]                     # BGR → RGB
    depth_m = depth_raw.astype(np.float32) * scale  # units → metres
    np.savez_compressed(out_path, color=np.ascontiguousarray(rgb), depth=depth_m)
    print(f"saved {out_path}: color{rgb.shape} depth{depth_m.shape} "
          f"scale={scale:.6f} depth_median={float(np.median(depth_m[depth_m > 0])):.2f}m")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g1_snap.npz")
