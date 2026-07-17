"""Tool: take one camera snapshot and report the objects YOLO detects (+ distance).

Lightweight, dependency-light vision. Two capture paths:

  1. RealSense (preferred): pyrealsense2 grabs a colour frame and a depth frame
     *aligned to the colour lens*, so YOLO runs on the colour image and each
     detection's distance is read straight from the depth at its box centre.
  2. Fallback: if no RealSense/depth camera is present, ffmpeg grabs a plain
     colour frame from a webcam — same detection, but no distance.

Detection itself is stock yolov8n.onnx under onnxruntime (CPU); NMS and box
decode are plain numpy. No PyTorch/OpenCV/ultralytics at runtime.

Model: models/yolov8n.onnx (stock COCO, 80 classes).
"""

import glob
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(WORKSPACE, "models", "yolov8n.onnx")

IMG_SIZE = 640           # model input side
CAP_W, CAP_H = 640, 480  # frame we ask the camera for
CONF_THRESH = 0.35
IOU_THRESH = 0.45
MAX_DETECTIONS = 40      # cap what we hand back to the LLM
RS_WARMUP = 10           # frames to let RealSense auto-exposure settle

# letterbox geometry for a CAP_W x CAP_H frame padded into IMG_SIZE x IMG_SIZE
_GAIN = min(IMG_SIZE / CAP_W, IMG_SIZE / CAP_H)
_PAD_X = int(round((IMG_SIZE - CAP_W * _GAIN) / 2))
_PAD_Y = int(round((IMG_SIZE - CAP_H * _GAIN) / 2))

# Stock COCO class names, in model output order.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

TOOL = {
    "type": "function",
    "function": {
        "name": "look",
        "description": (
            "Take a single snapshot from the camera and report the objects detected "
            "in it (people, bottles, laptops, chairs, etc. — 80 common object types), "
            "including how far away each one is in metres when a depth camera is "
            "available. Use whenever the user asks what you can see, to look at the "
            "camera, whether a specific thing or person is in view, or how far away "
            "something is."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "Optional single object the user is asking about, e.g. "
                        "'person', 'bottle', 'laptop'. Omit for a general 'what do you see'."
                    ),
                }
            },
            "required": [],
        },
    },
}

# Lazily-initialised onnxruntime session (loading the model takes ~100ms).
_session = None


def _letterbox(img, fill):
    """Pad a CAP_H x CAP_W image into IMG_SIZE x IMG_SIZE (same geometry for RGB/depth)."""
    if img.ndim == 3:
        canvas = np.full((IMG_SIZE, IMG_SIZE, img.shape[2]), fill, dtype=img.dtype)
    else:
        canvas = np.full((IMG_SIZE, IMG_SIZE), fill, dtype=img.dtype)
    canvas[_PAD_Y:_PAD_Y + CAP_H, _PAD_X:_PAD_X + CAP_W] = img
    return canvas


def _blob(rgb_letterboxed):
    """HWC uint8 RGB (640x640) → NCHW float32 [0,1] for the model."""
    chw = rgb_letterboxed.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(chw)


def _capture_realsense():
    """RealSense colour + depth-aligned-to-colour. Returns (blob, depth_lb_metres).

    Raises if no RealSense is connected/usable so the caller can fall back.
    """
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, CAP_W, CAP_H, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, CAP_W, CAP_H, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    try:
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)  # warp depth into the colour frame
        frames = None
        for _ in range(RS_WARMUP):
            frames = align.process(pipe.wait_for_frames())
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())
        depth_raw = np.asanyarray(frames.get_depth_frame().get_data())
    finally:
        pipe.stop()

    rgb = color_bgr[:, :, ::-1]                       # BGR → RGB
    depth_m = depth_raw.astype(np.float32) * scale    # units → metres
    frame_lb = _letterbox(rgb, 114)
    return _blob(frame_lb), _letterbox(depth_m, 0.0), frame_lb


def _detect_camera():
    """First video node advertising a colour format (MJPG/YUYV); override via SEE_CAMERA_DEVICE."""
    override = os.environ.get("SEE_CAMERA_DEVICE")
    if override:
        return override
    for dev in sorted(glob.glob("/dev/video*")):
        try:
            out = subprocess.run(
                ["v4l2-ctl", "-d", dev, "--list-formats"],
                capture_output=True, text=True, timeout=5,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "MJPG" in out or "YUYV" in out:
            return dev
    return "/dev/video0"


def _capture_ffmpeg():
    """Colour-only fallback via ffmpeg. Returns (blob, None) — no depth."""
    device = _detect_camera()
    vf = (
        f"scale={IMG_SIZE}:{IMG_SIZE}:force_original_aspect_ratio=decrease,"
        f"pad={IMG_SIZE}:{IMG_SIZE}:(ow-iw)/2:(oh-ih)/2:color=gray"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "v4l2", "-input_format", "yuyv422",
        "-video_size", f"{CAP_W}x{CAP_H}", "-i", device,
        "-frames:v", "1", "-vf", vf,
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    expected = IMG_SIZE * IMG_SIZE * 3
    if len(proc.stdout) != expected:
        err = proc.stderr.decode("utf-8", "replace").strip()[:300]
        raise RuntimeError(f"camera capture failed ({device}): {err or 'no frame'}")
    frame = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE, 3)
    return _blob(frame), None, frame


def _capture_g1():
    """Capture RGBD on the G1 robot's Orin over ssh; infer here.

    The RealSense is on the robot's Orin, not this PC, so we run the deployed
    grabber there (python3.8 + pyrealsense2), pull back one compressed npz, and
    letterbox it exactly like _capture_realsense — the (blob, depth_lb) contract
    and downstream detection are identical. Configure via env:
      G1_ORIN            ssh target (default unitree@192.168.0.87)
      G1_CAPTURE_SCRIPT  remote grabber path (default /home/unitree/g1_capture_rgbd.py)
    """
    orin = os.environ.get("G1_ORIN", "unitree@192.168.0.87")
    remote_script = os.environ.get("G1_CAPTURE_SCRIPT", "/home/unitree/g1_capture_rgbd.py")
    remote_npz = "/tmp/g1_snap.npz"

    grab = subprocess.run(
        ["ssh", orin, "python3", remote_script, remote_npz],
        capture_output=True, text=True, timeout=45,
    )
    if grab.returncode != 0:
        raise RuntimeError(f"orin capture failed: {(grab.stderr or grab.stdout).strip()[:300]}")

    with tempfile.TemporaryDirectory() as tmp:
        local_npz = os.path.join(tmp, "snap.npz")
        pull = subprocess.run(
            ["scp", "-q", f"{orin}:{remote_npz}", local_npz],
            capture_output=True, text=True, timeout=30,
        )
        if pull.returncode != 0:
            raise RuntimeError(f"scp of snapshot failed: {pull.stderr.strip()[:300]}")
        data = np.load(local_npz)
        rgb, depth_m = data["color"], data["depth"]

    frame_lb = _letterbox(rgb, 114)
    return _blob(frame_lb), _letterbox(depth_m, 0.0), frame_lb


def _capture():
    """Prefer RealSense (gives depth); fall back to ffmpeg colour-only.

    Returns (blob, depth_lb_or_None, frame_lb) where frame_lb is the letterboxed
    640x640 uint8 RGB the model saw — used to render the annotated look() frame.
    SEE_CAMERA_SOURCE=g1 captures from the G1 robot's Orin over ssh instead of a
    camera attached to this machine.
    """
    if os.environ.get("SEE_CAMERA_SOURCE") == "g1":
        return _capture_g1()
    try:
        return _capture_realsense()
    except Exception:
        return _capture_ffmpeg()


def _nms(boxes, scores, iou_thresh):
    """Plain-numpy non-max suppression; returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def _sample_distance(depth_lb, cx, cy):
    """Median non-zero depth (metres) in a small patch around a box centre, or None."""
    if depth_lb is None:
        return None
    x, y = int(round(cx)), int(round(cy))
    for r in (6, 12, 20):
        y0, y1 = max(0, y - r), min(IMG_SIZE, y + r)
        x0, x1 = max(0, x - r), min(IMG_SIZE, x + r)
        v = depth_lb[y0:y1, x0:x1]
        v = v[v > 0]
        if v.size >= 5:
            return round(float(np.median(v)), 2)
    return None


def _postprocess(output, depth_lb):
    """YOLOv8 (1,84,8400) → [(name, conf, orig_cx, dist_m_or_None, box_lb)] by conf.

    box_lb is [x1, y1, x2, y2] in letterboxed 640x640 space, so it draws directly
    onto frame_lb from _capture()."""
    pred = np.squeeze(output).T          # (8400, 84)
    scores_all = pred[:, 4:]
    class_ids = scores_all.argmax(axis=1)
    confs = scores_all.max(axis=1)
    mask = confs > CONF_THRESH
    if not mask.any():
        return []
    pred, class_ids, confs = pred[mask], class_ids[mask], confs[mask]

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    dets = []
    for cid in np.unique(class_ids):
        idx = np.where(class_ids == cid)[0]
        keep = _nms(boxes[idx], confs[idx], IOU_THRESH)
        for k in idx[keep]:
            orig_cx = (cx[k] - _PAD_X) / _GAIN          # back to the original frame
            dist = _sample_distance(depth_lb, cx[k], cy[k])
            dets.append((COCO_NAMES[int(cid)], float(confs[k]), float(orig_cx), dist,
                         boxes[k].tolist()))
    dets.sort(key=lambda d: d[1], reverse=True)
    return dets[:MAX_DETECTIONS]


def _position(cx_original):
    third = CAP_W / 3
    if cx_original < third:
        return "left"
    if cx_original > 2 * third:
        return "right"
    return "center"


def _describe(name, conf, cx, dist):
    bits = f"{conf:.2f}, {_position(cx)}"
    if dist is not None:
        bits += f", {dist:.1f} m"
    return f"{name} ({bits})"


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def _annotate(frame_lb, dets):
    """Draw detection boxes + labels on the letterboxed RGB frame → a PIL.Image.

    Pillow is imported here, not at module top, so the core detection path keeps
    its no-OpenCV/no-Pillow footprint — the draw only loads when a caller has
    opted into saving/showing the frame.
    """
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.ascontiguousarray(frame_lb).astype("uint8"), "RGB")
    draw = ImageDraw.Draw(im)
    for name, conf, _cx, dist, box in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        label = f"{name} {conf:.2f}" + (f" {dist:.1f}m" if dist is not None else "")
        ty = y1 - 11 if y1 >= 11 else y1 + 1
        draw.rectangle([x1, ty, x1 + 7 * len(label), ty + 11], fill=(0, 120, 0))
        draw.text((x1 + 1, ty), label, fill=(255, 255, 255))
    return im


def _show_annotated(frame_lb, dets):
    """Best-effort: save and/or flash the annotated look() frame. Never raises.

    Enabled per-call via env (the robot `look` tool sets these; whisper's own
    `look` leaves them unset, so its behaviour is unchanged):
      SEE_CAMERA_SAVE_DIR   save a timestamped PNG here (paper archive)
      SEE_CAMERA_POPUP=1    flash the frame in a borderless top-most window
      SEE_CAMERA_POPUP_MS   how long to show it (default 1000 ms)
      SEE_CAMERA_POPUP_X/Y  window top-left on screen (default 0,80 = left edge)
      SEE_CAMERA_POPUP_SCALE enlarge/shrink factor for the popup (default 1.0)
    """
    save_dir = os.environ.get("SEE_CAMERA_SAVE_DIR")
    popup = os.environ.get("SEE_CAMERA_POPUP") == "1"
    if frame_lb is None or not (save_dir or popup):
        return
    try:
        im = _annotate(frame_lb, dets)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, time.strftime("look_%Y%m%d_%H%M%S.png"))
        else:
            path = os.path.join(tempfile.gettempdir(), "g1_look_latest.png")
        im.save(path)
        if popup:
            helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_popup.py")
            args = [sys.executable, helper, path,
                    os.environ.get("SEE_CAMERA_POPUP_MS", "1000"),
                    os.environ.get("SEE_CAMERA_POPUP_X", "0"),
                    os.environ.get("SEE_CAMERA_POPUP_Y", "80"),
                    os.environ.get("SEE_CAMERA_POPUP_SCALE", "1.0")]
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # visualization is a demo aid; it must never break detection


def run(target=None):
    if not os.path.isfile(MODEL_PATH):
        return "error: model not found at models/yolov8n.onnx (see INSTALL.md vision setup)"

    try:
        session = _get_session()
        blob, depth_lb, frame_lb = _capture()
        output = session.run(None, {session.get_inputs()[0].name: blob})[0]
        dets = _postprocess(output, depth_lb)
    except Exception as e:  # keep tool errors as strings — never kill the session
        return f"error: {e}"

    _show_annotated(frame_lb, dets)  # best-effort save/flash of the annotated frame

    if not dets:
        summary = "Camera snapshot: no recognizable objects detected."
    else:
        parts = [_describe(*d[:4]) for d in dets]
        summary = f"Camera snapshot ({len(dets)} detected): " + ", ".join(parts) + "."

    if target:
        t = target.strip().lower()
        hits = [d for d in dets if t in d[0] or d[0] in t]
        verdict = f"YES — {len(hits)} {t}(s) in view" if hits else f"NO — no {t} detected"
        summary = f"Looking for '{target}': {verdict}. {summary}"

    return summary
