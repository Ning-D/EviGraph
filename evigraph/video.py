"""Frame I/O with OpenCV (decord segfaults next to torch in our environments)."""
import cv2
import numpy as np

# Long videos otherwise spawn one decode thread per core and thrash on lock contention.
cv2.setNumThreads(4)


def _resize(frame, max_side):
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return frame


def frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def sample_frames(video_path, idx, max_side=512):
    """RGB frames at the given native frame indices (unreadable indices are skipped)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        frames.append(_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_side))
    cap.release()
    return frames


def read_1fps(video_path, max_side=512):
    """Sequentially decode and keep ~1 frame per second (every round(fps)-th frame),
    as in AKS's feature extraction. Returns (PIL frames, native frame indices)."""
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, int(round(fps)))
    frames, idx, i = [], [], 0
    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        if i % step == 0:
            h, w = bgr.shape[:2]
            s = max_side / max(h, w)
            if s < 1.0:
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            idx.append(i)
        i += 1
    cap.release()
    return frames, idx


def open_video(video_path):
    """-> (cv2.VideoCapture, fps, duration_s)"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if total > 0 else 0.0
    return cap, fps, dur


def grab(cap, fps, sec, max_side):
    """RGB frame at time `sec` (seconds), or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return None
    return _resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_side)


def snapshot(cap, fps, ts, dur, snap_sec, snap_n, max_side):
    """`snap_n` frames spread evenly over [ts - snap_sec, ts + snap_sec]."""
    lo, hi = max(0, ts - snap_sec), min(dur, ts + snap_sec)
    if hi - lo < 0.3:
        return []
    out = []
    for t in np.linspace(lo, hi, snap_n):
        fr = grab(cap, fps, float(t), max_side)
        if fr is not None:
            out.append(fr)
    return out
