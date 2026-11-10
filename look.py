"""Single rendered views and panorama contact sheets, for people and AI agents.

view_jpeg(clip, t, yaw, pitch, fov, dist)  -> a levelled view in any direction at any time (flow seam)
sheet_jpeg(clip, n)                         -> n levelled equirect thumbnails with yaw ticks and times
"""
import math
import numpy as np
import cv2
import torch

from osvcut import Lens, dev, level_frames, view_dirs, view_dirs_eye, cam_rot, grade, nadir_patch
from stitch import load_calib
from seam import FlowStitch
from eval_seams import grab_pair

_cache = {}


def _rig(path):
    if path not in _cache:
        cal = load_calib(path)
        _cache.clear()
        _cache[path] = (cal, [Lens(cal, L) for L in cal["lenses"]], level_frames(path))
    return _cache[path]


def _render(path, t, d_out):
    cal, lenses, (ts, Ms) = _rig(path)
    imgs = grab_pair(path, t, cal["W"])
    if imgs is None:
        raise ValueError("could not decode frame at t=%.2f" % t)
    M = Ms[min(np.searchsorted(ts, t), len(Ms) - 1)]
    e = FlowStitch(cal, lenses, flow_ema=0)
    e.update(imgs)
    return e.render(imgs, d_out @ torch.tensor(M, dtype=torch.float32, device=dev).T)


def _jpeg(img_t, q=88):
    im = (img_t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def view_jpeg(clip, t, yaw, pitch, fov=90.0, dist=0.0, w=960, h=540, grade_=None, nadir=0.0, q=88):
    """grade_ / nadir: the studio's colour grade and nadir patch, so a still matches the export."""
    t = max(0.0, min(t, max(clip.get("duration", 0) - 0.05, 0)))
    C = torch.tensor(cam_rot(yaw, pitch), dtype=torch.float32, device=dev)
    if dist > 1e-3:
        base, valid = view_dirs_eye(w, h, fov, dist)
    else:
        base, valid = view_dirs("flat", w, h, min(fov, 170.0)), None
    d = base @ C.T
    out = _render(clip["path"], t, d)
    if nadir > 0:
        out = nadir_patch(out, d, float(nadir))
    if grade_:
        out = grade(out, grade_)
    if valid is not None:
        out = out * valid
    return _jpeg(out, q)


def sheet_jpeg(clip, n=6, W=1024):
    dur = clip.get("duration", 0)
    times = [dur * (i + .5) / n for i in range(n)]
    tiles = []
    base = view_dirs("equirect", W, W // 2)
    for t in times:
        im = (_render(clip["path"], t, base).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()
        for yaw in range(-180, 181, 45):
            x = int((yaw / 360 + .5) * W)
            cv2.line(im, (x, 0), (x, 12), (0, 255, 255), 2)
            cv2.putText(im, str(yaw), (max(x - 14, 0), 28), 0, .5, (0, 255, 255), 1)
        cv2.putText(im, f"t={t:.1f}s", (8, W // 2 - 12), 0, .8, (0, 0, 255), 2)
        tiles.append(im)
    rows = [np.hstack(tiles[i:i + 2] + [np.zeros_like(tiles[0])] * (2 - len(tiles[i:i + 2]))) for i in range(0, len(tiles), 2)]
    return cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def holder_direction(clip, t, lift=40.0, cap=60.0):
    """Where the person holding the (invisible) selfie stick is: the stick runs along the camera body's
    -Z axis; the face sits above the hand, so lift the pitch.  A starting point, not a detector."""
    _, _, (ts, Ms) = _rig(clip["path"])
    M = Ms[min(np.searchsorted(ts, t), len(Ms) - 1)]
    d = M.T @ np.array([0.0, 0.0, -1.0])
    yaw = math.degrees(math.atan2(d[0], d[1]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))
    return dict(t=t, stick_yaw=round(yaw, 1), stick_pitch=round(pitch, 1), yaw=round(yaw, 1),
                pitch=round(min(cap, pitch + lift), 1), fov=100)


def holder_track(clip, t0, t1, step=0.5, lift=40.0, smooth=1.0, cap=60.0):
    """Keyframes that keep the selfie-stick holder in view between t0 and t1 (IMU only)."""
    ts_ = np.arange(t0, t1 + 1e-6, step)
    pts = [holder_direction(clip, float(t), lift, cap) for t in ts_]
    yaw = np.degrees(np.unwrap(np.radians([p["yaw"] for p in pts])))
    pitch = np.array([p["pitch"] for p in pts])
    k = max(1, int(round(smooth / step)))
    if k > 1:
        pad = lambda a: np.pad(a, (k // 2, k - 1 - k // 2), mode="edge")
        yaw = np.convolve(pad(yaw), np.ones(k) / k, "valid")
        pitch = np.convolve(pad(pitch), np.ones(k) / k, "valid")
    keys = [dict(t=round(float(t), 3), yaw=round(float(((y + 180) % 360) - 180), 1), pitch=round(float(p), 1), fov=100,
                 ease="linear") for t, y, p in zip(ts_, yaw, pitch)]
    if keys:
        keys[-1].pop("ease")
    return keys
