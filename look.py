"""Single rendered views and panorama contact sheets, for people and AI agents.

view_jpeg(clip, t, yaw, pitch, fov, dist)  -> a levelled view in any direction at any time (flow seam)
sheet_jpeg(clip, n)                         -> n levelled equirect thumbnails with yaw ticks and times
"""
import math
import numpy as np
import cv2
import torch

from osvcut import Lens, dev, level_frames, view_dirs, view_dirs_eye, cam_rot
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


def view_jpeg(clip, t, yaw, pitch, fov=90.0, dist=0.0, w=960, h=540):
    t = max(0.0, min(t, max(clip.get("duration", 0) - 0.05, 0)))
    C = torch.tensor(cam_rot(yaw, pitch), dtype=torch.float32, device=dev)
    if dist > 1e-3:
        base, valid = view_dirs_eye(w, h, fov, dist)
    else:
        base, valid = view_dirs("flat", w, h, fov), None
    out = _render(clip["path"], t, base @ C.T)
    if valid is not None:
        out = out * valid
    return _jpeg(out)


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


def holder_direction(clip, t, lift=40.0):
    """Where the person holding the (invisible) selfie stick is: the stick runs along the camera body's
    -Z axis; the face sits above the hand, so lift the pitch.  A starting point, not a detector."""
    _, _, (ts, Ms) = _rig(clip["path"])
    M = Ms[min(np.searchsorted(ts, t), len(Ms) - 1)]
    d = M.T @ np.array([0.0, 0.0, -1.0])
    yaw = math.degrees(math.atan2(d[0], d[1]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))
    return dict(t=t, stick_yaw=round(yaw, 1), stick_pitch=round(pitch, 1), yaw=round(yaw, 1),
                pitch=round(min(60.0, pitch + lift), 1), fov=100)
