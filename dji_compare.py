"""Same-footage comparison with DJI Studio: align a DJI Studio 360 export to our stitch and sample both identically.

Put DJI Studio's equirectangular exports of whole clips (360 video, export starts at the clip start) into
EXPORT_DIR, named like the source clip:  <OSV name without extension>.mp4  (e.g. CAM_20251205150235_0074_D.mp4).

    python dji_compare.py list                 # which library clips have a DJI export
    python dji_compare.py align <clip id> <t>  # fit the rotation at time t and print the residual

DJI Studio may level, orient and shift its panorama differently from ours, so the mapping from our camera body
frame to its equirect is fitted per frame: E_dji(Q d) ~ S_ours(d) for body directions d, with Q initialised from
our IMU levelling (Q = Rz(yaw) M^T, coarse yaw search) and refined in 3 DoF on a 512x256 grey grid, away from the
seam band (where the two stitches legitimately differ).
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from osvcut import Lens, dev, level_frames, view_dirs

EXPORT_DIR = os.environ.get("OSV_DJI_EXPORTS", r"E:\Videos\_dji_studio_exports")


def exports(lib):
    """{clip id: export path} for library clips that have a DJI Studio export."""
    if not os.path.isdir(EXPORT_DIR):
        return {}
    have = {os.path.splitext(f)[0].lower(): os.path.join(EXPORT_DIR, f) for f in os.listdir(EXPORT_DIR)
            if f.lower().endswith((".mp4", ".mov"))}
    out = {}
    for c in lib.clips.values():
        k = os.path.splitext(c["name"])[0].lower()
        if k in have:
            out[c["id"]] = have[k]
    return out


def grab_eq(path, t, W=None):
    """One frame of an equirect video as (3,H,W) float on the GPU (optionally scaled to width W)."""
    vf = ["-vf", f"scale={W}:{W // 2}:flags=area"] if W else []
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip().split(",")
    w, h = (W, W // 2) if W else (int(pr[0]), int(pr[1]))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path] + vf +
                         ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    if len(raw) < w * h * 3:
        return None
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dev).view(h, w, 3).permute(2, 0, 1).float() / 255


def sample_eq(img, d):
    """Sample an equirect image (3,H,W) at unit directions d (...,3); +Y is the centre, +Z up (osvcut convention)."""
    lon = torch.atan2(d[..., 0], d[..., 1])
    lat = torch.asin(d[..., 2].clamp(-1, 1))
    gx = lon / math.pi
    gy = -lat / (math.pi / 2)
    grid = torch.stack([gx, gy], -1).reshape(1, -1, 1, 2)
    o = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="border", align_corners=False)
    return o[0, :, :, 0].reshape(3, *d.shape[:-1])


def rotvec(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def Rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _grey_blur(x):
    g = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2])[None, None]
    k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=x.device)
    k = (k[:, None] * k[None]) / 256
    return F.conv2d(F.pad(g, (2, 2, 2, 2), mode="replicate"), k[None, None])[0, 0]


def fit_rotation(ours_body, dji_eq, M, W=512):
    """ours_body: our stitch sampled on the body-frame equirect grid (3,W/2,W) (direction grid view_dirs('equirect')).
    Returns (Q, residual) with Q mapping body directions to DJI's sphere."""
    d = view_dirs("equirect", W, W // 2)
    beta = torch.rad2deg(torch.asin(d[..., 1].clamp(-1, 1)))
    lat = torch.rad2deg(torch.asin(d[..., 2].clamp(-1, 1)))
    keep = (beta.abs() > 6) & (lat.abs() < 75)               # away from the seam ring and the poles
    a = _grey_blur(ours_body)
    def loss(Q):
        b = _grey_blur(sample_eq(dji_eq, d @ torch.tensor(Q, dtype=torch.float32, device=dev).T))
        e = (a - b).abs()[keep]
        return float(e.mean())
    base = M.T
    best = min(((loss(Rz(y) @ base), y) for y in range(0, 360, 4)))
    y0 = best[1]
    Q0 = Rz(y0) @ base
    # refine: small 3-DoF rotation on top, Nelder-Mead on the rotation vector (degrees)
    from scipy.optimize import minimize
    r = minimize(lambda v: loss(rotvec(np.radians(v)) @ Q0), np.zeros(3), method="Nelder-Mead",
                 options=dict(initial_simplex=np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]], float),
                              xatol=0.02, fatol=1e-5, maxiter=300))
    Q = rotvec(np.radians(r.x)) @ Q0
    return Q, float(r.fun)


def ours_body_eq(clip, t, W=512):
    """Our flow stitch on the body-frame equirect grid at time t."""
    from stitch import load_calib
    from seam import FlowStitch
    from eval_seams import grab_pair
    cal = load_calib(clip["path"])
    lenses = [Lens(cal, L) for L in cal["lenses"]]
    imgs = grab_pair(clip["path"], t, cal["W"])
    if imgs is None:
        return None
    e = FlowStitch(cal, lenses, flow_ema=0)
    e.update(imgs)
    return e.render(imgs, view_dirs("equirect", W, W // 2)), imgs, cal, lenses


def level_M(clip, t):
    ts, Ms = level_frames(clip["path"])
    return np.asarray(Ms[min(np.searchsorted(ts, t), len(Ms) - 1)])


def align(clip, export, t):
    r = ours_body_eq(clip, t)
    dji = grab_eq(export, t, 1024)
    if r is None or dji is None:
        return None
    Q, res = fit_rotation(r[0], dji, level_M(clip, t))
    return Q, res


if __name__ == "__main__":
    import library as libmod
    lib = libmod.Library([])
    if sys.argv[1] == "list":
        ex = exports(lib)
        print(f"{len(ex)} clips with a DJI Studio export in {EXPORT_DIR}")
        for cid, p in ex.items():
            print(cid, p)
    elif sys.argv[1] == "align":
        c = lib.clips[sys.argv[2]]
        ex = exports(lib)
        Q, res = align(c, ex[c["id"]], float(sys.argv[3]))
        print(json.dumps(dict(Q=np.round(Q, 5).tolist(), residual=round(res, 5))))
