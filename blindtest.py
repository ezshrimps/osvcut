"""Blind pairwise seam comparison (perceptual study).

A trial = one frame of one clip, one seam location (preferably with real parallax), rendered as a
rectilinear close-up by two different seam methods; left/right order is random.  Votes are stored
with the hidden method names so win rates (and a Bradley-Terry fit) can be computed later.
"""
import json
import math
import os
import random
import threading
import time
import uuid

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BT_DIR = os.path.join(HERE, "cache", "blindtest")
VOTES = os.path.join(BT_DIR, "votes.jsonl")
os.makedirs(BT_DIR, exist_ok=True)

METHODS = ["legacy", "smart", "flow"]
_pairs = {}
_lock = threading.Lock()


def _render(clip, t, alpha_deg, methods, size=(720, 540), fov=55.0):
    import cv2
    import torch
    from osvcut import Lens, dev
    from stitch import load_calib
    from seam import SmartStitch, FlowStitch
    from eval_seams import grab_pair, legacy_render

    cal = load_calib(clip["path"])
    W = cal["W"]
    lenses = [Lens(cal, L) for L in cal["lenses"]]
    imgs = grab_pair(clip["path"], t, W)
    if imgs is None:
        return None
    a = math.radians(alpha_deg)
    fwd = np.array([math.cos(a), 0.0, math.sin(a)])
    # keep the picture upright (body Z up) where possible; near the top/bottom of the seam ring use +Y
    zup = np.array([0.0, 0.0, 1.0])
    up = zup - fwd * (fwd @ zup)
    up = up / np.linalg.norm(up) if np.linalg.norm(up) > 0.3 else np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    Wd, Hd = size
    f = (Wd / 2) / math.tan(math.radians(fov) / 2)
    xs = (np.arange(Wd) + .5 - Wd / 2) / f
    ys = -(np.arange(Hd) + .5 - Hd / 2) / f
    X, Y = np.meshgrid(xs, ys)
    d = fwd[None, None] + X[..., None] * right[None, None] + Y[..., None] * up[None, None]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    d = torch.tensor(d, dtype=torch.float32, device=dev)
    out = {}
    for m in methods:
        if m == "legacy":
            o = legacy_render(lenses, imgs, d)
        else:
            e = SmartStitch(cal, lenses) if m == "smart" else FlowStitch(cal, lenses, flow_ema=0)
            for _ in range(2):
                e.update(imgs)
            o = e.render(imgs, d)
        out[m] = (o.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()
    return out


def _pick_alpha(clip, t):
    """Prefer an azimuth where the two lenses disagree geometrically (parallax)."""
    import torch
    import cv2
    from osvcut import Lens, dev
    from stitch import load_calib
    from seam import SmartStitch
    from eval_seams import grab_pair, strip_dirs, luma, hard_columns, RES, BAND
    cal = load_calib(clip["path"])
    lenses = [Lens(cal, L) for L in cal["lenses"]]
    imgs = grab_pair(clip["path"], t, cal["W"])
    if imgs is None:
        return None
    e = SmartStitch(cal, lenses)
    D = strip_dirs()
    cA, mA = e._sample(0, imgs[0], D)
    cB, mB = e._sample(1, imgs[1], D)
    both = ((mA > 0.5) & (mB > 0.5)).cpu().numpy()
    betas = np.arange(BAND, -BAND - 1e-6, -RES)
    hard = hard_columns(luma(cA ** (1 / 2.2)), luma(cB ** (1 / 2.2)), both, betas)
    cols = np.nonzero(hard)[0]
    if len(cols) and random.random() < 0.8:
        return float(random.choice(cols) * RES)
    ok = np.nonzero(both.any(0))[0]
    return float(random.choice(ok) * RES) if len(ok) else random.uniform(0, 360)


def make_pair(lib):
    import cv2
    clips = [c for c in lib.clips.values() if c.get("duration", 0) > 3 and c.get("lenses") == 2]
    for _ in range(5):
        c = random.choice(clips)
        t = round(random.uniform(0.5, c["duration"] - 0.5), 2)
        alpha = _pick_alpha(c, t)
        if alpha is None:
            continue
        m1, m2 = random.sample(METHODS, 2)
        imgs = _render(c, t, alpha, [m1, m2])
        if not imgs:
            continue
        pid = uuid.uuid4().hex[:12]
        files = {}
        for side, m in (("left", m1), ("right", m2)):
            fn = os.path.join(BT_DIR, f"{pid}_{side}.jpg")
            cv2.imwrite(fn, imgs[m], [cv2.IMWRITE_JPEG_QUALITY, 92])
            files[side] = "/cache/blindtest/" + os.path.basename(fn)
        with _lock:
            _pairs[pid] = dict(pid=pid, clip=c["id"], t=t, alpha=round(alpha, 2), left=m1, right=m2, created=time.time())
        return dict(pid=pid, left=files["left"], right=files["right"])
    return None


def vote(pid, choice):
    with _lock:
        p = _pairs.pop(pid, None)
    if not p or choice not in ("left", "right", "same"):
        return False
    rec = dict(p, choice=choice, voted=time.time())
    with open(VOTES, "a", encoding="utf8") as fh:
        fh.write(json.dumps(rec) + "\n")
    for side in ("left", "right"):
        fn = os.path.join(BT_DIR, f"{pid}_{side}.jpg")
        if os.path.exists(fn):
            os.remove(fn)
    return True


def stats():
    recs = []
    if os.path.exists(VOTES):
        recs = [json.loads(l) for l in open(VOTES, encoding="utf8") if l.strip()]
    wins = {m: 0.0 for m in METHODS}
    games = {m: 0 for m in METHODS}
    pair = {}
    for r in recs:
        a, b = r["left"], r["right"]
        games[a] += 1
        games[b] += 1
        k = tuple(sorted((a, b)))
        pr = pair.setdefault(k, {"n": 0, k[0]: 0.0, k[1]: 0.0})
        pr["n"] += 1
        if r["choice"] == "same":
            wins[a] += .5; wins[b] += .5; pr[a] += .5; pr[b] += .5
        else:
            w = a if r["choice"] == "left" else b
            wins[w] += 1
            pr[w] += 1
    # Bradley-Terry scores (MM algorithm), ties as half wins
    s = {m: 1.0 for m in METHODS}
    for _ in range(200):
        for m in METHODS:
            den = 0.0
            for (x, y), pr in pair.items():
                if m in (x, y):
                    o = y if m == x else x
                    den += pr["n"] / (s[m] + s[o])
            if den > 0 and wins[m] > 0:
                s[m] = wins[m] / den
        z = sum(s.values())
        s = {m: v * len(s) / z for m, v in s.items()}
    return dict(n=len(recs), win_rate={m: (wins[m] / games[m] if games[m] else None) for m in METHODS},
                bt=s, pairs={f"{k[0]} vs {k[1]}": v for k, v in pair.items()})
