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
ALGO = "E50"          # seam defaults the "flow" renders use (research log); stored with every vote
_pairs = {}
_lock = threading.Lock()


def _render(clip, t, alpha_deg, methods, size=(720, 540), fov=55.0):
    from osvcut import Lens
    from stitch import load_calib
    from seam import SmartStitch, FlowStitch
    from eval_seams import grab_pair, legacy_render

    cal = load_calib(clip["path"])
    W = cal["W"]
    lenses = [Lens(cal, L) for L in cal["lenses"]]
    imgs = grab_pair(clip["path"], t, W)
    if imgs is None:
        return None
    d = _dirs(alpha_deg, size, fov)
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


def _dirs(alpha_deg, size, fov):
    """Upright close-up on the seam ring at azimuth alpha, in the body frame (the seam stays in view)."""
    import torch
    from osvcut import dev
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
    return torch.tensor(d, dtype=torch.float32, device=dev)


def _render_video(clip, t0, alpha_deg, methods, dst, secs=3.0, fps=25, size=(720, 540), fov=55.0):
    """The same close-up as a short video per method, with the engines' normal temporal state (as in an export)."""
    import subprocess
    from osvcut import Lens
    from stitch import load_calib
    from seam import SmartStitch, FlowStitch
    from eval_temporal import frames
    cal = load_calib(clip["path"])
    lenses = [Lens(cal, L) for L in cal["lenses"]]
    d = _dirs(alpha_deg, size, fov)
    # eval_temporal.frames keeps every step-th source frame; encode at the rate that results (real-time playback)
    src = float(eval(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=r_frame_rate", "-of", "csv=p=0", clip["path"]], capture_output=True,
                                    text=True).stdout.split()[0]))
    rate = src / max(1, round(src / fps))
    eng = {m: SmartStitch(cal, lenses) if m == "smart" else FlowStitch(cal, lenses) for m in methods}
    enc = {m: subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{size[0]}x{size[1]}", "-r", f"{rate:.4f}", "-i", "-", "-c:v", "libx264",
                                "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                dst[m]], stdin=subprocess.PIPE) for m in methods}
    n = 0
    for _, imgs in frames(clip["path"], t0, secs, fps, cal["W"]):
        for m in methods:
            eng[m].update(imgs)
            o = eng[m].render(imgs, d)
            enc[m].stdin.write((o.clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
        n += 1
    for p in enc.values():
        p.stdin.close()
        p.wait()
    return n >= rate * secs * 0.8


_vq = []                    # ready video pairs
_vfill = threading.Lock()


def make_video_pair(lib):
    """Flow seam vs. DP cut on a 3-s loop: are differences in motion (flicker, swimming) visible?"""
    clips = [c for c in lib.clips.values() if c.get("duration", 0) > 6 and c.get("lenses") == 2]
    for _ in range(5):
        c = random.choice(clips)
        t0 = round(random.uniform(0.5, c["duration"] - 4), 2)
        alpha = _pick_alpha(c, t0 + 1.5)
        if alpha is None:
            continue
        m1, m2 = random.sample(["flow", "smart"], 2)
        pid = uuid.uuid4().hex[:12]
        dst = {m1: os.path.join(BT_DIR, f"{pid}_left.mp4"), m2: os.path.join(BT_DIR, f"{pid}_right.mp4")}
        if not _render_video(c, t0, alpha, [m1, m2], dst):
            continue
        with _lock:
            _pairs[pid] = dict(pid=pid, kind="video", clip=c["id"], t=t0, alpha=round(alpha, 2), left=m1, right=m2,
                               created=time.time())
        return dict(pid=pid, kind="video", left=f"/cache/blindtest/{pid}_left.mp4",
                    right=f"/cache/blindtest/{pid}_right.mp4")
    return None


def next_video(lib, gpu_lock, keep=2):
    """A ready video pair (rendering takes ~10-30 s, so a background thread keeps `keep` in stock)."""
    def fill():
        if not _vfill.acquire(blocking=False):
            return
        try:
            while len(_vq) < keep:
                with gpu_lock:
                    pr = make_video_pair(lib)
                if not pr:
                    break
                _vq.append(pr)
        finally:
            _vfill.release()
    if _vq:
        pr = _vq.pop(0)
    else:
        with gpu_lock:
            pr = make_video_pair(lib)
    threading.Thread(target=fill, daemon=True).start()
    return pr


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
    rec = dict(p, choice=choice, voted=time.time(), algo=ALGO)
    with open(VOTES, "a", encoding="utf8") as fh:
        fh.write(json.dumps(rec) + "\n")
    for side in ("left", "right"):
        for ext in ("jpg", "mp4"):
            fn = os.path.join(BT_DIR, f"{pid}_{side}.{ext}")
            if os.path.exists(fn):
                os.remove(fn)
    return True


def stats(kind="still"):
    recs = []
    if os.path.exists(VOTES):
        recs = [json.loads(l) for l in open(VOTES, encoding="utf8") if l.strip()]
    recs = [r for r in recs if r.get("kind", "still") == kind]
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
