"""Multi-clip seam benchmark.

    python eval_seams.py [--clips 20] [--frames 5] [--seed 0] [--out results/seam_bench.json]

All metrics are computed on the seam strip in the camera body frame (azimuth alpha 0..360 deg,
latitude beta in [-band, band] around the plane between the lenses, 0.1 deg/px), only where both
lenses are valid, and relative to single-lens renders of the same directions:

  ghost_ssim : mean over |beta| < 3 of max(SSIM(out, slave-only), SSIM(out, master-only)).
               A clean output locally looks like one real photograph -> high; a ghosted mix of two
               misaligned images resembles neither -> low.
  edge_dev   : |log( E_out / E_ref )| where E = mean |d/dbeta| of luma in |beta| < 1.5 (across the seam)
               and E_ref the mean of the two single-lens values. Blur (ghost) -> E_out < E_ref; a hard
               visible cut -> E_out > E_ref. 0 is ideal.
  step       : median over 10-deg azimuth bins of | |log(L_lo/L_hi)|_out - |log(L_lo/L_hi)|_ref |, with
               L_lo / L_hi the mean luma in beta in [-3,-1] / [1,3]; the reference is the single lens that
               covers both sides better. Brightness jump in excess of the natural scene gradient.
  ms         : update + render time for one 3840x1920 equirect frame.
"""
import argparse
import json
import math
import os
import random
import subprocess
import time

import cv2
import numpy as np
import torch

from osvcut import Lens, dev, view_dirs
from stitch import load_calib
from seam import SmartStitch, FlowStitch
import library as libmod

METHODS = ["legacy", "smart", "flow_center", "flow_dp", "flow_dp_r15"]


def make_engine(name, cal, lenses):
    if name == "smart":
        return SmartStitch(cal, lenses)
    opts = dict(flow_ema=0)
    if name == "flow_center":
        opts["center"] = "overlap"
    elif name == "flow_dp":
        opts["center"] = "dp"
    elif name == "flow_dp_r15":
        opts.update(center="dp", ramp=1.5)
    return FlowStitch(cal, lenses, **opts)


RES = 0.1
BAND = 5.0


def strip_dirs():
    a = torch.arange(0, 360, RES, device=dev) * math.pi / 180
    b = torch.arange(BAND, -BAND - 1e-6, -RES, device=dev) * math.pi / 180
    B, A = torch.meshgrid(b, a, indexing="ij")
    return torch.stack([torch.cos(B) * torch.cos(A), torch.sin(B), torch.cos(B) * torch.sin(A)], -1)


def luma(t):  # (3,H,W) gamma [0,1] -> HxW float32 numpy
    t = t.clamp(0, 1)
    return (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]).cpu().numpy().astype(np.float32)


def ssim_map(x, y, s=1.5):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    g = lambda z: cv2.GaussianBlur(z, (0, 0), s)
    mx, my = g(x), g(y)
    vx, vy, cxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    return ((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2))


def grab_pair(f, t, W):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", f, "-filter_complex", "[0:v:0][0:v:1]hstack",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    if len(raw) < W * 2 * W * 3:
        return None
    im = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dev).view(W, 2 * W, 3).permute(2, 0, 1).float() / 255
    return [im[None, :, :, :W], im[None, :, :, W:]]


def legacy_render(lenses, imgs, d, parts=False):
    cs, ws = [], []
    for L, img in zip(lenses, imgs):
        c, w = L.sample(img, d)
        cs.append(c); ws.append(w)
    n = ws[0] + ws[1] + 1e-6
    out = ((cs[0] * ws[0] + cs[1] * ws[1]) / n).clamp(0, 1)
    if parts:
        return out, cs[0].clamp(0, 1), cs[1].clamp(0, 1), ws[0] / n, ws[1] / n
    return out


def disagreement(pA, pB, wA, wB, both):
    """Contribution disagreement: where both lenses contribute, how different are the two
    (possibly warped, gain-compensated) contributions? weight 4 wA wB peaks at an even mix."""
    w = (4 * wA * wB).cpu().numpy() * both
    if w.sum() < 50:
        return float("nan")
    return float((w * np.abs(luma(pA) - luma(pB))).sum() / w.sum())


def metrics(O, A, B, both, betas):
    core = (np.abs(betas)[:, None] < 3) & both
    if core.sum() < 2000:
        return None
    S = np.maximum(ssim_map(O, A), ssim_map(O, B))
    ghost = float(S[core].mean())
    seam = (np.abs(betas)[:, None] < 1.5) & both
    dB = lambda X: np.abs(np.diff(X, axis=0, append=X[-1:]))
    Eo, Ea, Eb = dB(O)[seam].mean(), dB(A)[seam].mean(), dB(B)[seam].mean()
    edge = float(abs(math.log((Eo + 1e-4) / (0.5 * (Ea + Eb) + 1e-4))))
    lo = (betas >= -3) & (betas <= -1)
    hi = (betas >= 1) & (betas <= 3)
    steps = []
    nb = int(360 / RES / 100)  # 10-degree bins
    for k in range(36):
        cols = slice(k * 100, (k + 1) * 100)
        m_lo, m_hi = both[lo][:, cols], both[hi][:, cols]
        if m_lo.sum() < 300 or m_hi.sum() < 300:
            continue
        r = lambda X: abs(math.log((X[lo][:, cols][m_lo].mean() + .02) / (X[hi][:, cols][m_hi].mean() + .02)))
        steps.append(abs(r(O) - min(r(A), r(B))))
    step = float(np.median(steps)) if steps else float("nan")
    return dict(ghost_ssim=ghost, edge_dev=edge, step=step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=r"E:\Videos")
    ap.add_argument("--out", default="results/seam_bench.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    lib = libmod.Library([a.root])
    if not lib.clips:
        lib.scan()
    pool = sorted([c for c in lib.clips.values() if c.get("duration", 0) > 3 and c.get("lenses") == 2],
                  key=lambda c: c["id"])
    clips = random.Random(a.seed).sample(pool, min(a.clips, len(pool)))
    D = strip_dirs()
    betas = np.arange(BAND, -BAND - 1e-6, -RES)
    eq = view_dirs("equirect", 3840, 1920)
    rows = []
    for ci, c in enumerate(clips):
        cal = load_calib(c["path"])
        W = cal["W"]
        lenses = [Lens(cal, L) for L in cal["lenses"]]
        probe = SmartStitch(cal, lenses)
        times = [0.5 + (c["duration"] - 1.0) * (k + .5) / a.frames for k in range(a.frames)]
        for t in times:
            imgs = grab_pair(c["path"], t, W)
            if imgs is None:
                continue
            cA, mA = probe._sample(0, imgs[0], D)
            cB, mB = probe._sample(1, imgs[1], D)
            A = luma(cA ** (1 / 2.2))
            B = luma(cB ** (1 / 2.2))
            both = ((mA > 0.5) & (mB > 0.5)).cpu().numpy()
            for name in METHODS:
                torch.cuda.synchronize()
                t0 = time.time()
                if name == "legacy":
                    _ = legacy_render(lenses, imgs, eq)
                else:
                    e = make_engine(name, cal, lenses)
                    e.update(imgs)
                    _ = e.render(imgs, eq)
                torch.cuda.synchronize()
                ms = (time.time() - t0) * 1000
                O, pA, pB, wA, wB = (legacy_render(lenses, imgs, D, parts=True) if name == "legacy"
                                     else e.render(imgs, D, parts=True))
                m = metrics(luma(O), A, B, both, betas)
                if m is None:
                    continue
                m["cd"] = disagreement(pA, pB, wA, wB, both)
                rows.append(dict(clip=c["id"], event=c.get("event"), mode="8K" if c.get("lens_px", 0) >= 3840 else "4K",
                                 t=round(t, 2), method=name, ms=round(ms, 1), **m))
        print(f"[{ci + 1}/{len(clips)}] {c['name']} done ({len([r for r in rows if r['clip'] == c['id']]) // len(METHODS)} frames)", flush=True)
    json.dump(rows, open(a.out, "w", encoding="utf8"), indent=1, ensure_ascii=False)
    # summary
    print("\nmethod   n   ghost_ssim↑      edge_dev↓       step↓          ms")
    for name in METHODS:
        R = [r for r in rows if r["method"] == name]
        f = lambda k: (np.nanmean([r[k] for r in R]), np.nanstd([r[k] for r in R]))
        cd, g, e, s, ms = f("cd"), f("ghost_ssim"), f("edge_dev"), f("step"), f("ms")
        print(f"{name:7s} {len(R):3d}  {cd[0]:.4f}±{cd[1]:.4f}  {g[0]:.4f}±{g[1]:.4f}  {e[0]:.4f}±{e[1]:.4f}  {s[0]:.4f}±{s[1]:.4f}  {ms[0]:6.0f}")
    # paired wins
    by = {}
    for r in rows:
        by.setdefault((r["clip"], r["t"]), {})[r["method"]] = r
    for k in ("cd", "ghost_ssim", "edge_dev", "step"):
        better = (lambda x, y: x > y) if k == "ghost_ssim" else (lambda x, y: x < y)
        for m1, m2 in (("flow_dp", "legacy"), ("flow_dp", "smart"), ("flow_dp", "flow_center"), ("smart", "legacy")):
            pairs = [v for v in by.values() if m1 in v and m2 in v and not math.isnan(v[m1][k]) and not math.isnan(v[m2][k])]
            w = sum(better(v[m1][k], v[m2][k]) for v in pairs)
            print(f"  {k:10s} {m1} beats {m2}: {w}/{len(pairs)}")


if __name__ == "__main__":
    main()
