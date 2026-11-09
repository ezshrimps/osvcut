"""E29: which calibration-to-stream scale registers the two lenses best?

For each clip / frame, both lenses are sampled on the seam ring (beta in [-4, 4] deg, 0.1 deg/px) with the
stream scale multiplied by kappa; the score is the correlation of the two lenses' luma where both are valid
(the body-occluded sector drops out by itself).  Reports the best kappa relative to the current scale (W/3840) and the
score of the old scale (digital_focal_length / fx).

    python tools_scale_check.py [n_clips] [frames_per_clip] [seed] [4K/8K]
"""
import math
import os
import random
import sys

import numpy as np
import torch

from osvcut import Lens, dev
from stitch import load_calib, lens_scale
from seam import SmartStitch
from eval_seams import grab_pair
import library as libmod

KAPPAS = np.round(np.arange(0.975, 1.0251, 0.0025), 4)


def ring(res=0.1, band=4.0):
    a = torch.arange(0, 360, res, device=dev) * math.pi / 180
    b = torch.arange(band, -band - 1e-6, -res, device=dev) * math.pi / 180
    B, A = torch.meshgrid(b, a, indexing="ij")
    return torch.stack([torch.cos(B) * torch.cos(A), torch.sin(B), torch.cos(B) * torch.sin(A)], -1)


def score(e, lenses, imgs, D, scales):
    for L, s in zip(lenses, scales):
        L.s = s
    g, m = [], []
    for i in (0, 1):
        c, v = e._sample(i, imgs[i], D)
        g.append(((c.clamp(0, 1) ** (1 / 2.2)).mean(0)))
        m.append(v > 0.5)
    both = m[0] & m[1]
    if both.sum() < 1000:
        return float("nan")
    a, b = g[0][both], g[1][both]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    lib = libmod.Library([r"E:\Videos"])
    mode = sys.argv[4] if len(sys.argv) > 4 else ""          # "4K" / "8K" to restrict the pool
    pool = sorted([c for c in lib.clips.values() if c.get("lenses") == 2 and c.get("duration", 0) > 3
                   and (not mode or ("8K" if c.get("lens_px", 0) >= 3840 else "4K") == mode)], key=lambda c: c["id"])
    clips = random.Random(seed).sample(pool, min(n, len(pool)))
    D = ring()
    rows = []
    for c in clips:
        cal = load_calib(c["path"])
        lenses = [Lens(cal, L) for L in cal["lenses"]]
        e = SmartStitch(cal, lenses)
        new = [lens_scale(cal, L) for L in cal["lenses"]]
        old = [cal["dfl"] / L["fx"] for L in cal["lenses"]]
        for k in range(nf):
            t = 0.5 + (c["duration"] - 1.0) * (k + .5) / nf
            imgs = grab_pair(c["path"], t, cal["W"])
            if imgs is None:
                continue
            sc = [score(e, lenses, imgs, D, [s * kp for s in new]) for kp in KAPPAS]
            so = score(e, lenses, imgs, D, old)
            j = int(np.nanargmax(sc))
            rows.append(dict(clip=c["name"], mode="8K" if cal["W"] >= 3840 else "4K", t=round(t, 1),
                             best_kappa=float(KAPPAS[j]), corr_best=sc[j], corr_new=sc[list(KAPPAS).index(1.0)],
                             corr_old=so, old_over_new=old[0] / new[0]))
            r = rows[-1]
            print(f"{r['clip']} {r['mode']} t={r['t']:7.1f}  best kappa {r['best_kappa']:.4f}  corr old {so:.3f}  "
                  f"new {r['corr_new']:.3f}  best {r['corr_best']:.3f}", flush=True)
    bk = np.array([r["best_kappa"] for r in rows])
    co, cn = np.array([r["corr_old"] for r in rows]), np.array([r["corr_new"] for r in rows])
    print(f"\nframes {len(rows)}  best kappa: median {np.median(bk):.4f}, IQR [{np.percentile(bk, 25):.4f}, {np.percentile(bk, 75):.4f}]"
          f"  (old scale = {np.mean([r['old_over_new'] for r in rows]):.4f} x new)")
    print(f"correlation at zero shift: old {np.nanmean(co):.3f}  new {np.nanmean(cn):.3f}  (new better in {int((cn > co).sum())}/{len(rows)})")
    os.makedirs("results", exist_ok=True)
    import json
    json.dump(rows, open("results/scale_check.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
