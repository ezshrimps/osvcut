"""E30: alignment in the narrow-overlap (body-occluded) sector.

For each frame, both engines render the seam strip (+-4 deg, 0.1 deg/px); we keep the warped contribution of each
lens, pA and pB, and measure mean |Y(pA) - Y(pB)| on textured pixels inside the narrow sector (raw overlap < 2.5 deg)
where both warped samples are valid for *both* engines (same pixels for every method).  Lower = better aligned.
This is a photometric measure, independent of the correlation search that builds the narrow prior.

    python eval_narrow.py [--clips 12] [--frames 3] [--seed 2]      (NARROW_BAND=-2: lower limit of the search band)
"""
import argparse
import json
import math
import random

import numpy as np
import torch

from osvcut import Lens, dev
from stitch import load_calib
from seam import FlowStitch
from eval_seams import grab_pair
import library as libmod


def ring(res=0.1, band=4.0):
    a = torch.arange(0, 360, res, device=dev) * math.pi / 180
    b = torch.arange(band, -band - 1e-6, -res, device=dev) * math.pi / 180
    B, A = torch.meshgrid(b, a, indexing="ij")
    return torch.stack([torch.cos(B) * torch.cos(A), torch.sin(B), torch.cos(B) * torch.sin(A)], -1), b * 180 / math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default="results/narrow_bench.json")
    a = ap.parse_args()
    lib = libmod.Library([r"E:\Videos"])
    pool = sorted([c for c in lib.clips.values() if c.get("lenses") == 2 and c.get("duration", 0) > 3
                   and c.get("lens_px", 0) < 3840], key=lambda c: c["id"])
    clips = random.Random(a.seed).sample(pool, min(a.clips, len(pool)))
    D, betas = ring()
    Y = lambda c: (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
    rows = []
    for c in clips:
        cal = load_calib(c["path"])
        lenses = [Lens(cal, L) for L in cal["lenses"]]
        for k in range(a.frames):
            t = 0.5 + (c["duration"] - 1.0) * (k + .5) / a.frames
            imgs = grab_pair(c["path"], t, cal["W"])
            if imgs is None:
                continue
            out = {}
            for opt in ("off", "search"):
                e = FlowStitch(cal, lenses, flow_ema=0, conf="photo", narrow=opt)
                e.narrow_band = float(__import__("os").environ.get("NARROW_BAND", "-5"))
                e.update(imgs)
                _, pA, pB, _, _ = e.render(imgs, D, parts=True)
                mA, mB = e._tls.masks
                out[opt] = (Y(pA), Y(pB), (mA > .5) & (mB > .5))
            _, rA = e._sample(0, imgs[0], D)
            _, rB = e._sample(1, imgs[1], D)
            width = ((rA > .5) & (rB > .5)).sum(0).float() * 0.1
            narrow = (width < 2.5)[None, :]
            gA = out["off"][0]
            tex = (torch.abs(gA[1:-1, 2:] - gA[1:-1, :-2]) + torch.abs(gA[2:, 1:-1] - gA[:-2, 1:-1]))
            tex = torch.nn.functional.pad(tex, (1, 1, 1, 1))
            common = out["off"][2] & out["search"][2] & narrow & (betas.abs() < 3)[:, None] & (tex > 0.02)
            n = int(common.sum())
            if n < 300:
                continue
            r = dict(clip=c["name"], t=round(t, 1), n=n)
            for opt, (A_, B_, _) in out.items():
                r[opt] = float(torch.abs(A_ - B_)[common].mean())
            rows.append(r)
            print(f"{c['name']} t={t:7.1f} n={n:5d}  off {r['off']:.4f}  search {r['search']:.4f}", flush=True)
    off = np.array([r["off"] for r in rows]); se = np.array([r["search"] for r in rows])
    from scipy.stats import wilcoxon
    print(f"\nframes {len(rows)}: off {off.mean():.4f}  search {se.mean():.4f}  better {int((se < off).sum())}/{len(rows)}  "
          f"p={wilcoxon(se, off).pvalue:.2g}")
    json.dump(rows, open(a.out, "w", encoding="utf8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
