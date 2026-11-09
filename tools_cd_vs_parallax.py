"""E41: contribution disagreement as a function of lens-to-lens parallax.

For each frame (E11 clip sample), the seam strip is cut into 10-deg azimuth bins.  Per bin: the raw lens-to-lens
displacement (median |Farneback flow| over both-valid pixels with |beta| < 3, on unwarped strips) and the CD of the
DP cut and of the flow seam over the same pixels.  Bins are pooled by displacement.

    python tools_cd_vs_parallax.py [clips] [frames] [out.dat]
"""
import random
import sys

import cv2
import numpy as np
import torch

from osvcut import Lens, dev, view_dirs
from stitch import load_calib
from seam import SmartStitch, FlowStitch
from eval_seams import strip_dirs, luma, grab_pair, RES
import library as libmod


def main():
    nc = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    out = sys.argv[3] if len(sys.argv) > 3 else "results/cd_vs_parallax.dat"
    lib = libmod.Library([r"E:\Videos"])
    pool = sorted([c for c in lib.clips.values() if c.get("duration", 0) > 3 and c.get("lenses") == 2], key=lambda c: c["id"])
    clips = random.Random(0).sample(pool, min(nc, len(pool)))          # same draw as eval_seams.py (seed 0)
    D = strip_dirs()
    betas = np.arange(5.0, -5.0 - 1e-6, -RES)
    core = (np.abs(betas) < 3)[:, None]
    recs = []                                                           # (disparity, cd_cut, cd_flow, n)
    for c in clips:
        cal = load_calib(c["path"])
        lenses = [Lens(cal, L) for L in cal["lenses"]]
        for k in range(nf):
            t = 0.5 + (c["duration"] - 1.0) * (k + .5) / nf
            imgs = grab_pair(c["path"], t, cal["W"])
            if imgs is None:
                continue
            probe = SmartStitch(cal, lenses)
            cA, mA = probe._sample(0, imgs[0], D)
            cB, mB = probe._sample(1, imgs[1], D)
            A, B = luma(cA ** (1 / 2.2)), luma(cB ** (1 / 2.2))
            both = ((mA > .5) & (mB > .5)).cpu().numpy()
            a8, b8 = (np.clip(A, 0, 1) * 255).astype(np.uint8), (np.clip(B, 0, 1) * 255).astype(np.uint8)
            Fl = cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 3, 15, 4, 5, 1.1, 0)
            mag = np.hypot(Fl[..., 0], Fl[..., 1]) * RES
            parts = {}
            import os
            if os.environ.get("RAMPS"):         # compare two flow-seam ramp widths instead of cut vs flow (E42)
                r1, r2 = [float(x) for x in os.environ["RAMPS"].split(",")]
                pairs = (("cut", FlowStitch(cal, lenses, flow_ema=0, conf="photo", ramp=r1)),
                         ("flow", FlowStitch(cal, lenses, flow_ema=0, conf="photo", ramp=r2)))
            else:
                pairs = (("cut", SmartStitch(cal, lenses)), ("flow", FlowStitch(cal, lenses, flow_ema=0, conf="photo")))
            for name, eng in pairs:
                eng.update(imgs)
                _, pA, pB, wA, wB = eng.render(imgs, D, parts=True)
                w = (4 * wA * wB).cpu().numpy()
                parts[name] = (w, np.abs(luma(pA) - luma(pB)))
            W = A.shape[1]
            for b in range(36):
                sl = slice(b * W // 36, (b + 1) * W // 36)
                m = (both & core)[:, sl]
                if m.sum() < 400:
                    continue
                disp = float(np.median(mag[:, sl][m]))
                row = [disp]
                for name in ("cut", "flow"):
                    w, d = parts[name]
                    ww = w[:, sl] * both[:, sl]
                    row.append(float((ww * d[:, sl]).sum() / max(ww.sum(), 1e-6)) if ww.sum() > 20 else np.nan)
                recs.append(row)
        print(f"{c['name']}: {len(recs)} bins so far", flush=True)
    R = np.array(recs)
    R = R[~np.isnan(R).any(1)]
    edges = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 10]
    with open(out, "w") as fo:
        fo.write("lo hi mid n cut flow rel\n")
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (R[:, 0] >= lo) & (R[:, 0] < hi)
            if s.sum() < 10:
                continue
            cut, flow = R[s, 1].mean(), R[s, 2].mean()
            mid = float(np.median(R[s, 0]))
            fo.write(f"{lo} {hi} {mid:.3f} {int(s.sum())} {cut:.4f} {flow:.4f} {flow / cut:.3f}\n")
            print(f"disparity {lo:4.1f}-{hi:4.1f} deg (median {mid:.2f}, n={int(s.sum()):4d}): cut {cut:.4f}  flow {flow:.4f}  ratio {flow / cut:.2f}")


if __name__ == "__main__":
    main()
