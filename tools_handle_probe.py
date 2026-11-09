"""Handle-side sector probe: where is the both-valid overlap narrow, and what does the default seam do there?

    python tools_handle_probe.py <clip.OSV> <t> <out.jpg>
Writes a strip (alpha x beta in [-8, 8] deg, 0.05 deg/px) around the narrow sector: slave-only, master-only, output.
"""
import math
import sys

import cv2
import numpy as np
import torch

from osvcut import Lens, dev
from stitch import load_calib
from seam import FlowStitch
from eval_seams import grab_pair


def strip(a0, a1, b, res):
    a = torch.arange(a0, a1, res, device=dev) * math.pi / 180
    bb = torch.arange(b, -b - 1e-6, -res, device=dev) * math.pi / 180
    B, A = torch.meshgrid(bb, a, indexing="ij")
    return torch.stack([torch.cos(B) * torch.cos(A), torch.sin(B), torch.cos(B) * torch.sin(A)], -1)


def main(f, t, out, **opts):
    cal = load_calib(f)
    lenses = [Lens(cal, L) for L in cal["lenses"]]
    imgs = grab_pair(f, float(t), cal["W"])
    e = FlowStitch(cal, lenses, flow_ema=0, conf="photo", **opts)
    e.update(imgs)
    w = e.ov_w.cpu().numpy()
    # narrow sector from the overlap mask on the coarse flow strip
    D = e.fstrip
    _, mA = e._sample(0, imgs[0], D)
    _, mB = e._sample(1, imgs[1], D)
    both = ((mA > .5) & (mB > .5)).cpu().numpy()
    width = both.sum(0) * e.fres
    narrow = width < 2.5
    al = np.arange(e.fW) * e.fres
    print("overlap width percentiles (deg):", np.percentile(width, [1, 5, 25, 50]).round(2))
    print("narrow (<2.5 deg) columns: %.1f deg of azimuth" % (narrow.sum() * e.fres))
    if not narrow.any():
        return
    # centre of the narrow sector (circular mean)
    ang = np.deg2rad(al[narrow])
    c = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360
    print("narrow sector centre alpha = %.1f" % c)
    print("flow |d_beta| in the sector: max %.2f deg; ramp half-width min %.2f" % (
        np.abs(e.flow[1].cpu().numpy()[:, narrow]).max(), w[narrow].min()))
    Ds = strip(c - 50, c + 50, 8, 0.1)
    O, pA, pB, wA, wB = e.render(imgs, Ds, parts=True)
    im = lambda x: (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8)
    rows = [im(pA), im(pB), im(O)]
    # seam centre line on a copy of the output
    cen = e.ov_c.cpu().numpy()
    O2 = rows[2].copy()
    for x in range(O2.shape[1]):
        a = (c - 50 + x * 0.1) % 360
        bc = np.interp(a, al, cen, period=360)
        y = int(round((8 - bc) / 0.1))
        if 0 <= y < O2.shape[0]:
            O2[y, x] = (0, 0, 255)
    cv2.imwrite(out, np.vstack(rows + [O2]))
    np.save(out + ".npy", dict(c=c, width=width, al=al), allow_pickle=True)


def resolve(name):
    """Accept a path or a clip file name from the library index (avoids shell code-page issues)."""
    import json, os
    if os.path.exists(name):
        return name
    L = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "library.json"), encoding="utf8"))
    C = L.get("clips", L) if isinstance(L, dict) else L
    return [c["path"] for c in (C.values() if isinstance(C, dict) else C) if c["name"] == name][0]


if __name__ == "__main__":
    main(resolve(sys.argv[1]), *sys.argv[2:4])
