"""E35: seam quality of the *live* preview, emulated offline.

The studio shader stitches the proxy with tracks stored in the rig (2 Hz, quantised): seam path, exposure field and,
since rig v4, one (d_alpha, d_beta) displacement per 2-deg azimuth bin.  This tool reproduces that on proxy frames
with the tracks decoded from the rig (nearest sample in time), and scores it with the E11 metrics against
single-lens strips:  live-cut (path + 1-deg blend, no morph, as before v4)  vs  live-morph (v4).

    python tools_live_eval.py [frames_per_clip]
"""
import base64
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

from osvcut import Lens, dev, yuv420_to_rgb
from stitch import load_calib
from seam import FlowStitch, _circ_interp
from eval_seams import strip_dirs, luma, metrics, disagreement, hard_columns

HERE = os.path.dirname(os.path.abspath(__file__))


def frame_at(proxy, t, lp):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", proxy, "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "yuv420p", "-"], capture_output=True).stdout
    if len(raw) < lp * 2 * lp * 3 // 2:
        return None
    im = yuv420_to_rgb(raw, lp, 2 * lp)
    return [im[None, :, :, :lp], im[None, :, :, lp:]]


def main():
    nf = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    lib = json.load(open(os.path.join(HERE, "cache", "library.json"), encoding="utf8"))
    lib = lib if isinstance(lib, list) else lib.get("clips", lib)
    lib = {c["id"]: c for c in (lib.values() if isinstance(lib, dict) else lib)}
    D = strip_dirs()
    betas = np.arange(5.0, -5.0 - 1e-6, -0.1)
    rows = []
    for rj in sorted(glob.glob(os.path.join(HERE, "cache", "proxies", "*.json"))):
        r = json.load(open(rj))
        sm = r.get("seam") or {}
        if r.get("version", 0) < 4 or not sm.get("flow"):
            continue
        cid = os.path.basename(rj)[:-5]
        cal = load_calib(lib[cid]["path"])
        lp = r["lens_px"]
        pc = dict(cal, W=lp)
        lenses = [Lens(pc, L) for L in pc["lenses"]]
        n, bins, gb = sm["n"], sm["bins"], sm["gain_bins"]
        P = np.frombuffer(base64.b64decode(sm["path"]), np.uint8).reshape(n, bins).astype(np.float32)
        G = np.frombuffer(base64.b64decode(sm["gains"]), np.uint8).reshape(n, gb + 1, 3).astype(np.float32)
        Fl = np.frombuffer(base64.b64decode(sm["flow"]), np.uint8).reshape(n, bins, 2).astype(np.float32)
        dur = r.get("duration", n / sm["rate"])
        for k in range(nf):
            # evaluation times on a fixed grid (independent of the track rate), tracks interpolated linearly
            # between the two neighbouring samples exactly like the shader's texture lookup
            t = (k + 0.5) / nf * min(dur - 0.2, (n - 1) / sm["rate"]) + float(os.environ.get("TSHIFT", "0"))
            x = t * sm["rate"]; i0 = int(np.clip(np.floor(x), 0, n - 1)); i1 = min(n - 1, i0 + 1); fr = float(x - np.floor(x))
            lerp = lambda X: X[i0] * (1 - fr) + X[i1] * fr
            imgs = frame_at(rj[:-5] + ".mp4", t, lp)
            if imgs is None:
                continue
            eng = FlowStitch(pc, lenses, flow_ema=0)
            cA, mA = eng._sample(0, imgs[0], D)
            cB, mB = eng._sample(1, imgs[1], D)
            A, B = luma(cA ** (1 / 2.2)), luma(cB ** (1 / 2.2))
            both = ((mA > .5) & (mB > .5)).cpu().numpy()
            hard = hard_columns(A, B, both, betas)
            # tracks -> engine state (fW = 1800 columns at 0.2 deg)
            up = lambda v: np.interp(np.arange(eng.fW) * eng.fres, np.arange(bins) * 360 / bins + 1, v, period=360)
            path = up((lerp(P) / 255 - 0.5) * 9.0)
            lr = (lerp(G) / 255 - 0.5) * 2.0
            eng.logr = lr[:gb].T.copy()                 # local bins; the global mean is their mean in the engine
            FlI = lerp(Fl)
            fa, fb = [up((FlI[:, j] / 255 - 0.5) * 2 * sm.get("flow_range", 4.0)) for j in (0, 1)]
            eng.ov_c = torch.tensor(path, dtype=torch.float32, device=dev)
            res = {}
            if os.environ.get("ORACLE"):        # tracks computed on this very frame (upper bound for the track rate)
                o = FlowStitch(pc, lenses, conf="photo", flow1d=True, flow_ema=0)
                o.update(imgs)
                path = np.asarray(o.fpath); fa, fb = [o.flow[j, 0].cpu().numpy() for j in (0, 1)]
                eng.logr = np.array(o.logr)
                eng.ov_c = torch.tensor(path, dtype=torch.float32, device=dev)
            for name, half, fl in (("live_cut", 0.5, 0.0), ("live_morph", 0.75, 1.0)):
                eng.ov_w = torch.full((eng.fW,), half, device=dev)
                f2 = np.stack([fa, fb])[:, None, :].repeat(eng.fH, 1) * fl
                eng.flow = torch.tensor(f2, dtype=torch.float32, device=dev)
                O, pA, pB, wA, wB = eng.render(imgs, D, parts=True)
                m = metrics(luma(O), A, B, both, betas)
                if m is None:
                    continue
                m["cd"] = disagreement(pA, pB, wA, wB, both)
                bh = both & hard[None, :]
                m["cd_hard"] = disagreement(pA, pB, wA, wB, bh) if hard.mean() > 0.01 else float("nan")
                mh = metrics(luma(O), A, B, bh, betas) if hard.mean() > 0.01 else None
                m["edge_hard"] = mh["edge_dev"] if mh else float("nan")
                res[name] = m
            if len(res) == 2:
                rows.append(dict(clip=cid, t=round(t, 1), **{f"{a}_{b}": v for a, d in res.items() for b, v in d.items()}))
                print(f"{cid} t={t:6.1f}  CD cut {res['live_cut']['cd']:.4f}  morph {res['live_morph']['cd']:.4f}  "
                      f"edge {res['live_cut']['edge_dev']:.3f} / {res['live_morph']['edge_dev']:.3f}", flush=True)
    from scipy.stats import wilcoxon
    for k in ("cd", "cd_hard", "edge_dev", "edge_hard", "ghost_ssim"):
        a = np.array([r[f"live_cut_{k}"] for r in rows]); b = np.array([r[f"live_morph_{k}"] for r in rows])
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() > 5:
            better = (b < a) if k != "ghost_ssim" else (b > a)
            print(f"{k:10s} cut {a[ok].mean():.4f}  morph {b[ok].mean():.4f}  morph better {int(better[ok].sum())}/{ok.sum()}  "
                  f"p={wilcoxon(a[ok], b[ok]).pvalue:.2g}")
    json.dump(rows, open(os.path.join(HERE, "results", "live_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
