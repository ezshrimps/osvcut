"""E15: is lens-to-lens parallax on the ground explained by a gravity-aligned ground plane?
Fits flow_beta (deg) = k * sin(-elevation) + c on confident strip pixels below the horizon."""
import sys, json, numpy as np, torch
from osvcut import Lens, dev, level_frames
from stitch import load_calib
from seam import FlowStitch
from eval_seams import grab_pair
import library as libmod
lib = libmod.Library([r"E:\Videos"])
ids = sys.argv[1:]
for cid in ids:
    c = lib.get(cid); f = c["path"]
    cal = load_calib(f); W = cal["W"]; lenses = [Lens(cal, L) for L in cal["lenses"]]
    ts, Ms = level_frames(f)
    ks = []
    for k in range(8):
        t = 1 + (c["duration"] - 2) * (k + .5) / 8
        imgs = grab_pair(f, t, W)
        if imgs is None: continue
        e = FlowStitch(cal, lenses, flow_ema=0, center="overlap"); e.update(imgs)
        Fb = -e.last_raw[..., 1] * e.fres            # beta component, degrees
        Fa = e.last_raw[..., 0] * e.fres
        conf = e.last_conf
        M = Ms[min(np.searchsorted(ts, t), len(Ms) - 1)]
        D = e.fstrip.cpu().numpy()                   # body dirs (fH,fW,3)
        out = D @ M                                  # d_out = M^T d_body  (row-vector form)
        el = np.degrees(np.arcsin(np.clip(out[..., 2], -1, 1)))
        sel = (conf > 0.6) & (el < -15)
        up = (conf > 0.6) & (el > 15)
        if sel.sum() < 300: continue
        g = np.sin(np.radians(-el[sel])); y = Fb[sel]
        A = np.stack([g, np.ones_like(g)], 1); (kk, cc), *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ [kk, cc]; r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ks.append(kk)
        print(f"{c['name']} t={t:6.1f}: n={sel.sum():5d} k={kk:+.3f} c={cc:+.3f} R2={r2:.2f} | median |Fb| ground {np.median(np.abs(y)):.2f} sky/up {np.median(np.abs(Fb[up])) if up.sum() else float('nan'):.2f} | |Fa| ground {np.median(np.abs(Fa[sel])):.2f}")
    if ks: print(f"  -> k median {np.median(ks):+.3f}, spread {np.std(ks):.3f}\n")
