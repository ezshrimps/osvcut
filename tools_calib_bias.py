"""E19: is there a *systematic* (calibration) misalignment along the seam?

Per 10-degree azimuth bin, the median confident lens-to-lens flow (d_alpha, d_beta) in each of N frames;
then across frames: mean, std and a consistency t-statistic.  A calibration error shows up as a bias
that keeps its sign from frame to frame; parallax and noise do not.

    python tools_calib_bias.py <clip-id-or-path> [frames]
"""
import sys, numpy as np, torch
from osvcut import Lens, dev
from stitch import load_calib
from seam import FlowStitch
from eval_seams import grab_pair


def main():
    f = sys.argv[1]
    if len(f) == 12 and all(ch in "0123456789abcdef" for ch in f):
        import library as _lib
        f = _lib.Library([r"E:\Videos"]).get(f)["path"]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    cal = load_calib(f); W = cal["W"]; lenses = [Lens(cal, L) for L in cal["lenses"]]
    import subprocess
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f],
                               capture_output=True, text=True).stdout)
    e = FlowStitch(cal, lenses, flow_ema=0, center="overlap")
    H, Wd = e.fH, e.fW
    nb = 36
    per = np.full((n, nb, 2), np.nan)
    for k in range(n):
        t = 1 + (dur - 2) * (k + .5) / n
        imgs = grab_pair(f, t, W)
        if imgs is None:
            continue
        e.update(imgs)
        conf = e.last_conf > 0.6
        Fa = e.last_raw[..., 0] * e.fres
        Fb = -e.last_raw[..., 1] * e.fres
        core = np.zeros_like(conf); core[H // 4: 3 * H // 4] = True
        for b in range(nb):
            sl = (slice(None), slice(b * Wd // nb, (b + 1) * Wd // nb))
            m = (conf & core)[sl]
            if m.sum() > 40:
                per[k, b] = [np.median(Fa[sl][m]), np.median(Fb[sl][m])]
    mean = np.nanmean(per, 0); sd = np.nanstd(per, 0); cnt = np.sum(~np.isnan(per[..., 0]), 0)
    tstat = mean / (sd / np.sqrt(np.maximum(cnt, 1))[:, None] + 1e-6)
    print(" az   n   d_alpha mean±sd    d_beta mean±sd    |t|_beta")
    for b in range(nb):
        if cnt[b] >= 5:
            print(f"{b*10:4d} {cnt[b]:3d}   {mean[b,0]:+.2f}±{sd[b,0]:.2f}      {mean[b,1]:+.2f}±{sd[b,1]:.2f}      {abs(tstat[b,1]):5.1f}")
    ok = cnt >= 5
    print(f"bins with |t|>4 (systematic): alpha {int(np.sum(np.abs(tstat[ok,0])>4))}/{ok.sum()}, beta {int(np.sum(np.abs(tstat[ok,1])>4))}/{ok.sum()}")
    print(f"rms of per-bin mean bias: {np.sqrt(np.nanmean(mean[ok]**2)):.3f} deg; median per-frame |flow| sd: {np.nanmedian(sd[ok]):.3f}")
    np.save(sys.argv[3] if len(sys.argv) > 3 else "results/_calib_bias.npy", per)


if __name__ == "__main__":
    main()
