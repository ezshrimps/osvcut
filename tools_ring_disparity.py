"""E29: across-seam disparity around the ring from a wide search (not dense flow).

Lens A (just below the seam, beta in [-5, 0.6]) is compared with lens B shifted by s in [-3, 6] deg; the score
is the zero-mean normalised cross-correlation of band-passed luma (difference of Gaussians, 0.3 / 1.6 deg)
over a 6-deg azimuth window, counted only where both samples are valid.  Per 10-deg bin and frame the
median best shift over confident columns (ZNCC > 0.5) is kept; the output is mean and s.d. over frames.

    python tools_ring_disparity.py <clip name or path> [frames] [out.dat]
    OSV_LEGACY_SCALE=1 python tools_ring_disparity.py ...      # the old stream scale
"""
import math
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from osvcut import Lens, dev
from stitch import load_calib
from seam import SmartStitch
from eval_seams import grab_pair
from tools_handle_probe import resolve, strip

RES, BB = 0.2, 10.0
SHIFTS = np.arange(-3.0, 6.01, RES)


def _blur(Z, sg):
    r = int(3 * sg)
    g = torch.exp(-torch.arange(-r, r + 1, device=dev).float() ** 2 / (2 * sg ** 2))
    g = g / g.sum()
    Z = F.conv2d(Z[None, None], g.view(1, 1, -1, 1), padding=(r, 0))
    return F.conv2d(Z, g.view(1, 1, 1, -1), padding=(0, r))[0, 0]


def bandpass(X, V, s1=1.5, s2=8.0):
    nb = lambda sg: _blur(X * V, sg) / _blur(V, sg).clamp(min=1e-3)
    return nb(s1) - nb(s2)


def ring_disparity(e, imgs, D, betas):
    Y = lambda c: (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]).clamp(0, 1) ** (1 / 2.2)
    cA, mA = e._sample(0, imgs[0], D)
    cB, mB = e._sample(1, imgs[1], D)
    vA, vB = (mA > .5).float(), (mB > .5).float()
    A, B = bandpass(Y(cA), vA), bandpass(Y(cB), vB)
    band = ((betas <= 0.6) & (betas >= -5)).float()[:, None]
    k = int(3 / RES)

    def box(x):                               # circular box filter along azimuth
        x = torch.cat([x[-k:], x, x[:k]])
        c = torch.cumsum(torch.cat([x[:1] * 0, x]), 0)
        return c[2 * k + 1:] - c[:-2 * k - 1]

    S = []
    for s in SHIFTS:
        r = int(round(s / RES))               # B at beta + s
        Bs, vBs = torch.roll(B, r, 0), torch.roll(vB, r, 0)
        if r > 0:
            vBs[:r] = 0
        elif r < 0:
            vBs[r:] = 0
        m = vA * vBs * band
        n = box(m.sum(0))
        sa, sb = box((m * A).sum(0)), box((m * Bs).sum(0))
        saa, sbb, sab = box((m * A * A).sum(0)), box((m * Bs * Bs).sum(0)), box((m * A * Bs).sum(0))
        nn = n.clamp(min=1)
        z = (sab - sa * sb / nn) / ((saa - sa * sa / nn) * (sbb - sb * sb / nn)).clamp(min=1e-8).sqrt()
        S.append(torch.where(n > 240, z, torch.full_like(z, -1)))
    S = torch.stack(S).cpu().numpy()
    return SHIFTS[S.argmax(0)], S.max(0)


def main():
    f = resolve(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out = sys.argv[3] if len(sys.argv) > 3 else None
    cal = load_calib(f)
    lenses = [Lens(cal, L) for L in cal["lenses"]]
    e = SmartStitch(cal, lenses)
    D = strip(0, 360, BB, RES)
    betas = BB - torch.arange(D.shape[0], device=dev) * RES
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f],
                               capture_output=True, text=True).stdout)
    nb = 36
    per = np.full((n, nb), np.nan)
    for i in range(n):
        imgs = grab_pair(f, 1 + (dur - 2) * (i + .5) / n, cal["W"])
        if imgs is None:
            continue
        d, z = ring_disparity(e, imgs, D, betas)
        for b in range(nb):
            sl = slice(b * len(d) // nb, (b + 1) * len(d) // nb)
            ok = z[sl] > 0.5
            if ok.sum() >= 5:
                per[i, b] = np.median(d[sl][ok])
    mean, sd, cnt = np.nanmean(per, 0), np.nanstd(per, 0), np.sum(~np.isnan(per), 0)
    ok = cnt >= 5
    print(f"bins with >= 5 frames: {ok.sum()}/36; ring mean {np.nanmean(mean[ok]):+.2f} deg, "
          f"median |bin mean| {np.nanmedian(np.abs(mean[ok])):.2f} deg, median per-frame s.d. {np.nanmedian(sd[ok]):.2f}")
    if out:
        with open(out, "w") as fo:
            fo.write("az db sdb n\n")
            for b in range(nb):
                if ok[b]:
                    fo.write(f"{b * 10 + 5} {mean[b]:.3f} {sd[b]:.3f} {cnt[b]}\n")


if __name__ == "__main__":
    main()
