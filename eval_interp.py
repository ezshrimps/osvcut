"""E60: frame interpolation against real intermediate frames (100-fps clips).

From each run of consecutive frames 0..g, frames 1..g-1 are predicted from 0 and g by
  repeat  (nearest end — what slow motion beyond the native rate did before),
  blend   (linear cross-fade),
  flow    (interp.Interpolator),
and compared with the real frames (PSNR on a 1280x720 crop of the lens centre, and on the 20 % of pixels that
change most between the ends, where motion is).

    python eval_interp.py [--clips 12] [--runs 4] [--gaps 2,4]
"""
import argparse
import json
import random
import subprocess

import numpy as np
import torch

import library as libmod
from interp import Interpolator

dev = "cuda" if torch.cuda.is_available() else "cpu"
CW, CH = 1280, 720


def run_frames(path, t, n):
    vf = f"crop={CW}:{CH}:(iw-{CW})/2:(ih-{CH})/2"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-map", "0:v:0", "-vf", vf,
                          "-frames:v", str(n), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    k = len(raw) // (CW * CH * 3)
    if k < n:
        return None
    a = np.frombuffer(raw, np.uint8)[:n * CW * CH * 3].reshape(n, CH, CW, 3)
    return [torch.from_numpy(x.copy()).to(dev).permute(2, 0, 1).float() / 255 for x in a]


def psnr(x, y, m=None):
    e = (x - y).pow(2).mean(0)
    e = e[m] if m is not None else e
    return float(10 * torch.log10(1 / (e.mean() + 1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--gaps", default="2,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/e60_interp.json")
    a = ap.parse_args()
    lib = libmod.Library([])
    pool = sorted([c for c in lib.clips.values() if round(c.get("fps", 0)) >= 100 and c.get("duration", 0) > 5],
                  key=lambda c: c["id"])
    rng = random.Random(a.seed)
    clips = rng.sample(pool, min(a.clips, len(pool)))
    ip = Interpolator()
    rows = []
    for c in clips:
        for _ in range(a.runs):
            t = rng.uniform(1, c["duration"] - 2)
            for g in map(int, a.gaps.split(",")):
                fr = run_frames(c["path"], t, g + 1)
                if fr is None:
                    continue
                A, B = fr[0], fr[g]
                d = (A - B).abs().mean(0)
                m = d >= torch.quantile(d.flatten()[::7], 0.8)
                ph = [k / g for k in range(1, g)]
                pred = {"repeat": [A if p < 0.5 else B for p in ph],
                        "blend": [(1 - p) * A + p * B for p in ph],
                        "flow": ip.between(A, B, ph)}
                for k, p in enumerate(ph):
                    T = fr[k + 1]
                    rows.append(dict(clip=c["id"], t=round(t, 2), gap=g, p=round(p, 3), motion=float(d.mean()),
                                     **{f"{n}": psnr(v[k], T) for n, v in pred.items()},
                                     **{f"{n}_mov": psnr(v[k], T, m) for n, v in pred.items()}))
        print(c["id"], len(rows), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    from scipy.stats import wilcoxon
    for g in sorted({r["gap"] for r in rows}):
        R = [r for r in rows if r["gap"] == g]
        print(f"gap {g}: n={len(R)}")
        for key in ("", "_mov"):
            s = "  ".join(f"{n}{key} {np.mean([r[n + key] for r in R]):.2f}" for n in ("repeat", "blend", "flow"))
            fb = [r["flow" + key] - r["blend" + key] for r in R]
            print("  ", s, f"| flow>blend {sum(x > 0 for x in fb)}/{len(fb)} p={wilcoxon(fb).pvalue:.2g}")


if __name__ == "__main__":
    main()
