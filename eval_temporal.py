"""Multi-clip temporal seam benchmark (no video encoding in the loop).

    python eval_temporal.py [--clips 8] [--seconds 4] [--seed 1] [--out results/temporal_bench.json]

For each clip, a window of `seconds` is decoded at 25 fps and rendered to a 1920x960 levelled
equirect by every method, plus a slave-only render (no seam) as the reference. Per frame pair:
flicker = mean|dI| inside |beta| < 2.5 deg / mean|dI| in -12 < beta < -5 (same side, same frames).
Reported per clip and method; the quantity of interest is flicker(method) - flicker(reference).
"""
import argparse
import json
import os
import random
import subprocess

import numpy as np
import torch

from osvcut import Lens, dev, level_frames, view_dirs
from stitch import load_calib
from seam import SmartStitch, FlowStitch
import library as libmod

W_OUT, H_OUT = 1920, 960


def frames(f, t0, secs, fps, W):
    src_fps = float(eval(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                         "stream=r_frame_rate", "-of", "csv=p=0", f], capture_output=True,
                                        text=True).stdout.strip().split("\n")[0]))
    step = max(1, round(src_fps / fps))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(t0), "-t", str(secs), "-i", f, "-filter_complex",
                          f"[0:v:0][0:v:1]hstack,select='not(mod(n\\,{step}))'", "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    fb = W * 2 * W * 3
    i = 0
    while True:
        raw = p.stdout.read(fb)
        if len(raw) < fb:
            break
        im = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dev).view(W, 2 * W, 3).permute(2, 0, 1).float() / 255
        yield t0 + i * step / src_fps, [im[None, :, :, :W], im[None, :, :, W:]]
        i += 1
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=4)
    ap.add_argument("--fps", type=float, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/temporal_bench.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    lib = libmod.Library([r"E:\Videos"])
    pool = sorted([c for c in lib.clips.values() if c.get("duration", 0) > a.seconds + 2 and c.get("lenses") == 2],
                  key=lambda c: c["id"])
    clips = random.Random(a.seed).sample(pool, min(a.clips, len(pool)))
    eq = view_dirs("equirect", W_OUT, H_OUT)
    methods = ["single", "legacy", "smart", "flow"]
    results = []
    for c in clips:
        f = c["path"]
        cal = load_calib(f)
        W = cal["W"]
        lenses = [Lens(cal, L) for L in cal["lenses"]]
        ts, Ms = level_frames(f)
        t0 = random.Random(c["id"]).uniform(1, c["duration"] - a.seconds - 1)
        engines = {"smart": SmartStitch(cal, lenses), "flow": FlowStitch(cal, lenses)}
        prev = {m: None for m in methods}
        acc = {m: [] for m in methods}
        for t, imgs in frames(f, t0, a.seconds, a.fps, W):
            M = Ms[min(np.searchsorted(ts, t), len(Ms) - 1)]
            d = eq @ torch.tensor(M, dtype=torch.float32, device=dev).T
            db = d.cpu().numpy()
            beta = np.degrees(np.arcsin(np.clip(db[..., 1], -1, 1)))
            seam = np.abs(beta) < 2.5
            near = (beta < -5) & (beta > -12)
            for m in methods:
                if m == "single":
                    o = lenses[0].sample(imgs[0], d)[0]
                elif m == "legacy":
                    cs = [L.sample(img, d) for L, img in zip(lenses, imgs)]
                    o = (cs[0][0] * cs[0][1] + cs[1][0] * cs[1][1]) / (cs[0][1] + cs[1][1] + 1e-6)
                else:
                    engines[m].update(imgs)
                    o = engines[m].render(imgs, d)
                g = (0.299 * o[0] + 0.587 * o[1] + 0.114 * o[2]).clamp(0, 1).cpu().numpy() * 255
                if prev[m] is not None:
                    dI = np.abs(g - prev[m])
                    acc[m].append(dI[seam].mean() / (dI[near].mean() + 1e-6))
                prev[m] = g
        row = dict(clip=c["id"], t0=round(t0, 2), n=len(acc["single"]),
                   **{m: float(np.mean(v)) if v else float("nan") for m, v in acc.items()})
        results.append(row)
        print(f"{c['name']}: " + "  ".join(f"{m} {row[m]:.3f}" for m in methods), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)
    R = {m: np.array([r[m] for r in results]) for m in methods}
    print("\nexcess flicker vs. single-lens reference (mean over clips, [min, max]):")
    for m in methods[1:]:
        d = R[m] - R["single"]
        print(f"  {m:7s} {np.nanmean(d):+.3f}  [{np.nanmin(d):+.3f}, {np.nanmax(d):+.3f}]  "
              f"clips above reference: {int(np.nansum(d > 0))}/{len(d)}")


if __name__ == "__main__":
    main()
