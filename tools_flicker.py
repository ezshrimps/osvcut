"""Seam quality metrics on rendered equirect videos (same source clip, levelled the same way).

flicker : temporal change inside the seam band / temporal change just outside it on the
          slave side.  Compare against the single-lens reference (no seam) - not against 1.0.
step    : brightness step across the seam: median over azimuth bins and frames of
          |log(mean luma just below / just above the seam)|, in percent.
"""
import sys, subprocess, numpy as np
from osvcut import attitudes, body_frames, view_dirs


def frames(f, n, W=960, H=480):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-frames:v", str(n), "-vf", f"scale={W}:{H}",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)


def main(src, vids, fps=50.0, N=300, dur=10.2):
    ts, qs = attitudes(src, 0, dur)
    Ms = body_frames(ts, qs, 1.0, "level")
    dirs = view_dirs("equirect", 960, 480).cpu().numpy()
    out = {}
    for v in vids:
        F = frames(v, N)
        fl, st = [], []
        for i in range(1, len(F)):
            M = Ms[min(np.searchsorted(ts, i / fps), len(Ms) - 1)]
            db = dirs @ M.T
            beta = np.degrees(np.arcsin(np.clip(db[..., 1], -1, 1)))
            alpha = np.degrees(np.arctan2(db[..., 2], db[..., 0])) % 360
            seam = np.abs(beta) < 2.5
            near = (beta < -5) & (beta > -12)
            d = np.abs(F[i] - F[i - 1])
            fl.append(d[seam].mean() / (d[near].mean() + 1e-6))
            if i % 10 == 0:
                lo = (beta > -2.5) & (beta < -1.0)
                hi = (beta > 1.0) & (beta < 2.5)
                b = (alpha // 10).astype(int)
                for k in range(36):
                    a, c = F[i][lo & (b == k)], F[i][hi & (b == k)]
                    if len(a) > 30 and len(c) > 30:
                        st.append(abs(np.log((a.mean() + 2) / (c.mean() + 2))))
        out[v] = (np.mean(fl), np.median(st) * 100)
        print(f"{v:34s} flicker {np.mean(fl):.3f}   step {np.median(st) * 100:5.2f}%")
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
