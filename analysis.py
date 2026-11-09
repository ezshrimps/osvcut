"""Per-clip analysis for the timeline: IMU motion curve and speech transcript (cached).

    python analysis.py transcript <clip.OSV> <out.json>
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(HERE, "cache", "analysis")
os.makedirs(ADIR, exist_ok=True)


def motion_path(cid):
    return os.path.join(ADIR, f"{cid}_motion.json")


def transcript_path(cid):
    return os.path.join(ADIR, f"{cid}_transcript.json")


def motion(clip, rate=10.0):
    """Angular speed of the camera (deg/s) and a shake score, sampled at `rate` Hz."""
    out = motion_path(clip["id"])
    if os.path.exists(out):
        return json.load(open(out, encoding="utf8"))
    from djmd import frame_attitudes
    ts, qs = frame_attitudes(clip["path"])
    if len(ts) < 3:
        res = dict(rate=rate, t0=0, speed=[], shake=[], tilt=[])
    else:
        q = qs / np.linalg.norm(qs, axis=1, keepdims=True)
        # angle between consecutive orientations
        dots = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(0, 1)
        ang = np.degrees(2 * np.arccos(dots))
        dt = np.diff(ts)
        dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.01
        w = ang / dt                                     # deg/s, per frame
        tm = ts[1:]
        grid = np.arange(0, ts[-1], 1 / rate)
        wi = np.interp(grid, tm, w)
        k = max(1, int(0.3 * rate))
        speed = np.convolve(wi, np.ones(k) / k, "same")
        # shake: high-frequency part of the per-frame angular speed (1/rate windows, minus 1 s trend)
        fr = 1 / np.median(dt)
        kk = max(1, int(fr))
        trend = np.convolve(w, np.ones(kk) / kk, "same")
        hf = np.abs(w - trend)
        shake = np.interp(grid, tm, np.convolve(hf, np.ones(max(1, int(fr / rate))) / max(1, int(fr / rate)), "same"))
        # tilt of the camera body up-axis from vertical (deg), via the empirical IMU->body map
        P = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1.]])
        wq, xq, yq, zq = q.T
        # third row of R(q) = world z expressed... R^T z = (2(xz-wy), 2(yz+wx), 1-2(x^2+y^2))
        up_imu = np.stack([2 * (xq * zq - wq * yq), 2 * (yq * zq + wq * xq), 1 - 2 * (xq ** 2 + yq ** 2)], 1)
        up_b = up_imu @ P.T
        tilt = np.degrees(np.arccos(np.clip(up_b[:, 2], -1, 1)))
        res = dict(rate=rate, t0=0,
                   speed=np.round(speed, 1).tolist(),
                   shake=np.round(shake, 1).tolist(),
                   tilt=np.round(np.interp(grid, ts, tilt), 1).tolist())
    json.dump(res, open(out, "w", encoding="utf8"))
    return res


def transcribe(src, out, model="small"):
    from faster_whisper import WhisperModel
    wav = out + ".wav"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-map", "0:a:0", "-ac", "1", "-ar", "16000", wav],
                       capture_output=True)
    if r.returncode != 0 or not os.path.exists(wav):
        json.dump(dict(language=None, segments=[], note="no audio"), open(out, "w", encoding="utf8"), ensure_ascii=False)
        print("progress 1/1", flush=True)
        return
    m = WhisperModel(model, device="cuda", compute_type="float16")
    segs, info = m.transcribe(wav, vad_filter=True)
    dur = max(info.duration, 0.1)
    res = []
    for s in segs:
        res.append(dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip()))
        print(f"progress {int(s.end * 10)}/{int(dur * 10)}", flush=True)
    json.dump(dict(language=info.language, segments=res), open(out, "w", encoding="utf8"), ensure_ascii=False)
    os.remove(wav)
    print("progress 1/1", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "transcript":
        transcribe(sys.argv[2], sys.argv[3])
