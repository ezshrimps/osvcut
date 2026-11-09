"""Follow a subject through 360 footage and emit camera keyframes.

    python track.py job.json

job.json: {"src", "t0", "t1", "yaw", "pitch", "box_w", "box_h", "fov"?, "fps"?, "out", "debug"?}
  yaw/pitch : centre of the subject in the levelled panorama (degrees, viewer convention)
  box_w/h   : angular size of the subject box (degrees)
  out       : JSON written with {"keys": [{t, yaw, pitch, fov}], "lost_at": t|null, "track": [...]}

The subject is tracked in a small rectilinear view rendered around it; whenever it drifts away from
the centre the view is re-aimed and the tracker re-initialised, so it can follow all the way round.
"""
import json
import math
import subprocess
import sys

import cv2
import numpy as np
import torch

from osvcut import Lens, dev, level_frames, view_dirs, cam_rot
from stitch import load_calib
from seam import SmartStitch


def angles_of(d):
    return math.degrees(math.atan2(d[0], d[1])), math.degrees(math.asin(max(-1, min(1, d[2]))))


def run(job):
    f = job["src"]
    t0, t1 = float(job["t0"]), float(job["t1"])
    fps = float(job.get("fps", 15))
    cal = load_calib(f)
    W = cal["W"]
    lenses = [Lens(cal, L) for L in cal["lenses"]]
    eng = SmartStitch(cal, lenses)
    ts, Ms = level_frames(f, job.get("stabilize", "level"), job.get("smooth", 1.0))
    bw, bh = float(job["box_w"]), float(job["box_h"])
    vfov = float(np.clip(max(bw, bh * 4 / 3) * 3.0, 35, 100))      # subject ~1/3 of the tracking view
    VW, VH = 640, 480
    fpx = (VW / 2) / math.tan(math.radians(vfov) / 2)
    base = view_dirs("flat", VW, VH, vfov)
    srcfps = float(eval(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=r_frame_rate", "-of", "csv=p=0", f],
                                       capture_output=True, text=True).stdout.strip().split("\n")[0]))
    step = max(1, round(srcfps / fps))
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(t0), "-to", str(t1), "-i", f, "-filter_complex",
                            f"[0:v:0][0:v:1]hstack,select='not(mod(n\\,{step}))'", "-fps_mode", "passthrough",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    dbg = None
    if job.get("debug"):
        dbg = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{VW}x{VH}",
                                "-r", str(srcfps / step), "-i", "-", "-c:v", "h264_nvenc", "-cq", "28", job["debug"]],
                               stdin=subprocess.PIPE)
    cy, cp = float(job["yaw"]), float(job["pitch"])
    tracker = None
    box = None
    track, lost_at, misses = [], None, 0
    total = max(1, int((t1 - t0) * srcfps / step))
    fb = W * 2 * W * 3
    i = 0

    def render(imgs, M, yaw, pitch):
        C = cam_rot(yaw, pitch)
        d = base @ torch.tensor(M @ C, dtype=torch.float32, device=dev).T
        im = (eng.render(imgs, d) * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()
        return im, C

    while True:
        raw = dec.stdout.read(fb)
        if len(raw) < fb:
            break
        t = t0 + i * step / srcfps
        if i % 5 == 0:
            print(f"progress {i}/{total}", flush=True)
        i += 1
        im = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dev).view(W, 2 * W, 3).permute(2, 0, 1).float() / 255
        imgs = [im[None, :, :, :W], im[None, :, :, W:]]
        if i % 10 == 1:
            eng.update(imgs)
        M = Ms[min(np.searchsorted(ts, t), len(Ms) - 1)]
        view, C = render(imgs, M, cy, cp)
        if tracker is None:
            w = 2 * fpx * math.tan(math.radians(bw) / 2)
            h = 2 * fpx * math.tan(math.radians(bh) / 2)
            box = (int(VW / 2 - w / 2), int(VH / 2 - h / 2), max(8, int(w)), max(8, int(h)))
            tracker = cv2.TrackerMIL_create()
            tracker.init(view, box)
            ok = True
        else:
            ok, nb = tracker.update(view)
            if ok:
                box = tuple(int(v) for v in nb)
        if ok:
            misses = 0
            bx = box[0] + box[2] / 2 - VW / 2
            by = box[1] + box[3] / 2 - VH / 2
            dv = np.array([bx / fpx, 1.0, -by / fpx])
            dv /= np.linalg.norm(dv)
            ty, tp = angles_of(C @ dv)
            track.append((round(t, 3), ty, tp))
            # re-aim when the subject leaves the central region
            if abs(bx) > VW * 0.18 or abs(by) > VH * 0.18:
                cy, cp = ty, tp
                view, C = render(imgs, M, cy, cp)
                box = (int(VW / 2 - box[2] / 2), int(VH / 2 - box[3] / 2), box[2], box[3])
                tracker = cv2.TrackerMIL_create()
                tracker.init(view, box)
        else:
            misses += 1
            if misses > fps * 1.0:
                lost_at = round(t, 3)
                break
        if dbg:
            v = view.copy()
            cv2.rectangle(v, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 255) if ok else (0, 0, 255), 2)
            cv2.putText(v, f"t={t:.2f}", (8, 24), 0, .7, (255, 255, 255), 2)
            dbg.stdin.write(v.tobytes())
    dec.kill()
    if dbg:
        dbg.stdin.close()
        dbg.wait()
    # smooth (unwrap yaw), then decimate to keyframes every 0.25 s
    keys = []
    if track:
        T = np.array([p[0] for p in track])
        Y = np.degrees(np.unwrap(np.radians([p[1] for p in track])))
        P = np.array([p[2] for p in track])
        k = max(1, int(round(fps * 0.6)))
        pad = lambda a: np.pad(a, (k // 2, k - 1 - k // 2), mode="edge")
        Ys = np.convolve(pad(Y), np.ones(k) / k, "valid")
        Ps = np.convolve(pad(P), np.ones(k) / k, "valid")
        fov = float(job.get("fov", 90))
        tk = np.arange(T[0], T[-1] + 1e-6, 0.25)
        for tt in tk:
            yy = float(np.interp(tt, T, Ys))
            keys.append(dict(t=round(float(tt), 3), yaw=round(((yy + 180) % 360) - 180, 2),
                             pitch=round(float(np.interp(tt, T, Ps)), 2), fov=fov))
    out = dict(keys=keys, lost_at=lost_at, track=track, vfov=vfov)
    json.dump(out, open(job["out"], "w", encoding="utf8"))
    print(f"progress {total}/{total}", flush=True)
    print(f"tracked {len(track)} frames, {len(keys)} keys, lost_at={lost_at}", flush=True)


if __name__ == "__main__":
    run(json.load(open(sys.argv[1], encoding="utf8")))
