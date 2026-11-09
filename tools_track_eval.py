"""E62: label-free benchmark of subject following (forward-backward error).

Subjects: person-sized (8x16 deg), strongly textured boxes near the horizon of a levelled panorama.  Each
follower tracks the person forward for SECS seconds, then backward from where it ended; the angular distance
between the start and the return point is the forward-backward error (Kalal et al. 2010) - a tracker that drifts
or jumps to the background does not come back.  Also reported: the share of runs that end lost.

Followers:
  mil      the studio's tracker: MIL in a 640x480 view, re-aimed (and MIL re-initialised) when the subject leaves
           the central region
  ncc      the view is re-centred on the subject every frame; the subject is found by normalised cross-correlation
           of a slowly updated template within a search window around the centre

    python tools_track_eval.py [--clips 20] [--secs 3]
"""
import argparse
import json
import math
import random
import subprocess

import cv2
import numpy as np
import torch

import library as libmod
from osvcut import Lens, dev, level_frames, view_dirs, cam_rot
from stitch import load_calib
from seam import SmartStitch

VW, VH = 640, 480


def angles_of(d):
    return math.degrees(math.atan2(d[0], d[1])), math.degrees(math.asin(max(-1, min(1, d[2]))))


def ang_dist(a, b):
    def v(y, p):
        y, p = math.radians(y), math.radians(p)
        return np.array([math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)])
    return math.degrees(math.acos(float(np.clip(v(*a) @ v(*b), -1, 1))))


class Scene:
    def __init__(self, clip, t0, secs, fps=15):
        f = clip["path"]
        self.cal = load_calib(f)
        W = self.W = self.cal["W"]
        lenses = [Lens(self.cal, L) for L in self.cal["lenses"]]
        self.eng = SmartStitch(self.cal, lenses)
        ts, Ms = level_frames(f)
        tl, Ml = level_frames(f, "lock")            # world-fixed heading (camera turns compensated by the IMU)
        src = float(eval(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                         "stream=r_frame_rate", "-of", "csv=p=0", f], capture_output=True,
                                        text=True).stdout.split()[0]))
        step = max(1, round(src / fps))
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{secs:.3f}", "-i", f, "-filter_complex",
                            f"[0:v:0][0:v:1]hstack,select='not(mod(n\\,{step}))'", "-fps_mode", "passthrough",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        fb = W * 2 * W * 3
        self.frames, self.Ms, self.Ml = [], [], []
        for i in range(len(p) // fb):
            im = torch.frombuffer(bytearray(p[i * fb:(i + 1) * fb]), dtype=torch.uint8).to(dev).view(W, 2 * W, 3)
            self.frames.append(im)
            t = t0 + i * step / src
            self.Ms.append(np.asarray(Ms[min(np.searchsorted(ts, t), len(Ms) - 1)]))
            self.Ml.append(np.asarray(Ml[min(np.searchsorted(tl, t), len(Ml) - 1)]))
        self.eng.update(self.imgs(0))

    def imgs(self, i):
        im = self.frames[i].permute(2, 0, 1).float() / 255
        return [im[None, :, :, :self.W], im[None, :, :, self.W:]]

    def view(self, i, yaw, pitch, base, world=False):
        C = cam_rot(yaw, pitch)
        M = self.Ml[i] if world else self.Ms[i]
        d = base @ torch.tensor(M @ C, dtype=torch.float32, device=dev).T
        return (self.eng.render(self.imgs(i), d) * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy(), C

    def pano(self, i, W=2048):
        d = view_dirs("equirect", W, W // 2) @ torch.tensor(self.Ms[i], dtype=torch.float32, device=dev).T
        return (self.eng.render(self.imgs(i), d) * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()


def _dir(y, p):
    y, p = math.radians(y), math.radians(p)
    return np.array([math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)])


def to_world(scene, i, yaw, pitch):
    return angles_of(scene.Ml[i].T @ scene.Ms[i] @ _dir(yaw, pitch))


def to_level(scene, i, yaw, pitch):
    return angles_of(scene.Ms[i].T @ scene.Ml[i] @ _dir(yaw, pitch))


def follow(scene, order, yaw, pitch, bw, bh, method):
    """Track through scene frames in `order` (positions in the levelled frame of the first / last frame);
    methods ending in _w track in the world-fixed-heading frame.  Returns (final yaw, pitch, lost)."""
    world = method.endswith("_w")
    method = method[:-2] if world else method
    order = list(order)
    if world:
        yaw, pitch = to_world(scene, order[0], yaw, pitch)
        ly, lp, lost = _follow(scene, order, yaw, pitch, bw, bh, method, True)
        return (*to_level(scene, order[-1], ly, lp), lost)
    return _follow(scene, order, yaw, pitch, bw, bh, method, False)


def _follow(scene, order, yaw, pitch, bw, bh, method, world):
    vfov = float(np.clip(max(bw, bh * 4 / 3) * 3.0, 35, 100))
    fpx = (VW / 2) / math.tan(math.radians(vfov) / 2)
    base = view_dirs("flat", VW, VH, vfov)
    w = max(8, int(2 * fpx * math.tan(math.radians(bw) / 2)))
    h = max(8, int(2 * fpx * math.tan(math.radians(bh) / 2)))
    cy, cp = yaw, pitch
    ly, lp = yaw, pitch                                          # last tracked position
    misses, tracker, box, tmpl = 0, None, None, None
    for n, i in enumerate(order):
        view, C = scene.view(i, cy, cp, base, world)
        if n == 0:
            box = (VW // 2 - w // 2, VH // 2 - h // 2, w, h)
            if method == "mil":
                tracker = cv2.TrackerMIL_create(); tracker.init(view, box)
            else:
                g = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                tmpl = g[box[1]:box[1] + h, box[0]:box[0] + w].astype(np.float32)
            continue
        if method == "mil":
            ok, nb = tracker.update(view)
            if ok:
                box = tuple(int(v) for v in nb)
        else:
            g = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY).astype(np.float32)
            R = int(max(w, h) * 0.6)                     # search window around the centre
            x0, y0 = max(0, VW // 2 - w // 2 - R), max(0, VH // 2 - h // 2 - R)
            x1, y1 = min(VW, VW // 2 + w // 2 + R), min(VH, VH // 2 + h // 2 + R)
            res = cv2.matchTemplate(g[y0:y1, x0:x1], tmpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            ok = mx > 0.35
            if ok:
                box = (x0 + loc[0], y0 + loc[1], w, h)
                patch = g[box[1]:box[1] + h, box[0]:box[0] + w]
                if patch.shape == tmpl.shape:
                    tmpl = 0.9 * tmpl + 0.1 * patch          # slow update: follows pose changes, resists drift
        if not ok:
            misses += 1
            if misses > 15:
                return ly, lp, True
            continue
        misses = 0
        bx = box[0] + box[2] / 2 - VW / 2
        by = box[1] + box[3] / 2 - VH / 2
        dv = np.array([bx / fpx, 1.0, -by / fpx]); dv /= np.linalg.norm(dv)
        ty, tp = angles_of(C @ dv)
        ly, lp = ty, tp
        if method == "mil":
            if abs(bx) > VW * 0.18 or abs(by) > VH * 0.18:          # the studio's re-aim rule
                cy, cp = ty, tp
                view, C = scene.view(i, cy, cp, base, world)
                box = (VW // 2 - box[2] // 2, VH // 2 - box[3] // 2, box[2], box[3])
                tracker = cv2.TrackerMIL_create(); tracker.init(view, box)
        else:
            cy, cp = ty, tp                                          # re-centre every frame
    return ly, lp, False


def subjects(pano, rng, bw=8.0, bh=16.0, tries=60):
    """Person-sized boxes (bw x bh degrees) within 25 deg of the horizon, the most textured of `tries` random
    candidates (the OpenCV build here has no pedestrian detector; forward-backward error needs no labels)."""
    H, W = pano.shape[:2]
    g = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    en = np.sqrt(gx * gx + gy * gy)
    best = []
    for _ in range(tries):
        yaw, pitch = rng.uniform(-180, 180), rng.uniform(-20, 10)
        cx, cy = int((yaw / 360 + 0.5) * W), int((0.5 - pitch / 180) * H)
        w2, h2 = int(bw / 360 * W / 2), int(bh / 180 * H / 2)
        if cx - w2 < 0 or cx + w2 >= W:
            continue
        best.append((float(en[cy - h2:cy + h2, cx - w2:cx + w2].mean()), yaw, pitch))
    best.sort(reverse=True)
    return [(y, p, bw, bh, e) for e, y, p in best[:1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="mil,ncc,mil_w,ncc_w")
    ap.add_argument("--out", default="results/e62_track.json")
    a = ap.parse_args()
    lib = libmod.Library([])
    pool = sorted([c for c in lib.clips.values() if c.get("duration", 0) > a.secs + 2 and c.get("lenses") == 2],
                  key=lambda c: c["id"])
    rng = random.Random(a.seed)
    rng.shuffle(pool)
    rows = []
    methods = a.methods.split(",")
    for c in pool:
        if len({r["clip"] for r in rows}) >= a.clips:
            break
        t0 = rng.uniform(0.5, c["duration"] - a.secs - 1)
        sc = Scene(c, t0, a.secs)
        if len(sc.frames) < 10:
            continue
        P = subjects(sc.pano(0), rng)
        if not P:
            continue
        yaw, pitch, bw, bh, s = max(P, key=lambda p: p[4])
        n = len(sc.frames)
        row = dict(clip=c["id"], t0=round(t0, 2), yaw=round(yaw, 1), pitch=round(pitch, 1), bw=round(bw, 1),
                   bh=round(bh, 1), score=round(s, 2))
        for m in methods:
            fy, fp, lost = follow(sc, range(n), yaw, pitch, bw, bh, m)
            if lost:
                row[m] = dict(lost="forward")
                continue
            by_, bp, lost = follow(sc, range(n - 1, -1, -1), fy, fp, bw, bh, m)
            row[m] = dict(lost="backward") if lost else dict(fb=round(ang_dist((yaw, pitch), (by_, bp)), 3),
                                                              moved=round(ang_dist((yaw, pitch), (fy, fp)), 2))
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    for m in methods:
        fbs = [r[m]["fb"] for r in rows if "fb" in r[m]]
        lost = sum("lost" in r[m] for r in rows)
        print(f"{m}: n={len(rows)} lost={lost} FB median {np.median(fbs):.2f} deg, mean {np.mean(fbs):.2f}, "
              f"<1 deg {sum(x < 1 for x in fbs)}/{len(rows)}")


if __name__ == "__main__":
    main()
