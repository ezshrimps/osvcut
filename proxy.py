"""Fast dual-fisheye proxy + rig description for live stitching in the browser.

    python proxy.py <clip.OSV> <out.mp4> <out_rig.json> [lens_px]

The proxy keeps both fisheyes side by side (slave left, master right), downscaled to lens_px and
30 fps, decoded / scaled on the GPU.  The rig JSON carries what a shader needs to stitch and level it:
per-lens Kannala-Brandt model and rotation in proxy pixels, the usable radius per image angle, a
global exposure balance, and the levelling rotation as quaternions at 15 Hz.
"""
import json
import os
import math
import subprocess
import sys

import numpy as np


def rot_to_quat(M):
    """3x3 rotation -> (w, x, y, z)."""
    t = np.trace(M)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
    i = int(np.argmax(np.diag(M)))
    if i == 0:
        s = math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        return [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
    if i == 1:
        s = math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        return [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
    s = math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
    return [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]


def transcode(src, out, lens_px, duration):
    chain = lambda i, lab: (f"[0:v:{i}]fps=30,scale_cuda={lens_px}:{lens_px}:format=nv12,hwdownload,format=nv12[{lab}]")
    gpu = ["ffmpeg", "-v", "error", "-y", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", src,
           "-filter_complex", f"{chain(0, 'a')};{chain(1, 'b')};[a][b]hstack,format=yuv420p[v]"]
    cpu = ["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex",
           f"[0:v:0][0:v:1]hstack,scale={2 * lens_px}:{lens_px},fps=30,format=yuv420p[v]"]
    tail = ["-map", "[v]", "-map", "0:a?", "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "30", "-g", "30",
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", out]
    for cmd in (gpu + tail, cpu + tail):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in p.stdout:
            if line.startswith("out_time_us="):
                try:
                    us = int(line.split("=")[1])
                    print(f"progress {int(min(1, us / 1e6 / max(duration, 0.1)) * 900)}/1000", flush=True)
                except ValueError:
                    pass
        if p.wait() == 0:
            return
    raise SystemExit("transcode failed")


def seam_track(proxy, cal, dur, lens_px, rate=2.0, bins=180):
    """Min-difference seam path + per-azimuth exposure field sampled at `rate` Hz from the proxy itself
    (calibration rescaled to proxy pixels), quantised to bytes.
    path: beta* in [-4.5, 4.5] deg -> u8;  gains: log(slave/master) per 5-deg bin + global mean -> u8;
    flow: per 2-deg bin (d_alpha, d_beta) in [-4, 4] deg -> u8 pairs (E35: one displacement per azimuth keeps
    ~90 % of the flow seam's agreement gain, so the live shader can morph too)."""
    import base64
    import copy
    import torch
    from osvcut import Lens, yuv420_to_rgb
    from seam import FlowStitch
    pc = copy.deepcopy(cal)
    pc["dfl"] = cal["dfl"] * lens_px / cal["W"]
    pc["W"] = lens_px
    lenses = [Lens(pc, L) for L in pc["lenses"]]
    # 2 Hz samples are far apart: no temporal smoothing (the per-frame EMAs would lag by ~1 s; E36)
    eng = FlowStitch(pc, lenses, conf="photo", flow1d=True, flow_ema=0.0, seam_ema=0.0, gain_ema=0.0, lam_t=0.0)
    # sample k must be the frame at k/rate (the shader looks samples up at exactly that time); the fps filter
    # picks frames about half an interval later (E36), so select the first frame at or after each k/rate instead
    sel = f"select='gte(t\\,selected_n/{rate}-0.01)'"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", proxy, "-vf", sel, "-fps_mode", "passthrough", "-f", "rawvideo",
                          "-pix_fmt", "yuv420p", "-"], stdout=subprocess.PIPE)
    fb = lens_px * 2 * lens_px * 3 // 2
    rows_p, rows_g, rows_f = bytearray(), bytearray(), bytearray()
    n, total = 0, max(1, int(dur * rate))
    while True:
        raw = p.stdout.read(fb)
        if len(raw) < fb:
            break
        im = yuv420_to_rgb(raw, lens_px, 2 * lens_px)
        eng.update([im[None, :, :, :lens_px], im[None, :, :, lens_px:]])
        pth = np.asarray(eng.fpath).reshape(bins, -1).mean(1)      # beta* per 2-deg bin (flow seam's path)
        fl = eng.flow[:, 0, :].cpu().numpy().reshape(2, bins, -1).mean(2).T   # (bins, 2): d_alpha, d_beta
        rows_f += bytes(np.clip(np.round((np.clip(fl, -4, 4) / 4 * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8).ravel())
        rows_p += bytes(np.clip(np.round((pth / 4.5 * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8))
        lr = np.asarray(eng.logr)                                  # (3, 72)
        lrg = np.concatenate([lr, lr.mean(1, keepdims=True)], 1).T  # (73, 3)
        rows_g += bytes(np.clip(np.round((np.clip(lrg, -1, 1) * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8).ravel())
        n += 1
        if n % 5 == 0:
            print(f"progress {900 + int(90 * min(1, n / total))}/1000", flush=True)
    p.wait()
    return dict(rate=rate, n=n, bins=bins, gain_bins=72,
                path=base64.b64encode(bytes(rows_p)).decode(), gains=base64.b64encode(bytes(rows_g)).decode(),
                flow=base64.b64encode(bytes(rows_f)).decode(), flow_range=4.0)


def rig(src, lens_px, proxy=None):
    import torch
    from osvcut import Lens, level_frames
    from stitch import load_calib, qmat, lens_scale
    from seam import SmartStitch, build_mask
    from eval_seams import grab_pair
    cal = load_calib(src)
    W = cal["W"]
    p = lens_px / W
    lenses = []
    for L in cal["lenses"]:
        s = lens_scale(cal, L)
        _, (cx, cy, rlim) = build_mask(L, cal)
        lenses.append(dict(R=qmat(L["q"]).round(7).tolist(), k=[float(v) for v in L["k"]],
                           f=[L["fx"] * s * p, L["fy"] * s * p], c=[cx * p, cy * p],
                           rlim=[round(float(v) * p, 2) for v in rlim]))
    # global exposure balance from a few frames
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                               capture_output=True, text=True).stdout or 0)
    eng = SmartStitch(cal, [Lens(cal, L) for L in cal["lenses"]])
    lrs = []
    for k in range(3):
        imgs = grab_pair(src, max(0.0, dur * (k + 0.5) / 3 - 0.05), W)
        if imgs is None:
            continue
        eng.logr = None
        eng.update(imgs)
        lrs.append(np.array(eng.logr).mean(1))
    lr = np.mean(lrs, 0) if lrs else np.zeros(3)
    ts, Ms = level_frames(src)
    grid = np.arange(0, ts[-1] if len(ts) else 0, 1 / 15.0)
    qs = []
    for t in grid:
        M = Ms[min(np.searchsorted(ts, t), len(Ms) - 1)]
        qs.append([round(v, 5) for v in rot_to_quat(np.asarray(M))])
    # 5 Hz tracks are measurably better than 2 Hz in the live preview (E38); long clips stay at 2 Hz to keep the rig small
    hz = float(os.environ.get("OSV_TRACK_HZ", 5.0 if dur <= 600 else 2.0))
    seam = seam_track(proxy, cal, dur, lens_px, rate=hz, bins=int(os.environ.get("OSV_TRACK_BINS", 180))) if proxy else None
    return dict(version=4, lens_px=lens_px, width=2 * lens_px, height=lens_px, lenses=lenses, seam=seam,
                gain_slave=np.exp(-lr / 2).round(4).tolist(), gain_master=np.exp(lr / 2).round(4).tolist(),
                level_rate=15.0, level_q=qs, duration=dur, theta_cut_deg=98.0)


if __name__ == "__main__":
    src, out, out_rig = sys.argv[1:4]
    lens_px = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 1536
    final = sys.argv[5] if len(sys.argv) > 5 else None      # publish the proxy under this name as soon as it exists
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                               capture_output=True, text=True).stdout or 0)
    transcode(src, out, lens_px, dur)
    if final:
        # stage 1: the clip is viewable live right away (fixed seam); stage 2 adds the seam / flow track
        os.replace(out, final)
        out = final
        json.dump(rig(src, lens_px, proxy=None), open(out_rig, "w"), separators=(",", ":"))
        print("progress 900/1000", flush=True)
    r = rig(src, lens_px, proxy=out)
    json.dump(r, open(out_rig + ".tmp", "w"), separators=(",", ":"))
    os.replace(out_rig + ".tmp", out_rig)
    print("progress 1000/1000", flush=True)
