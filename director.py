"""Shot direction for the one-click edit: where to look and how the virtual camera moves.

Rules taken from how 360 footage is usually reframed by hand (DJI Studio / Insta360 guides, travel-vlog practice):
- aim at something: scenery shots look where the panorama has the most detail / light / colour, away from the person
  holding the stick and away from the floor;
- one motivated move per shot, eased, slow (under ~10 deg/s): pan across the subject, push in, pull out, or tilt up
  to reveal something tall; never the same move twice in a row;
- natural field of view (about 80-90 deg horizontal for 16:9; wider only for openers), level horizon;
- a little-planet opener that unfolds into the first shot and a pull-back into a little planet at the end;
- hard cuts inside a scene, a whip pan (fast turn out of one shot, into the next in the same direction) when the
  day or the part of the day changes.
"""
import math
import os

import numpy as np

EW, EH = 512, 256                          # interest map: equirect resolution


def base_fov(size):
    """Horizontal field of view of a 'normal' shot for the output aspect."""
    w, h = size
    return 84.0 if w >= h else 58.0


def interest_map(clip, t):
    """Per-pixel interest of the levelled panorama at time t (EH x EW, rows = +90..-90 deg pitch, cols = -180..180
    yaw) and the direction of the selfie stick (yaw, pitch)."""
    import cv2
    import torch
    import look
    from osvcut import view_dirs
    d = view_dirs("equirect", EW, EH)
    img = look._render(clip["path"], t, d)
    rgb = img.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    L = rgb.mean(2)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 6)
    mx, mn = rgb.max(2), rgb.min(2)
    sat = cv2.GaussianBlur((mx - mn) / (mx + 1e-3), (0, 0), 6)
    lum = cv2.GaussianBlur(L, (0, 0), 6)
    I = edge / (edge.mean() + 1e-6) * (0.4 + lum / (lum.mean() + 1e-6)) + 0.5 * sat * lum / (lum.mean() + 1e-6)
    lat = np.linspace(90, -90, EH)[:, None]
    I = I * np.exp(-((lat - 5) / 35.0) ** 2)                 # around (slightly above) the horizon, not sky / floor
    h = look.holder_direction(clip, t, 0.0)
    sy, sp = math.radians(h["stick_yaw"]), math.radians(h["stick_pitch"])
    lon = np.radians(np.linspace(-180, 180, EW, endpoint=False) + 180 / EW)[None, :]
    la = np.radians(lat)
    cosd = np.sin(la) * math.sin(sp) + np.cos(la) * math.cos(sp) * np.cos(lon - sy)
    I = I * np.where(cosd > math.cos(math.radians(50)), 0.15, 1.0)   # the holder's body around the stick
    del torch
    return I, (h["stick_yaw"], h["stick_pitch"])


def aim(I, window=70.0):
    """Best direction to look at: centre yaw of the most interesting `window`-degree slice, its interest-weighted
    pitch, how much more interesting its right side is than its left (for the pan direction) and its strength."""
    col = I.sum(0)
    k = int(EW * window / 360)
    ext = np.concatenate([col, col[:k]])
    win = np.convolve(ext, np.hanning(k), "valid")[:EW]   # tapered: the subject ends up in the middle
    i0 = int(np.argmax(win))
    ci = (i0 + k // 2) % EW
    yaw = -180 + 360 * (ci + 0.5) / EW
    cols = [(i0 + j) % EW for j in range(k)]
    sub = I[:, cols]
    lat = np.linspace(90, -90, EH)
    w = sub.sum(1)
    w = np.maximum(w - np.median(w), 0) + 1e-9               # what stands out, not the even background
    pitch = float((w * lat).sum() / (w.sum() + 1e-9))
    left, right = sub[:, :k // 2].sum(), sub[:, k // 2:].sum()
    return dict(yaw=round(float(yaw), 1), pitch=round(max(-20.0, min(25.0, pitch)), 1),
                side=1 if right >= left else -1, strength=float(win[i0] / (col.sum() * np.hanning(k).sum() / EW + 1e-9)))


MOVES = ("pan", "push", "pull", "tilt")


def scenery_keys(a, size, d, prev_move=None, pan_dir=1):
    """Keys (shot time) for a scenery shot of d seconds aimed at a = aim(...)."""
    f = base_fov(size)
    y, p = a["yaw"], a["pitch"]
    order = ["tilt"] if p > 8 else []
    order += ["pan", "push", "pull"]
    move = next(m for m in order if m != prev_move)
    if move == "pan":
        s = a["side"] * pan_dir if pan_dir else a["side"]
        span = min(36.0, 8.0 * d)                            # <= 8 deg/s
        k0, k1 = dict(yaw=y - s * span / 2, pitch=p, fov=f), dict(yaw=y + s * span / 2, pitch=p, fov=f)
    elif move == "push":
        k0, k1 = dict(yaw=y, pitch=p, fov=f * 1.18), dict(yaw=y, pitch=p, fov=f * 0.85)
    elif move == "pull":
        k0, k1 = dict(yaw=y, pitch=p, fov=f * 0.85), dict(yaw=y, pitch=p, fov=f * 1.18)
    else:                                                    # tilt up: reveal something tall
        k0, k1 = dict(yaw=y, pitch=max(-25.0, p - 18), fov=f), dict(yaw=y, pitch=min(35.0, p + 6), fov=f)
    return move, [dict(t=0.0, **k0), dict(t=round(d, 3), **k1)]


YUNET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_detection_yunet_2023mar.onnx")
_yunet = {}


def detect_faces(bgr, score=0.7):
    """Faces in a BGR image with OpenCV's YuNet: [(x, y, w, h, score)], biggest first.  [] without the model."""
    import cv2
    if not os.path.exists(YUNET):
        return []
    h, w = bgr.shape[:2]
    if "d" not in _yunet:
        _yunet["d"] = cv2.FaceDetectorYN.create(YUNET, "", (w, h), score, 0.3, 50)
    det = _yunet["d"]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None:
        return []
    out = [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])) for f in faces]
    return sorted(out, key=lambda f: -f[2] * f[3])


def face_direction(clip, t, yaw, pitch, fov=130.0, n=640):
    """Look around (yaw, pitch) with a wide view and return the speaker's face as dict(yaw, pitch, size), size = the
    face's angular height in degrees; None when no face is found."""
    import torch
    import look
    from osvcut import view_dirs, cam_rot, dev
    C = torch.tensor(cam_rot(yaw, pitch), dtype=torch.float32, device=dev)
    d = view_dirs("flat", n, n, fov) @ C.T
    img = look._render(clip["path"], t, d)
    bgr = np.ascontiguousarray((img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1])
    faces = detect_faces(bgr)
    if not faces:
        return None
    big = faces[0][2] * faces[0][3]                 # the speaker is the stick holder: of the bigger faces, the one
    near = [f for f in faces if f[2] * f[3] >= 0.25 * big]   # nearest the IMU guess (centre), not the biggest
    x, y, w, h, _ = min(near, key=lambda f: math.hypot(f[0] + f[2] / 2 - n / 2, f[1] + f[3] / 2 - n / 2))
    v = d[int(min(n - 1, max(0, y + h / 2))), int(min(n - 1, max(0, x + w / 2)))].cpu().numpy()
    top = d[int(max(0, y)), int(min(n - 1, max(0, x + w / 2)))].cpu().numpy()
    fy = math.degrees(math.atan2(v[0], v[1]))
    fp = math.degrees(math.asin(max(-1.0, min(1.0, float(v[2])))))
    size = 2 * math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v, top))))))
    return dict(yaw=round(fy, 1), pitch=round(fp, 1), size=round(size, 1))


def face_track(clip, keys, t0, size, step=0.8, min_hits=0.5):
    """Re-aim holder keys (shot time) onto the speaker's face: probe every `step` s around the IMU guess (or around
    the last face seen), key the face direction itself, put the eye line a little above the centre and widen the
    view when the face would fill the frame.  None when a face is seen in fewer than min_hits of the probes."""
    if not keys or not os.path.exists(YUNET):
        return None
    d = keys[-1]["t"]
    n = max(2, int(round(d / step)) + 1)
    times = [d * i / (n - 1) for i in range(n)]
    pts, last = [], None
    for ts in times:
        k = _at(keys, ts)
        f = None
        for y, p in ([(k["yaw"], k["pitch"])] + ([last] if last else [])):
            try:
                f = face_direction(clip, t0 + min(ts, d - 0.05), y, p)
            except Exception:                       # noqa: BLE001  (undecodable frame)
                f = None
            if f:
                break
        if f:
            last = (f["yaw"], f["pitch"])
        pts.append((ts, k, f))
    hits = [f for _, _, f in pts if f]
    if len(hits) < min_hits * len(pts):
        return None
    fsize = float(np.median([f["size"] for f in hits]))
    # yaw / pitch per probe: the face, or the IMU guess shifted by the nearest face's offset
    ys, ps = [], []
    for i, (ts, k, f) in enumerate(pts):
        if f:
            ys.append(f["yaw"]); ps.append(f["pitch"])
            continue
        j = min((j for j in range(len(pts)) if pts[j][2]), key=lambda j: abs(j - i))
        kj, fj = pts[j][1], pts[j][2]
        ys.append(k["yaw"] + _wrap(fj["yaw"] - kj["yaw"])); ps.append(k["pitch"] + fj["pitch"] - kj["pitch"])
    ys = np.degrees(np.unwrap(np.radians(ys)))
    if len(ys) >= 3:                                 # light 3-tap smoothing against detector jitter
        ys = np.convolve(np.pad(ys, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
        ps = np.convolve(np.pad(ps, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
    w, h = size
    out = []
    for ts, y, p in zip(times, ys, ps):
        fov = _at(keys, ts)["fov"]
        vfov = math.degrees(2 * math.atan(math.tan(math.radians(fov) / 2) * h / w))
        if fsize > 0.32 * vfov:                     # a face filling the frame: widen so head + shoulders fit
            fov = min(110.0, fov * fsize / (0.32 * vfov))
            vfov = math.degrees(2 * math.atan(math.tan(math.radians(fov) / 2) * h / w))
        out.append(dict(t=round(ts, 3), yaw=round(_wrap(float(y)), 1),
                        pitch=round(max(-60.0, min(80.0, float(p) - 0.12 * vfov)), 1), fov=round(fov, 1)))
    return out


def talk_keys(keys, size):
    """Holder-tracking keys (shot time) with a natural field of view and a slow push in."""
    f = base_fov(size)
    n = len(keys)
    out = []
    for i, k in enumerate(keys):
        out.append(dict(k, fov=round(f * (1.3 - 0.15 * i / max(1, n - 1)), 1)))
    return out


def _at(keys, t):
    from osvcut import interp_kf
    k = interp_kf(keys, t)
    return dict(yaw=round(k["yaw"], 2), pitch=round(k["pitch"], 2), fov=round(k["fov"], 2), dist=round(k.get("dist", 0), 3))


def _wrap(y):
    return ((y + 180) % 360) - 180


def whip_out(keys, d, turn=80.0, dur=0.35):
    """The end of a shot turns fast (accelerating) by `turn` degrees: the first half of a whip-pan transition."""
    t0 = max(0.0, d - dur)
    k0 = _at(keys, t0)
    keep = [dict(k) for k in keys if k["t"] < t0 - 1e-3]
    end = dict(k0, t=round(d, 3), yaw=_wrap(k0["yaw"] + turn))
    return keep + [dict(k0, t=round(t0, 3), ease="linear"), end]


def whip_in(keys, turn=80.0, dur=0.35):
    """The start of a shot arrives from a fast turn (decelerating): the second half of a whip pan."""
    k1 = _at(keys, dur)
    keep = [dict(k) for k in keys if k["t"] > dur + 1e-3]
    return [dict(k1, t=0.0, yaw=_wrap(k1["yaw"] - turn), ease="linear"), dict(k1, t=round(dur, 3))] + keep


def planet_open(keys, unfold=2.6, settle=3.4):
    """Little planet that spins and unfolds into the shot (the studio's 'little planet opener')."""
    k = _at(keys, settle)
    keep = [dict(x) for x in keys if x["t"] > settle + 1e-3]
    return [dict(t=0.0, yaw=_wrap(k["yaw"] - 120), pitch=-90.0, fov=300.0, dist=1.0),
            dict(t=1.2, yaw=_wrap(k["yaw"] - 40), pitch=-90.0, fov=260.0, dist=1.0),
            dict(t=unfold, yaw=k["yaw"], pitch=k["pitch"], fov=k["fov"], dist=0.15),
            dict(k, t=settle, dist=0.0)] + keep


def planet_close(keys, d, dur=2.4):
    """The last shot pulls back into a little planet."""
    t0 = max(0.0, d - dur)
    k = _at(keys, t0)
    keep = [dict(x) for x in keys if x["t"] < t0 - 1e-3]
    return keep + [dict(k, t=round(t0, 3)), dict(t=round(t0 + dur * 0.55, 3), yaw=_wrap(k["yaw"] + 30), pitch=-60.0,
                                                  fov=200.0, dist=0.6),
                   dict(t=round(d, 3), yaw=_wrap(k["yaw"] + 80), pitch=-90.0, fov=300.0, dist=1.0)]
