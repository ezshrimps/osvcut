"""Review sheets: what is in each segment of a film, for the author to tick what matters.

    python review.py build <name> <shots.json> [lines.json]

shots.json: a directed shot list (directed_edit format).  lines.json (optional): {"<shot index>": [{start, end,
text}, ...]} transcript lines to show (default: the shot's `lines`, else the clip transcript).
Writes cache/review/<name>/data.json and one JPEG per candidate element:
- people: faces found every second in four directions, grouped by direction; the clearest view of each;
- things: the most detailed / lit directions of the panorama at the start, middle and end (signs, stages, screens);
- the current framing: what each beat of the shot looks at now.
The studio page viewer/review.html?set=<name> lists the segments with their lines (translations / summaries can be
added to data.json by whoever understood them), the candidates with a tick box and a comment each, and saves the
author's marks to cache/review/<name>/marks.json (POST /api/review/<name>).
"""
import json
import math
import os
import sys
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "review")


def _view(clip, t, yaw, pitch, fov, w=400, h=300):
    import look
    return cv2.imdecode(np.frombuffer(look.view_jpeg(clip, t, yaw, pitch, fov, w=w, h=h, q=85), np.uint8), 1)


def _ang(a, b):
    return math.hypot(((a[0] - b[0] + 180) % 360 - 180) * math.cos(math.radians((a[1] + b[1]) / 2)), a[1] - b[1])


def people(clip, a, b, step=1.0, fov=120.0, n=512):
    """Faces seen in four directions every `step` s, grouped by direction: [(yaw, pitch, size, t, hits)]."""
    import torch
    import director
    import look
    from osvcut import view_dirs, cam_rot, dev
    hits = []
    t = a + 0.2
    while t < b:
        for y0 in (0, 90, 180, -90):
            C = torch.tensor(cam_rot(y0, 0), dtype=torch.float32, device=dev)
            d = view_dirs("flat", n, n, fov) @ C.T
            try:
                img = look._render(clip["path"], t, d)
            except Exception:                               # noqa: BLE001  (undecodable frame)
                continue
            bgr = np.ascontiguousarray((img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1])
            for f in director.detect_faces(bgr):
                x, y, w, h = f[:4]
                if max(abs(x + w / 2 - n / 2), abs(y + h / 2 - n / 2)) > 0.42 * n:
                    continue                                # near the probe edge: the next probe sees it better
                v = d[int(min(n - 1, y + h / 2)), int(min(n - 1, x + w / 2))].cpu().numpy()
                top = d[int(max(0, y)), int(min(n - 1, x + w / 2))].cpu().numpy()
                hits.append((math.degrees(math.atan2(v[0], v[1])), math.degrees(math.asin(max(-1, min(1, v[2])))),
                             2 * math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v, top)))))), t))
        t += step
    groups = []
    for h in sorted(hits, key=lambda h: -h[2]):             # biggest first: each group keeps its clearest view
        g = next((g for g in groups if _ang((h[0], h[1]), (g[0], g[1])) < 25), None)
        if g:
            g[4] += 1
        else:
            groups.append([h[0], h[1], h[2], h[3], 1])
    return [tuple(g) for g in groups if g[4] >= 2]


def things(clip, times, k=3):
    """The k most interesting directions (away from each other) at each time: [(yaw, pitch, t)]."""
    import director
    out = []
    for t in times:
        try:
            I, _ = director.interest_map(clip, t)
        except Exception:                                   # noqa: BLE001
            continue
        I = I.copy()
        for _ in range(k):
            am = director.aim(I, window=50)
            out.append((am["yaw"], am["pitch"], t))
            c = int((am["yaw"] + 180) / 360 * director.EW)
            r = int(director.EW * 60 / 360)
            for j in range(c - r, c + r):
                I[:, j % director.EW] = 0
    return out


def build(name, shots, lines_by_shot=None):
    import analysis
    lib = {c["id"]: dict(c, path=c["src"]) for c in
           json.load(urllib.request.urlopen("http://127.0.0.1:8360/api/library"))["clips"]}
    folder = os.path.join(OUT, name)
    os.makedirs(folder, exist_ok=True)
    segs, film_t = [], 0.0
    for i, sh in enumerate(shots):
        c = lib[sh["clip_id"]]
        a, b = float(sh["start"]), float(sh["end"])
        if lines_by_shot and str(i) in lines_by_shot:
            lines = lines_by_shot[str(i)]
        elif sh.get("lines"):
            lines = sh["lines"]
        else:
            tp = analysis.transcript_path(c["id"])
            tx = json.load(open(tp, encoding="utf8")) if os.path.exists(tp) else {}
            lines = [g for g in tx.get("segments", []) if g["end"] > a and g["start"] < b]
        agreed = {round(float(l["start"]), 1) for l in sh.get("lines") or []}
        cands = []

        def add(kind, label, t, yaw, pitch, fov):
            cid = f"{i:03d}_{len(cands):02d}"
            cv2.imwrite(os.path.join(folder, cid + ".jpg"), _view(c, t, yaw, pitch, fov), [cv2.IMWRITE_JPEG_QUALITY, 85])
            cands.append(dict(id=cid, kind=kind, label=label, t=round(t, 1), yaw=round(yaw, 1), pitch=round(pitch, 1),
                              fov=round(fov, 1)))
        if sh.get("kind") == "talk" or lines:
            for y, p, size, t, n in sorted(people(c, a, b), key=lambda g: -g[4])[:6]:
                add("person", f"人物 · 方向 {y:.0f}° · 出现 {n} 次", t, y, p - 5, max(45.0, min(90.0, size * 3.2)))
        for y, p, t in things(c, [a + 0.3, (a + b) / 2, b - 0.3])[:6]:
            if not any(_ang((y, p), (x["yaw"], x["pitch"])) < 30 and abs(x["t"] - t) < 3 for x in cands):
                add("thing", f"画面细节 · 方向 {y:.0f}°", t, y, p, 75.0)
        for bt in sh.get("beats") or ([dict(at=a, focus=sh["focus"], fov=sh.get("fov"))] if sh.get("focus") else []):
            t = max(a, min(b - 0.1, float(bt.get("at", a)) + 0.5))
            import director
            fo = bt["focus"]
            add("framing", f"我现在的取景 · 第 {t - a:.1f} 秒", t, fo["yaw"], fo["pitch"],
                director.beat_fov(bt.get("fov"), (1920, 1080)))
        segs.append(dict(i=i, film_t=round(film_t, 1), clip_id=c["id"], clip=c.get("name"), start=a, end=b,
                         kind=sh.get("kind"), chapter=sh.get("chapter"), note=sh.get("note", ""),
                         lines=[dict(start=round(float(l["start"]), 2), end=round(float(l["end"]), 2), text=l["text"],
                                     agreed=round(float(l["start"]), 1) in agreed) for l in lines],
                         candidates=cands))
        film_t += b - a
        print(f"segment {i + 1}/{len(shots)}: {len(cands)} candidates", flush=True)
    data_path = os.path.join(folder, "data.json")
    old = json.load(open(data_path, encoding="utf8")) if os.path.exists(data_path) else {}
    for s, o in zip(segs, old.get("segments", [])):          # keep translations / summaries already written
        for k in ("summary", "zh"):
            if k in o:
                s[k] = o[k]
    json.dump(dict(name=name, segments=segs), open(data_path, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("->", data_path)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) >= 4 and sys.argv[1] == "build":
        lb = json.load(open(sys.argv[4], encoding="utf8")) if len(sys.argv) > 4 else None
        build(sys.argv[2], json.load(open(sys.argv[3], encoding="utf8")), lb)
    else:
        print(__doc__)
