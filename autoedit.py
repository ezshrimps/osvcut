"""One-click edit: pick highlight segments from several clips and turn them into a sequence job.

Segments come from the speech transcript (talk) and the IMU motion curve (calm = steady and slow); talk segments
are framed on the selfie-stick holder from the IMU (look.holder_track), calm ones with a slow pan along the
heading.  Selection is greedy by score, spread over the clips, then put back in clip / time order.
"""
import json
import os

def suggest(motion, tx, dur):
    """Candidate segments of one clip: talk (merged transcript lines), calm (steady, slow) and whip (fast turn)."""
    S = []
    if tx and tx.get("segments"):
        cur = None
        for s in tx["segments"]:
            if cur and s["start"] - cur["end"] < 1.5 and s["end"] - cur["start"] <= 8:
                cur["end"] = s["end"]; cur["text"] += " " + s["text"]
            else:
                if cur:
                    S.append(cur)
                cur = dict(kind="talk", start=s["start"], end=s["end"], text=s["text"])
        if cur:
            S.append(cur)
    if motion and motion.get("speed"):
        sp, sh, r = motion["speed"], motion["shake"], motion["rate"]
        q = sorted(sh)[int(len(sh) * 0.6)] if sh else 0
        def runs(pred, minlen):
            out, st = [], -1
            for i in range(len(sp) + 1):
                ok = i < len(sp) and pred(i)
                if ok and st < 0: st = i
                if not ok and st >= 0:
                    if (i - st) / r >= minlen: out.append((st / r, i / r))
                    st = -1
            return out
        for a, b in runs(lambda i: sp[i] < 20 and sh[i] <= max(8, q), 2.5):
            S.append(dict(kind="calm", start=a, end=min(b, a + 8), text="steady shot (scenery / look-around)"))
        for a, b in runs(lambda i: sp[i] > 45, 0.5):
            S.append(dict(kind="whip", start=max(0, a - .3), end=min(dur, b + .3), text="fast rotation (whip transition)"))
    return sorted(S, key=lambda s: s["start"])


KIND_SCORE = {"talk": 3.0, "calm": 2.0}


def plan(clips, seconds=30.0, aspect="16:9", fps=30, transition="fade", captions="burn", min_len=2.5, max_len=6.0):
    """clips: library clip dicts (with path, id, duration).  Returns a sequence job body (specs, size, fps, ...)."""
    import analysis
    import look
    size = {"16:9": [1920, 1080], "9:16": [1080, 1920], "1:1": [1080, 1080], "4:5": [1080, 1350]}.get(aspect, [1920, 1080])
    cands = []
    for ci, c in enumerate(clips):
        tx = None
        tp = analysis.transcript_path(c["id"])
        if os.path.exists(tp):
            tx = json.load(open(tp, encoding="utf8"))
        try:
            mo = analysis.motion(c)
        except Exception:                                   # noqa: BLE001  (no IMU track)
            mo = None
        for s in suggest(mo, tx, c.get("duration", 0)):
            if s["kind"] not in KIND_SCORE:
                continue
            a, b = float(s["start"]), min(float(s["end"]), float(s["start"]) + max_len)
            b = min(b, c.get("duration", b) - 0.1)
            if b - a < min_len:
                continue
            score = KIND_SCORE[s["kind"]] + 0.2 * (b - a)
            cands.append(dict(ci=ci, clip=c, kind=s["kind"], start=a, end=b, score=score, tx=tx))
    # greedy: best first, at most one segment per clip per round, no overlaps, until the target length
    chosen, total, used = [], 0.0, {}
    for rnd in range(20):
        progress = False
        for ci in range(len(clips)):
            pool = sorted([x for x in cands if x["ci"] == ci and x not in chosen
                           and all(x["end"] <= y["start"] or x["start"] >= y["end"] for y in chosen if y["ci"] == ci)],
                          key=lambda x: -x["score"])
            if not pool or total >= seconds or used.get(ci, 0) > rnd:
                continue
            x = pool[0]
            chosen.append(x); total += x["end"] - x["start"]; used[ci] = used.get(ci, 0) + 1; progress = True
        if total >= seconds or not progress:
            break
    chosen.sort(key=lambda x: (x["ci"], x["start"]))
    specs = []
    for x in chosen:
        c, a, b = x["clip"], x["start"], x["end"]
        if x["kind"] == "talk":
            keys = look.holder_track(c, a, b)
            for k in keys:
                k["t"] = round(k["t"] - a, 3)
        else:                                               # slow pan along the heading
            keys = [dict(t=0, yaw=-12.0, pitch=-5.0, fov=100.0, ease="linear"), dict(t=round(b - a, 2), yaw=12.0, pitch=-5.0, fov=100.0)]
        sp = dict(src=c["path"], **{"in": round(a, 2), "out": round(b, 2)}, proj="flat", size=size, fps=fps, seam="flow",
                  seam_opts={"conf": "photo"}, cq=20, keys=keys)
        if captions in ("burn", "soft") and x["kind"] == "talk" and x["tx"]:
            caps = [dict(start=s["start"], end=s["end"], text=s["text"]) for s in x["tx"].get("segments", [])
                    if s["end"] > a and s["start"] < b]
            if caps:
                sp.update(captions=caps, captions_mode=captions)
        specs.append(sp)
    job = dict(specs=specs, size=size, fps=fps, seconds=round(total, 1),
               picked=[dict(clip=x["clip"]["id"], kind=x["kind"], start=round(x["start"], 1), end=round(x["end"], 1)) for x in chosen])
    if transition and transition != "cut":
        job["transition"] = dict(type=transition, duration=0.5)
    return job
