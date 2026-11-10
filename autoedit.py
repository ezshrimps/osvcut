"""One-click edit: pick highlight segments from several clips and turn them into a sequence job.

Segments come from the speech transcript (talk) and the IMU motion curve (calm = steady and slow); talk segments
are framed on the selfie-stick holder from the IMU (look.holder_track), calm ones with a slow pan along the
heading.  Clips are taken in capture order; by default the cuts are spread evenly over the footage timeline
(see plan), then put back in capture / time order.
"""
import json
import math
import os
import re
from datetime import datetime

def suggest(motion, tx, dur, calm_every=None):
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
            # calm_every: also offer later windows of a long steady run, so long clips can be sampled along their length
            starts = [a] if not calm_every else [a + k * calm_every for k in range(int((b - a - 2.5) // calm_every) + 1)]
            for s0 in starts:
                S.append(dict(kind="calm", start=s0, end=min(b, s0 + 8), text="steady shot (scenery / look-around)"))
        for a, b in runs(lambda i: sp[i] > 45, 0.5):
            S.append(dict(kind="whip", start=max(0, a - .3), end=min(dur, b + .3), text="fast rotation (whip transition)"))
    return sorted(S, key=lambda s: s["start"])


KIND_SCORE = {"talk": 3.0, "calm": 2.0}
TALK_LIFT = 58.0                                    # deg above the stick: the face, not the chest (talk framing)
TALK_MAX = 8.0                                      # s: longest talk cut (whole transcript sentences)
SCENERY_LEN = (3.5, 3.0, 4.5, 3.0, 4.0)             # s: scenery cut lengths, varied so the rhythm is not a metronome
CALM_EVERY = 15.0                                   # s between calm windows offered inside one long steady run
PARTS = [(0, 5, "凌晨"), (5, 12, "早"), (12, 18, "午"), (18, 24, "晚")]
SIZES = {"16:9": [1920, 1080], "9:16": [1080, 1920], "1:1": [1080, 1080], "4:5": [1080, 1350]}


def shot_time(c):
    """Capture time: the library's `shot` (from the file name), else a timestamp in the name, else the file mtime."""
    s = c.get("shot")
    if s:
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            pass
    m = re.search(r"(20\d{12})", c.get("name") or os.path.basename(c.get("path", "")))
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(c["path"]))
    except (OSError, KeyError):
        return datetime.min


def period(dt):
    """'4/4 晚' - day and part of day of a capture time."""
    if dt == datetime.min:
        return "unknown"
    return f"{dt.month}/{dt.day} " + next(p for a, b, p in PARTS if a <= dt.hour < b)


def _words(text):
    return len(re.findall(r"[㐀-鿿]|[A-Za-z0-9']+", text or ""))


def plan(clips, seconds=30.0, aspect="16:9", fps=30, transition="auto", captions="burn", min_len=2.5, max_len=6.0,
         music=None, coverage="even", min_clip_seconds=2.0, progress=None):
    """clips: library clip dicts (path, id, duration, shot, audio, lenses).  Returns a sequence job body (specs, size,
    fps, ...) plus a reviewable plan: cuts (clip-time keys and captions, render_sequence format), coverage by part of
    day, warnings, missing transcripts and skipped clips.

    coverage="even": the clips are laid end to end in capture order and this footage timeline is cut into equal
    buckets, one cut per bucket, so every part of the trip is represented in proportion to how much was filmed.
    coverage="highlights": best-scoring segments anywhere.  progress(done, total, stage) is called while working."""
    import analysis
    import look
    size = SIZES.get(aspect, [1920, 1080])
    skipped, usable = [], []
    for c in clips:
        if c.get("lenses") != 2:
            skipped.append(dict(clip=c["id"], reason="not a dual-lens 360 clip"))
        elif float(c.get("duration") or 0) < min_clip_seconds:
            skipped.append(dict(clip=c["id"], reason=f"too short ({float(c.get('duration') or 0):.1f} s)"))
        else:
            usable.append(c)
    usable.sort(key=lambda c: (shot_time(c), c.get("name", "")))
    offs, T = [], 0.0
    for c in usable:
        offs.append(T)
        T += float(c["duration"])
    cands, missing = [], []
    for ci, c in enumerate(usable):
        dur = float(c["duration"])
        tx = None
        tp = analysis.transcript_path(c["id"])
        if os.path.exists(tp):
            tx = json.load(open(tp, encoding="utf8"))
        elif c.get("audio"):
            missing.append(c["id"])
        try:
            mo = analysis.motion(c)
        except Exception:                                   # noqa: BLE001  (no IMU track)
            mo = None
        for s in suggest(mo, tx, dur, calm_every=CALM_EVERY):
            if s["kind"] not in KIND_SCORE:
                continue
            if s["kind"] == "talk":
                # whole sentences: a little air before the first word and after the last, never cut mid-sentence
                a, b = max(0.0, float(s["start"]) - 0.15), float(s["end"]) + 0.35
                if b - a > TALK_MAX:
                    ends = [g["end"] + 0.35 for g in tx["segments"] if a < g["end"] and g["end"] + 0.35 - a <= TALK_MAX]
                    b = max(ends) if ends else a + TALK_MAX
            else:                                            # varied lengths, so the rhythm is not a metronome
                a = float(s["start"])
                b = min(float(s["end"]), a + max_len, a + max(min_len, SCENERY_LEN[len(cands) % len(SCENERY_LEN)]))
            b = min(b, dur - 0.1)
            if b - a < min_len:
                continue
            score = KIND_SCORE[s["kind"]] + 0.2 * (b - a)
            if s["kind"] == "talk" and _words(s.get("text")) < 4:           # "嗯", "好" ...
                score -= 1.0
            cands.append(dict(ci=ci, clip=c, kind=s["kind"], start=a, end=b, score=score, tx=tx, text=s.get("text", ""),
                              pos=offs[ci] + a))
        if progress:
            progress(ci + 1, len(usable), "analysing")

    def length(x):
        return x["end"] - x["start"]

    n = max(1, round(seconds / ((min_len + max_len) / 2)))
    B = max(T / n, 1e-6)
    cap = {ci: max(1, math.ceil(float(c["duration"]) / B)) for ci, c in enumerate(usable)}   # no clip hogs the film

    def fits(x, chosen):
        same = [y for y in chosen if y["ci"] == x["ci"]]
        return len(same) < cap[x["ci"]] and all(x["end"] <= y["start"] or x["start"] >= y["end"] for y in same)

    chosen = []
    if coverage == "highlights":
        total = 0.0
        for x in sorted(cands, key=lambda x: -x["score"]):
            if total >= seconds:
                break
            if fits(x, chosen):
                chosen.append(x); total += length(x)
    else:
        for k in range(n):                                  # one cut per bucket of the footage timeline
            pool = [x for x in cands if k * B <= x["pos"] < (k + 1) * B and fits(x, chosen)]
            if pool:
                chosen.append(max(pool, key=lambda x: x["score"]))
        day = [shot_time(c).date() if shot_time(c) != datetime.min else None for c in usable]
        for d in dict.fromkeys(day):                       # every day of the trip appears, however little was filmed
            if d not in {day[y["ci"]] for y in chosen}:
                pool = [x for x in cands if day[x["ci"]] == d and fits(x, chosen)]
                if pool:
                    chosen.append(max(pool, key=lambda x: x["score"]))
        total = sum(length(x) for x in chosen)
        lab = [period(shot_time(c)) for c in usable]
        share = {}
        for ci, c in enumerate(usable):
            share[lab[ci]] = share.get(lab[ci], 0.0) + float(c["duration"]) / T
        while len(chosen) > 1:
            # over the target: drop a cut so that the cuts per part of day stay closest to the footage shares (never a
            # day's last cut while others can go), most crowded first
            cnt, dcnt = {}, {}
            for y in chosen:
                cnt[lab[y["ci"]]] = cnt.get(lab[y["ci"]], 0) + 1
                dcnt[day[y["ci"]]] = dcnt.get(day[y["ci"]], 0) + 1

            def err_after(p):
                return sum(abs(share[q] - (cnt.get(q, 0) - (q == p)) / (len(chosen) - 1)) for q in share)
            gap = {id(x): min(abs(x["pos"] - y["pos"]) for y in chosen if y is not x) for x in chosen}
            x = min((x for x in chosen if total - length(x) >= seconds),
                    key=lambda x: (dcnt[day[x["ci"]]] == 1, err_after(lab[x["ci"]]), gap[id(x)], x["score"]), default=None)
            if x is None:
                break
            chosen.remove(x); total -= length(x)
        while total < seconds:                              # empty buckets: fill far from what is already in
            ids = {id(y) for y in chosen}
            pool = [x for x in cands if id(x) not in ids and fits(x, chosen)]
            if not pool:
                break
            x = max(pool, key=lambda x: min((abs(x["pos"] - y["pos"]) for y in chosen), default=T) / B + 0.1 * x["score"])
            chosen.append(x); total += length(x)
    chosen.sort(key=lambda x: (x["ci"], x["start"]))       # capture order
    auto = transition == "auto"
    if auto and len(chosen) > 1:                            # room for the little-planet opener and ending
        for x in (chosen[0], chosen[-1]):
            if x["kind"] != "talk":
                x["end"] = x["start"] + min(float(x["clip"]["duration"]) - 0.1 - x["start"], max(5.0, x["end"] - x["start"]))
    # beat sync: with background music, every part lasts a whole number of bars (2 beats) plus the transition,
    # so each transition starts on a beat; the music is then started at its first beat (sequence.py)
    grid = None
    if music and music.get("path"):
        try:
            from beats import beat_grid
            g = beat_grid(music["path"])
            if g and g["confidence"] >= 0.2:
                grid = g
        except Exception:                                  # noqa: BLE001
            grid = None
    if grid:
        d = 0.5 if transition and transition not in ("cut", "auto") else 0.0
        bar = 2 * grid["period"]
        for x in chosen:
            room = x["clip"].get("duration", x["end"]) - 0.1 - x["start"]
            m = max(1, int(round((x["end"] - x["start"] - d) / bar)))
            while m > 1 and m * bar + d > room:
                m -= 1
            x["end"] = x["start"] + min(m * bar + d, room)
    import director
    specs, cuts = [], []
    chap = [period(shot_time(x["clip"])) for x in chosen]
    prev_move = None
    for i, x in enumerate(chosen):
        c, a, b = x["clip"], x["start"], x["end"]
        d = round(b - a, 3)
        if x["kind"] == "talk":
            keys = director.talk_keys([dict(k, t=round(k["t"] - a, 3)) for k in look.holder_track(c, a, b, lift=TALK_LIFT, cap=80.0)], size)
            move = "holder"
            fk = director.face_track(c, keys, a, size)      # YuNet: follow the speaker's face, IMU as the fallback
            if fk:
                keys, move = fk, "face"
        else:                                               # aim at the most interesting direction, one eased move
            try:
                I, _ = director.interest_map(c, a + d / 2)
                am = director.aim(I)
            except Exception:                               # noqa: BLE001  (no frame / no IMU: look ahead)
                am = dict(yaw=0.0, pitch=0.0, side=1, strength=0.0)
            move, keys = director.scenery_keys(am, size, d, prev_move)
        prev_move = move
        trans_in = trans_out = "cut"
        if auto:
            if i == 0 and d >= 3.0:
                keys, trans_in = director.planet_open(keys, unfold=min(2.6, d * 0.5), settle=min(3.4, d * 0.68)), "planet"
            elif i > 0 and chap[i] != chap[i - 1]:
                keys, trans_in = director.whip_in(keys), "whip"
            if i == len(chosen) - 1 and d >= 3.0 and len(chosen) > 1:
                keys, trans_out = director.planet_close(keys, d, dur=min(2.4, d * 0.5)), "planet"
            elif i + 1 < len(chosen) and chap[i + 1] != chap[i]:
                keys, trans_out = director.whip_out(keys, d), "whip"
        x["move"], x["trans"] = move, (trans_in, trans_out)
        keys_abs = [dict(k, t=round(k["t"] + a, 3)) for k in keys]
        sp = dict(src=c["path"], **{"in": round(a, 2), "out": round(b, 2)}, proj="flat", size=size, fps=fps, seam="flow",
                  seam_opts={"conf": "photo"}, cq=20, keys=keys)
        caps = []
        if captions in ("burn", "soft") and x["kind"] == "talk" and x["tx"]:
            caps = [dict(start=s["start"], end=s["end"], text=s["text"]) for s in x["tx"].get("segments", [])
                    if s["end"] > a and s["start"] < b]
            if caps:
                sp.update(captions=caps, captions_mode=captions)
        specs.append(sp)
        st = shot_time(c)
        cuts.append(dict(clip_id=c["id"], start=round(a, 2), end=round(b, 2), kind=x["kind"],
                         shot=st.isoformat(sep=" ") if st != datetime.min else None, text=x["text"][:120], keys=keys_abs,
                         move=x["move"], transition_in=x["trans"][0], transition_out=x["trans"][1],
                         **({"captions": caps} if caps else {})))
        if progress:
            progress(i + 1, len(chosen), "framing")
    # coverage report by day and part of day (capture time of the clip)
    groups = {}
    for c in usable:
        g = groups.setdefault(period(shot_time(c)), dict(period=period(shot_time(c)), cuts=0, footage_s=0.0))
        g["footage_s"] += float(c["duration"])
    for x in chosen:
        groups[period(shot_time(x["clip"]))]["cuts"] += 1
    cov = [dict(g, footage_s=round(g["footage_s"], 1), footage_share=round(g["footage_s"] / T, 3) if T else 0,
                cut_share=round(g["cuts"] / len(chosen), 3) if chosen else 0) for g in groups.values()]
    total = sum(length(x) for x in chosen)
    warnings = []
    if missing:
        warnings.append(f"{len(missing)} of {len(usable)} clips with audio have no speech transcript, so their talk segments"
                        + (" and captions" if captions in ("burn", "soft") else "") + " are not available. "
                        "Transcribe them first, then plan again.")
    if skipped:
        warnings.append(f"{len(skipped)} clips skipped: " + ", ".join(f"{s['clip']} ({s['reason']})" for s in skipped[:12])
                        + (" ..." if len(skipped) > 12 else ""))
    for g in cov:
        if g["cuts"] == 0 and g["footage_share"] >= 0.1:
            warnings.append(f"{g['period']} ({g['footage_share'] * 100:.0f}% of the footage) has no usable talk or steady segment.")
    if captions in ("burn", "soft") and chosen and not any("captions" in cu for cu in cuts):
        warnings.append("captions were requested but none of the picked cuts has transcribed speech: the film will have no subtitles.")
    if usable and total < 0.8 * seconds:
        warnings.append(f"only {total:.0f} s of usable segments found (target {seconds:.0f} s).")
    if not usable:
        warnings.append("no usable clips.")
    times = [shot_time(c) for c in usable if shot_time(c) != datetime.min]
    job = dict(specs=specs, size=size, fps=fps, seconds=round(total, 1),
               picked=[dict(clip=x["clip"]["id"], kind=x["kind"], start=round(x["start"], 1), end=round(x["end"], 1),
                            shot=cu["shot"], text=cu["text"], **({"captions": True} if "captions" in cu else {}))
                       for x, cu in zip(chosen, cuts)],
               cuts=cuts, coverage=cov, warnings=warnings, missing_transcripts=missing, skipped=skipped,
               n_clips=len(usable), footage_s=round(T, 1), coverage_mode="highlights" if coverage == "highlights" else "even",
               span=[min(times).isoformat(sep=" "), max(times).isoformat(sep=" ")] if times else None)
    if auto:
        job["fade_out"] = 1.0                               # hard cuts / whips are in the keys; the film fades to black
    elif transition and transition != "cut":
        job["transition"] = dict(type=transition, duration=0.5)
    if grid:
        job["beat"] = grid
    return job
