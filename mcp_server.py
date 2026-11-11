"""MCP server: lets an AI assistant browse, look around in, and cut 360 OSV footage.

Runs over stdio (JSON-RPC 2.0, one message per line) and talks to the osvcut studio (`python server.py`),
default http://127.0.0.1:8360 (override with OSVCUT_URL).  If the studio is not running, the first call starts it
(library folders from OSVCUT_ROOTS, ';'-separated, default E:\\Videos; Python from OSVCUT_PYTHON, default this one;
OSVCUT_AUTOSTART=0 turns this off).  The studio keeps running after the MCP server exits.

    claude mcp add osvcut -- python /path/to/osvcut/mcp_server.py

Tools: list_clips, search_speech, transcribe, clip_overview, look, holder_keys, track_subject, plan_edit, auto_edit,
render_cut, render_sequence, job_status.
Angles: yaw 0 = camera forward, +90 = right, ±180 = behind; pitch +90 = up. Times are seconds in the clip.
"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("OSVCUT_URL", "http://127.0.0.1:8360").rstrip("/")
HERE = os.path.dirname(os.path.abspath(__file__))
AUTOSTART = os.environ.get("OSVCUT_AUTOSTART", "1") != "0"
ROOTS = [r for r in os.environ.get("OSVCUT_ROOTS", r"E:\Videos").split(";") if r.strip()]
PYTHON = os.environ.get("OSVCUT_PYTHON") or sys.executable


# ---------------------------------------------------------------- http helpers
def _refused(e):
    r = getattr(e, "reason", e)
    return isinstance(r, ConnectionRefusedError) or getattr(r, "winerror", None) == 10061 or getattr(r, "errno", None) in (61, 111, 10061)


def _start_command(port):
    roots = " ".join(f'"{r}"' for r in ROOTS)
    return f'cd "{HERE}" && "{PYTHON}" server.py {port} {roots}'


def _ensure_studio():
    """Start the studio when nothing listens at BASE (local URLs only); wait up to 30 s for it to answer."""
    u = urllib.parse.urlparse(BASE)
    port = str(u.port or 80)
    if not AUTOSTART or u.hostname not in ("127.0.0.1", "localhost"):
        why = "autostart is off (OSVCUT_AUTOSTART=0)" if not AUTOSTART else "it is not a local address"
        raise RuntimeError(f"osvcut studio is not running at {BASE} and was not started ({why}). "
                           f"Start it with: {_start_command(port)}")
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    log = open(os.path.join(HERE, "cache", "studio.log"), "ab")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    try:
        p = subprocess.Popen([PYTHON, "server.py", port, *ROOTS], cwd=HERE, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, **kw)
    except OSError as e:
        raise RuntimeError(f"osvcut studio is not running at {BASE} and could not be started: {e}. "
                           f"Start it with: {_start_command(port)}")
    t_end = time.time() + 30
    while time.time() < t_end:
        if p.poll() is not None:
            break
        try:
            with urllib.request.urlopen(BASE + "/api/jobs", timeout=3):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    tail = ""
    try:
        tail = open(os.path.join(HERE, "cache", "studio.log"), encoding="utf8", errors="replace").read()[-400:].strip()
    except OSError:
        pass
    raise RuntimeError(f"osvcut studio is not running at {BASE} and could not be started "
                       f"({'it exited' if p.poll() is not None else 'no answer within 30 s'}"
                       + (f"; log: {tail}" if tail else "") + f"). Start it with: {_start_command(port)}")


def _call(fn):
    try:
        return fn()
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as e:
        if not _refused(e):
            raise
    _ensure_studio()                       # nothing listening: start the studio once, then retry
    return fn()


def _get(path, raw=False, timeout=300):
    def go():
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.read()
    data = _call(go)
    return data if raw else json.loads(data.decode("utf8"))


def _post(path, body, timeout=60):
    def go():
        req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:            # the studio answers errors as JSON {"error": ...}
            return e.read()
    return json.loads(_call(go).decode("utf8"))


def _clip(cid):
    c = _get(f"/api/clip/{cid}")
    if "error" in c:
        raise ValueError(f"unknown clip id {cid}")
    return c


def _fmt(s):
    s = int(round(s or 0))
    return f"{s // 60}:{s % 60:02d}"


def _image(jpeg):
    return {"type": "image", "data": base64.b64encode(jpeg).decode(), "mimeType": "image/jpeg"}


def _text(t):
    return {"type": "text", "text": t}


# ---------------------------------------------------------------- tools
def _match(clips, query="", category=""):
    q = (query or "").lower()
    return [c for c in clips if (not category or category in (c.get("category") or ""))
            and (not q or q in f"{c.get('event', '')} {c['name']} {c.get('shot', '')}".lower())]


def _hms(s):
    s = int(round(s or 0))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else _fmt(s)


def _resolve(clip_ids=None, query=""):
    """Clip ids from an explicit list or a list_clips-style query."""
    if clip_ids:
        return list(clip_ids)
    if not query:
        raise ValueError("give clip_ids or query")
    ids = [c["id"] for c in _match(_get("/api/library")["clips"], query)]
    if not ids:
        raise ValueError(f"no clips match '{query}'")
    return ids


def _untranscribed(clips):
    return [c["id"] for c in clips if c.get("audio") and c.get("duration", 0) > 1 and "speech" not in c]


def list_clips(query="", category="", limit=30):
    lib = _get("/api/library")
    out = _match(lib["clips"], query, category)
    lines = [f"{len(out)} clips match (showing {min(limit, len(out))}); library has {len(lib['clips'])}."]
    if lib.get("scanning"):
        lines.append(f"Library scan still running ({lib.get('scanned', 0)}/{lib.get('total', 0)} files): results may be incomplete.")
    if out:
        days = {}
        for c in sorted(out, key=lambda c: c.get("shot") or ""):
            d = (c.get("shot") or "unknown")[:10]
            n, s = days.get(d, (0, 0.0))
            days[d] = (n + 1, s + (c.get("duration") or 0))
        tx = _untranscribed(out)
        lines.append(f"Total footage {_hms(sum(c.get('duration') or 0 for c in out))}; by day: "
                     + " · ".join(f"{d} {n} clips {_hms(s)}" for d, (n, s) in days.items()))
        lines.append(f"No audio: {sum(not c.get('audio') for c in out)} · not transcribed: {len(tx)}"
                     + (" (transcribe them for talk segments, captions and search_speech)" if tx else "")
                     + f" · shorter than 2 s: {sum((c.get('duration') or 0) < 2 for c in out)}")
    for c in out[:limit]:
        flags = []
        if c.get("previews"):
            flags.append("preview")
        if c.get("proxy"):
            flags.append("live")
        if not c.get("audio"):
            flags.append("no-audio")
        lines.append(f"{c['id']} | {c.get('event', '')} | {c['name']} | {_fmt(c.get('duration'))} | "
                     f"{(c.get('shot') or '').replace('T', ' ')} | {'8K' if c.get('lens_px', 0) >= 3840 else '4K'}"
                     + (f" | {','.join(flags)}" if flags else "")
                     + (f" | says: {c['speech']['text'][:80]}" if c.get("speech") and c["speech"].get("n") else ""))
    return [_text("\n".join(lines))]


def search_speech(query, limit=30):
    r = _get("/api/search?" + urllib.parse.urlencode({"q": query}))
    lines = [f"{len(r['hits'])} matches for '{query}' (transcripts available for {r['transcribed']} of {r['with_audio']} clips with audio)."]
    for h in r["hits"][:limit]:
        lines.append(f"{h['clip']} | {h.get('event', '')} | {h['name']} | {h['start']:.1f}-{h['end']:.1f}s | {h['text']}")
    return [_text(chr(10).join(lines))]


from autoedit import suggest as _suggest   # shared with the studio's one-click edit


def clip_overview(clip_id, frames=6, wait_transcript=90):
    c = _clip(clip_id)
    sheet = _get(f"/api/clip/{clip_id}/sheet?n={int(frames)}", raw=True)
    motion = _get(f"/api/clip/{clip_id}/motion")
    tx, t_end = None, time.time() + wait_transcript
    while time.time() < t_end:
        tx = _get(f"/api/clip/{clip_id}/transcript")
        if tx.get("status") == "done":
            break
        time.sleep(2)
    lines = [f"Clip {c['id']}: {c.get('event', '')} / {c['name']}, {_fmt(c.get('duration'))}, "
             f"{'8K' if c.get('lens_px', 0) >= 3840 else '4K'} mode, {c.get('fps')} fps, shot {c.get('shot')}.",
             f"Contact sheet: {int(frames)} horizon-levelled 360 panoramas; x axis = yaw (-180..180, 0 = camera forward), "
             "labelled at the top; time in red."]
    if motion and motion.get("speed"):
        sp = motion["speed"]
        lines.append(f"Camera rotation speed: median {sorted(sp)[len(sp) // 2]:.0f} deg/s, max {max(sp):.0f} deg/s.")
    if tx and tx.get("status") == "done":
        segs = tx.get("segments", [])
        lines.append(f"Transcript ({tx.get('language') or 'n/a'}, {len(segs)} segments):" if segs else
                     f"Transcript: {tx.get('note') or 'no speech detected'}.")
        lines += [f"  [{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segs[:200]]
    else:
        lines.append("Transcript: still running; call clip_overview again later.")
    sug = _suggest(motion, tx if tx and tx.get("status") == "done" else None, c.get("duration", 0))
    if sug:
        lines.append("Suggested segments:")
        lines += [f"  {s['kind']:5s} {s['start']:.1f}-{s['end']:.1f}  {s['text'][:80]}" for s in sug[:40]]
    return [_image(sheet), _text("\n".join(lines))]


def look(clip_id, t, yaw=0.0, pitch=0.0, fov=90.0, little_planet=False, width=960, height=540, at=""):
    if at == "holder":                 # aim at the selfie-stick holder (from the IMU), then render
        h = _get(f"/api/clip/{clip_id}/holder?t={t}")
        yaw, pitch, fov = h["yaw"], h["pitch"], h["fov"]
    if little_planet:
        pitch, fov, dist = -90.0, 300.0, 1.0
        width = height = max(width, height)
    else:
        dist = 0.0
    q = urllib.parse.urlencode(dict(t=t, yaw=yaw, pitch=pitch, fov=fov, dist=dist, w=int(width), h=int(height)))
    img = _get(f"/api/clip/{clip_id}/view?{q}", raw=True)
    return [_image(img), _text(f"view at t={t:.2f}s yaw={yaw:.0f} pitch={pitch:.0f} fov={fov:.0f}"
                               + (" (little planet)" if little_planet else ""))]


def _wait_job(jid, timeout):
    t_end = time.time() + timeout
    while time.time() < t_end:
        j = next((j for j in _get("/api/jobs") if j["id"] == jid), None)
        if j and j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(2)
    return next((j for j in _get("/api/jobs") if j["id"] == jid), None)


def track_subject(clip_id, t0, t1, yaw, pitch, box_w=12.0, box_h=30.0, fov=90.0):
    c = _clip(clip_id)
    j = _post("/api/track", dict(src=c["src"], t0=t0, t1=t1, yaw=yaw, pitch=pitch, box_w=box_w, box_h=box_h, fov=fov))
    if "id" not in j:
        raise ValueError(j.get("error") or "request failed")
    j = _wait_job(j["id"], 600)
    if not j or j["status"] != "done":
        return [_text(f"tracking {j['status'] if j else 'timed out'}: {(j or {}).get('error') or ''}")]
    res = _get(j["output"])
    keys = res["keys"]
    msg = (f"Tracked {len(res['track'])} frames; {len(keys)} keyframes "
           + (f"(subject lost at {res['lost_at']:.1f}s). " if res.get("lost_at") else "(followed to the end). ")
           + "Use them as `keys` in render_cut (times are clip seconds; render_cut makes them relative). keys = "
           + json.dumps([{k: v for k, v in kk.items()} for kk in keys]))
    return [_text(msg)]


def holder_keys(clip_id, start, end):
    keys = _get(f"/api/clip/{clip_id}/holder?" + urllib.parse.urlencode({"t0": start, "t1": end}))
    return [_text(f"{len(keys)} keyframes that keep the selfie-stick holder in view ({start:.1f}-{end:.1f}s), from the IMU; "
                  "pass them as keys to render_cut. keys = " + json.dumps(keys))]


def _spec(clip_id, start, end, keys=None, size=(1920, 1080), fps=30, speed=1.0, seam="flow"):
    c = _clip(clip_id)
    keys = keys or [{"t": start, "yaw": 0, "pitch": 0, "fov": 90}]
    ks = []
    for k in sorted(keys, key=lambda k: k["t"]):
        kk = dict(k)
        kk["t"] = round(max(0.0, float(k["t"]) - start), 3)        # spec keys are relative to `in`
        ks.append(kk)
    s = dict(src=c["src"], **{"in": float(start), "out": float(end)}, proj="flat", size=list(size), fps=fps, seam=seam, keys=ks)
    if abs(speed - 1) > 1e-6:
        s["speed"] = speed
    return s


def render_cut(clip_id, start, end, keys=None, aspect="16:9", fps=30, speed=1.0, title="", projection="flat", nadir_patch=0, grade=None, captions="off", freeze_at=None, freeze_seconds=2.0, freeze_turn=360.0,
               slow_from=None, slow_to=None, slow_speed=0.25, title_card=""):
    size = {"16:9": [1920, 1080], "9:16": [1080, 1920], "1:1": [1080, 1080], "4:5": [1080, 1350], "2.39:1": [1920, 816]}.get(aspect, [1920, 1080])
    spec = _spec(clip_id, start, end, keys, size, fps, speed)
    if nadir_patch:
        spec["nadir_patch"] = float(nadir_patch)
    if grade:
        spec["grade"] = {k: float(v) for k, v in grade.items() if k in ("exposure", "contrast", "saturation", "temperature")}
    if title_card:                     # centred title over the first 3 s
        spec["title"] = {"text": str(title_card), "seconds": 3.0}
    if slow_from is not None and slow_to is not None:   # local slow motion inside the cut
        spec["slow"] = {"from": float(slow_from), "to": float(slow_to), "speed": float(slow_speed)}
    if freeze_at is not None:          # freeze orbit: picture stops at freeze_at while the camera turns
        spec["freeze"] = dict(at=float(freeze_at), dur=float(freeze_seconds), turn=float(freeze_turn))
    if captions in ("burn", "soft"):   # subtitles from the clip's speech transcript (clip time = source time)
        tx = _get(f"/api/clip/{clip_id}/transcript")
        segs = [dict(start=x["start"], end=x["end"], text=x["text"]) for x in (tx.get("segments") or [])
                if x["end"] > start and x["start"] < end]
        if segs:
            spec.update(captions=segs, captions_mode=captions)
    if projection == "360":            # full levelled 360 video with spherical metadata (YouTube / VR)
        spec.update(proj="equirect", size=[3840, 1920], spherical=True)
        spec.pop("keys", None)
    j = _post("/api/jobs", dict(kind="export", spec=spec, title=title or f"AI cut {clip_id} {start:.1f}-{end:.1f}"))
    if "id" not in j:
        raise ValueError(j.get("error") or "request failed")
    return [_text(f"export job {j['id']} queued ({spec['out'] - spec['in']:.1f}s source"
                  + (f" at {speed}x" if speed != 1 else "") + f", {size[0]}x{size[1]}). Poll job_status.")]


def render_sequence(cuts, title="", aspect="16:9", fps=30, projection="flat", transition="cut", transition_seconds=0.5,
                    captions="burn"):
    size = {"16:9": [1920, 1080], "9:16": [1080, 1920], "1:1": [1080, 1080], "4:5": [1080, 1350], "2.39:1": [1920, 816]}.get(aspect, [1920, 1080])
    specs = []
    for cu in cuts:
        sp = _spec(cu["clip_id"], cu["start"], cu["end"], cu.get("keys"), size, fps, cu.get("speed", 1.0))
        if captions in ("burn", "soft") and cu.get("captions"):       # e.g. the captions of a plan_edit cut
            sp.update(captions=[dict(start=float(x["start"]), end=float(x["end"]), text=str(x["text"])) for x in cu["captions"]],
                      captions_mode=captions)
        specs.append(sp)
    if projection == "360":
        size = [3840, 1920]
        for sp in specs:
            sp.update(proj="equirect", size=size, spherical=True); sp.pop("keys", None)
    j = _post("/api/jobs", dict(kind="sequence", specs=specs, title=title or f"AI sequence ({len(specs)} cuts)", size=size, fps=fps,
                                         **({"transition": {"type": transition, "duration": float(transition_seconds)}} if transition != "cut" else {})))
    if "id" not in j:
        raise ValueError(j.get("error") or "request failed")
    return [_text(f"sequence job {j['id']} queued with {len(specs)} cuts. Poll job_status.")]


def transcribe(clip_ids=None, query="", wait_seconds=0):
    ids = _resolve(clip_ids, query)
    r = _post("/api/transcribe_all", dict(clip_ids=ids))
    if r.get("error"):
        raise ValueError(r["error"])
    msg = (f"queued {r.get('queued', 0)} of {len(ids)} clips for transcription (skipped: {r.get('skipped_done', 0)} already done, "
           f"{r.get('skipped_no_audio', 0)} without audio or < 1 s, {r.get('skipped_busy', 0)} already being transcribed).")
    if not r.get("job"):
        return [_text(msg + " Nothing to wait for.")]
    if wait_seconds:
        j = _wait_job(r["job"], wait_seconds)
        msg += f"\njob {r['job']}: {j['status'] if j else 'unknown'} {(j or {}).get('progress', 0) * 100:.0f}%"
        if j and j["status"] == "done":
            msg += ". Transcripts are ready: plan_edit / auto_edit can now use talk segments and captions."
        elif j and j.get("error"):
            msg += f"\nerror: {j['error']}"
        else:
            msg += f". Still running: job_status('{r['job']}', wait_seconds=...)."
    else:
        msg += f" job {r['job']}; about 5-20 s per minute of audio. Wait with job_status('{r['job']}', wait_seconds=...)."
    return [_text(msg)]


def _fmt_plan(p, full):
    """Readable plan: header, coverage, cut list, warnings; with full, the cuts JSON for render_sequence."""
    rows = p.get("cuts") if full else p.get("picked")
    rows = rows or []
    span = " → ".join(s[:16] for s in p["span"]) + ", " if p.get("span") else ""
    lines = [f"Plan for {p.get('n_clips', 0)} clips ({span}{_hms(p.get('footage_s'))} footage) → {p.get('seconds', 0)} s, "
             f"{len(rows)} cuts, coverage {p.get('coverage_mode', 'even')}"]
    if p.get("coverage"):
        lines.append("Coverage: " + " · ".join(f"{g['period']} {g['cuts']} cuts ({g['cut_share'] * 100:.0f}% of cuts / "
                                               f"{g['footage_share'] * 100:.0f}% of footage)" for g in p["coverage"]))
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    if kinds:
        lines.append("Kinds: " + " · ".join(f"{k} {n}" for k, n in kinds.items()))
    if rows:
        lines.append("Cuts:")
        for i, r in enumerate(rows[:40]):
            cid = r.get("clip_id") or r.get("clip")
            lines.append(f"  {i + 1:2d}. {(r.get('shot') or '')[:16]} {cid} {r['start']:.1f}-{r['end']:.1f}s {r['kind']}"
                         + (" [captions]" if r.get("captions") else "")
                         + (f"  {r['text'][:60]}" if r.get("text") and r["kind"] == "talk" else ""))
        if len(rows) > 40:
            lines.append(f"  ... {len(rows) - 40} more")
    if p.get("warnings"):
        lines.append("Warnings:")
        lines += [f"  - {w}" for w in p["warnings"]]
    if p.get("missing_transcripts"):
        ids = p["missing_transcripts"]
        lines.append(f"  → transcribe(clip_ids={json.dumps(ids[:80])}{' + more' if len(ids) > 80 else ''}) then plan again, "
                     "or auto_edit(..., transcribe_missing=true).")
    if full and rows:
        lines.append("cuts (pass to render_sequence as is, or edit first; keys and captions are in clip seconds) = "
                     + json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def plan_edit(clip_ids=None, query="", seconds=0, aspect="16:9", captions="burn", coverage="even", min_clip_seconds=2.0,
              wait_seconds=120):
    ids = _resolve(clip_ids, query)
    j = _post("/api/jobs", dict(kind="plan", clip_ids=ids, seconds=float(seconds), aspect=aspect, captions=captions,
                                coverage=coverage, min_clip_seconds=float(min_clip_seconds)))
    if "id" not in j:
        raise ValueError(j.get("error") or "request failed")
    if wait_seconds:
        j = _wait_job(j["id"], wait_seconds) or j
    if j["status"] != "done":
        return [_text(f"plan job {j['id']} is {j['status']} ({j.get('stage') or ''})"
                      + (f": {j['error']}" if j.get("error") else f"; wait with job_status('{j['id']}', wait_seconds=...)."))]
    return [_text(_fmt_plan(_get(j["output"]), True))]


def auto_edit(clip_ids=None, query="", seconds=0, aspect="16:9", transition="auto", captions="burn", coverage="even",
              title="", music_path="", transcribe_missing=False, min_clip_seconds=2.0):
    ids = _resolve(clip_ids, query)
    j = _post("/api/jobs", dict(kind="autoedit", clip_ids=ids, seconds=float(seconds), aspect=aspect, transition=transition,
                                captions=captions, coverage=coverage, title=title or None, music_path=music_path or None,
                                transcribe_missing=bool(transcribe_missing), min_clip_seconds=float(min_clip_seconds)))
    if "id" not in j:
        raise ValueError(j.get("error") or "request failed")
    msg = (f"auto edit job {j['id']} started ({j['status']}); it plans, then renders. "
           f"Use job_status('{j['id']}', wait_seconds=...) - planning takes ~1-3 s per clip, rendering a few minutes.")
    if not transcribe_missing:
        lib = {c["id"]: c for c in _get("/api/library")["clips"]}
        tx = _untranscribed([lib[i] for i in ids if i in lib])
        if tx:
            msg += (f"\nNote: {len(tx)} of {len(ids)} clips are not transcribed, so they give no talk segments"
                    + (" or captions" if captions != "off" else "") + ". Pass transcribe_missing=true (or run transcribe first) to include them.")
    return [_text(msg)]


def directed_edit(shots, aspect="16:9", transition="auto", captions="burn", title=""):
    ids = list(dict.fromkeys(str(s.get("clip_id")) for s in shots))
    j = _post("/api/jobs", dict(kind="autoedit", clip_ids=ids, shots=shots, aspect=aspect, transition=transition,
                                captions=captions, title=title or None))
    if "id" not in j:
        raise ValueError(j.get("error") or "request failed")
    return [_text(f"directed edit job {j['id']} started ({j['status']}): {len(shots)} shots. "
                  f"Use job_status('{j['id']}', wait_seconds=...); framing takes ~5-20 s per talk shot, then it renders.")]


def job_status(job_id, wait_seconds=0):
    j = _wait_job(job_id, wait_seconds) if wait_seconds else next((j for j in _get("/api/jobs") if j["id"] == job_id), None)
    if not j:
        return [_text(f"no job {job_id}")]
    path = os.path.join(HERE, j["output"].lstrip("/").replace("/", os.sep)) if j.get("output") else ""
    msg = (f"{j['kind']} '{j['title']}': {j['status']} {j['progress'] * 100:.0f}%"
           + (f" ({j['stage']})" if j.get("stage") and j["status"] in ("planning", "transcribing") else "")
           + (f", eta {j['eta']:.0f}s" if j.get("eta") and j["status"] == "running" else "")
           + (f"\noutput: {path}" if j["status"] == "done" and j["kind"] != "plan" else "")
           + (f"\nerror: {j['error']}" if j.get("error") else ""))
    if j["kind"] == "plan" and j["status"] == "done":
        msg += "\n" + _fmt_plan(_get(j["output"]), True)
    elif j.get("plan"):
        msg += "\n" + _fmt_plan(j["plan"], False)
    return [_text(msg)]


TOOLS = {
    "list_clips": (list_clips, "Search the 360 footage library (DJI Osmo 360 .OSV). Returns id | event | file | duration | shot time | mode | flags.",
                   {"query": {"type": "string", "description": "substring of event folder, file name or date"},
                    "category": {"type": "string"}, "limit": {"type": "integer", "default": 30}}, []),
    "search_speech": (search_speech, "Find clips by what is said in them (speech transcripts). Returns clip id, time range and the sentence.",
                      {"query": {"type": "string"}, "limit": {"type": "integer", "default": 30}}, ["query"]),
    "clip_overview": (clip_overview, "Understand a clip: levelled 360 contact sheet (image; yaw ticks on top, the thin red line is the lens seam - keep subjects off it), speech transcript, camera motion and suggested segments.",
                      {"clip_id": {"type": "string"}, "frames": {"type": "integer", "default": 6}}, ["clip_id"]),
    "look": (look, "Render what a virtual camera sees at time t looking at (yaw, pitch) with field of view fov; little_planet=true for a stereographic overview; at='holder' aims roughly at the person holding the selfie stick (computed from the IMU, refine with yaw/pitch afterwards).",
             {"clip_id": {"type": "string"}, "t": {"type": "number"}, "yaw": {"type": "number"}, "pitch": {"type": "number", "default": 0},
              "fov": {"type": "number", "default": 90}, "little_planet": {"type": "boolean", "default": False},
              "width": {"type": "integer", "default": 960}, "height": {"type": "integer", "default": 540},
              "at": {"type": "string", "enum": ["", "holder"], "default": ""}}, ["clip_id", "t"]),
    "holder_keys": (holder_keys, "Selfie mode: camera keyframes that keep the person holding the selfie stick in view between start and end (no detection; from the IMU).",
                    {"clip_id": {"type": "string"}, "start": {"type": "number"}, "end": {"type": "number"}}, ["clip_id", "start", "end"]),
    "track_subject": (track_subject, "Follow a subject seen at (yaw, pitch) at time t0 until t1 and return camera keyframes that keep it centred. box_w/box_h = subject size in degrees.",
                      {"clip_id": {"type": "string"}, "t0": {"type": "number"}, "t1": {"type": "number"}, "yaw": {"type": "number"},
                       "pitch": {"type": "number"}, "box_w": {"type": "number", "default": 12}, "box_h": {"type": "number", "default": 30},
                       "fov": {"type": "number", "default": 90}}, ["clip_id", "t0", "t1", "yaw", "pitch"]),
    "render_cut": (render_cut, "Export a flat video from one clip between start and end (clip seconds) following camera keyframes [{t, yaw, pitch, fov, dist?, ease?}] (t in clip seconds). dist 0 = normal lens, 1 = little planet.",
                   {"clip_id": {"type": "string"}, "start": {"type": "number"}, "end": {"type": "number"},
                    "keys": {"type": "array", "items": {"type": "object"}}, "aspect": {"type": "string", "enum": ["16:9", "9:16", "1:1", "4:5", "2.39:1"], "default": "16:9"},
                    "fps": {"type": "integer", "default": 30}, "speed": {"type": "number", "default": 1.0}, "title": {"type": "string"},
                    "grade": {"type": "object", "description": "colour grade: exposure (EV, -2..2), contrast (-0.5..0.5), saturation (-1..1), temperature (-1..1, warm > 0)",
                              "properties": {"exposure": {"type": "number"}, "contrast": {"type": "number"}, "saturation": {"type": "number"}, "temperature": {"type": "number"}}},
                    "freeze_at": {"type": "number", "description": "freeze orbit ('bullet time'): clip time where the picture stops while the camera turns (not with speed != 1)"},
                    "slow_from": {"type": "number", "description": "local slow motion: clip time where it starts (normal speed elsewhere; not with speed or freeze)"},
                    "slow_to": {"type": "number"}, "slow_speed": {"type": "number", "default": 0.25},
                    "title_card": {"type": "string", "description": "text shown as a centred title over the first 3 s"},
                    "freeze_seconds": {"type": "number", "default": 2.0}, "freeze_turn": {"type": "number", "default": 360.0, "description": "degrees of yaw during the freeze"},
                    "captions": {"type": "string", "enum": ["off", "burn", "soft"], "default": "off",
                                 "description": "subtitles from the speech transcript: burned into the picture, or a soft track + .srt"},
                    "nadir_patch": {"type": "number", "default": 0, "description": "cover the area below the camera (hand / stick) with a smooth patch of this radius in degrees, e.g. 12"},
                    "projection": {"type": "string", "enum": ["flat", "360"], "default": "flat",
                                   "description": "flat = reframed video following keys; 360 = full levelled 360 video (YouTube/VR), keys ignored"}},
                   ["clip_id", "start", "end"]),
    "render_sequence": (render_sequence, "Export several cuts (possibly from different clips) joined into one film. cuts = [{clip_id, start, end, keys?, speed?, captions?}] (the `cuts` of plan_edit fit as is).",
                        {"cuts": {"type": "array", "items": {"type": "object"}}, "title": {"type": "string"},
                         "captions": {"type": "string", "enum": ["off", "burn", "soft"], "default": "burn", "description": "how to show the captions a cut carries"},
                         "projection": {"type": "string", "enum": ["flat", "360"], "default": "flat"},
                         "transition": {"type": "string", "enum": ["cut", "fade", "fadeblack", "wipeleft", "circleopen"], "default": "cut"},
                         "transition_seconds": {"type": "number", "default": 0.5},
                         "aspect": {"type": "string", "enum": ["16:9", "9:16", "1:1", "4:5", "2.39:1"], "default": "16:9"}, "fps": {"type": "integer", "default": 30}},
                        ["cuts"]),
    "transcribe": (transcribe, "Speech-to-text for clips that have no transcript yet (needed for talk segments, captions and search_speech). Give clip_ids or a list_clips-style query. Returns a job id; wait_seconds > 0 waits for it.",
                   {"clip_ids": {"type": "array", "items": {"type": "string"}}, "query": {"type": "string"},
                    "wait_seconds": {"type": "integer", "default": 0}}, []),
    "plan_edit": (plan_edit, "Plan a film from many clips WITHOUT rendering: returns the picked cuts in capture order, how they cover each day / part of day, warnings (missing transcripts, skipped clips) and a `cuts` JSON you can edit and pass to render_sequence. coverage='even' spreads cuts over the whole footage timeline in proportion to how much was filmed; 'highlights' takes the best-scoring segments anywhere.",
                  {"clip_ids": {"type": "array", "items": {"type": "string"}}, "query": {"type": "string", "description": "instead of clip_ids: substring of event folder, file name or date"},
                   "seconds": {"type": "number", "default": 0, "description": "target length; 0 = no target: every worthwhile moment, each conversation kept whole (length never costs a shot its completeness)"}, "aspect": {"type": "string", "enum": ["16:9", "9:16", "1:1", "4:5"], "default": "16:9"},
                   "captions": {"type": "string", "enum": ["off", "burn", "soft"], "default": "burn"},
                   "coverage": {"type": "string", "enum": ["even", "highlights"], "default": "even"},
                   "min_clip_seconds": {"type": "number", "default": 2}, "wait_seconds": {"type": "integer", "default": 120}}, []),
    "auto_edit": (auto_edit, "One-click film from several clips (= plan_edit + render in one background job; returns at once): talk segments (whole sentences) framed on the selfie-stick holder's face with captions; scenery aimed at the most detailed / lit direction with one eased move each (pan, push in, pull out, tilt up); transition auto = hard cuts, a whip pan where the day / part of day changes, a little-planet opener and ending, fade to black. Follow it with job_status. Use plan_edit first to review the cuts.",
                  {"clip_ids": {"type": "array", "items": {"type": "string"}}, "query": {"type": "string", "description": "instead of clip_ids: substring of event folder, file name or date"},
                   "seconds": {"type": "number", "default": 0, "description": "target length; 0 = no target: every worthwhile moment, each conversation kept whole (length never costs a shot its completeness)"},
                   "aspect": {"type": "string", "enum": ["16:9", "9:16", "1:1", "4:5"], "default": "16:9"},
                   "transition": {"type": "string", "enum": ["auto", "cut", "fade", "fadeblack", "wipeleft", "circleopen"], "default": "auto"},
                   "captions": {"type": "string", "enum": ["off", "burn", "soft"], "default": "burn"},
                   "coverage": {"type": "string", "enum": ["even", "highlights"], "default": "even"}, "title": {"type": "string"},
                   "music_path": {"type": "string", "description": "local audio file (mp3/m4a/aac/wav/flac/ogg) for background music; cuts are snapped to its beats"},
                   "transcribe_missing": {"type": "boolean", "default": False, "description": "transcribe clips without a transcript first (slower, but adds talk segments and captions)"},
                   "min_clip_seconds": {"type": "number", "default": 2}}, []),
    "directed_edit": (directed_edit, "Render a film from YOUR shot list - the way to make a film that is about something. First understand the "
                      "footage: read what is said (search_speech, clip_overview transcripts), look at the panoramas (clip_overview, look) and decide "
                      "what each moment means and where in the 360 its subject is (who is talking, what they talk about or look at). Then give "
                      "the shots in film order with that subject as focus {yaw, pitch} (yaw/pitch as in clip_overview: 0 = camera forward, "
                      "+ = right / up). The tool does the craft: talk shots follow the speaking face searched around the focus (mouth motion "
                      "picks the speaker), other shots aim at the focus with one eased move (or the one you name), every shot is turned a little "
                      "so the lens seam (red line in clip_overview) stays out of frame unless the subject is on it, captions from the "
                      "transcript, transition auto = hard cuts, whip pans between chapters, little-planet open / close, fade to black.",
                      {"shots": {"type": "array", "items": {"type": "object", "properties": {
                          "clip_id": {"type": "string"}, "start": {"type": "number"}, "end": {"type": "number"},
                          "kind": {"type": "string", "enum": ["talk", "calm"], "description": "talk = frame the speaker; calm = scenery / action"},
                          "focus": {"type": "object", "properties": {"yaw": {"type": "number"}, "pitch": {"type": "number"}},
                                    "description": "where the subject of this shot is"},
                          "move": {"type": "string", "enum": ["pan", "push", "pull", "tilt", "hold"]},
                          "chapter": {"type": "string", "description": "shots of a new chapter start with a whip pan"},
                          "caption": {"type": "boolean", "description": "burn the transcript in (default: talk shots only)"},
                          "note": {"type": "string", "description": "why this shot is in the film (kept in the plan)"},
                          "beats": {"type": "array", "description": "choreograph a long shot: watch it through (look at several times), "
                                    "then give one beat per thing worth seeing - {at: clip seconds, focus: {yaw, pitch}, fov: number | close | "
                                    "normal | wide | ultra, track: follow the speaking face around the focus}. The camera moves between beats in "
                                    "0.8 s. E.g. someone points at the table -> a wide beat on the table and the people around it.",
                                    "items": {"type": "object"}},
                          "fov": {"description": "field of view of a calm shot: degrees or close / normal / wide / ultra (wide for a dome or a crowd)"},
                          "lines": {"type": "array", "description": "captions checked by ear [{start, end, text}] in clip seconds; "
                                    "replace the transcript for this shot ([] = no captions)", "items": {"type": "object"}}},
                          "required": ["clip_id", "start", "end"]}},
                       "aspect": {"type": "string", "enum": ["16:9", "9:16", "1:1", "4:5"], "default": "16:9"},
                       "transition": {"type": "string", "enum": ["auto", "cut", "fade", "fadeblack", "wipeleft", "circleopen"], "default": "auto"},
                       "captions": {"type": "string", "enum": ["off", "burn", "soft"], "default": "burn"},
                       "title": {"type": "string"}}, ["shots"]),
    "job_status": (job_status, "Status of a job (render, track, transcription, plan, auto edit); wait_seconds > 0 blocks until it finishes (up to that long). Plan and auto-edit jobs also show their cuts, coverage and warnings.",
                   {"job_id": {"type": "string"}, "wait_seconds": {"type": "integer", "default": 0}}, ["job_id"]),
}


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if method == "initialize":
        return dict(protocolVersion=msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                    capabilities={"tools": {}}, serverInfo={"name": "osvcut", "version": "0.2"})
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [dict(name=n, description=d, inputSchema={"type": "object", "properties": props, "required": req})
                          for n, (_, d, props, req) in TOOLS.items()]}
    if method == "tools/call":
        p = msg.get("params", {})
        fn = TOOLS.get(p.get("name"), (None,))[0]
        if not fn:
            return {"content": [_text(f"unknown tool {p.get('name')}")], "isError": True}
        try:
            return {"content": fn(**(p.get("arguments") or {}))}
        except Exception as e:
            return {"content": [_text(f"error: {e}")], "isError": True}
    if mid is not None:
        raise KeyError(method)
    return None


def main():
    # MCP stdio is UTF-8; Windows consoles default to a legacy code page
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" not in msg:          # notification
            continue
        try:
            res = handle(msg)
            out = {"jsonrpc": "2.0", "id": msg["id"], "result": res}
        except KeyError as e:
            out = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"method not found: {e}"}}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
