"""MCP server: lets an AI assistant browse, look around in, and cut 360 OSV footage.

Runs over stdio (JSON-RPC 2.0, one message per line) and talks to a running osvcut studio
(`python server.py`), default http://127.0.0.1:8360 (override with OSVCUT_URL).

    claude mcp add osvcut -- python /path/to/osvcut/mcp_server.py

Tools: list_clips, clip_overview, look, track_subject, render_cut, render_sequence, job_status.
Angles: yaw 0 = camera forward, +90 = right, ±180 = behind; pitch +90 = up. Times are seconds in the clip.
"""
import base64
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = os.environ.get("OSVCUT_URL", "http://127.0.0.1:8360").rstrip("/")
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- http helpers
def _get(path, raw=False, timeout=300):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        data = r.read()
    return data if raw else json.loads(data.decode("utf8"))


def _post(path, body, timeout=60):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf8"))


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
def list_clips(query="", category="", limit=30):
    lib = _get("/api/library")
    q = (query or "").lower()
    out = []
    for c in lib["clips"]:
        if category and category not in (c.get("category") or ""):
            continue
        if q and q not in f"{c.get('event', '')} {c['name']} {c.get('shot', '')}".lower():
            continue
        out.append(c)
    lines = [f"{len(out)} clips match (showing {min(limit, len(out))}); library has {len(lib['clips'])}."]
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


def render_cut(clip_id, start, end, keys=None, aspect="16:9", fps=30, speed=1.0, title="", projection="flat", nadir_patch=0, grade=None, captions="off", freeze_at=None, freeze_seconds=2.0, freeze_turn=360.0):
    size = {"16:9": [1920, 1080], "9:16": [1080, 1920], "1:1": [1080, 1080], "4:5": [1080, 1350], "2.39:1": [1920, 816]}.get(aspect, [1920, 1080])
    spec = _spec(clip_id, start, end, keys, size, fps, speed)
    if nadir_patch:
        spec["nadir_patch"] = float(nadir_patch)
    if grade:
        spec["grade"] = {k: float(v) for k, v in grade.items() if k in ("exposure", "contrast", "saturation", "temperature")}
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


def render_sequence(cuts, title="", aspect="16:9", fps=30, projection="flat", transition="cut", transition_seconds=0.5):
    size = {"16:9": [1920, 1080], "9:16": [1080, 1920], "1:1": [1080, 1080], "4:5": [1080, 1350], "2.39:1": [1920, 816]}.get(aspect, [1920, 1080])
    specs = [_spec(cu["clip_id"], cu["start"], cu["end"], cu.get("keys"), size, fps, cu.get("speed", 1.0)) for cu in cuts]
    if projection == "360":
        size = [3840, 1920]
        for sp in specs:
            sp.update(proj="equirect", size=size, spherical=True); sp.pop("keys", None)
    j = _post("/api/jobs", dict(kind="sequence", specs=specs, title=title or f"AI sequence ({len(specs)} cuts)", size=size, fps=fps,
                                         **({"transition": {"type": transition, "duration": float(transition_seconds)}} if transition != "cut" else {})))
    if "id" not in j:
        raise ValueError(j.get("error") or "request failed")
    return [_text(f"sequence job {j['id']} queued with {len(specs)} cuts. Poll job_status.")]


def auto_edit(clip_ids, seconds=30, aspect="16:9", transition="fade", captions="burn", title=""):
    j = _post("/api/jobs", dict(kind="autoedit", clip_ids=list(clip_ids), seconds=float(seconds), aspect=aspect,
                                transition=transition, captions=captions, title=title or None))
    if "id" not in j:
        raise ValueError(j.get("error") or "request failed")
    return [_text(f"auto edit started: job {j['id']} -> {j.get('output')}. Use job_status to wait for it.")]


def job_status(job_id, wait_seconds=0):
    j = _wait_job(job_id, wait_seconds) if wait_seconds else next((j for j in _get("/api/jobs") if j["id"] == job_id), None)
    if not j:
        return [_text(f"no job {job_id}")]
    path = os.path.join(HERE, j["output"].lstrip("/").replace("/", os.sep)) if j.get("output") else ""
    return [_text(f"{j['kind']} '{j['title']}': {j['status']} {j['progress'] * 100:.0f}%"
                  + (f", eta {j['eta']:.0f}s" if j.get("eta") and j["status"] == "running" else "")
                  + (f"\noutput: {path}" if j["status"] == "done" else "")
                  + (f"\nerror: {j['error']}" if j.get("error") else ""))]


TOOLS = {
    "list_clips": (list_clips, "Search the 360 footage library (DJI Osmo 360 .OSV). Returns id | event | file | duration | shot time | mode | flags.",
                   {"query": {"type": "string", "description": "substring of event folder, file name or date"},
                    "category": {"type": "string"}, "limit": {"type": "integer", "default": 30}}, []),
    "search_speech": (search_speech, "Find clips by what is said in them (speech transcripts). Returns clip id, time range and the sentence.",
                      {"query": {"type": "string"}, "limit": {"type": "integer", "default": 30}}, ["query"]),
    "clip_overview": (clip_overview, "Understand a clip: levelled 360 contact sheet (image), speech transcript, camera motion and suggested segments.",
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
                    "freeze_seconds": {"type": "number", "default": 2.0}, "freeze_turn": {"type": "number", "default": 360.0, "description": "degrees of yaw during the freeze"},
                    "captions": {"type": "string", "enum": ["off", "burn", "soft"], "default": "off",
                                 "description": "subtitles from the speech transcript: burned into the picture, or a soft track + .srt"},
                    "nadir_patch": {"type": "number", "default": 0, "description": "cover the area below the camera (hand / stick) with a smooth patch of this radius in degrees, e.g. 12"},
                    "projection": {"type": "string", "enum": ["flat", "360"], "default": "flat",
                                   "description": "flat = reframed video following keys; 360 = full levelled 360 video (YouTube/VR), keys ignored"}},
                   ["clip_id", "start", "end"]),
    "render_sequence": (render_sequence, "Export several cuts (possibly from different clips) joined into one film. cuts = [{clip_id, start, end, keys?, speed?}].",
                        {"cuts": {"type": "array", "items": {"type": "object"}}, "title": {"type": "string"},
                         "projection": {"type": "string", "enum": ["flat", "360"], "default": "flat"},
                         "transition": {"type": "string", "enum": ["cut", "fade", "fadeblack", "wipeleft", "circleopen"], "default": "cut"},
                         "transition_seconds": {"type": "number", "default": 0.5},
                         "aspect": {"type": "string", "enum": ["16:9", "9:16", "1:1", "4:5", "2.39:1"], "default": "16:9"}, "fps": {"type": "integer", "default": 30}},
                        ["cuts"]),
    "auto_edit": (auto_edit, "One-click film from several clips: picks talk segments (framed on the selfie-stick holder, with captions) and steady scenery (slow pan), joins them with transitions. Good first draft; refine with render_sequence.",
                  {"clip_ids": {"type": "array", "items": {"type": "string"}}, "seconds": {"type": "number", "default": 30},
                   "aspect": {"type": "string", "enum": ["16:9", "9:16", "1:1", "4:5"], "default": "16:9"},
                   "transition": {"type": "string", "enum": ["cut", "fade", "fadeblack", "wipeleft", "circleopen"], "default": "fade"},
                   "captions": {"type": "string", "enum": ["off", "burn", "soft"], "default": "burn"}, "title": {"type": "string"}},
                  ["clip_ids"]),
    "job_status": (job_status, "Status of a render/track job; wait_seconds > 0 blocks until it finishes (up to that long).",
                   {"job_id": {"type": "string"}, "wait_seconds": {"type": "integer", "default": 0}}, ["job_id"]),
}


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if method == "initialize":
        return dict(protocolVersion=msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                    capabilities={"tools": {}}, serverInfo={"name": "osvcut", "version": "0.1"})
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
