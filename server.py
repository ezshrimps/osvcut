"""osvcut studio server: static files (with Range), OSV library API, GPU job queue.

    python server.py [port] [library_root ...]
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
from serve import RangeHandler          # noqa: E402
import library as libmod                # noqa: E402
import analysis                         # noqa: E402
import blindtest                        # noqa: E402
import look                             # noqa: E402

PREVIEWS = os.path.join(HERE, "cache", "previews")
TRACKS = os.path.join(HERE, "cache", "tracks")
PROXIES = os.path.join(HERE, "cache", "proxies")
MUSIC = os.path.join(HERE, "cache", "music")
EXPORTS = os.path.join(HERE, "exports")
SPECS = os.path.join(HERE, "cache", "jobspecs")
for d in (PREVIEWS, EXPORTS, SPECS, TRACKS, PROXIES, MUSIC):
    os.makedirs(d, exist_ok=True)

# modules this process imports and renders with (blind test, look, one-click planning); exports run in fresh
# subprocesses.  If they change on disk after start, the studio asks for a restart instead of silently using old code.
_INPROC = ("seam.py", "osvcut.py", "stitch.py", "look.py", "blindtest.py", "autoedit.py", "proxy.py")
_START_MTIME = {m: os.path.getmtime(os.path.join(os.path.dirname(os.path.abspath(__file__)), m))
                for m in _INPROC if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), m))}


def stale_modules():
    here = os.path.dirname(os.path.abspath(__file__))
    return [m for m, t in _START_MTIME.items() if os.path.getmtime(os.path.join(here, m)) > t + 1]


PREVIEW_PRESETS = {
    "fast": dict(size=[1920, 960], fps=25, cq=26, seam="smart"),
    "standard": dict(size=[2880, 1440], fps=30, cq=23, seam="flow", seam_opts={"conf": "photo"}),
    "high": dict(size=[3840, 1920], fps=50, cq=20, seam="flow"),
}

LIB = None
JOBS = {}
JOBQ = queue.Queue()
FAILED_THUMBS = set()


def rel_url(path):
    return "/" + os.path.relpath(path, HERE).replace("\\", "/")


def preview_file(cid, preset):
    return os.path.join(PREVIEWS, f"{cid}_{preset}.mp4")


_TX_CACHE = {}


def speech_summary(cid):
    """(number of segments, first ~120 characters said) from the cached transcript, memoised by mtime."""
    tp = analysis.transcript_path(cid)
    if not os.path.exists(tp):
        return None
    mt = os.path.getmtime(tp)
    hit = _TX_CACHE.get(cid)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        from textnorm import clean_segments
        segs = clean_segments(json.load(open(tp, encoding="utf8")).get("segments", []))
    except Exception:
        segs = []
    txt = " ".join(sg["text"] for sg in segs)
    val = dict(n=len(segs), text=(txt[:120] + "…") if len(txt) > 120 else txt)
    _TX_CACHE[cid] = (mt, val)
    return val


def clip_public(c):
    out = {k: v for k, v in c.items() if k != "path"}
    sp = speech_summary(c["id"])
    if sp is not None:
        out["speech"] = sp
    out["previews"] = {p: rel_url(preview_file(c["id"], p)) for p in PREVIEW_PRESETS
                       if os.path.exists(preview_file(c["id"], p))}
    out["thumb"] = f"/api/thumb/{c['id']}.jpg"
    px, rj = os.path.join(PROXIES, c["id"] + ".mp4"), os.path.join(PROXIES, c["id"] + ".json")
    if os.path.exists(px) and os.path.exists(rj):
        out["proxy"], out["rig"] = rel_url(px), rel_url(rj)
    out["src"] = c["path"].replace("\\", "/")
    return out


# ---- jobs -----------------------------------------------------------------------
def new_job(kind, title, spec, clip_id=None, cmd=None, output=None):
    jid = uuid.uuid4().hex[:10]
    spec_path = None
    if spec is not None:
        spec_path = os.path.join(SPECS, jid + ".json")
        json.dump(spec, open(spec_path, "w", encoding="utf8"), ensure_ascii=False, indent=1)
        cmd = [sys.executable, "-u", "osvcut.py", spec_path]
        output = spec["dst"]
    job = dict(id=jid, kind=kind, title=title, clip=clip_id, status="queued", progress=0.0,
               created=time.time(), started=None, finished=None, output=rel_url(output),
               spec=spec_path, cmd=cmd, error=None, eta=None)
    JOBS[jid] = job
    JOBQ.put(jid)
    return job


def worker():
    while True:
        jid = JOBQ.get()
        job = JOBS.get(jid)
        if not job or job["status"] == "cancelled":
            continue
        job["status"], job["started"] = "running", time.time()
        try:
            p = subprocess.Popen(job["cmd"], cwd=HERE,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf8", errors="replace")
            job["_proc"] = p
            tail = []
            for line in p.stdout:
                m = re.match(r"progress (\d+)/(\d+)", line)
                if m:
                    done, tot = int(m.group(1)), int(m.group(2))
                    job["progress"] = min(1.0, done / max(tot, 1))
                    el = time.time() - job["started"]
                    if job["progress"] > 0.02:
                        job["eta"] = el / job["progress"] - el
                elif "Warning" not in line and "mod(n" not in line and line.strip():
                    tail.append(line.rstrip())
                    tail = tail[-12:]
            p.wait()
            if job["status"] == "cancelled":
                pass
            elif p.returncode == 0:
                job["status"], job["progress"] = "done", 1.0
            else:
                job["status"], job["error"] = "failed", "\n".join(tail[-6:]) or f"exit {p.returncode}"
        except Exception:
            job["status"], job["error"] = "failed", traceback.format_exc()[-800:]
        job["finished"] = time.time()
        job.pop("_proc", None)


def thumb_worker():
    """Pre-generate thumbnails for the whole library, newest clips first."""
    while True:
        time.sleep(2)
        if LIB.scanning and not LIB.clips:
            continue
        todo = [c for c in LIB.listing() if not os.path.exists(libmod.thumb_path(c["id"]))
                and c["id"] not in FAILED_THUMBS and c.get("duration", 0) > 0]
        if not todo:
            time.sleep(10)
            continue
        # don't fight a running render for the GPU
        if any(j["status"] == "running" for j in JOBS.values()):
            time.sleep(3)
            continue
        c = todo[0]
        try:
            libmod.make_thumb(c)
        except Exception:
            FAILED_THUMBS.add(c["id"])


# ---- HTTP -------------------------------------------------------------------------
class Handler(RangeHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        if p == "/":
            self.send_response(302)
            self.send_header("Location", "/viewer/")
            self.end_headers()
            return
        if p == "/api/library":
            return self._json(dict(scanning=LIB.scanning, scanned=LIB.scanned, total=LIB.total,
                                   clips=[clip_public(c) for c in LIB.listing()],
                                   presets=PREVIEW_PRESETS, stale=stale_modules()))
        m = re.match(r"^/api/thumb/([0-9a-f]{12})\.jpg$", p)
        if m:
            c = LIB.get(m.group(1))
            if not c:
                return self._json({"error": "unknown clip"}, 404)
            try:
                path = libmod.make_thumb(c)
            except Exception as e:
                FAILED_THUMBS.add(c["id"])
                return self._json({"error": str(e)}, 500)
            data = open(path, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(data)
            return
        m = re.match(r"^/api/clip/([0-9a-f]{12})$", p)
        if m:
            c = LIB.get(m.group(1))
            return self._json(clip_public(c) if c else {"error": "unknown clip"}, 200 if c else 404)
        m = re.match(r"^/api/clip/([0-9a-f]{12})/holder$", p)
        if m:
            c = LIB.get(m.group(1))
            if not c:
                return self._json({"error": "unknown clip"}, 404)
            from urllib.parse import parse_qs
            qs_ = parse_qs(u.query)
            if "t1" in qs_:
                return self._json(look.holder_track(c, float(qs_.get("t0", ["0"])[0]), float(qs_["t1"][0])))
            t = float(qs_.get("t", ["0"])[0])
            return self._json(look.holder_direction(c, t))
        m = re.match(r"^/api/clip/([0-9a-f]{12})/(view|sheet)$", p)
        if m:
            c = LIB.get(m.group(1))
            if not c:
                return self._json({"error": "unknown clip"}, 404)
            from urllib.parse import parse_qs
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                with libmod._thumb_lock:
                    data = look.view_jpeg(c, float(q.get("t", 0)), float(q.get("yaw", 0)), float(q.get("pitch", 0)),
                                          float(q.get("fov", 90)), float(q.get("dist", 0)),
                                          min(7680, int(q.get("w", 960))), min(7680, int(q.get("h", 540))),
                                          grade_=json.loads(q["grade"]) if q.get("grade") else None,
                                          nadir=float(q.get("nadir", 0)), q=int(q.get("q", 88))) \
                        if m.group(2) == "view" else look.sheet_jpeg(c, int(q.get("n", 6)))
            except Exception:
                return self._json({"error": traceback.format_exc()[-600:]}, 500)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            if q.get("dl"):      # still export from the studio: download with a readable file name
                self.send_header("Content-Disposition", 'attachment; filename="%s"' % re.sub(r"[^\w.-]", "_", q["dl"]))
            self.end_headers()
            self.wfile.write(data)
            return
        m = re.match(r"^/api/clip/([0-9a-f]{12})/(motion|transcript|wave)$", p)
        if m:
            c = LIB.get(m.group(1))
            if not c:
                return self._json({"error": "unknown clip"}, 404)
            if m.group(2) == "wave":
                try:
                    return self._json(analysis.waveform(c))
                except Exception as e:  # noqa: BLE001
                    return self._json({"error": str(e)}, 500)
            if m.group(2) == "motion":
                try:
                    return self._json(analysis.motion(c))
                except Exception as e:
                    return self._json({"error": str(e)}, 500)
            out = analysis.transcript_path(c["id"])
            if os.path.exists(out):
                from textnorm import clean_segments
                tx = json.load(open(out, encoding="utf8"))
                tx["segments"] = clean_segments(tx.get("segments", []))
                return self._json(dict(status="done", **tx))
            if not c.get("audio"):
                return self._json(dict(status="done", language=None, segments=[], note="no audio"))
            j = next((j for j in JOBS.values() if j["kind"] == "transcript" and j["clip"] == c["id"]
                      and j["status"] in ("queued", "running")), None)
            if not j:
                j = new_job("transcript", f"{c['name']} · 语音转写", None, c["id"],
                            cmd=[sys.executable, "-u", "analysis.py", "transcript", c["path"], out], output=out)
            return self._json(dict(status=j["status"], job=j["id"], progress=j["progress"]))
        if p == "/api/search":
            from urllib.parse import parse_qs
            from textnorm import norm
            q = norm((parse_qs(u.query).get("q", [""])[0] or "").strip())
            hits = []
            if q:
                for c in LIB.listing():
                    tp = analysis.transcript_path(c["id"])
                    if not os.path.exists(tp):
                        continue
                    try:
                        from textnorm import clean_segments
                        segs = clean_segments(json.load(open(tp, encoding="utf8")).get("segments", []))
                    except Exception:
                        continue
                    for sg in segs:
                        if q in norm(sg["text"]):
                            hits.append(dict(clip=c["id"], name=c["name"], event=c.get("event"), start=sg["start"],
                                             end=sg["end"], text=sg["text"]))
            done = sum(1 for c in LIB.clips.values() if os.path.exists(analysis.transcript_path(c["id"])))
            todo = sum(1 for c in LIB.clips.values() if c.get("audio") and c.get("duration", 0) > 1)
            return self._json(dict(q=q, hits=hits[:200], transcribed=done, with_audio=todo))
        if p == "/api/storage":
            def size(d, ext=None):
                tot, n = 0, 0
                if os.path.isdir(d):
                    for fn in os.listdir(d):
                        fp = os.path.join(d, fn)
                        if os.path.isfile(fp) and (ext is None or fn.endswith(ext)):
                            tot += os.path.getsize(fp); n += 1
                return dict(bytes=tot, files=n)
            return self._json(dict(proxies=size(PROXIES), previews=size(PREVIEWS), thumbs=size(libmod.THUMBS),
                                   exports=size(EXPORTS, ".mp4"), analysis=size(analysis.ADIR)))
        if p == "/api/blindtest/stats":
            kind = "video" if "kind=video" in (u.query or "") else "dji" if "kind=dji" in (u.query or "") else "still"
            st = blindtest.stats(kind)
            import dji_compare
            st["dji_exports"] = len(dji_compare.exports(LIB))
            return self._json(st)
        if p == "/api/jobs":
            return self._json([{k: v for k, v in j.items() if not k.startswith("_")}
                               for j in sorted(JOBS.values(), key=lambda j: -j["created"])])
        if p == "/api/exports":
            items = []
            for fn in sorted(os.listdir(EXPORTS), reverse=True):
                if fn.endswith(".mp4") and not any(j["status"] in ("queued", "running") and
                                                   j["output"].endswith("/" + fn) for j in JOBS.values()):
                    side = os.path.join(EXPORTS, fn[:-4] + ".json")
                    meta = json.load(open(side, encoding="utf8")) if os.path.exists(side) else {}
                    items.append(dict(file="/exports/" + fn, name=fn, size=os.path.getsize(os.path.join(EXPORTS, fn)),
                                      title=meta.get("title", fn), aspect=meta.get("size")))
            return self._json(items)
        return super().do_GET()

    def do_POST(self):
        p = unquote(urlparse(self.path).path)
        if p == "/api/jobs":
            b = self._body()
            kind = b.get("kind")
            if kind == "preview":
                c = LIB.get(b.get("clip", ""))
                if not c:
                    return self._json({"error": "unknown clip"}, 404)
                preset = b.get("preset", "standard")
                pr = PREVIEW_PRESETS.get(preset)
                if not pr:
                    return self._json({"error": "unknown preset"}, 400)
                for j in JOBS.values():
                    if j["clip"] == c["id"] and j["kind"] == "preview" and j["status"] in ("queued", "running") \
                            and j["output"] == rel_url(preview_file(c["id"], preset)):
                        return self._json(j)
                spec = dict(src=c["path"], **{"in": 0, "out": round(c["duration"], 3)}, proj="equirect",
                            dst=preview_file(c["id"], preset), **pr)
                job = new_job("preview", f"{c['event'] or ''} · {c['name']} · 预览({preset})", spec, c["id"])
                return self._json({k: v for k, v in job.items() if not k.startswith("_")})
            if kind == "proxy":
                c = LIB.get(b.get("clip", ""))
                if not c:
                    return self._json({"error": "unknown clip"}, 404)
                px, rj = os.path.join(PROXIES, c["id"] + ".mp4"), os.path.join(PROXIES, c["id"] + ".json")
                for j in JOBS.values():
                    if j["clip"] == c["id"] and j["kind"] == "proxy" and j["status"] in ("queued", "running"):
                        return self._json({k: v for k, v in j.items() if not k.startswith("_")})
                tmp = px[:-4] + ".part.mp4"
                job = new_job("proxy", f"{c['name']} · 实时预览代理", None, c["id"],
                              cmd=[sys.executable, "-u", "-c",
                                   "import sys,os,subprocess;"
                                   "r=subprocess.run([sys.executable,'-u','proxy.py',sys.argv[1],sys.argv[2],sys.argv[3],'1536',sys.argv[4]]).returncode;"
                                   "os.replace(sys.argv[2],sys.argv[4]) if r==0 and os.path.exists(sys.argv[2]) else None;sys.exit(r)",
                                   c["path"], tmp, rj, px], output=px)
                # jump the queue: proxies are short and unblock the viewer
                with JOBQ.mutex:
                    JOBQ.queue.remove(job["id"])
                    JOBQ.queue.appendleft(job["id"])
                return self._json({k: v for k, v in job.items() if not k.startswith("_")})
            if kind == "export":
                spec = b.get("spec") or {}
                src = spec.get("src", "")
                cid = libmod.clip_id(src) if src else None
                if not src or not os.path.exists(src):
                    return self._json({"error": "source not found"}, 400)
                title = (b.get("title") or os.path.basename(src)).strip()
                safe = re.sub(r'[\\/:*?"<>|]+', "_", title)[:60]
                fn = f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}.mp4"
                spec["dst"] = os.path.join(EXPORTS, fn)
                spec.setdefault("seam", "flow")
                json.dump(dict(title=title, size=spec.get("size"), spec=spec),
                          open(spec["dst"][:-4] + ".json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
                job = new_job("export", title, spec, cid)
                return self._json({k: v for k, v in job.items() if not k.startswith("_")})
            if kind == "autoedit":
                # one-click edit: pick highlights from the given clips, frame them, then run as a sequence
                import autoedit
                clips = [LIB.get(cid) for cid in (b.get("clip_ids") or [])]
                clips = [c for c in clips if c and c.get("lenses") == 2]
                if not clips:
                    return self._json({"error": "no usable clips"}, 400)
                plan = autoedit.plan(clips, seconds=float(b.get("seconds", 30)), aspect=b.get("aspect", "16:9"),
                                     transition=b.get("transition", "fade"), captions=b.get("captions", "burn"),
                                     music=b.get("music"))
                if not plan["specs"]:
                    return self._json({"error": "no talk or steady segments found in these clips"}, 400)
                if plan.get("beat") and b.get("music"):
                    b["music"] = dict(b["music"], start=plan["beat"]["phase"])
                b = dict(kind="sequence", specs=plan["specs"], size=plan["size"], fps=plan["fps"], music=b.get("music"),
                         transition=plan.get("transition"), title=b.get("title") or f"一键成片 · {len(plan['specs'])} 段")
                kind = "sequence"
            if kind == "sequence":
                specs = b.get("specs") or []
                if not specs or any(not os.path.exists(x.get("src", "")) for x in specs):
                    return self._json({"error": "empty sequence or missing source"}, 400)
                title = (b.get("title") or f"剪辑台 · {len(specs)} 段").strip()
                safe = re.sub(r'[\\/:*?"<>|]+', "_", title)[:60]
                dst = os.path.join(EXPORTS, f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}.mp4")
                jid_spec = os.path.join(SPECS, "seq_" + uuid.uuid4().hex[:8] + ".json")
                music = b.get("music") or None
                if music and not (os.path.abspath(music["path"]).startswith(os.path.abspath(MUSIC)) and os.path.exists(music["path"])):
                    return self._json({"error": "music file not found"}, 400)
                tr = b.get("transition") or None
                if tr and tr.get("type") not in ("fade", "fadeblack", "wipeleft", "circleopen"):
                    tr = None
                json.dump(dict(specs=specs, dst=dst, size=b.get("size"), fps=b.get("fps"), music=music, transition=tr, title_card=b.get("title_card")),
                          open(jid_spec, "w", encoding="utf8"), ensure_ascii=False, indent=1)
                json.dump(dict(title=title, size=b.get("size") or specs[0].get("size"), sequence=specs),
                          open(dst[:-4] + ".json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
                job = new_job("sequence", title, None, None, cmd=[sys.executable, "-u", "sequence.py", jid_spec], output=dst)
                return self._json({k: v for k, v in job.items() if not k.startswith("_")})
            return self._json({"error": "unknown kind"}, 400)
        if p == "/api/debug/snapshot":
            # dev aid: save a canvas dataURL to cache/debug/<name>.png (never served publicly, gitignored)
            import base64
            b = self._body()
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", b.get("name", "snap"))[:60]
            data = b.get("data", "").split(",", 1)[-1]
            d = os.path.join(HERE, "cache", "debug")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, name + ".png"), "wb").write(base64.b64decode(data))
            return self._json({"ok": True})
        if p == "/api/blindtest/next":
            try:
                kind = self._body().get("kind")
                if kind == "video":                 # flow vs. cut in motion, pre-rendered in the background
                    pr = blindtest.next_video(LIB, libmod._thumb_lock)
                elif kind == "dji":                 # ours vs. DJI Studio's export of the same clip
                    with libmod._thumb_lock:
                        pr = blindtest.make_dji_pair(LIB)
                else:
                    with libmod._thumb_lock:      # share the GPU politely with thumbnail generation
                        pr = blindtest.make_pair(LIB)
            except Exception:
                return self._json({"error": traceback.format_exc()[-600:]}, 500)
            return self._json(pr or {"error": "no pair"}, 200 if pr else 500)
        if p == "/api/blindtest/vote":
            b = self._body()
            return self._json({"ok": blindtest.vote(b.get("pid", ""), b.get("choice", ""))})
        if p == "/api/track":
            b = self._body()
            src = b.get("src", "")
            if not src or not os.path.exists(src):
                return self._json({"error": "source not found"}, 400)
            jid = uuid.uuid4().hex[:10]
            out = os.path.join(TRACKS, jid + ".json")
            job_spec = {k: b[k] for k in ("t0", "t1", "yaw", "pitch", "box_w", "box_h", "fov") if k in b}
            job_spec.update(src=src, out=out, fps=15)
            spec_path = os.path.join(SPECS, "track_" + jid + ".json")
            json.dump(job_spec, open(spec_path, "w", encoding="utf8"), ensure_ascii=False)
            job = new_job("track", f"{os.path.basename(src)} · 跟随主体 {b.get('t0', 0):.1f}s→{b.get('t1', 0):.1f}s", None,
                          libmod.clip_id(src), cmd=[sys.executable, "-u", "track.py", spec_path], output=out)
            return self._json({k: v for k, v in job.items() if not k.startswith("_")})
        m = re.match(r"^/api/jobs/([0-9a-f]+)/cancel$", p)
        if m:
            j = JOBS.get(m.group(1))
            if not j:
                return self._json({"error": "unknown job"}, 404)
            j["status"] = "cancelled"
            proc = j.get("_proc")
            if proc:
                proc.kill()
            return self._json({"ok": True})
        if p == "/api/upload/music":
            # raw body = the audio file; ?name= original file name
            from urllib.parse import parse_qs
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            name = re.sub(r'[\\/:*?"<>|]+', "_", q.get("name", "music"))[-80:]
            if not re.search(r"\.(mp3|m4a|aac|wav|flac|ogg)$", name, re.I):
                return self._json({"error": "unsupported audio type"}, 400)
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 300 * 1024 * 1024:
                return self._json({"error": "file too large or empty"}, 400)
            dst = os.path.join(MUSIC, f"{uuid.uuid4().hex[:8]}_{name}")
            with open(dst, "wb") as fh:
                left = n
                while left:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    fh.write(chunk); left -= len(chunk)
            return self._json({"ok": True, "path": dst, "name": name})
        if p == "/api/storage/clear":
            b = self._body()
            what = b.get("what")
            dirs = {"proxies": PROXIES, "previews": PREVIEWS, "thumbs": libmod.THUMBS}
            if what not in dirs:
                return self._json({"error": "unknown cache"}, 400)
            busy = {j["clip"] for j in JOBS.values() if j["status"] in ("queued", "running")}
            removed = 0
            for fn in os.listdir(dirs[what]):
                if fn[:12] in busy:
                    continue
                try:
                    os.remove(os.path.join(dirs[what], fn)); removed += 1
                except OSError:
                    pass
            return self._json({"ok": True, "removed": removed})
        if p == "/api/transcribe_all":
            if any(j["kind"] == "transcript_batch" and j["status"] in ("queued", "running") for j in JOBS.values()):
                return self._json({"ok": True, "queued": 0, "note": "already running"})
            busy = {j["clip"] for j in JOBS.values() if j["kind"] == "transcript" and j["status"] in ("queued", "running")}
            items = [dict(src=c["path"], out=analysis.transcript_path(c["id"])) for c in LIB.listing()
                     if c.get("audio") and c.get("duration", 0) > 1 and not os.path.exists(analysis.transcript_path(c["id"]))
                     and c["id"] not in busy]
            if not items:
                return self._json({"ok": True, "queued": 0})
            lst = os.path.join(SPECS, "txbatch_" + uuid.uuid4().hex[:8] + ".json")
            json.dump(items, open(lst, "w", encoding="utf8"), ensure_ascii=False)
            new_job("transcript_batch", f"批量语音转写 · {len(items)} 段", None, None,
                    cmd=[sys.executable, "-u", "analysis.py", "transcript_batch", lst], output=lst)
            return self._json({"ok": True, "queued": len(items)})
        if p == "/api/rescan":
            threading.Thread(target=LIB.scan, daemon=True).start()
            return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)


def main():
    global LIB
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8360
    roots = sys.argv[2:] or [r"E:\Videos"]
    LIB = libmod.Library(roots)
    threading.Thread(target=LIB.scan, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=thumb_worker, daemon=True).start()
    print(f"osvcut studio on http://localhost:{port}/viewer/  library: {roots}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
