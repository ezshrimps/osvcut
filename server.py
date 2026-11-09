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

PREVIEWS = os.path.join(HERE, "cache", "previews")
EXPORTS = os.path.join(HERE, "exports")
SPECS = os.path.join(HERE, "cache", "jobspecs")
for d in (PREVIEWS, EXPORTS, SPECS):
    os.makedirs(d, exist_ok=True)

PREVIEW_PRESETS = {
    "fast": dict(size=[1920, 960], fps=25, cq=26, seam="smart"),
    "standard": dict(size=[2880, 1440], fps=30, cq=23, seam="flow"),
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


def clip_public(c):
    out = {k: v for k, v in c.items() if k != "path"}
    out["previews"] = {p: rel_url(preview_file(c["id"], p)) for p in PREVIEW_PRESETS
                       if os.path.exists(preview_file(c["id"], p))}
    out["thumb"] = f"/api/thumb/{c['id']}.jpg"
    out["src"] = c["path"].replace("\\", "/")
    return out


# ---- jobs -----------------------------------------------------------------------
def new_job(kind, title, spec, clip_id=None):
    jid = uuid.uuid4().hex[:10]
    spec_path = os.path.join(SPECS, jid + ".json")
    json.dump(spec, open(spec_path, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    job = dict(id=jid, kind=kind, title=title, clip=clip_id, status="queued", progress=0.0,
               created=time.time(), started=None, finished=None, output=rel_url(spec["dst"]),
               spec=spec_path, error=None, eta=None)
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
            p = subprocess.Popen([sys.executable, "-u", "osvcut.py", job["spec"]], cwd=HERE,
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
                                   presets=PREVIEW_PRESETS))
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
            return self._json({"error": "unknown kind"}, 400)
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
