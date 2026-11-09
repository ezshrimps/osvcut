"""OSV library: scan folders, cache per-clip metadata, make stitched panorama thumbnails."""
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
THUMBS = os.path.join(CACHE, "thumbs")
INDEX = os.path.join(CACHE, "library.json")
os.makedirs(THUMBS, exist_ok=True)

NAME_RE = re.compile(r"CAM_(\d{14})_(\d{4})_D\.OSV$", re.I)


def clip_id(path):
    return hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode("utf8")).hexdigest()[:12]


def probe(path):
    j = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name,codec_tag_string,width,height,r_frame_rate:format=duration",
         "-of", "json", path], capture_output=True, text=True).stdout or "{}")
    vs = [s for s in j.get("streams", []) if s.get("codec_name") == "hevc"]
    au = [s for s in j.get("streams", []) if s.get("codec_type") == "audio"]
    fr = vs[0]["r_frame_rate"].split("/") if vs else ["0", "1"]
    return dict(duration=float(j.get("format", {}).get("duration", 0) or 0),
                lens_px=vs[0]["width"] if vs else 0,
                fps=round(float(fr[0]) / float(fr[1] or 1), 3),
                audio=bool(au), lenses=len(vs))


class Library:
    def __init__(self, roots):
        self.roots = roots
        self.clips = {}
        self.lock = threading.Lock()
        self.scanning = False
        self.scanned = 0
        self.total = 0
        if os.path.exists(INDEX):
            try:
                self.clips = {c["id"]: c for c in json.load(open(INDEX, encoding="utf8"))}
            except Exception:
                self.clips = {}

    # ---- index ----------------------------------------------------------------
    def _walk(self):
        for root in self.roots:
            for d, _, files in os.walk(root):
                for fn in files:
                    if fn.upper().endswith(".OSV"):
                        yield root, os.path.join(d, fn)

    def scan(self):
        with self.lock:
            if self.scanning:
                return
            self.scanning = True
        try:
            found = list(self._walk())
            self.total, self.scanned = len(found), 0
            seen = set()
            for root, p in found:
                st = os.stat(p)
                cid = clip_id(p)
                seen.add(cid)
                old = self.clips.get(cid)
                if not (old and old.get("size") == st.st_size and old.get("mtime") == int(st.st_mtime)):
                    rel = os.path.relpath(os.path.dirname(p), root)
                    parts = [x for x in rel.replace("\\", "/").split("/") if x not in (".", "原片")]
                    m = NAME_RE.search(os.path.basename(p))
                    shot = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").isoformat() if m else None
                    info = dict(id=cid, path=p, name=os.path.basename(p), size=st.st_size,
                                mtime=int(st.st_mtime), event=" / ".join(parts[-2:]) if parts else "",
                                category=parts[0] if parts else "", shot=shot)
                    try:
                        info.update(probe(p))
                    except Exception as e:  # unreadable clip: keep it listed, flagged
                        info.update(duration=0, error=str(e))
                    with self.lock:
                        self.clips[cid] = info
                self.scanned += 1
                if self.scanned % 20 == 0:
                    self.save()
            with self.lock:
                for cid in [c for c in self.clips if c not in seen]:
                    del self.clips[cid]
            self.save()
        finally:
            self.scanning = False

    def save(self):
        with self.lock:
            data = list(self.clips.values())
        tmp = INDEX + ".tmp"
        json.dump(data, open(tmp, "w", encoding="utf8"), ensure_ascii=False)
        os.replace(tmp, INDEX)

    def get(self, cid):
        return self.clips.get(cid)

    def listing(self):
        with self.lock:
            return sorted(self.clips.values(), key=lambda c: (c.get("shot") or "", c["name"]), reverse=True)


# ---- thumbnails ---------------------------------------------------------------
_thumb_lock = threading.Lock()


def thumb_path(cid):
    return os.path.join(THUMBS, cid + ".jpg")


def make_thumb(clip, W=768):
    """Stitched, horizon-levelled equirect thumbnail (W x W/2) from ~30% into the clip."""
    out = thumb_path(clip["id"])
    if os.path.exists(out):
        return out
    with _thumb_lock:
        if os.path.exists(out):
            return out
        import numpy as np
        import torch
        import cv2
        from osvcut import Lens, dev, attitudes, body_frames, view_dirs
        from stitch import load_calib, grab
        from seam import SmartStitch
        f = clip["path"]
        dur = clip.get("duration") or 0
        t = min(max(dur * 0.3, 0), max(dur - 0.2, 0))
        cal = load_calib(f)
        lenses = [Lens(cal, L) for L in cal["lenses"]]
        try:
            ts, qs = attitudes(f, max(0, t - 0.5), t + 0.5)
            M = body_frames(ts, qs, 1.0, "level")[min(np.searchsorted(ts, t), len(ts) - 1)]
        except Exception:
            M = np.eye(3)
        imgs = [torch.from_numpy(grab(f, t, s, cal["W"])[:, :, ::-1].copy()).to(dev).permute(2, 0, 1)[None].float() / 255
                for s in (0, 1)]
        d = view_dirs("equirect", W, W // 2) @ torch.tensor(M, dtype=torch.float32, device=dev).T
        e = SmartStitch(cal, lenses)
        e.update(imgs)
        im = (e.render(imgs, d) * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        cv2.imwrite(out + ".tmp.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 82])
        os.replace(out + ".tmp.jpg", out)
        return out
