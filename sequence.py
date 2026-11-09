"""Render a sequence of osvcut specs and join them into one MP4.

    python sequence.py job.json

job.json: {"specs": [spec, ...], "dst": "out.mp4", "size": [w, h], "fps": n}
Every part is rendered with the sequence's size / fps, given a silent stereo track if its source has
no audio, and the parts are joined losslessly with the concat demuxer.
"""
import json
import os
import re
import subprocess
import sys
import tempfile


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return bool(out.strip())


def main(job_path):
    job = json.load(open(job_path, encoding="utf8"))
    specs, dst = job["specs"], job["dst"]
    size, fps = job.get("size") or specs[0].get("size", [1920, 1080]), job.get("fps") or specs[0].get("fps", 50)
    here = os.path.dirname(os.path.abspath(__file__))
    tmp = tempfile.mkdtemp(prefix="osvseq_", dir=os.path.join(here, "cache"))
    weights = [max(0.1, float(s["out"]) - float(s["in"])) for s in specs]
    total_w = sum(weights)
    done_w = 0.0
    parts = []
    for i, s in enumerate(specs):
        s = dict(s, size=size, fps=fps, force_fps=True, dst=os.path.join(tmp, f"part{i:03d}.mp4"))
        sp = os.path.join(tmp, f"part{i:03d}.json")
        json.dump(s, open(sp, "w", encoding="utf8"), ensure_ascii=False)
        p = subprocess.Popen([sys.executable, "-u", os.path.join(here, "osvcut.py"), sp], cwd=here,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf8", errors="replace")
        for line in p.stdout:
            m = re.match(r"progress (\d+)/(\d+)", line)
            if m:
                frac = int(m.group(1)) / max(1, int(m.group(2)))
                print(f"progress {int(1000 * (done_w + frac * weights[i]) / total_w)}/1000", flush=True)
        if p.wait() != 0:
            print(f"part {i} failed", flush=True)
            sys.exit(1)
        part = s["dst"]
        if not has_audio(part):
            fixed = part[:-4] + "_a.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", part, "-f", "lavfi", "-i",
                            "anullsrc=channel_layout=stereo:sample_rate=48000", "-map", "0:v", "-map", "1:a",
                            "-c:v", "copy", "-c:a", "aac", "-shortest", fixed], check=True)
            part = fixed
        parts.append(part)
        done_w += weights[i]
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w", encoding="utf8") as fh:
        for p in parts:
            fh.write("file '" + p.replace("\\", "/").replace("'", "'\\''") + "'\n")
    music = job.get("music")
    joined = dst if not music else os.path.join(tmp, "joined.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", joined], check=True)
    if music:
        # background music: loop/trim to the film, mix under the original sound, fade out at the end
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", joined],
                                   capture_output=True, text=True).stdout or 0)
        mv, ov = float(music.get("volume", 0.6)), float(music.get("original", 0.5))
        fade = min(3.0, dur / 4)
        fc = (f"[1:a]aloop=loop=-1:size=2e9,atrim=0:{dur:.3f},volume={mv},afade=t=out:st={max(0, dur - fade):.3f}:d={fade:.3f}[m];"
              f"[0:a]volume={ov}[o];[o][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", joined, "-i", music["path"], "-filter_complex", fc,
                        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", dst], check=True)
    if specs and all(sp.get("spherical") and sp.get("proj") == "equirect" for sp in specs):
        import spherical                       # every cut is a 360 render: tag the joined film too
        spherical.inject(dst)
    for p in os.listdir(tmp):
        os.remove(os.path.join(tmp, p))
    os.rmdir(tmp)
    print("progress 1000/1000", flush=True)
    print("sequence ->", dst, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
