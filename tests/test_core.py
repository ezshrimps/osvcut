"""Footage-free unit tests for the core maths.  Run:  python tests/test_core.py   (or pytest)."""
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---------------------------------------------------------------- protobuf decoding
def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(fn, wt, payload):
    key = _varint((fn << 3) | wt)
    if wt == 0:
        return key + _varint(payload)
    if wt == 5:
        return key + struct.pack("<f", payload)
    if wt == 2:
        return key + _varint(len(payload)) + payload
    raise ValueError(wt)


def test_protobuf_flat_and_nested():
    from djmd import parse, tree, get
    inner = _field(1, 5, 1.5) + _field(2, 5, -2.0)
    msg = _field(3, 0, 300) + _field(4, 2, inner) + _field(5, 2, b"hello")
    flat = parse(msg)
    assert [f[0] for f in flat] == [3, 4, 5]
    assert flat[0][2] == 300
    t = tree(msg)
    assert abs(get(t, 4, 1) - 1.5) < 1e-6 and abs(get(t, 4, 2) + 2.0) < 1e-6
    assert get(t, 5) == "hello"


def test_concatenated_samples_split_by_field3():
    """Each djmd sample is a ProductMeta; later samples only carry field 3, so a concatenation parsed
    as one message yields one field-3 per frame (the trick behind frame_attitudes)."""
    from djmd import parse
    frames = [_field(3, 2, _field(1, 2, _field(2, 0, 1000 * i))) for i in range(5)]
    stream = _field(1, 2, b"\x08\x01") + _field(2, 2, b"\x08\x02") + b"".join(frames)
    assert sum(1 for f in parse(stream) if f[0] == 3) == 5


# ---------------------------------------------------------------- rotations
def test_qmat_orthonormal_and_quat_roundtrip():
    from stitch import qmat
    from proxy import rot_to_quat
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = qmat(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6
        q2 = np.array(rot_to_quat(R))
        assert np.allclose(qmat(q2), R, atol=1e-6)


def test_cam_rot_conventions():
    from osvcut import cam_rot
    fwd = np.array([0, 1.0, 0])
    assert np.allclose(cam_rot(90, 0) @ fwd, [1, 0, 0], atol=1e-9)      # +yaw turns right
    assert np.allclose(cam_rot(0, 90) @ fwd, [0, 0, 1], atol=1e-9)      # +pitch looks up


# ---------------------------------------------------------------- projections
def _eye_offset_forward(theta, d, f):
    return f * (1 + d) * math.sin(theta) / (d + math.cos(theta))


def test_eye_offset_inverse_matches_forward():
    """view_dirs_eye inverts r/f = (1+d) sin(th) / (d + cos(th)) (same maths as the viewer shader)."""
    import torch
    from osvcut import view_dirs_eye
    for d in (0.0, 0.4, 1.0):
        for fov in (90.0, 150.0, 300.0 if d == 1.0 else 150.0):
            W, H = 64, 48
            dirs, valid = view_dirs_eye(W, H, fov, d)
            h = math.radians(min(fov, 2 * math.degrees(math.acos(-d)) - 1)) / 2
            f = (W / 2) * (d + math.cos(h)) / ((1 + d) * math.sin(h))
            v = dirs.cpu().numpy()
            th = np.arccos(np.clip(v[..., 1], -1, 1))
            px = np.arange(W) + .5 - W / 2
            py = -(np.arange(H) + .5 - H / 2)
            r = np.hypot(*np.meshgrid(px, py))
            ok = valid.cpu().numpy()
            pred = np.array([_eye_offset_forward(t, d, f) for t in th[ok]])
            assert np.allclose(pred, r[ok], atol=1e-2), (d, fov)


def test_equirect_centre_is_forward():
    from osvcut import view_dirs
    d = view_dirs("equirect", 8, 4).cpu().numpy()
    c = (d[1, 3] + d[1, 4] + d[2, 3] + d[2, 4]) / 4
    assert np.argmax(c) == 1 and c[1] > 0.8 and abs(c[0]) < 1e-6 and abs(c[2]) < 1e-6


# ---------------------------------------------------------------- keyframes
def test_interp_kf_smoothstep_linear_and_wrap():
    from osvcut import interp_kf
    ks = [{"t": 0, "yaw": 170, "pitch": 0, "fov": 90}, {"t": 1, "yaw": -170, "pitch": 10, "fov": 90}]
    m = interp_kf(ks, 0.5)
    assert abs(((m["yaw"] + 180) % 360) - 180 - 180) < 1e-6 or abs(abs(m["yaw"]) - 180) < 1e-6   # shortest way
    assert abs(m["pitch"] - 5) < 1e-9
    q = interp_kf(ks, 0.25)["pitch"]
    lin = interp_kf([dict(ks[0], ease="linear"), ks[1]], 0.25)["pitch"]
    assert abs(lin - 2.5) < 1e-9 and q < lin           # smoothstep starts slower than linear


# ---------------------------------------------------------------- seam DP
def test_dp_closed_path_is_closed_and_follows_valley():
    from seam import _dp_closed_path
    H, W = 21, 200
    rows = (10 + 6 * np.sin(np.linspace(0, 2 * np.pi, W, endpoint=False))).round().astype(int)
    cost = np.ones((H, W))
    cost[rows, np.arange(W)] = 0
    p = _dp_closed_path(cost)
    assert np.all(np.abs(np.diff(p)) <= 1) and abs(int(p[0]) - int(p[-1])) <= 1
    assert np.mean(p == rows) > 0.9


def test_circular_helpers():
    from seam import _fill_circular, _circ_smooth
    v = np.array([1.0, 0, 0, 3.0])
    ok = np.array([True, False, False, True])
    f = _fill_circular(v, ok)
    assert np.allclose(f, [1, 5 / 3, 7 / 3, 3])
    s = _circ_smooth(np.r_[np.ones(10), np.zeros(10)], 2)
    assert s.shape == (20,) and abs(s.sum() - 10) < 1e-6


# ---------------------------------------------------------------- colour
def test_yuv420_to_rgb_grey_and_primaries():
    from osvcut import yuv420_to_rgb
    H, W = 4, 8
    def frame(Y, U, V):
        return bytes(np.full(H * W, Y, np.uint8)) + bytes(np.full(H * W // 4, U, np.uint8)) + bytes(np.full(H * W // 4, V, np.uint8))
    g = yuv420_to_rgb(frame(126, 128, 128), H, W).cpu().numpy()
    assert np.allclose(g, (126 - 16) / 219, atol=1e-3)
    white = yuv420_to_rgb(frame(235, 128, 128), H, W).cpu().numpy()
    assert np.allclose(white, 1.0, atol=1e-3)
    red = yuv420_to_rgb(frame(63, 102, 240), H, W).cpu().numpy()[:, 0, 0]    # BT.709 limited-range red
    assert red[0] > 0.95 and red[1] < 0.05 and red[2] < 0.05


# ---------------------------------------------------------------- nadir patch
def test_nadir_patch_covers_cap_only():
    import torch
    from osvcut import view_dirs, nadir_patch
    d = view_dirs("equirect", 64, 32)
    torch.manual_seed(0)
    img = torch.rand(3, 32, 64, device=d.device)
    out = nadir_patch(img, d, 20.0)
    el = torch.rad2deg(torch.asin((-d[..., 2]).clamp(-1, 1)))
    above = el < 69                                     # outside cap + 1 deg soft edge: untouched
    assert torch.allclose(out[:, above], img[:, above])
    pole = el > 85                                      # near the pole: smooth fill, little variance
    assert out[:, pole].std(1).max() < 0.1 and img[:, pole].std(1).min() > 0.2


# ---------------------------------------------------------------- colour grade
def test_grade_identity_and_exposure():
    import torch
    from osvcut import grade
    x = torch.rand(3, 8, 8)
    assert torch.equal(grade(x, {}), x)
    y = grade(torch.full((3, 2, 2), 0.25), {"exposure": 1.0})
    assert torch.allclose(y, torch.full_like(y, 0.25 * 2 ** (1 / 2.2)), atol=1e-4)    # +1 EV doubles linear light
    g = grade(torch.tensor([0.2, 0.5, 0.8]).view(3, 1, 1), {"saturation": -1.0})
    assert float(g.std()) < 1e-6                                                         # fully desaturated


# ---------------------------------------------------------------- spherical metadata
def test_spherical_injection_both_layouts():
    import shutil, subprocess, tempfile
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("     (skipped: ffmpeg not found)")
        return
    from spherical import inject
    d = tempfile.mkdtemp()
    src = os.path.join(d, "a.mp4"); fast = os.path.join(d, "b.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=256x128:rate=10:duration=1",
                    "-c:v", "libx264", src], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-c", "copy", "-movflags", "+faststart", fast], check=True)
    for f in (src, fast):
        inject(f)
        inject(f)                                           # idempotent
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream_side_data=projection",
                              "-of", "csv=p=0", f], capture_output=True, text=True).stdout
        assert "equirectangular" in out, out
        n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", f], capture_output=True, text=True).stdout
        assert n.strip().rstrip(",") == "10", n


# ---------------------------------------------------------------- sequence transitions
def test_crossfade_length():
    import shutil, subprocess, tempfile
    if not shutil.which("ffmpeg"):
        print("     (skipped: ffmpeg not found)")
        return
    from sequence import crossfade, duration
    d = tempfile.mkdtemp()
    parts = []
    for i, col in enumerate(("red", "blue", "green")):
        p = os.path.join(d, f"p{i}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color={col}:size=160x90:rate=25:duration=1",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", p], check=True)
        parts.append(p)
    out = os.path.join(d, "x.mp4")
    crossfade(parts, out, 25, "fade", 0.3)
    assert abs(duration(out) - (3 - 2 * 0.3)) < 0.1, duration(out)


# ---------------------------------------------------------------- one-click edit: segment suggestions
def test_suggest_talk_calm_whip():
    from autoedit import suggest
    tx = {"segments": [{"start": 1.0, "end": 2.0, "text": "a"}, {"start": 2.5, "end": 3.5, "text": "b"},
                       {"start": 20.0, "end": 21.0, "text": "c"}]}
    rate = 10.0
    speed = [5.0] * 100 + [80.0] * 10 + [5.0] * 90          # calm, a 1-s whip at 10 s, calm again
    motion = {"rate": rate, "speed": speed, "shake": [1.0] * len(speed), "tilt": [0.0] * len(speed)}
    S = suggest(motion, tx, 20.0)
    kinds = [x["kind"] for x in S]
    talk = [x for x in S if x["kind"] == "talk"]
    assert talk[0]["start"] == 1.0 and talk[0]["end"] == 3.5 and talk[0]["text"] == "a b"   # merged (gap < 1.5 s)
    assert len(talk) == 2 and "whip" in kinds and "calm" in kinds
    w = [x for x in S if x["kind"] == "whip"][0]
    assert 9.5 <= w["start"] <= 10.0 and 11.0 <= w["end"] <= 11.4
    assert all(x["end"] - x["start"] <= 8.0 + 1e-9 for x in S if x["kind"] == "calm")


# ---------------------------------------------------------------- captions and title cards
def test_srt_time_and_caption_mapping():
    import shutil, subprocess, tempfile
    from osvcut import _srt_time, add_captions
    assert _srt_time(0) == "00:00:00,000" and _srt_time(3723.4567) == "01:02:03,457"
    if not shutil.which("ffmpeg"):
        print("     (skipped: ffmpeg not found)")
        return
    d = tempfile.mkdtemp()
    v = os.path.join(d, "v.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=gray:size=320x180:rate=25:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", v], check=True)
    caps = [{"start": 9.0, "end": 9.5, "text": "before"}, {"start": 10.5, "end": 11.5, "text": "inside"},
            {"start": 12.8, "end": 14.0, "text": "tail"}, {"start": 20, "end": 21, "text": "after"}]
    n = add_captions(v, caps, 10.0, 13.0, speed=0.5, mode="soft", size=(320, 180))
    srt = open(v[:-4] + ".srt", encoding="utf8").read()
    assert n == 2 and "inside" in srt and "tail" in srt and "before" not in srt and "after" not in srt
    assert "00:00:01,000 --> 00:00:03,000" in srt          # (10.5-10)/0.5 .. (11.5-10)/0.5


def test_title_card_burns_and_keeps_length():
    import shutil, subprocess, tempfile
    if not shutil.which("ffmpeg"):
        print("     (skipped: ffmpeg not found)")
        return
    from osvcut import add_title
    d = tempfile.mkdtemp()
    v = os.path.join(d, "v.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:size=320x180:rate=25:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", v], check=True)
    add_title(v, "Title", (320, 180), seconds=2.0, subtitle="sub")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", v],
                               capture_output=True, text=True).stdout)
    assert abs(dur - 3.0) < 0.1, dur
    def peak_at(t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", v, "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "gray", "-"], capture_output=True).stdout
        return max(raw) if raw else 0
    assert peak_at(1.0) > 200 and peak_at(2.6) < 40          # white text on black during the card, none after
    assert not os.path.exists(v[:-4] + "_title.ass")


# ---------------------------------------------------------------- beat grid
def test_beat_grid_on_a_metronome():
    import shutil, subprocess, tempfile
    if not shutil.which("ffmpeg"):
        print("     (skipped: ffmpeg not found)")
        return
    from beats import beat_grid
    d = tempfile.mkdtemp()
    click, track = os.path.join(d, "c.wav"), os.path.join(d, "m.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.05",
                    "-af", "apad=pad_dur=0.4", click], check=True)                      # 0.45 s per click: 133.3 BPM
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-stream_loop", "40", "-i", click, "-t", "15",
                    "-af", "adelay=200|200", track], check=True)
    g = beat_grid(track)
    assert abs(g["period"] - 0.45) < 0.005, g
    assert min(abs(g["phase"] - 0.2), abs(g["phase"] - 0.2 - 0.45)) < 0.03, g


# ---------------------------------------------------------------- search normalisation
def test_traditional_simplified_folding():
    from textnorm import norm
    assert norm("這個鳥") == norm("这个鸟")
    assert norm("我們說 Boat") == "我们说 boat"


def test_hallucination_filter():
    from textnorm import clean_segments
    segs = [{"text": "You"}, {"text": "Thanks for watching!"}, {"text": "Eu me assino!"},
            {"text": "Where's our boat?"}, {"text": "You are right"},
            {"text": "uh", "no_speech": 0.9, "logprob": -1.2}, {"text": "ok", "no_speech": 0.9, "logprob": -0.3}]
    assert [s["text"] for s in clean_segments(segs)] == ["Where's our boat?", "You are right", "ok"]


# ---------------------------------------------------------------- frame interpolation
def test_flow_interpolation_moving_texture():
    """A textured image shifted by 8 px: the flow midpoint is close to the true 4-px shift, unlike a cross-fade."""
    import numpy as np
    import torch
    import cv2
    from interp import Interpolator
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.random((200, 320)).astype(np.float32), (0, 0), 2.0)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    def shifted(dx):
        return torch.from_numpy(np.ascontiguousarray(tex[40:160, 40 - dx:280 - dx]))[None].repeat(3, 1, 1)
    a, b, mid = shifted(0), shifted(8), shifted(4)
    out = Interpolator(scale=1.0).between(a, b, [0.5])[0]
    inner = (slice(None), slice(10, -10), slice(20, -20))
    e_flow = float((out - mid)[inner].abs().mean())
    e_blend = float(((a + b) / 2 - mid)[inner].abs().mean())
    assert e_flow < 0.3 * e_blend, (e_flow, e_blend)


# ---------------------------------------------------------------- runner
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
