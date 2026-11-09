# osvcut

**Open-source toolkit for editing DJI Osmo 360 `.OSV` footage like ordinary MP4 — built so that AI agents (and people) can read, stitch, reframe and cut 360° video from scripts.**

> Status: research prototype, under active development. Windows + NVIDIA GPU tested.

## Why

DJI Studio (and plug-ins such as OpenOSV) are GUI-first. To let an AI edit 360 footage you need:

- a **plain-file pipeline**: OSV in → stitched / levelled / reframed MP4 out, driven by a JSON spec;
- an **AI-readable view** of a clip: levelled panorama contact sheets with yaw ticks + speech transcript;
- a **browser studio** to look around, set camera keyframes and export, with the same projection maths as the renderer.

## What works today

| Area | Status |
|---|---|
| OSV parsing | ffmpeg reads OSV directly (2 × HEVC Main10 fisheye + AAC + `djmd` protobuf). `djmd.py` is a schema-free protobuf decoder: lens calibration (Kannala-Brandt 5), extrinsics, per-frame IMU attitude |
| Stitching | Calibrated dual-fisheye → equirect on the GPU (PyTorch). 4K (1920²/lens) and 8K (3840²/lens) modes |
| Seam engine (`seam.py`) | Lens-validity masks (image circle + body/handle occlusion arc + frame edges), per-azimuth exposure/colour compensation in linear light, **both lenses half-aligned with optical flow and a ±0.75° morph along a minimum-disagreement closed path** (default) or min-difference DP seam; single-lens fast path outside the seam band |
| Stabilisation | Horizon levelling from the IMU (empirically derived IMU→body axis map) + smoothed heading |
| Reframing | Virtual camera keyframes (yaw / pitch / fov, smoothstep easing) → 16:9 / 9:16 / 1:1 MP4 via NVENC |
| Studio (`server.py` + `viewer/`) | Library of every OSV under a folder with stitched, levelled panorama thumbnails; one-click preview generation (3 quality presets) with progress; WebGL 360 viewer (perspective / wide / little-planet / crystal-ball), minimap with view footprint, horizon grid, seam-inspection overlay, A/B seam comparison |
| Geometry note | Calibration intrinsics are stored for a 3840-px lens image; the stream scale is `W/3840` (`stitch.lens_scale`). Reading `digital_focal_length/fx` instead (1.3 % larger) misregisters the lenses by ~2.8° around the seam — see `docs/RESEARCH_LOG.md` E29 |
| Live preview | Opening any clip auto-builds a dual-fisheye proxy (GPU decode/scale, ~4× real time) plus a rig JSON (lens model, usable radii, exposure balance, 15 Hz levelling quaternions); the WebGL shader stitches and levels it in real time, including the flow morph (5 Hz per-azimuth displacement track in the rig; `B` toggles morph / cut). Exports use the full flow seam |
| Timeline | IMU motion curve, shake markers, speech segments (faster-whisper transcript, clickable), seam-difficulty band (lens-to-lens parallax), keyframes, in/out points (`I` / `O`), undo / redo (`Ctrl+Z` / `Ctrl+Y`) |
| Stills | `P` in the studio: the current moment and view rendered server-side with the full flow seam at 3840 px on the long side (aspect, grade and nadir patch as set), ~2.3 s per 4K still; also `GET /api/clip/<id>/view` |
| Subject follow | Shift+drag a box around a person → GPU-rendered tracking views + OpenCV MIL tracker with automatic re-aiming → dense linear keyframes, previewed live and exportable |
| Export | Keyframed yaw / pitch / fov / distortion (little planet etc.) to 16:9 / 9:16 / 1:1 / 4:5 / 2.39:1 MP4, slow motion / fast forward (optical-flow frame interpolation below the source frame rate, `interp.py`), multi-clip sequences with transitions (fade / fade-to-black / wipe / circle) and background music, burned-in or soft captions from the transcript, colour grade (exposure / contrast / saturation / temperature, identical in the live preview), freeze orbit ("bullet time": the picture stops while the camera turns, silence inserted), local slow motion of a moment, nadir patch over the hand / stick, or a full levelled 360 MP4 with spherical metadata (YouTube / VR), also as a batch per library group |
| One-click edit | Per library group: picks talk segments (framed on the selfie-stick holder from the IMU, captioned) and steady scenery (slow pan), adds transitions and music (cuts snapped to the music's beat grid, `beats.py`) → a short film draft (`autoedit.py`, studio dialog, MCP `auto_edit`) |
| Metrics | `tools_flicker.py` – seam flicker vs. a single-lens no-seam reference; `seamdiag.py` – per-azimuth misalignment and exposure ratio along the seam; `tools_scale_check.py`, `tools_ring_disparity.py` – stream-scale check and wide-search across-seam disparity |

## Results (seam research, 100 frames from 20 clips)

| Method | Contribution disagreement ↓ | CD, parallax-heavy ↓ | Edge dev. ↓ | Exposure step ↓ | Flicker vs. no-seam ref. |
|---|---|---|---|---|---|
| Feathered crossfade | 0.064 | 0.094 | 0.229 | 0.017 | −0.211 (blur) |
| Min-difference DP cut | 0.045 | 0.068 | **0.029** | 0.011 | −0.033 |
| **osvcut flow seam** | **0.025** | **0.036** | 0.034 (n.s.) | **0.009** | −0.014 |

The flow seam's advantage grows with parallax (CD ratio to the cut 0.76 → 0.42 from zero to ~0.9°). The largest single improvement was a geometry fix: the calibration-to-stream scale is `W/3840`, not `digital_focal_length/fx` (E29). Full protocol, negative results and every number: [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) (E1–E50), [`REPORT.md`](REPORT.md) (Chinese), [`paper/`](paper/) (arXiv draft).

## Quick start

Requirements: Python 3.10+, `torch` with CUDA, `opencv-python`, `numpy`, ffmpeg/ffprobe (with `h264_nvenc`) on PATH. Optional: `numba` (≈50× faster seam path search; a pure-numpy fallback gives identical results).

```bash
# studio: library + 360 viewer + render queue
python server.py 8360 "E:\Videos"
# open http://localhost:8360/viewer/

# render from a JSON spec
python osvcut.py my_cut.json
```

Example spec:

```json
{
  "src": "E:/Videos/clip/CAM_20251127120236_0033_D.OSV",
  "in": 36, "out": 49.2,
  "proj": "flat", "size": [1920, 1080], "fps": 50,
  "seam": "flow", "stabilize": "level",
  "keys": [
    {"t": 0,   "yaw": 165, "pitch": 8,   "fov": 70},
    {"t": 4,   "yaw": 10,  "pitch": -15, "fov": 105}
  ],
  "dst": "out.mp4"
}
```

`proj`: `flat` | `equirect` · `seam`: `flow` (default) | `smart` | `legacy` · `stabilize`: `level` | `lock` | `none`

Optional: `spherical` (with `proj: equirect`: tag the MP4 as 360 video for YouTube / VR players), `speed` (0.25–4; slow motion uses every source frame), `cq` (quality), `seam_async` (default true), `keys[].dist` (0 rectilinear … 1 little planet), `keys[].ease` (`linear`). `nadir_patch` (degrees: cover the hand / stick below the camera), `grade` (`exposure` EV, `contrast`, `saturation`, `temperature`; same maths as the studio preview). `freeze` (`{at, dur, turn}`: freeze orbit), `slow` (`{from, to, speed}`: local slow motion, audio time-stretched), `title` (`{text, subtitle, seconds}`: centred title card with fades), `captions` (`[{start, end, text}]` in source seconds, e.g. from the transcript) with `captions_mode` `burn` (libass) or `soft` (mov_text track + .srt).

## Use it from an AI assistant (MCP)

`mcp_server.py` exposes the studio to MCP clients (Claude Code, Claude Desktop, …). Start the studio, then:

```bash
claude mcp add osvcut -- python /path/to/osvcut/mcp_server.py
```

| Tool | What the assistant gets |
|---|---|
| `list_clips` | search the library (event, file, date, duration, mode) |
| `search_speech` | find clips and times by what is said (batch Whisper transcripts) |
| `clip_overview` | levelled 360 contact sheet (image) + transcript + camera motion + suggested segments |
| `look` | a rendered view at any time and direction (or a little-planet overview) — the assistant can look around |
| `track_subject` | follow a person from (t0, yaw, pitch) → camera keyframes |
| `render_cut` / `render_sequence` | export one cut / several cuts joined (aspect, fps, speed) |
| `job_status` | progress and output path |

## Files

| File | Purpose |
|---|---|
| `djmd.py` | Generic protobuf decoder for DJI `djmd` metadata tracks |
| `stitch.py` | Calibration loader, rotation helpers |
| `seam.py` | Seam engine: masks, exposure compensation, DP seam, optical-flow morphing |
| `osvcut.py` | Renderer: spec → MP4 |
| `sheet.py` | AI-readable panorama contact sheets |
| `library.py`, `server.py`, `serve.py` | Library index, thumbnails, job queue, HTTP API |
| `analysis.py` | Motion curve + transcripts (cached) |
| `track.py` | Subject follow → camera keyframes |
| `mcp_server.py`, `look.py` | MCP server for AI assistants; single views and contact sheets |
| `proxy.py` | Dual-fisheye proxy + rig JSON for live browser stitching |
| `sequence.py` | Multi-clip sequence render + lossless concat |
| `blindtest.py` | Blind pairwise seam study |
| `eval_seams.py`, `eval_temporal.py`, `analyze_bench.py` | Benchmarks |
| `viewer/index.html` | Browser studio (single file, WebGL2) |
| `bezel.py`, `seamdiag.py`, `compare_seam.py`, `tools_*.py` | Research / diagnostics |
| `REPORT.md` | Research notes (Chinese) |
| `docs/使用指南.md` | Studio user guide (Chinese) |
| `docs/RESEARCH_LOG.md` | Experiment log E1–E20 |

## Acknowledgements

Format knowledge cross-checked against the documentation of [OpenOSV](https://github.com/Kemerd/OpenOSV) (Apache-2.0). No OpenOSV code is included.

## Licence

Apache-2.0
