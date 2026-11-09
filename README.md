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
| Seam engine (`seam.py`) | Lens-validity masks (image circle + body/handle occlusion arc + frame edges), per-azimuth exposure/colour compensation in linear light, **optical-flow morphing across an overlap-adaptive transition** (default) or min-difference DP seam |
| Stabilisation | Horizon levelling from the IMU (empirically derived IMU→body axis map) + smoothed heading |
| Reframing | Virtual camera keyframes (yaw / pitch / fov, smoothstep easing) → 16:9 / 9:16 / 1:1 MP4 via NVENC |
| Studio (`server.py` + `viewer/`) | Library of every OSV under a folder with stitched, levelled panorama thumbnails; one-click preview generation (3 quality presets) with progress; WebGL 360 viewer (perspective / wide / little-planet / crystal-ball), minimap with view footprint, horizon grid, A/B seam comparison |
| Live preview | Opening any clip auto-builds a dual-fisheye proxy (GPU decode/scale, ~4× real time) plus a rig JSON (lens model, usable radii, exposure balance, 15 Hz levelling quaternions); the WebGL shader stitches and levels it in real time. Exports still use the full flow seam |
| Timeline | IMU motion curve, shake markers, speech segments (faster-whisper transcript, clickable), keyframes, in/out points (`I` / `O`) |
| Subject follow | Shift+drag a box around a person → GPU-rendered tracking views + OpenCV MIL tracker with automatic re-aiming → dense linear keyframes, previewed live and exportable |
| Export | Keyframed yaw / pitch / fov / distortion (little planet etc.) to 16:9 / 9:16 / 1:1 / 4:5 / 2.39:1 MP4, slow motion / fast forward, multi-clip sequences with background music, or a full levelled 360 MP4 with spherical metadata (YouTube / VR) |
| Metrics | `tools_flicker.py` – seam flicker vs. a single-lens no-seam reference; `seamdiag.py` – per-azimuth misalignment and exposure ratio along the seam |

## Quick start

Requirements: Python 3.10+, `torch` with CUDA, `opencv-python`, `numpy`, ffmpeg/ffprobe (with `h264_nvenc`) on PATH.

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

Optional: `spherical` (with `proj: equirect`: tag the MP4 as 360 video for YouTube / VR players), `speed` (0.25–4; slow motion uses every source frame), `cq` (quality), `seam_async` (default true), `keys[].dist` (0 rectilinear … 1 little planet), `keys[].ease` (`linear`).

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
