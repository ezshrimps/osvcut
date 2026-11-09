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
| Studio (`server.py` + `viewer/`) | Library of every OSV under a folder with stitched panorama thumbnails, one-click preview generation, WebGL 360 viewer (perspective / little-planet / crystal-ball), minimap with view footprint, horizon grid, keyframe editor, export queue with progress |
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

## Files

| File | Purpose |
|---|---|
| `djmd.py` | Generic protobuf decoder for DJI `djmd` metadata tracks |
| `stitch.py` | Calibration loader, rotation helpers |
| `seam.py` | Seam engine: masks, exposure compensation, DP seam, optical-flow morphing |
| `osvcut.py` | Renderer: spec → MP4 |
| `sheet.py` | AI-readable panorama contact sheets |
| `library.py`, `server.py`, `serve.py` | Library index, thumbnails, job queue, HTTP API |
| `viewer/index.html` | Browser studio (single file, WebGL2) |
| `bezel.py`, `seamdiag.py`, `compare_seam.py`, `tools_*.py` | Research / diagnostics |
| `REPORT.md` | Research notes (Chinese) |

## Acknowledgements

Format knowledge cross-checked against the documentation of [OpenOSV](https://github.com/Kemerd/OpenOSV) (Apache-2.0). No OpenOSV code is included.

## Licence

Apache-2.0
