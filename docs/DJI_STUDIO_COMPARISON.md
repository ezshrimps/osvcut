# osvcut vs. DJI Studio — feature comparison

Status: 2026-11-09. DJI Studio's column is taken from published reviews (sources below), **not** from running DJI Studio on
the same footage; a side-by-side on identical clips (stitch quality, export speed) is still to be done and needs DJI
Studio exports of the same clips. osvcut's column is what the code in this repository does today.

| Area | DJI Studio (per reviews) | osvcut | Notes |
|---|---|---|---|
| Stitching | In-camera / app stitching, reviewers find seams hard to see; Osmo 360 II adds "AI stitching"; one comparison reports occasional stitch lines and exposure mismatch | Calibrated stitch with a flow-aligned seam on a minimum-disagreement path, per-azimuth exposure balance, single-lens fast path; benchmarked (CD −45 % vs. a min-difference cut, 100/100 frames) and blind-tested (E58) | Same-footage comparison pending |
| Seam inspection | — | Seam overlay, A/B seam toggle, timeline seam-difficulty band (parallax), blind-test page | osvcut only |
| Horizon / stabilisation | Yes | IMU levelling from the OSV metadata, smoothed heading | |
| Reframing + keyframes | Yes | Keyframes (yaw / pitch / fov / distortion), easing, camera-move templates, undo/redo | |
| Subject tracking | "Intelligent tracking" (people, vehicles, pets) | Shift+drag a box → tracker with re-aiming → keyframes | DJI's tracker is likely stronger; ours is not benchmarked |
| Projections | Flat, little planet, etc. | Perspective, wide, little planet, crystal ball (continuous eye-offset model) | |
| Slow motion | Speed ramps; Osmo 360 II: **AI frame interpolation up to "64×"** | Frame-exact slow motion from high-fps footage, **optical-flow frame interpolation** below the source rate (0.125× / 0.0625× in the studio; +4–5 dB over cross-fading on held-out frames, E60), local slow-mo of a range, freeze orbit | DJI's is learned ("AI"); ours is classical flow — likely weaker at very large motion |
| Filters / colour | Filters | Exposure / contrast / saturation / temperature, presets, identical in preview and export | |
| Text / titles | Reviewer: missing text overlays | Title cards, burned-in or soft captions from the transcript | osvcut ahead |
| Transitions, multi-clip | Reviewer: missing transitions; background music present | Sequences with fade / wipe / circle transitions, music, project files | osvcut ahead |
| Speech | — | Transcript (faster-whisper), search the library by what was said, talk-segment suggestions | osvcut only |
| One-click edit | — | Draft film from a group of clips, beat-synced cuts | osvcut only |
| Stills | Snapshot | 4K stills with the full seam, still candidates (steadiest instants) | |
| 360 export | Up to 8K equirect (per one reviewer) | Levelled 360 MP4 with spherical metadata, batch per group; 3840×1920 (H.264) or 7680×3840 (HEVC, ~8× real time with the flow seam) | |
| Automation / AI agents | — | JSON render specs, MCP server (render, sequence, auto-edit, contact sheets, views) | osvcut only |
| Performance | Reviewers report it can be slow / buggy | Live WebGL stitch from proxies (opens in ~20 s for a 67-s clip), 360 export 1.6× real time | Same-machine timing pending |

## Gaps to close

1. Same-footage stitch comparison (needs DJI Studio exports).

## Sources

- Engadget, DJI Osmo 360 review: https://www.engadget.com/cameras/dji-osmo-360-review-a-capable-action-cam-thats-a-real-threat-to-insta360-120019592.html
- PetaPixel, DJI Osmo 360 review: https://petapixel.com/2025/10/24/dji-osmo-360-review-ready-for-adventure-against-stiff-competition
- DroneXL, Osmo 360 pre-release review: https://dronexl.co/de/2025/07/24/dji-osmo-360-pre-release-review/
- Digital Camera World, Osmo 360 II review: https://www.digitalcameraworld.com/cameras/360-cameras/dji-osmo-360-ii-review
- BGR, DJI Osmo 360 review: https://bgr.com/2195365/dji-osmo-360-review
