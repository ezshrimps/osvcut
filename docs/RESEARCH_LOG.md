# osvcut research log

Structured record of every experiment, written to be turned into a paper.
Each entry: **Question → Method → Data → Result → Conclusion → Reproduce**.
Hardware for all timings: RTX 4070 Ti (12 GB), Windows 11, PyTorch 2.10 + CUDA 12.6, ffmpeg 8.0.1.

## Dataset

Private corpus of 239 `.OSV` clips (431.7 GB, 8.08 h) from one DJI Osmo 360, 2025-09 → 2026-04:
226 clips in 4K mode (2 × 1920² HEVC Main10 fisheye, 100 fps, AAC), 12 in 8K mode (2 × 3840², 29.97 fps, no audio), 1 4K without audio.
Scenes: outdoor travel (sun, sky, hiking, crowds), indoor (restaurants, casinos, arcades), night events, sports.
Mostly handheld on an invisible selfie stick. Footage is not redistributable (people's faces); the code and the metrics are.

Clips used repeatedly below:
- **Y** – Yosemite falls, 10.2 s, 4K, crowd close to the camera, selfie stick tilted.
- **H** – Hawaii trail, 49.2 s, 4K, harsh backlight, friends walking within 0.3–1 m.
- **K** – Hong Kong restaurant, 10.0 s, 8K, static, table 0.3 m below.

---

## E1 — The OSV container is plain ISO-BMFF

**Q.** Can footage be read without vendor SDKs?
**M.** ffprobe all 239 clips; dump the `djmd` track; decode protobuf without a schema (`djmd.py`), cross-check field meanings against OpenOSV `docs/FORMAT.md`.
**R.** ffmpeg demuxes every clip: two HEVC Main10 tracks (slave = stream 0, optical axis −Y; master = stream 1, +Y), AAC, two `djmd` metadata tracks, two `dbgi` tracks, MJPEG cover. Sample 0 of the primary `djmd` track holds 24 calibration slots (Kannala-Brandt fx, fy, cx, cy, k1..k5 in a 3840² calibration space, extrinsic quaternion, 14-point occlusion polygon); every frame holds a fused attitude quaternion and accelerometer vector. The schema name in our firmware is `dvtm_oq101.proto`, not `dvtm_osmo360.proto`, with the same layout. 4K mode maps calibration to stream pixels with `s = digital_focal_length / fx` (530.79 / 1047.95 = 0.5065) about the image centre.
**C.** OSV is ordinary MP4 + protobuf; a scriptable pipeline needs only ffmpeg + a small decoder.
**Reproduce.** `python djmd.py <clip>`.

## E1b — Reading per-frame metadata fast

**Q.** ffprobe `-show_data` hex dumps are too slow for whole clips (minutes for long clips).
**M.** Stream-copy the `djmd` track (`ffmpeg -map 0:<djmd> -c copy -f data -`) and parse the concatenation as **one** protobuf message: every sample contributes exactly one top-level field 3 (FrameMeta), so frames split for free.
**R.** 10 s clip: 0.13 s (vs. several seconds); 19-min clip (115 443 frames): 22 s, then cached.
**Reproduce.** `djmd.frame_attitudes()`.

## E2 — IMU axes vs. lens body axes

**Q.** The per-frame attitude quaternion is in IMU axes; how do they map to the lens body frame (X right, Y forward = master, Z up)?
**M.** (a) Check R(q)ᵀ·ẑ against the normalised accelerometer vector. (b) Render one levelled equirect frame of Y for all 24 signed axis permutations of the up vector and pick the one with vertical trees and a straight horizon.
**R.** (a) R(q)ᵀ·ẑ = (0.639, 0.321, 0.699) vs. −acc = (0.639, 0.367, 0.676): consistent, so R maps IMU → world. (b) Exactly one candidate is level: **body = (−imu_y, −imu_x, imu_z)**. The map has det = −1 (a reflection), which is harmless because only directions are mapped.
**C.** Horizon levelling from metadata alone, no image analysis; heading follows a 1 s moving average of the forward vector.
**Reproduce.** `samples/grid.jpg` generation in the session notes; `osvcut.body_frames`.

## E3 — Root causes of visible seams and ghosting

**Q.** Why does a calibrated stitch still show seams and ghosts?
**M.** Unwrap both lenses along the seam band (±8° around the body y = 0 plane, 0.1°/px), Farneback flow between them per 15° azimuth block, luminance ratio per block (`seamdiag.py`).
**R.**
- Far content (ridges, clouds) aligns: calibration is correct at infinity.
- Misalignment median 0.77–0.85°, max 2.7° — all maxima on near objects (hand on the stick, people < 1 m). Calibration has rotation only, no inter-lens translation ⇒ **parallax**. DJI's "stitching distance" presets (0.7–1.6 m) differ only by ~0.2 % in focal length.
- **Exposure/colour mismatch**: slave/master luminance ratio 0.53–1.40 per azimuth (H, backlit).
- **Occlusion**: the 14-point polygon traces the camera body / handle at the bottom of both images; usable radius drops to ~91.5° there, and both lenses' occluded sectors face the same direction ⇒ overlap shrinks from ~10° to ~1° on the handle side. The occlusion edge is a 1–2° soft dark gradient.
- **Vignetting refuted**: the median radial luminance profile of H is flat (±4 %) out to 96°.
**C.** Four causes: parallax (physical), wide crossfade turning misalignment into ghosts, dark occlusion edge blended in, inter-lens exposure mismatch.
**Reproduce.** `python seamdiag.py <clip> <t> <tag>`.

## E4 — Smart seam (masks + gains + DP cut)

**M.** (1) Validity mask per lens = image circle (96.5°) ∩ coded frame ∩ inside the occlusion arc, eroded 1° and feathered; rays outside are clamped to the usable edge (blind-zone fill). (2) Per-azimuth (5° bins) slave/master ratio in linear light on the overlap, circularly smoothed, temporally EMA'd; global part applied to the whole hemisphere, local part faded out by 45° from the seam. (3) Min-cost closed path through a ±4° band (colour + gradient difference, invalid-pixel penalty, centre prior, temporal term), 1° blend.
**R.** Semi-transparent ghosts, dark smears and the sky step disappear; near objects crossing the seam are now *cut / duplicated* instead of ghosted. Update 10 ms, render 48 ms per 3840×1920 frame.

## E5 — Optical-flow morphing; DIS vs. Farneback on seam strips

**Q.** Can parallax be absorbed by warping both lenses toward each other across a transition zone?
**M.** Flow F between the two lens strips (A(p) ≈ B(p+F)); output at transition t∈[0,1]: A sampled at p − tF, B at p + (1−t)F, blended (1−t)/t. Forward–backward consistency → confidence; normalised convolution fills low-confidence areas; |F| ≤ 4°.
**R.** With OpenCV **DIS** (any preset) the flow had a systematic +2.2 to +3.4° across-seam bias on 90–120-row strips, even on clouds, producing streaks. **Farneback** on the same strips: 0.15–0.52° median on far content, as expected. With Farneback: no streaks; walking people stay single; hands are continuous.
**C.** Use Farneback for narrow strips (DIS's coarse-to-fine pyramid is unreliable when one image dimension is tiny).
**Reproduce.** `compare_seam.py`, debug in session notes (`_dbg_flow2.py`).

## E6 — Overlap-adaptive transition

**Q.** Residual streak near the handle side.
**M.** Coverage map (`tools_coverage.py`): which lens is valid per output pixel.
**R.** No blind zone on Y, but the both-valid band is ~1° wide on the handle side vs ~10° elsewhere. A fixed ±3° ramp pushed warped rays into invalid areas. Fix: per-azimuth overlap interval [lo, hi] from the masks; ramp centred in it with half-width min(3°, (hi−lo)/2); flow scaled by the available half-width.
**C.** The walking person on the handle side is restored except for a small arm streak.

## E7 — A seam flicker metric needs a no-seam reference

**Q.** Do the new seams flicker?
**M.** Ratio of mean |frame difference| inside |β|<2.5° to that in −12°<β<−5° (same side, same frames), on 300 frames of Y. Reference: a render from the slave lens only (no seam at all).
**R.**

| Method | Flicker ratio |
|---|---|
| Single lens, no seam (reference) | 1.519 |
| Legacy 12° crossfade | 1.155 |
| Smart cut | 1.516 |
| Flow morph (0.1°) | 1.422 |
| Flow morph (0.2°, current) | 1.402 |

Stronger temporal smoothing did not change the numbers (1.341 / 1.247).
**C.** The legacy value is *below* the reference because ghosting low-passes the seam band, not because it is stable. Against the proper reference, neither new method adds flicker. Always report seam metrics relative to a single-lens reference.
**Reproduce.** `tools_flicker.py <clip> <videos…>`.

## E8 — Flow resolution vs. speed

**R.** Flow strip at 0.2°/px instead of 0.1°/px and skipping the unused DP: update 141 → 29 ms/frame, no visible difference on Y or H. Full flow render ≈ 0.13 s/frame at 3840×1920 (contended GPU).

## E9 — Rolling shutter is not a factor (negative result)

**Q.** Does seam misalignment grow with angular speed (rolling-shutter mismatch between the two sensors)?
**M.** 40 frames across H; median and p90 |flow| over confident overlap pixels vs. IMU angular speed (±5 frames).
**R.** Speeds 5.5–532.6 °/s; median flow 0.04–0.21°; Pearson r = 0.09; slope ≈ 0.0000° per °/s.
**C.** The two sensors are effectively synchronised in readout; no RS correction is needed for the seam.
**Reproduce.** `python tools_rs_probe.py <clip> 40`.

## E10 — Subject following on stitched views

**M.** Render a rectilinear view (fov = 3× the subject box, 35–100°) around the subject each frame with the same levelling as exports; OpenCV MIL tracker (no external weights); re-aim and re-initialise when the box centre leaves the central 36 %; smooth 0.6 s; keyframes every 0.25 s with linear easing.
**R.** H 43.5→49 s: subject stays boxed through partial occlusion by another person's hand; 5.5 s of footage tracked in 8.5 s.
**Open.** Quantitative tracking accuracy needs annotated boxes.

## E11 — Multi-clip seam benchmark (v2)

**Q.** Which seam method is best across the corpus, especially where parallax is real?
**M.** `eval_seams.py`: 20 clips drawn with a fixed seed from the 239 (duration > 3 s), 5 evenly spaced frames each (n = 100). Metrics on the body-frame seam strip (±5°, 0.1°/px), only where both lenses are valid:
- **CD (contribution disagreement)**: Σ 4·w_A·w_B·|Y_A − Y_B| / Σ 4·w_A·w_B over the two (warped, gain-compensated) contributions — how much the two sources disagree wherever they are mixed. Method-agnostic: covers crossfades, cut lines and morph ramps alike.
- **Fidelity**: mean over |β|<3° of max(SSIM(out, slave-only), SSIM(out, master-only)). Biased toward cut methods (they copy source pixels by construction); reported for completeness.
- **Edge deviation**: |log(E_out / E_ref)| of across-seam gradient energy in |β|<1.5° vs the single-lens references.
- **Excess exposure step**: luminance step across the seam beyond the better single lens's natural step.
- **Hard subset**: azimuth columns whose median Farneback |flow| between the lenses exceeds 0.4° (dilated ±2°) — where parallax is visible.
- Runtime per 3840×1920 frame (update + render).

**R.** Means with 95 % bootstrap CIs:

| method | CD ↓ | CD hard ↓ | edge hard ↓ | fidelity hard ↑ | step ↓ | ms |
|---|---|---|---|---|---|---|
| legacy 12° crossfade | 0.1007 [.0930,.1087] | 0.1334 [.1266,.1399] | 0.1882 [.1808,.1957] | 0.853 | 0.0298 | 23 |
| smart (DP cut) | 0.0706 [.0663,.0748] | 0.0923 [.0883,.0966] | **0.0517** [.0438,.0601] | **0.967** | 0.0232 | 146 |
| flow, ramp centred in overlap | 0.0853 [.0791,.0914] | 0.1187 [.1129,.1246] | 0.0758 | 0.870 | 0.0275 | 169 |
| flow + DP path, ramp ±3° | 0.0711 [.0661,.0764] | 0.0972 [.0924,.1020] | 0.0863 | 0.884 | **0.0229** | 180 |
| **flow + DP path, ramp ±1.5°** | **0.0629** [.0588,.0670] | **0.0836** [.0798,.0875] | 0.0653 [.0563,.0747] | 0.929 | 0.0245 | 180 |

Paired (Wilcoxon signed-rank), flow+DP ±1.5° vs smart: CD better on 79/100 frames (p = 1.6e-13), CD-hard 82/100 (p = 1.7e-10); edge-hard worse on 56/100 (mean +0.014, p = 0.022); fidelity worse on all (expected bias). Vs legacy: better on every metric except step (n.s., p = 0.25).

**C.** Aligning both lenses half-way with the flow and placing a *narrow* morph ramp on the minimum-disagreement closed path gives the lowest disagreement overall and on the parallax-heavy subset; a wide ramp centred in the overlap is clearly worse (the ramp crosses regions where the two lenses differ for reasons flow cannot fix: edge MTF, chromatic aberration, noise). Pure DP cuts keep slightly crisper edges. Default is now flow + DP, ±1.5°.
**Limitations.** No ground truth for a centre-of-rig view; metrics are proxies. A perceptual study (pairwise preference on seam crops) is the next step.
**Reproduce.** `python eval_seams.py --clips 20 --frames 5 --seed 0 --out results/seam_bench_v2.json && python analyze_bench.py results/seam_bench_v2.json flow_dp_r15`.

## E12 — How much of the lens-to-lens residual is misalignment?

**M.** `tools_flow_residual.py`: textured core pixels, |A − B| with no warp / raw flow / processed flow; strips sampled at 0.2°/px directly vs. at 0.05°/px then area-downsampled.
**R.** Y: 38.4 → 34.1 (raw flow) gray levels; anti-aliased strips 36.0 → 32.0. H: 33.9 → 28.9; anti-aliased 29.7 → 25.5. Processed (smoothed, confidence-filled, clamped) flow keeps about half the gain.
**C.** Most of the remaining lens-to-lens difference on texture is not geometric (edge-of-field blur, chromatic aberration, sensor noise, aliasing); this is why the seam should run where the two lenses *agree* rather than through the overlap centre (motivates E11's DP placement).

## E13 — Temporal stability of the default (flow + DP, ±1.5°)

**M.** Same protocol as E7 on Y (300 frames).
**R.** single-lens reference 1.510 · smart cut 1.507 · flow centred 1.395 · **flow + DP ±1.5° 1.361**.
**C.** Moving the ramp onto a per-frame DP path (temporal term λ_t = 0.6, EMA 0.5) adds no flicker relative to the no-seam reference.

## E14 — Perceptual pairwise study (protocol; data collection open)

**M.** Studio tab 「拼缝盲测」 (`blindtest.py`): random clip, random time, azimuth on the seam ring drawn 80 % from parallax-heavy columns (E11 definition); a 720×540, 55° rectilinear close-up centred on the seam, upright (body Z up), rendered by two of {legacy, smart, flow}; left/right randomised; the rater answers left / same / right (keys ← ↓ →). Method names are hidden until the stats view. Votes are appended to `cache/blindtest/votes.jsonl` with clip, time, azimuth and the hidden methods; win rates and Bradley–Terry scores (ties = ½) are computed by `blindtest.stats()`.
**Status.** Waiting for human raters (the author of the methods does not vote).

## E15 — Gravity-aligned ground-plane parallax prior (negative result)

**Q.** The upright camera's seam ring passes under the user, where the ground is 1–1.5 m away. Can a ground plane (known "down" from the IMU) predict the lens-to-lens parallax, so the narrow handle-side overlap can be corrected without flow?
**M.** `tools_ground_probe.py`: on two pickleball-court clips (8 frames each), confident raw Farneback flow on the seam strip below −15° elevation; fit flow_β = k·sin(−elevation) + c (sin(−el)/h ∝ 1/depth for a plane at height h).
**R.** R² = 0.00–0.07 per frame; k = +0.155 ± 0.073 on one clip and −0.159 ± 0.157 on the other (sign flips); median |flow_β| on measurable ground 0.07–0.26°.
**C.** Where the overlap is wide enough to measure, ground parallax is small and not explained by a plane model; the visible ground breaks sit in the ~1° handle-side overlap where neither flow nor a weak geometric prior has support. Not pursued; the handle sector is a hardware limit (both lenses are occluded there).

## E16 — Multi-clip temporal stability

**M.** `eval_temporal.py`: 8 clips (seed 1), a random 4-s window each, decoded at 25 fps and rendered in memory to 1920×960 levelled equirect by every method and by the slave lens alone (no seam). Flicker ratio as in E7; reported as the difference to the single-lens reference of the same clip.
**R.** Excess flicker, mean over clips [min, max]: legacy −0.206 [−0.326, −0.096] (0/8 above reference); smart cut −0.032 [−0.272, +0.072] (3/8); flow + DP ±1.5° −0.039 [−0.200, +0.071] (3/8).
**C.** Confirms E7/E13 across clips: neither new seam adds measurable flicker (worst case +0.07 on a ratio of ≈1); the feathered stitch's lower value is blur, not stability.
**Reproduce.** `python eval_temporal.py --clips 8 --seconds 4 --seed 1`.

## E17 — Renderer throughput

**M.** 10 s of H rendered to a 2880×1440 equirect at 30 fps with the flow seam; wall-clock including decode and NVENC encode.
**R.** Original serial loop 39.8 s → threaded decode/encode + GPU yuv→rgb 30.8 s (NVDEC vs CPU decode made no difference: 30.8 vs 31.2 s, the loop is analysis-bound) → seam analysis every 2nd frame 23.9 s → **analysis in a worker thread, every frame, render uses the latest finished analysis (≤ 1 frame behind) 17.2 s** (2.3× faster than the original). Flicker on Y with async analysis 1.394 vs 1.361 synchronous (reference 1.510).
**Profile (per 2880×1440 frame, synchronous):** decode 22 ms (CPU) / 14 ms (NVDEC), upload 6 ms, flow analysis 48 ms (two Farneback passes on 1800×46 strips + DP), render 26 ms, download 4 ms.

## E18 — Rolling shutter revisited with a per-pixel model (negative result)

**Q.** E9 used the median flow over the whole ring, where an azimuth-dependent rolling-shutter effect could cancel. Does a per-pixel model explain the seam flow?
**M.** `tools_rs_model.py`: for each confident seam pixel with body direction u, the two lenses read it at rows y0, y1; with readout time τ the second lens sees it Δt = τ(y1−y0)/H later, so the predicted lens-to-lens displacement is −(ω×u)Δt with ω the body angular velocity from the IMU. Fit τ by least squares over ~1.5 M pixels from 30 frames; control: ω taken from shuffled frames.
**R.** Beach clip (median 33°/s, max 94°/s): τ = +4.7 ms, R² = −0.002 (control R² = −0.002). Trail clip (median 18°/s, max 585°/s): τ = −0.8 ms, R² = 0.000 (control 0.000). RMS measured flow 0.21–0.24° is unchanged by the model.
**C.** Confirms E9 with a model that cannot cancel: rolling shutter does not contribute measurably to seam misalignment on this camera.

## E19 — No systematic calibration bias along the seam

**Q.** A ~2° far-field step seen near the handle-side sector in one frame: calibration error at extreme field angles?
**M.** `tools_calib_bias.py`: per 10° azimuth bin, median confident flow in each of 30 frames; across frames: mean, s.d., t-statistic. A calibration error keeps its sign from frame to frame.
**R.** RMS of the per-bin mean bias 0.044° (beach) and 0.039° (trail), vs. per-frame s.d. ≈ 0.12°; bins with |t| > 4: 0/36 and 1/36.
**C.** The factory calibration is accurate around the whole ring; large local misalignments are frame-specific (near objects, water glints), i.e. parallax/appearance, not geometry. Online extrinsic refinement is not needed for this unit.

## E20 — Seam quality in the live browser preview

**M.** `proxy.py` now also samples the min-difference DP seam path (2° bins) and the per-azimuth exposure field (5° bins + global) at 2 Hz from the proxy itself (calibration rescaled to proxy pixels), quantised to bytes (98 samples, 88 KB for 49 s; +3 s generation time). The shader looks both up by playback time (linear between samples) and applies a 1° cut at β*(α) with local gains, instead of a fixed ±1.5° crossfade at β = 0.
**R.** Rendered views differ (mean |Δ| 12.8 grey levels at seam-centred views); the cut moves off near-field content (|β*| up to 2.7°). Parallax morphing remains export-only.

## E21 — Anti-aliased flow strips (negative result)

**M.** FlowStitch(ss=2): sample the flow strip at 0.1° and average-pool to 0.2° before Farneback (motivated by E12). Benchmark protocol of E11, n = 100, paired against the default.
**R.** CD 0.0628 vs 0.0629 (62/100, mean −0.0001, p = 0.028); CD-hard −0.0002 (p = 0.10); edge-hard −0.0007 (p = 0.5); fidelity-hard −0.0010 (p < 0.001).
**C.** Statistically detectable but practically nil; not adopted (default ss = 1). Aliasing matters for residual measurements (E12) but not for where the seam goes.

## E22 — An AI assistant edits through MCP only (case study)

**M.** Acting as an assistant with only the MCP tools (`mcp_server.py`): `clip_overview` on a beach clip (contact sheet + transcript: "Oh, that's a bird! … That's a freaking ugly pigeon"), `look` at t = 11 s, yaw 40° → subject slightly left, re-aimed to 25°; `track_subject` on the trail clip from (43.5 s, yaw 78°, pitch −3°) → 22 keyframes, followed to the end; `render_sequence` with a little-planet intro (dist 1 → 0.15 → 0) on the beach cut and the tracked keys on the trail cut, 9:16, 30 fps.
**R.** 11.5 s 1080×1920 film rendered and joined without manual edits; look calls 0.5–1.1 s, overview 2.5 s (cached transcript), tracking 5.5 s of footage in ~9 s. Found and fixed: tracked keys must carry linear easing when consumed outside the studio.
**C.** The tool surface (overview → look → track → render) is sufficient for an assistant to make framing decisions it can verify visually before rendering — the step that failed in the first, contact-sheet-only experiment (subject out of frame).

---

## Open questions / next experiments

1. ~~Multi-clip benchmark~~ → E11. Next: perceptual pairwise study on seam crops; DJI Studio / OpenOSV exports as external baselines.
2. Temporal metrics (flicker) on the benchmark clips for every method.
3. Near-field parallax on the handle side (1° overlap): depth-aware or temporal fill.
4. Tracking accuracy benchmark.
