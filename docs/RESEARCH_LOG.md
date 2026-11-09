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

## E23 — Two-band blending (optional)

**M.** FlowStitch(bands=2): low frequencies (Gaussian, σ = 3°, computed on a downsampled copy) cross over a ±6° ramp around the seam path, high frequencies keep the ±1.5° ramp. Paired against the default on the E11 protocol (n = 100).
**R.** CD unchanged (same contributions); edge deviation 0.0594 vs 0.0648 (79/100 frames better, p = 2.9e-6); exposure step 0.0233 vs 0.0245 (63/100, p = 0.047); fidelity −0.022 (low frequencies are mixed); edge-hard 73/100 frames better but mean +0.0003 (p = 0.016, mixed). Render overhead +2 ms after the downsampled blur (+166 ms with a full-resolution 190-tap blur).
**C.** A small, real reduction of brightness steps and seam edge energy; kept as an option (`"seam_opts": {"bands": 2}`), not the default.

## E24 — Photometric flow confidence instead of forward–backward consistency

**M.** FlowStitch(conf="photo"): confidence = exp(−(r/30)²)·clip(texture) with r the blurred residual |A − B(p+F)|, saving the backward Farneback pass. Calibration on one frame: correlation with the forward–backward confidence only r = 0.0–0.14 for any parameters (residual is dominated by non-geometric differences, E12). Paired benchmark (E11 protocol, n = 100).
**R.** Analysis 51 → 34 ms per update (total 188 → 170 ms per 3840×1920 frame). CD 0.0621 vs 0.0629 (82/100 better, p = 8.9e-10), CD-hard 80/100 better (p = 2.7e-9); edge deviation worse (0.0700 vs 0.0648, 32/100 better, p = 3.2e-4); fidelity and step n.s.
**C.** A speed/edge trade-off. Exports keep forward–backward confidence; the studio's standard preview preset uses the photometric one.

## E25 — GPU pyramidal Lucas–Kanade flow, and a caveat about the CD metric

**M.** `gpuflow.py`: dense pyramidal LK in PyTorch (3 levels on the 46-row strip, 4 iterations, Gaussian window σ = 2.5 px, Tikhonov damping 500 — undamped it diverged in flat regions), confidence from the structure tensor's smallest eigenvalue and the warped residual. Paired benchmark against Farneback (E11 protocol, n = 100).
**R.** Speed: no gain (≈ 15 ms per strip pair, launch-bound; total 190 vs 195 ms per frame). CD 0.0591 vs 0.0629 (82/100 better, p = 1.8e-13), CD-hard 84/100 better; but edge deviation 0.102 vs 0.065 (16/100 better, p = 6.5e-13), edge-hard 0.102 vs 0.065, fidelity-hard −0.013.
**C.** Not adopted. Methodological point: LK minimises exactly the photometric difference that CD measures, so it can lower CD by warping to match non-geometric differences (blur, noise) while adding seam edges. CD must always be reported together with edge deviation; a method that improves CD while worsening edges is suspect.

## E26 — Framing the selfie-stick holder from the IMU alone

**Q.** Without a person detector (OpenCV 5 ships none; downloading weights needs the user's consent), can the main subject be framed automatically?
**M.** With an invisible selfie stick the stick runs along the camera body's −Z axis, so the holder lies along M(t)ᵀ·(0,0,−1) in the levelled frame; the face sits above the hand, so the pitch is lifted by 40°. `look.holder_direction` / `holder_track` (0.5 s keys, 1 s smoothing, linear easing).
**R.** Checked visually on 3 single frames and a 12 s follow on the trail clip: the holder is in frame throughout (at the frame edge in 1/3 single frames, because the arm holds the stick sideways rather than towards the face).
**C.** A useful zero-model starting point (studio button and template, MCP `holder_keys` / `look(at="holder")`); not a replacement for detection-based framing. Quantitative evaluation needs annotated face positions.

## E27 — 8K mode

**M.** E11 protocol restricted to 8K clips (2 × 3840² lenses): 6 clips × 5 frames, n = 30 (`eval_seams.py --mode 8K`).
**R.** vs feathered: better on every metric (30/30 CD, p = 1.9e-9). vs DP cut: CD 17/30 better (−0.003, p = 0.047), CD-hard n.s. (p = 0.53), edge deviation slightly worse (n.s., p = 0.10), fidelity lower (by construction).
**C.** At 8K (mostly indoor, static scenes in this corpus) the flow seam is on par with the DP cut; its advantage is concentrated in 4K handheld footage with near-field parallax. Reported as such in the paper.

## E28 — Nadir patch

**Q.** The hand, the stick root and the operator's shadow fill the nadir of every handheld clip; the stitch cannot remove them (they are real content in both lenses). Can a cheap fill hide them in 360 exports and little-planet shots?
**M.** `osvcut.nadir_patch`: in the levelled frame, take the ring 1°–6° outside a cap of radius r around the nadir, average it into 36 azimuth bins (circularly smoothed), fill the cap with the per-azimuth colour fading to the ring mean towards the pole, and blend over a 2° soft edge. Spec key `nadir_patch: r`; studio option (10° / 15° / 20°), MCP `render_cut(nadir_patch=…)`.
**R.** On the trail clip (r = 18°) the hand and stick vanish into a smooth ground-coloured disc; in the little-planet projection the centre reads as a plain patch of ground. Cost is one masked reduction per frame (negligible). Unit test: pixels outside the cap are unchanged, the pole is near-uniform.
**C.** A cosmetic tool, not research: it hides rather than reconstructs. Inpainting the true ground (from earlier frames as the camera moves) is the natural next step and would need the per-frame ground-plane motion that E-ground showed to be unreliable.

## E29 — The 4K/8K stream scale was wrong by 1.3–1.6 % (major correction)

**Q.** Investigating the handle-side sector (E6, E15: ~1° overlap over ~97° of azimuth, where the holder stands), the holder's hand appeared twice. A wide vertical-disparity search (2-D ZNCC of band-passed luma between lens A below the seam and lens B above it, `tools_handle_probe.py`, `cache/debug/_wd.py`) reported ~2.8° — not only there, but at every azimuth, including far-field foliage in the wide overlap where Farneback reported ≈ 0.
**M.** Checks, in order: (1) the search code returns 0 when B := A; (2) decoding the two streams separately by frame index gives the same result as seek + hstack (not a sync problem); (3) direct correlation of the two lenses' strips at α = 160° (far foliage, both lenses valid): 0.10 at zero shift vs 0.68 at 2.8°; (4) a common radial scale factor κ on both lenses: correlation at zero shift peaks at κ ≈ 0.984–0.987 on three 4K clips (0.56 → 0.91, 0.48 → 0.74, 0.72 → 0.87) and at κ ≈ 0.99 on an 8K clip.
**R.** κ·s with s = digital_focal_length / fx (0.5065 / 0.5081 in 4K, 1.013 / 1.016 in 8K) is W/3840 within the search resolution: the intrinsics live in a 3840² calibration space and the stream scale is simply W/3840 (0.5 / 1.0). With the corrected scale the wide-overlap disparity is ±0.2° and the handle sector's 0.2–1.0° (true near-field parallax); the doubled hand and the broken rail at the holder disappear.
**C.** The old scale put every ray ~1.4° too far from the optical axis in each lens near 90°, i.e. a constant ~2.8° β misregistration around the whole ring. It went unnoticed because (a) a β-shift is invisible on structures that cross the seam at right angles, (b) the narrow DP-placed ramp hides much of it, and (c) Farneback on the 46-row strip does not see a 15-px vertical shift (raw median −0.07°) — the "+2–3.4° bias" that got DIS rejected in E5 was in fact the true offset. E3's "median misalignment 0.77–0.85°", E12 and E19 (no systematic calibration bias) were measured with that Farneback and are superseded. Fixed in `stitch.lens_scale` (all paths: renderer, seam engines, masks, proxy rigs → rig v3); `OSV_LEGACY_SCALE=1` reproduces the old geometry for A/B.
**Corpus check** (`tools_scale_check.py`, common κ on both lenses, correlation at zero shift where both lenses are valid): 4K, 12 random clips × 3 frames: best κ relative to W/3840 median 0.9975 (IQR 0.9975–1.0); correlation 0.546 → 0.765, better in 35/36 frames. 8K, 30 frames: median κ 1.000 (IQR 0.998–1.005); 0.700 → 0.829, better in 27/30. The old scale is κ = 1.013 in both modes.
**Benchmark** (E11 protocol, same 100 frames, old → new scale): CD feathered 0.101 → 0.064, DP cut 0.071 → 0.045, flow + DP (photometric confidence) 0.062 → 0.031 (better in 97/100, p = 7.6e-18); CD-hard 0.082 → 0.046; edge-hard 0.071 → 0.085 (n.s., p = 0.07). Flow + DP vs DP cut under the new scale: CD 0.031 vs 0.045.
**Re-run of all seam results with the corrected scale** (`results/seam_bench_v3.json`, same 100 frames):

| method | CD | CD-hard | edge-hard | fid-hard | step |
|---|---|---|---|---|---|
| feathered | 0.064 | 0.094 | 0.235 | 0.875 | 0.017 |
| DP cut | 0.045 | 0.068 | **0.043** | **0.977** | **0.011** |
| flow, centred | 0.036 | 0.057 | 0.087 | 0.919 | 0.015 |
| flow + DP ±3° | 0.032 | 0.049 | 0.075 | 0.879 | 0.013 |
| **flow + DP ±1.5° (default)** | **0.031** | **0.046** | 0.084 | 0.927 | 0.014 |

vs DP cut: CD 89/100 (p = 3.6e-17), CD-hard 92/100; the cut keeps crisper edges (+0.041, p = 4.2e-10) and a smaller step. ±1.5° vs ±3°: 71/100 (p = 2e-6) — smaller margin than before (less misregistration left for a wide ramp to blur). Anti-aliasing: no practical effect. Two-band: edge 0.043 vs 0.048 (p = 0.03), step 0.012 vs 0.014, fidelity −0.024. **GPU LK (E25) reverses**: now worse on CD (0.035 vs 0.031, 95/100) and edges (0.053 vs 0.048); E25's "LK games CD" example mostly reflected LK absorbing part of the scale misregistration. 8K (E27 protocol, 30 frames): CD 24/30 better than the cut (p = 5.5e-4), CD-hard n.s. (p = 0.11).
Temporal (E16 protocol, 8 clips, corrected scale): excess flicker vs single-lens reference flow −0.050 [−0.168, +0.013], DP cut −0.033 [−0.129, +0.041] (2/8 above the reference each), feathered −0.211.
**Superseded:** the numbers in E3 (misalignment), E5 (DIS "bias"), E11, E12, E19, E23–E25 and E27 were measured with the wrong scale.

## E30 — A correlation prior for the narrow-overlap sector (negative result)

**Q.** With the corrected scale, the wide correlation search (E29) still finds +0.5–1° (up to 2°) of across-seam disparity in the handle-side sector (α ≈ 220–320°, overlap < 2.5°), where the flow is near zero because the overlap is too narrow to measure it and the warp is scaled down by the narrow ramp. Positive disparity means duplicated content, which both lenses could remove by sampling *inside* their own valid areas. Does using the search as the across-seam flow there, with the warp spread over ±3° (blend transition kept narrow), align the sector better?
**M.** `FlowStitch(narrow="search")`: per azimuth, ZNCC search of band-passed luma (A on [−5, 0.6]°, B shifted −1…4°), smoothed by normalised convolution; weight 1 where the raw overlap is < 1°, 0 above 2.5°; negative (gap) values clamped to −0.2°; warp ramp max(w, 3°·weight). Evaluation (`eval_narrow.py`): mean |Y(pA) − Y(pB)| of the two warped contributions on textured pixels in the narrow sector where both warped samples are valid for both engines (the same pixels for both) — photometric, independent of the search. 12 random 4K clips × 3 frames.
**R.** off 0.0925 vs search 0.0984, better in 13/36 frames (p = 0.12); with the search band narrowed to [−2, 0.6]°: 0.1014, 13/36 (p = 0.06).
**C.** Not adopted (option kept, default off). The disparity measured a few degrees away from the seam is not the disparity *at* the 1° overlap (near ground and the holder's body change distance quickly with elevation), and a wide warp moves content that was already acceptable. In this sector, with the right geometry, the existing flow seam is as good as this prior can make it; the residual is real near-field parallax with too little shared view to fix from one frame.

## E31 — Renderer throughput, round 2

**M.** Per-stage timing of the export loop (1920×1080 flat, 30 fps, flow seam with photometric confidence, RTX 4070 Ti): decode of both streams 2.2 s per 5 s of footage (not the bottleneck); per frame yuv→rgb 5.4 ms, render 8.7 ms, readback 1.2 ms on the main thread; seam analysis (worker thread) ~45 ms, of which the minimum-cost closed path was 18 ms (a Python loop over 1 800 columns, with an array allocation per step) and Farneback 8 ms.
**R.** DP path: per-step allocation removed (numpy fallback 8.5 ms) and an optional Numba kernel with identical tie-breaking (0.32 ms; equal paths on random cost maps with ties). Export 14.6 → 25.8 fps (seam every frame), 16.3 → 32.4 fps (every 2nd frame); 2880×1440 standard preview 17.2 → 14.9 s per 10 s.
**C.** 1080p reframed exports now run faster than real time with `seam_every: 2`; the defaults keep analysis on every frame.

## E32 — Where does the flow seam's advantage come from? (warp vs. path; exposure step)

**Q.** Under the corrected scale the DP cut has a smaller exposure step than the flow seam (0.011 vs 0.014, 28/100, p = 3.4e-8). Is that the warp, the path, or the ramp width? And which part of the flow seam lowers CD?
**M.** E11 protocol on 10 clips × 4 frames; extra methods: `flow_dp_r15_nowarp` (flow seam with the displacement forced to ~0: same path search on half-aligned strips, same ±1.5° transition) and `smart_r15` (DP cut with a ±1.5° transition instead of its 1° blend).
**R.** CD: flow seam 0.0345, no-warp 0.0471, DP cut 0.0456, DP cut ±1.5° 0.0500 — the flow seam beats its own no-warp version on 40/40 frames. Edge-hard: 0.075 (flow), 0.049 (no warp), 0.042 (cut). Exposure step: 0.0130 / 0.0128 / 0.0123 / 0.0125, no pair significant on this subset (p ≥ 0.25).
**C.** The CD gain comes entirely from the flow warp, not from where the path runs; the warp is also what raises the edge deviation. The step difference is ≈ 0.3 % in brightness ratio and not attributable to the warp; practically invisible. No change.

## E33 — Warp strength sweep

**Q.** E32 shows the warp buys CD at the price of edge energy. Is a partial warp a better compromise?
**M.** `FlowStitch(warp=k)` scales the measured displacement; k ∈ {0, 0.5, 0.75, 1, 1.25}; E11 protocol, 10 clips × 4 frames (`results/e33_warp.json`, `paper/data/warp_sweep.dat`).
**R.** CD 0.052 / 0.045 / 0.039 / **0.035** / 0.037; edge-hard 0.059 / 0.093 / 0.085 / **0.075** / 0.094. k = 1 beats 0.75 on CD 40/40 and edge 26/40 (p ≤ 0.001), and 1.25 on CD 38/40 and edge 34/40 (p < 1e-7).
**C.** No compromise exists along this axis: a partial warp leaves two half-aligned copies of an edge, an over-warp creates new ones. Both metrics peak at the measured displacement, which also validates the flow's magnitude under the corrected scale. Default stays k = 1; the remaining edge cost versus the cut is the price of morphing, not of a mis-scaled flow.

## E34 — Gating small displacements (negative result)

**Q.** If the warp's edge cost came from noisy small flows, not warping displacements below a threshold should cut it at little CD cost.
**M.** `FlowStitch(gate=τ)`: smooth gate on |F| from τ/2 to τ; τ ∈ {0, 0.1, 0.2, 0.4}°; 10 clips × 4 frames (`results/e34_gate.json`).
**R.** CD 0.0345 / 0.0346 / 0.0355 / 0.0404; edge-hard 0.0750 / 0.0743 / 0.0737 / 0.0719 (edge gains ≤ 0.003, p ≈ 0.02; CD losses up to 0.006, 40/40).
**C.** Not adopted (option kept, default 0). The edge cost is not noise on small flows; it sits where the displacement is large, i.e. where morphing real near-field parallax inevitably stretches content. Together with E33, the flow seam's edge cost is intrinsic to morphing.

## E35 — Bringing the morph to the live preview

**Q.** The live shader (E20) only cuts along a tracked path; E32 showed the flow seam's gain comes from the warp. Can the shader warp too, with what it can afford (a small texture, no per-pixel flow)?
**M.** (1) Offline: `FlowStitch(flow1d=True)` collapses the flow to one confidence-weighted (d_alpha, d_beta) per azimuth column. (2) `proxy.seam_track` now runs that engine on the proxy at 2 Hz and stores the flow seam's path plus one displacement pair per 2° bin (u8, ±4°) → rig v4; the shader half-aligns both lenses with it over ±1.5° (like the export). (3) `tools_live_eval.py` emulates the shader on proxy frames with the decoded tracks and scores it with the E11 metrics, live-cut (v3) vs live-morph (v4).
**R.** (1) 10 clips × 4 frames: CD full flow 0.0345, per-column 0.0356, no warp 0.0471 — one displacement per column keeps ~91 % of the gain (beats the DP cut 31/40). (3) 5 clips with proxies, 40 frames: CD 0.057 → 0.050 (30/40, p = 8.7e-5), CD-hard 0.096 → 0.085 (29/39, p = 0.003); edge deviation n.s. (p ≥ 0.45). In the browser the morph changes only the pixels next to the seam.
**C.** Adopted (rig v4). The live preview now morphs like the export, at a smaller gain (2 Hz tracks, proxy resolution).

## E36 — Live-preview tracks were half an interval late

**Q.** E35's live gain (CD 0.057 → 0.050) was well below an oracle that computes the tracks on the evaluated frame itself (0.044 cut / 0.0375 morph). Why?
**M.** (1) Temporal smoothing: the track engine used the per-frame EMAs (flow, path, gains) and the path's temporal term at 2 Hz. (2) Time alignment: `tools_live_eval.py` with the evaluated frame shifted by ±0.25 s (`TSHIFT`).
**R.** (1) Turning the smoothing off changed little (morph 0.0504 → 0.0491). (2) Shifting by +0.25 s helped (0.0437), −0.25 s did not: ffmpeg's `fps=2` filter had been feeding the frame from about half an interval *after* each sample time, while the shader looks sample k up at k/rate. With `select='gte(t, selected_n/rate − 0.01)'` the best shift is 0 and live CD is 0.0486 (cut) / **0.0413** (morph), 31/40, p = 6e-4 — down from 0.057 for the v3 live preview.
**C.** Fixed in `proxy.seam_track` (exact sample times, no temporal smoothing between 2 Hz samples). The live preview's remaining gap to the oracle (0.041 vs 0.0375) is quantisation, 2° bins and interpolation between samples.

## E37 — Farneback parameters (no change)

**M.** `FlowStitch(fb_params=(levels, winsize, iterations, poly_n, poly_sigma))`, 10 clips × 4 frames, corrected scale: winsize 7 / 11 / 15 / 21, levels 2 / 3 / 4, poly 7/1.5.
**R.** CD 0.0351 / **0.0345** / 0.0342 / 0.0340 and edge-hard 0.0735 / **0.0750** / 0.0770 / 0.0789 for winsize 7–21; levels have no effect (the strip is 46 rows); poly 7/1.5: CD 0.0346, edge 0.0813.
**C.** The default (3, 11, 4, 5, 1.1) sits on the CD–edge trade-off; no setting improves both. Unchanged.

**E40 (negative, not adopted) — motion saliency as a detector-free subject finder.** Levelled 512×256 equirect at t and t + 0.3 s, blurred |difference| restricted to −35°…45° elevation, peak of a 8-px Gaussian. On a casino clip the peak sat on an animated advertising screen and on a beach clip on water glints; the people in both frames (two near the camera) were not the maximum. Rotation is levelled out but screens, water and foliage move too, and camera translation adds parallax motion; a frame-difference cue cannot tell people from them. A real detector is needed.

**Not done, needs the user:** person / face detection for subject framing. OpenCV 5 in this environment ships neither Haar cascades nor the HOG people detector, and `FaceDetectorYN` needs an ONNX model download.

## E38 — Track rate for the live preview

**M.** `tools_live_eval.py` now evaluates on a fixed time grid and interpolates the tracks linearly between neighbouring samples, as the shader's texture lookup does. 2 Hz vs 5 Hz rigs (`OSV_TRACK_HZ`), 5 clips, 12 frames each.
**R.** Matched frames (n = 36): live-morph CD 0.0415 (2 Hz) vs 0.0393 (5 Hz), 5 Hz better 24/36, p = 0.013; cut-only n.s.; edge-hard 0.157 vs 0.155 (p = 0.05). Cost: rig 2.5 → 6.3 MB and 119 → 198 s for a 19-min clip.
**C.** Default 5 Hz for clips up to 10 min, 2 Hz beyond (rig size).

## E39 — Azimuth resolution of the live tracks (no change)

**M.** 2° (180 bins) vs 1° (360 bins) per-azimuth tracks (`OSV_TRACK_BINS`), 4 clips, 48 matched frames.
**R.** Live-morph CD 0.0489 vs 0.0488 (1° better 31/48, p = 0.015), edge-hard 0.158 vs 0.157 (n.s.).
**C.** Consistent but negligible; not worth twice the track size. 2° bins kept. The remaining gap to tracks computed on the displayed frame is temporal interpolation, not azimuth resolution.

## E41 — Disagreement versus parallax

**M.** `tools_cd_vs_parallax.py`: the 100 E11 frames, seam ring cut into 10° bins (3 598 bins with ≥ 400 both-valid pixels in |β| < 3°); per bin the raw displacement (median Farneback magnitude on unwarped strips) and the CD of the DP cut and of the flow seam on the same pixels; pooled by displacement.
**R.** Ratio flow/cut: 0.94 (≈0°, n = 1182), 0.86 (0.16°), 0.85 (0.25°), 0.79 (0.38°), 0.67 (0.57°, n = 338), 0.57 (0.89°, n = 22). The cut's CD rises from 0.014 to 0.077; the flow seam's levels off near 0.05.
Re-run with the ±0.75° default (E42): ratio 0.92 / 0.83 / 0.80 / 0.72 / 0.60 / 0.48; the flow seam stays below 0.048 in every bin.
**C.** The morph's advantage is concentrated where parallax is large (near objects), exactly where seams are visible; at zero parallax the two methods agree. Figure in the paper (`paper/data/cd_vs_parallax.dat`).

## E42 — The transition was too wide for the corrected geometry (default ±1.5° → ±0.75°)

**Q.** E41 showed the flow seam's CD levelling off near 0.05 at large parallax. Would a wider ramp help there (adaptive width), or a narrower one everywhere?
**M.** (1) `tools_cd_vs_parallax.py` with `RAMPS=a,b`: per-parallax-bin CD of two ramp widths, 10 clips × 4 frames. (2) Full E11 benchmark of ±1.5 / 1.0 / 0.75 / 0.5° (photometric confidence) + DP cut (`results/e42_ramp.json`). (3) Temporal benchmark (E16) incl. ±0.75°.
**R.** (1) ±3° vs ±1.5°: worse in every bin, more at high parallax (ratio 1.05 → 1.14). ±0.75° vs ±1.5°: better in every bin, more at high parallax (0.97 → 0.88; 0.76 in the 10-bin top class). (2) CD 0.0309 / 0.0301 / **0.0298** / 0.0299; fidelity-hard 0.927 / 0.943 / 0.952 / 0.960; edge-hard 0.085 / 0.087 / 0.088 / 0.087 (n.s.); ±0.75 vs ±1.5: CD 59/100 (p = 0.002), CD-hard 66/100 (p = 2e-5), fidelity 99/100, step +0.0003 (p < 0.001, negligible). vs DP cut: CD 100/100, CD-hard 99/100. (3) Excess flicker −0.035 [−0.154, +0.040], 3/8 above reference — like the cut (−0.033). Live preview (shader ±0.75°): CD 0.0425, morph better than cut-only 56/60 (was 45/60).
**C.** Default ramp is now ±0.75° (engine, shader, live emulation). The ±1.5° choice (E11) was made under the wrong stream scale, when a wider ramp still had misregistration to blur. Paper and table updated. 8K re-run with ±0.75° (6 clips, 30 frames): CD vs DP cut 27/30 better (p = 5e-8), CD-hard 27/30 (p = 7e-6) — up from 24/30 and n.s. with ±1.5°; edges still favour the cut.

## E43 — The path's centre prior was harmful (removed)

**Q.** Like the ramp width (E42), the DP path's cost weights (gradient 0.25, centre prior 0.02 per degree) were set before the scale fix. Are they still right?
**M.** `FlowStitch(grad_w, centre_w)`; 10 clips x 4 frames: grad in {0, 0.25, 0.5, 1} x centre in {0, 0.01, 0.02, 0.1}; then the full 100-frame benchmark and the 8-clip temporal benchmark for the winner (`results/e43_*.json`).
**R.** Centre prior 0 is better on every metric; 0.01 already loses most of the gain (the term, up to 0.08 at 4 deg off-centre, is as large as the photometric cost of 0.02-0.05 and drags the path to the middle regardless of agreement). The gradient weight barely matters. Full benchmark (grad 0.25, centre 0) vs the E42 default: CD 0.0269 vs 0.0298 (99/100), CD-hard 0.0398 vs 0.0434 (93/100), edge-hard 0.061 vs 0.088 (76/100), edge 0.034 vs 0.045 (67/100), step 0.0095 vs 0.0145 (86/100), fidelity-hard 0.960 vs 0.952. vs DP cut: CD 100/100 (-40 %), step 0.0095 vs 0.0109 (68/100, p = 4.5e-4), overall edge 0.034 vs 0.029 (n.s., p = 0.18); the cut keeps crisper edges only on the hard subset (0.061 vs 0.043, p = 3e-4). Temporal: excess flicker -0.022 [-0.068, +0.029], 1/8 clips above the reference (cut -0.033, 2/8). Re-runs with the new default: 8K CD 30/30 vs the cut (p = 2e-9), hard 27/30, step 27/30, overall edge n.s.; E41 ratio 0.78 -> 0.56 (0.42 top bin); live preview morph 0.040 vs cut 0.048 (49/60).
**C.** Default `centre_w = 0`. With E29, E42 and E43 the flow seam is better than the DP cut on CD, exposure step and flicker, level on overall edge deviation, and behind only on edges at large parallax.

## E44 — Ramp width re-checked without the centre prior (no change)

**M.** ±0.5 / 0.75 / 1.0 / 1.5°, centre prior 0, 10 clips × 4 frames (`results/e44_ramp.json`).
**R.** CD 0.0298 / 0.0299 / 0.0302 / 0.0310; fidelity-hard 0.964 / 0.957 / 0.951 / 0.938; edge-hard 0.050 / 0.050 / 0.049 / 0.046; edge 0.031 / 0.032 / 0.032 / 0.032; step equal.
**C.** Without the centre prior the width matters much less than in E42; ±0.75° stays (wider buys a little hard-edge sharpness for CD and fidelity, narrower the reverse).

## E45 — The path's edge margin (0.6° → 0)

**Q.** The DP path paid 2.0 wherever it came closer than 0.6° to the overlap edge — a margin sized for the old ±1.5° ramp. Still needed with ±0.75° (and the ramp narrowing by itself near the edge)?
**M.** `FlowStitch(room_min)`: 0 / 0.1 / 0.2 / 0.4 / 0.6 / 1.0°, 10 clips × 4 frames; then 100 frames and the temporal benchmark for 0.
**R.** 40 frames: smaller is better on every metric (0 → CD 0.0290, edge-hard 0.043, edge 0.028, step 0.0095). 100 frames, 0 vs 0.6: CD 0.0263 vs 0.0269 (84/100, p = 3e-12), CD-hard 0.0390 vs 0.0398 (74/100), edge-hard 0.058 vs 0.061 (58/100, p = 0.026), fidelity-hard 0.963 vs 0.960 (93/100), step 0.0092 vs 0.0095 (mean lower, 22/100 frames, negligible), flicker −0.023 vs −0.022. vs DP cut: CD 100/100 (−41 %), step 69/100, edge n.s. (p = 0.28), edge-hard still behind (0.058 vs 0.043, p = 0.002).
Re-runs with the final settings: 8K CD 30/30 vs the cut, hard 29/30 (p = 8e-8), step 28/30, edge n.s. (p = 0.26); E41 ratio 0.76 → 0.57 (0.42 top bin); live morph 0.039 vs cut 0.046 (49/60).
**C.** Default `room_min = 0`. A modest gain; the 40-frame subset overstated it (edge-hard parity with the cut there, not on 100 frames).

## E46 — What is the remaining hard-subset edge gap? (negative)

**M.** (1) Signed log(E_out/E_ref) on hard seam pixels, 40 frames. (2) Lens resampling bilinear vs bicubic for every engine and the reference (`OSV_INTERP`), 40 frames.
**R.** (1) Cut +0.004 (60 % > 0), flow −0.003 (40 % > 0): no systematic blur or extra edge, just a wider spread. (2) Bicubic: edge-hard flow 0.0427 / cut 0.0420 (bilinear 0.0433 / 0.0423), CD up for both (0.0326 / 0.0497). On this subset the two are already level on hard edges; the 100-frame gap (0.058 vs 0.043) comes from a minority of frames.
**C.** Not resampling, not a systematic blur. Bilinear kept. The residual edge cost is frame-specific morphing of strong near-field parallax.

## E47 — Looking at the frames behind the hard-edge gap

**M.** The 100-frame results (E45 vs DP cut) sorted by the hard-subset edge-deviation difference; the 40° window of hard columns where the two outputs differ most, rendered side by side (`cache/debug`, not committed: the frames contain people).
**R.** Five frames carry 42 % of the total gap; one clip (casino) appears four times in the top eight. Worst frame: a red casino carpet just below the camera — dense, repetitive texture where the flow is ambiguous; the morph softens the texture slightly, the cut shows a faint horizontal break; neither is clearly worse. Second: a face at arm's length — the cut breaks a spectacle rim at the seam, the morph keeps it continuous, yet the metric scores the morph 0.296 vs 0.141.
**C.** In the frames that dominate the gap the morph is not visibly worse and is sometimes better; edge deviation penalises "different from the single-lens image" at near-field parallax, not visible defects. This is the limit of reference-based metrics here and what the blind study (E14) is for. Noted in the paper's limitations.

## E48 — Combined flow confidence (no change)

**M.** With the final defaults (±0.75°, no centre prior, no edge margin): photometric (`photo`), forward–backward (`fb`) and their product (`both`, aimed at repetitive texture matching the wrong period), 10 clips × 4 frames.
**R.** CD 0.0290 / 0.0291 / 0.0290; CD-hard 0.0418 / 0.0420 / 0.0418; edge-hard 0.0433 / 0.0425 / 0.0431; edge 0.0281 / 0.0285 / 0.0284; step 0.0095 / 0.0099 / 0.0096 — all within noise.
**C.** The confidence definition is no longer a limiting factor. `photo` kept (one flow instead of two). `conf="both"` remains available. The class default is now `photo` as well (exports did not pass `seam_opts` and used `fb`, i.e. two Farneback passes): a default 1080p30 reframed export runs at 35.7 fps, up from 25.8.

## E49 — Single-lens fast path outside the seam band

**M.** Per-stage timing of a 3840×1920 360 export: render 57.5 ms per frame (the bottleneck; readback 6 ms, rgb24 → NVENC 16 ms in another thread). Outside the band the transition can reach (max |path| + ramp + 0.25°, at least 4.5°), the other lens's weight is ≤ 1e-4, so `FlowStitch.render` now samples only the owning lens there (with the same exposure field) and runs the full morph on the band; pixels whose own mask is not fully valid fall back to the full path. Checked against the full computation on two clips (3840×1920, rotated view).
**R.** Render 57.5 → 22–28 ms; max |difference| 0.0035 (< 1 grey level), 0 pixels above 1/255. (A first version with a fixed 4.5° band differed on up to 1 220 pixels by up to 0.07: with the edge margin gone (E45) the path can run at the strip edge, so the transition reaches beyond 4.5°.) 360 export 25.6 → 16.1 s per 10 s (1.6× real time); 2880×1440 preview 14.9 → 10.9 s (27.6 fps).
**C.** Adopted (automatic for outputs > 400 k pixels, not for diagnostics with `parts=True`).

## E50 — Flow strip resolution 0.2 → 0.1°/px

**Q.** The flow strip samples at 0.2°/px; 4K lenses carry ≈ 0.11°/px and 8K ≈ 0.055°/px. Does a finer strip help?
**M.** `FlowStitch(fres)`; 8K: 6 clips × 5 frames; 4K: 10 × 4; then the full 100-frame benchmark, temporal and 8K re-runs with 0.1 as the default.
**R.** 8K: CD 0.0360 → 0.0347 (27/30, p = 1e-5), fidelity 20/30, edges n.s. 4K subset: 0.0290 → 0.0272 (35/40), fidelity 32/40, edges n.s. Full benchmark vs E45: CD 0.0263 → **0.0246** (93/100, p = 2e-16), CD-hard 0.0390 → 0.0360 (81/100), fidelity-hard 0.963 → 0.967 (79/100), edge and step n.s. vs DP cut: CD −45 % (100/100), CD-hard −47 % (100/100), step 66/100 better, edge n.s. (p = 0.11), edge-hard still behind (0.061 vs 0.043). Temporal: −0.014 [−0.070, +0.061], 2/8 above reference. 8K final: CD 30/30, step 27/30 vs the cut. Analysis +25 % (171 → 212 ms), but it runs asynchronously: 1080p export 41 fps, 360 export 1.52× real time (unchanged).
E41 re-run: ratio 0.74 → 0.52 (0.46 top bin), flow seam ≤ 0.041 in every bin. Live preview (proxy resolution): morph 0.039 vs cut 0.046 (48/60), unchanged.
**C.** Default `fres = 0.1`.

## E51 — Farneback window at the 0.1° strip (no change)

**M.** As E37 but on the 0.1°/px strip (the window now spans half the angle): winsize 7 / 11 / 15 / 21, levels 3 / 4; 10 clips × 4 frames (the `fbp_*` methods use a ±1.5° ramp, so only the comparison between windows is meaningful).
**R.** CD 0.0306 / 0.0303 / 0.0300 / 0.0299; edge-hard 0.038 / 0.038 / 0.040 / 0.047; edge 0.027 / 0.028 / 0.028 / 0.033; levels no effect.
**C.** Same trade-off as E37; winsize 11 kept.

## E52 — An even finer strip for 8K (no change)

**M.** 8K clips (6 × 5 frames): flow strip 0.1 vs 0.05°/px (8K lenses ≈ 0.055°/px).
**R.** CD 0.0347 vs 0.0351 (0.05 better 12/30, p = 0.16), fidelity and edges n.s.; analysis 406 → 636 ms.
**C.** 0.1° is enough for both modes; no per-mode resolution.

## E53 — Temporal smoothing parameters (no change)

**M.** `eval_temporal.py --variants` (8 clips, 4 s at 25 fps, final defaults otherwise): flow EMA 0 / 0.5 / 0.8, seam EMA 0.5 / 0.8, path temporal term λ_t 0 / 0.6 / 1.2.
**R.** Excess flicker vs the single-lens reference: default −0.014 (2/8 above); flow EMA 0 and 0.8 −0.014 (per-clip differences ≤ 0.003); seam EMA 0.8 −0.015; λ_t 1.2 −0.015; **λ_t 0 +0.057 [−0.035, +0.312], 5/8 above**.
**C.** The path's temporal term is what keeps the seam steady (without it the path jumps between near-equal minima); the EMAs barely matter at 25 fps. Defaults kept.

## E54 — Two-band blending with the final defaults (reverses)

**M.** `bands=2` vs 1 with the final defaults (±0.75°, no centre prior or margin, 0.1° flow), 10 clips × 4 frames.
**R.** CD identical by construction; edge 0.041 vs 0.030 (12/40 better, p = 0.008), edge-hard 0.057 vs 0.041 (p = 0.03), step 0.0111 vs 0.0091 (9/40, p = 0.001), fidelity-hard −0.025 (0/40).
**C.** Under E23's settings two-band blending smoothed edges and steps; with the final seam it is worse on every metric — its wide low-frequency crossfade had been masking defects that are now fixed at the source. Kept as an option, off.

## E55 — Flow attenuation where the ramp is narrow (kept)

**M.** E6 scales the flow by min(1, half-width / ramp) where the overlap forces a narrow transition. `atten=False` vs default with the final settings, 10 clips × 4 frames.
**R.** Without attenuation: CD 0.0264 vs 0.0272 (40/40), CD-hard 0.0367 vs 0.0388 (35/40), but edge 0.032 vs 0.030 (10/40 better, p = 4e-4), edge-hard 0.044 vs 0.041 (p = 0.04), fidelity −0.002.
**C.** The usual morph trade (agreement for edge energy), now at the narrow-overlap columns; attenuation kept because edges are what viewers see.

## E56 — Staged live proxies (time to first view)

**M.** `proxy.py` publishes the transcoded proxy and a rig without the seam track first, then computes the seam / flow track and replaces the rig; the studio picks up the stage-1 rig while the job runs and swaps in the full rig when it finishes.
**R.** 67-s clip: viewable after 20 s (was 66 s, the whole job); 44-s clip in the studio: live at 15 s with the fixed seam, full flow seam at 44 s.
**C.** Adopted. Where the remaining 20 s go: the GPU transcode runs at ≈ 4.8× real time (20 s of footage in 4.2 s, decode-bound: both 100-fps HEVC streams are decoded fully; `-skip_frame nonref` cannot be combined with the CUDA filter graph), and the stage-1 rig takes 4.2 s, about 2 s of it Python/torch imports and 1.5 s ffmpeg frame grabs. Little left to gain without restructuring.

---

## Open questions / next experiments

1. ~~Multi-clip benchmark~~ → E11. Next: perceptual pairwise study on seam crops; DJI Studio / OpenOSV exports as external baselines.
2. Temporal metrics (flicker) on the benchmark clips for every method.
3. Near-field parallax on the handle side (1° overlap): depth-aware or temporal fill.
4. Tracking accuracy benchmark.

## E57 — A finer flow strip for stills (0.05°/px; not adopted)

**Q.** Stills have no real-time budget, and 8K lenses carry ≈ 0.055°/px. Does halving the strip step again (E50: 0.2 → 0.1 helped at 8K) pay off for a single high-resolution frame?
**M.** `eval_seams.py --mode 8K --clips 6 --frames 5 --methods fres_010,fres_005` (same 30 frames as E50's 8K check); `analyze_bench.py`.
**R.** CD 0.0347 → 0.0351 (fres 0.05 better on 12/30, p = 0.16), CD-hard 0.0617 → 0.0630 (p = 0.26), edge deviation 0.110 → 0.126 (worse, 9/30, p = 0.005), ghost-SSIM +0.0005 (23/30, p = 0.004, negligible), step n.s.; analysis 587 → 1162 ms. At 0.05° the Farneback window covers half the angle, so the flow is noisier and the morph adds edge energy without removing more disagreement.
**C.** Not adopted; stills use the default 0.1°/px. The strip resolution is saturated at about the 4K lens pitch — finer flow would need a larger window (E51 found the window insensitive at 0.1°) or a different estimator.

## E58 — Blind pairwise study, first round (one rater)

**Q.** The metrics disagree about the flow seam versus the minimum-difference cut (CD clearly lower, edge deviation level overall and worse at large parallax, E47). Which do viewers prefer, and is the difference visible at all?
**M.** Studio blind test (`blindtest.py`, seam defaults E50): random clip and time, seam azimuth drawn from the parallax-heavy columns with probability 0.8, upright 55° close-up 720×540 rendered by two of {feathered, DP cut, flow}, sides randomised, answers left / same / right. One rater (the camera owner, not the author of the methods), 87 votes, 87 frames from 67 clips, median 6.7 s per vote.
**R.** Flow vs feathered: 16 wins, 1 loss, 7 same (sign test excluding ties p = 3e-4). DP cut vs feathered: 17 / 2 / 11 (p = 7e-4). **Flow vs DP cut: 1 / 0 / 32** — the rater saw no difference on 32 of 33 pairs. Bradley–Terry (ties as half wins): flow 1.40, cut 1.23, feathered 0.37. Side bias: 22 left, 15 right (p = 0.32).
**C.** Both seam-aware stitches are clearly preferred to feathering; between the flow seam and the cut there is no perceptible difference in still close-ups, even at parallax-heavy azimuths. The edge-deviation gap (E47) is therefore not visible, and neither is the CD advantage at this scale. Open: (a) more raters; (b) whether the two differ in motion (the temporal-stability numbers favour the flow seam on flicker, E53) — a video-pair round would test that; (c) larger magnification at 8K.

## E59 — Blind study, round two: video pairs (set up, collecting)

**Q.** E58 found the flow seam and the DP cut indistinguishable in stills. The temporal measurements favour the flow seam on seam flicker (E53); is that visible?
**M.** `blindtest.make_video_pair`: the same parallax-preferring seam close-up (body frame, 55°, 720×540) as a 3-s clip at ~25 fps, rendered by the flow seam and the DP cut with their normal temporal state (as in an export), libx264 CRF 14; sides random, both clips loop in lock-step in the studio. ~11 s per pair on the RTX GPU; a background thread keeps two pairs in stock. Votes carry `kind: "video"`; `/api/blindtest/stats?kind=video`.
**R.** Pending (target 30 votes).

## E60 — Flow frame interpolation for slow motion beyond the native frame rate

**Q.** Slower than the source rate (e.g. 0.125× of 100-fps footage at 30 fps), the renderer repeated frames, which stutters; DJI Studio advertises AI frame interpolation. How much does a classical flow interpolator gain?
**M.** `interp.Interpolator`: Farneback flow both ways on a half-resolution grey image, Super SloMo's quadratic approximation of the intermediate flows, backward warping of both ends, blending with time weights × forward-backward consistency. `eval_interp.py`: 12 random 100-fps clips × 4 runs, a 1280×720 crop of the lens centre; frames 1..g−1 predicted from frames 0 and g (g = 2, 4) and compared with the real frames (PSNR; also on the 20 % of pixels that change most). A first run had ffmpeg pad the start of runs with a duplicated frame (constant-frame-rate output); `-fps_mode passthrough` fixes it (the export reader already used it).
**R.**

| gap | n | repeat | blend | flow | flow > blend |
|---|---|---|---|---|---|
| 2 | 48 | 26.09 dB | 29.07 | **33.82** | 38/48, p = 1e-7 |
| 4 | 144 | 25.32 | 26.53 | **30.67** | 125/144, p = 2e-22 |
| 2, moving pixels | | 20.17 | 23.76 | **28.92** | 40/48, p = 5e-8 |
| 4, moving pixels | | 19.80 | 21.37 | **25.81** | 126/144, p = 9e-23 |

Wired into `osvcut.render` (`"interp": "flow"`, default whenever the output would repeat source frames; `"repeat"` restores the old behaviour): each source frame is rendered once and the in-between frames are interpolated from the rendered views (after grade / nadir patch). 1 s of 100-fps footage at 0.125× → 240 frames at 1280×720 in 10.5 s (repeat: 3.4 s). While testing, the local-slow-motion schedule turned out to start the slow range one output frame late (81 instead of 90 slow frames) and the time-stretched audio to come out 0.24 s short (atempo chains drop their tail), so `-shortest` cut the video: the step is now split at the range edges and the stretched segment is fed 0.1 s extra and cut to length — video and audio 3.700 s each (expected 3.7).
**C.** Adopted. Classical flow interpolation gains 4–5 dB over cross-fading and 5–8 dB over repetition on real footage; learned interpolators (RIFE etc.) would likely gain more at large motion but need a model download.

## E61 — Same-footage comparison with DJI Studio (tooling; awaiting exports)

**Q.** How does our stitch compare with DJI Studio's on identical footage? DJI Studio (desktop, 1.0.0) is installed but has no command-line export, so its 360 exports have to be made by hand.
**M.** `dji_compare.py`: matches exports to library clips by name; fits per frame the rotation Q from our body frame to DJI's panorama (initialised from our IMU levelling, 4° yaw search, Nelder-Mead on a 3-DoF rotation vector, grey 512×256 grid, seam band |β| < 6° and the poles excluded). Self-test: our own levelled export (cut seam) and a copy rotated with `v360` (yaw 37°, pitch 5°, roll −3°): unrotated fit 0.00° from the levelling, residual 0.0060; rotated residual 0.0064; close-ups sampled through the two fits agree at 33.2 dB (the copy has an extra encode). `blindtest.make_dji_pair`: same parallax-preferring seam close-up from both, our side resampled from our panorama rendered at DJI's export width.
**R.** Pending DJI Studio exports.
