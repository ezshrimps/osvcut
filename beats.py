"""Beat grid of a music file for beat-synced cuts: tempo (period) and phase of the first beat.

    python beats.py <audio file>

Onset strength = half-wave-rectified spectral flux of a log-magnitude STFT (11.025 kHz mono, 23 ms hop);
the period is the autocorrelation peak between 60 and 180 BPM (with a mild preference for 120 BPM); the phase
is the offset whose comb of beats collects the most onset strength.
"""
import json
import subprocess
import sys

import numpy as np

SR, HOP, WIN = 11025, 256, 1024


def onset_envelope(path, seconds=90):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-t", str(seconds), "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if len(x) < WIN * 4:
        return np.zeros(0)
    n = 1 + (len(x) - WIN) // HOP
    frames = np.lib.stride_tricks.as_strided(x, (n, WIN), (x.strides[0] * HOP, x.strides[0]))
    mag = np.log1p(30 * np.abs(np.fft.rfft(frames * np.hanning(WIN), axis=1)))
    flux = np.maximum(np.diff(mag, axis=0), 0).sum(1)
    flux -= np.convolve(flux, np.ones(16) / 16, "same")          # remove the slow loudness trend
    return np.maximum(flux, 0)


def beat_grid(path):
    env = onset_envelope(path)
    fps = SR / HOP
    if len(env) < fps * 4:
        return None
    e = env - env.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    lags = np.arange(len(ac)) / fps
    ok = (lags >= 60 / 180) & (lags <= 60 / 60)
    weight = np.exp(-0.5 * (np.log2(lags[ok] / 0.5) / 1.0) ** 2)   # mild prior around 120 BPM
    lag = np.arange(len(ac))[ok][np.argmax(ac[ok] * weight)]
    # refine period and phase below one frame: the comb (beats at phase + k*period) that collects the most onset
    # strength, with linear interpolation of the envelope
    # flux[i] fires when an onset enters the analysis window, i.e. at the window's end: (i*HOP + WIN) / SR
    t_env = (np.arange(len(env)) * HOP + WIN) / SR
    best, period, phase = -1.0, lag / fps, 0.0
    for p in np.arange(lag - 1.0, lag + 1.0001, 0.02) / fps:
        k = np.arange(int(t_env[-1] / p))
        for ph in np.arange(0, p, 0.25 / fps):
            sc = np.interp(ph + k * p, t_env, env).sum() / max(1, len(k))
            if sc > best:
                best, period, phase = sc, p, ph
    conf = float(ac[lag] / (ac[0] + 1e-9))
    return dict(period=round(float(period), 4), bpm=round(60 / float(period), 1), phase=round(float(phase), 3),
                confidence=round(conf, 3))


if __name__ == "__main__":
    print(json.dumps(beat_grid(sys.argv[1])))
