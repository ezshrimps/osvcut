"""E18: rolling-shutter model of the lens-to-lens seam flow.

For every confident seam-strip pixel with body direction u, the two lenses read u out at rows y0, y1;
if each sensor reads top->bottom in tau seconds, lens 1 sees u later by dt = tau * (y1 - y0) / H.
While the camera rotates with body angular velocity w, a world-fixed direction drifts in the body frame
by -w x u per second, so the expected flow (lens0 -> lens1) is  F_pred = -(w x u) * dt.
We fit the single parameter tau by least squares, report R^2, and repeat with w taken from a shuffled
frame as a control.

    python tools_rs_model.py <clip-id-or-path> [frames]
"""
import sys, math, json, numpy as np, torch
from osvcut import Lens, dev, attitudes
from stitch import load_calib, qmat
from seam import FlowStitch
from eval_seams import grab_pair

P = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1.]])


def body_rot(q):
    return qmat(q) @ P          # body -> world


def omega_body(ts, qs, t, h=0.02):
    i0 = np.searchsorted(ts, t - h); i1 = min(len(ts) - 1, np.searchsorted(ts, t + h))
    i0 = min(i0, i1 - 1)
    R0, R1 = body_rot(qs[i0]), body_rot(qs[i1])
    dR = R0.T @ R1                                    # body(t0) -> body(t1) expressed in body(t0)
    w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2
    return w / max(ts[i1] - ts[i0], 1e-4)             # rad/s, body frame


def lens_rows(L, d):
    dl = d @ L.R.T
    x, y, z = dl.unbind(-1)
    th = torch.atan2(torch.hypot(x, y), z); ph = torch.atan2(y, x); t2 = th * th; k = L.k
    r = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * (k[3] + t2 * k[4])))))
    return (L.W / 2 + (L.fy * r * torch.sin(ph) + L.cy - 1920) * L.s)   # stream row


def main():
    f = sys.argv[1]
    if len(f) == 12 and all(ch in "0123456789abcdef" for ch in f):
        import library as _lib
        f = _lib.Library([r"E:\Videos"]).get(f)["path"]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    cal = load_calib(f); W = cal["W"]; lenses = [Lens(cal, L) for L in cal["lenses"]]
    ts, qs = attitudes(f, -1, 1e12)
    dur = ts[-1]
    e = FlowStitch(cal, lenses, flow_ema=0, center="overlap")
    D = e.fstrip
    u = D.cpu().numpy().reshape(-1, 3)
    dy = ((lens_rows(lenses[1], D) - lens_rows(lenses[0], D)) / W).cpu().numpy().reshape(-1)
    beta = np.arcsin(np.clip(u[:, 1], -1, 1)); hz = u[:, 0] ** 2 + u[:, 2] ** 2
    rng = np.random.default_rng(0)
    Ys, Gs, Gc, speeds = [], [], [], []
    times = [1 + (dur - 2) * (k + .5) / n for k in range(n)]
    shuffled = rng.permutation(times)
    for t, ts_ctrl in zip(times, shuffled):
        imgs = grab_pair(f, t, W)
        if imgs is None:
            continue
        e.update(imgs)
        conf = e.last_conf.reshape(-1) > 0.6
        Fa = np.radians(e.last_raw[..., 0].reshape(-1) * e.fres)
        Fb = np.radians(-e.last_raw[..., 1].reshape(-1) * e.fres)
        def pred(tt):
            w = omega_body(ts, qs, tt)
            dlt = -np.cross(np.broadcast_to(w, u.shape), u) * dy[:, None]       # per unit tau
            ga = (u[:, 0] * dlt[:, 2] - u[:, 2] * dlt[:, 0]) / np.maximum(hz, 1e-6)
            gb = dlt[:, 1] / np.maximum(np.cos(beta), 1e-3)
            return ga, gb, np.degrees(np.linalg.norm(w))
        ga, gb, sp = pred(t)
        ca, cb, _ = pred(ts_ctrl)
        Ys.append(np.concatenate([Fa[conf], Fb[conf]]))
        Gs.append(np.concatenate([ga[conf], gb[conf]]))
        Gc.append(np.concatenate([ca[conf], cb[conf]]))
        speeds.append(sp)
    Y, G, Gc = np.concatenate(Ys), np.concatenate(Gs), np.concatenate(Gc)
    def fit(G):
        tau = float((G * Y).sum() / max((G * G).sum(), 1e-12))
        r2 = 1 - ((Y - tau * G) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
        return tau, r2
    tau, r2 = fit(G); tc, r2c = fit(Gc)
    # explained misalignment: rms of predicted vs measured
    print(f"frames {len(Ys)}, pixels {len(Y)//2}, angular speed median {np.median(speeds):.0f} deg/s (max {np.max(speeds):.0f})")
    print(f"fit:     tau = {tau*1000:+.2f} ms   R^2 = {r2:.3f}")
    print(f"control: tau = {tc*1000:+.2f} ms   R^2 = {r2c:.3f}   (angular velocity from shuffled frames)")
    print(f"rms measured flow {np.degrees(np.sqrt((Y**2).mean())):.3f} deg -> residual after RS model {np.degrees(np.sqrt(((Y-tau*G)**2).mean())):.3f} deg")


if __name__ == "__main__":
    main()
