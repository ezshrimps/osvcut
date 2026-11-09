"""Smart seam engine for dual-fisheye OSV footage.

1. Validity masks per lens: usable image circle, coded-frame edges, and the
   body/handle occlusion arc from the calibration polygon (feathered).
2. Exposure/colour compensation: per-azimuth gain field measured on the
   overlap band in linear light, smoothed around the seam and over time.
3. Narrow seam: a minimum-difference closed path through the overlap band
   (dynamic programming, temporally regularised), blended over ~1 degree.

All geometry lives in the camera body frame (master lens +Y, slave -Y), so the
same seam serves every output projection.
"""
import math
import threading
import numpy as np
import cv2
import torch
import torch.nn.functional as F

GAMMA = 2.2
TH_CUT = 98.0  # hard cut (deg): beyond this the lens polynomial is not trusted


def _radius_px(L, cal, theta_deg):
    th = math.radians(theta_deg)
    t2 = th * th
    k = L["k"]
    r = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * (k[3] + t2 * k[4])))))
    return r * L["fx"] * cal["dfl"] / L["fx"]


def build_mask(L, cal, theta_max=96.5, margin_deg=1.0):
    """Float mask in stream pixel space (1 = usable, feathered to 0 at the edges), plus the
    usable radius per image angle (720 bins) used to clamp rays into the valid area."""
    W = cal["W"]
    s = cal["dfl"] / L["fx"]
    cx, cy = W / 2 + (L["cx"] - 1920) * s, W / 2 + (L["cy"] - 1920) * s
    m = np.zeros((W, W), np.uint8)
    sh = 4  # sub-pixel drawing
    cv2.circle(m, (int(cx * 16), int(cy * 16)), int(_radius_px(L, cal, theta_max) * 16), 255, -1, cv2.LINE_AA, sh)
    m[:2, :] = 0
    m[-2:, :] = 0
    m[:, :2] = 0
    m[:, -2:] = 0
    if L.get("occ") and len(L["occ"]) >= 14:
        pts = np.array([[W / 2 + (x - 1920) * s, W / 2 + (y - 1920) * s] for x, y in L["occ"]], np.float64)
        arc = pts[1:14]  # ordered along the bottom of the image circle
        c = np.array([cx, cy])
        far = [c + (p - c) * 1.8 for p in (arc[-1], pts[7], arc[0])]
        poly = np.vstack([arc, far])
        cv2.fillPoly(m, [np.round(poly * 16).astype(np.int32)], 0, cv2.LINE_AA, sh)
    px_per_deg = cal["dfl"] * math.pi / 180
    margin = max(1, int(round(margin_deg * px_per_deg)))
    m = cv2.erode(m, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    # usable radius per angle (image coords, y down), from the hard mask
    phis = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    rs = np.arange(0, W * 0.75, 0.5)
    X = (cx + np.cos(phis)[:, None] * rs).astype(np.float32)
    Y = (cy + np.sin(phis)[:, None] * rs).astype(np.float32)
    P = cv2.remap(m, X, Y, cv2.INTER_NEAREST, borderValue=0) > 127
    first_out = np.argmax(~P, 1)
    rlim = np.maximum(rs[np.maximum(first_out - 2, 0)], 1.0)
    m = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), margin * 0.5)
    return m, (cx, cy, rlim)


def _circ_interp(vals, pos):
    """vals (..., N) sampled at integer positions on a circle; pos float tensor in [0, N)."""
    n = vals.shape[-1]
    i0 = torch.floor(pos)
    f = pos - i0
    i0 = i0.long() % n
    i1 = (i0 + 1) % n
    return vals[..., i0] * (1 - f) + vals[..., i1] * f


def _fill_circular(v, ok):
    """Fill invalid bins of a circular 1-D array by linear interpolation between valid ones."""
    n = len(v)
    if ok.sum() == 0:
        return np.zeros_like(v)
    idx = np.nonzero(ok)[0]
    xs = np.concatenate([idx - n, idx, idx + n])
    ys = np.concatenate([v[idx]] * 3)
    return np.interp(np.arange(n), xs, ys)


def _circ_smooth(v, sigma):
    if sigma <= 0:
        return v
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    vv = np.concatenate([v[-r:], v, v[:r]])
    return np.convolve(vv, k, "valid")


def _dp_closed_path(cost, pen=0.002):
    """Minimum-cost closed path across the columns of cost (H x W), step <= 1 row/column."""
    H, W = cost.shape
    c0 = int(np.argmin(cost.min(0)))
    r0 = int(np.argmin(cost[:, c0]))
    C = np.roll(cost, -c0, axis=1)
    C = np.concatenate([C, C[:, :1]], 1)
    acc = np.full(H, np.inf)
    acc[r0] = C[r0, 0]
    back = np.zeros((W + 1, H), np.int8)
    rows = np.arange(H)
    for j in range(1, W + 1):
        up = np.concatenate([[np.inf], acc[:-1]]) + pen
        dn = np.concatenate([acc[1:], [np.inf]]) + pen
        st = np.stack([up, acc, dn])
        k = st.argmin(0)
        acc = st[k, rows] + C[:, j]
        back[j] = k - 1
    path = np.zeros(W + 1, np.int64)
    path[W] = r0
    for j in range(W, 0, -1):
        path[j - 1] = path[j] + back[j][path[j]]
    return np.roll(path[:W], c0)


class SmartStitch:
    def __init__(self, cal, lenses, device=None, band=4.0, ares=0.5, bres=0.1, blend=1.0,
                 gain_bins=72, gain_ema=0.85, seam_ema=0.5, lam_t=0.6):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dev = device
        self.lenses = lenses
        self._tls = threading.local()      # per-thread scratch (analysis may run in a worker thread)
        self.masks, self.limits = [], []
        for L in cal["lenses"]:
            m, (cx, cy, rlim) = build_mask(L, cal)
            self.masks.append(torch.from_numpy(m).to(device)[None, None])
            self.limits.append((cx, cy, torch.tensor(rlim, dtype=torch.float32, device=device)))
        self.band, self.ares, self.bres, self.blend = band, ares, bres, blend
        self.nb, self.gain_ema, self.seam_ema, self.lam_t = gain_bins, gain_ema, seam_ema, lam_t
        a = torch.arange(0, 360, ares, device=device) * math.pi / 180
        b = torch.arange(band, -band - 1e-6, -bres, device=device) * math.pi / 180
        B, A = torch.meshgrid(b, a, indexing="ij")
        self.strip = torch.stack([torch.cos(B) * torch.cos(A), torch.sin(B), torch.cos(B) * torch.sin(A)], -1)
        self.Hb, self.Wa = self.strip.shape[:2]
        self.logr = None   # (3, nb) log(slave / master) per azimuth bin
        self.path = None   # (Wa,) seam latitude in degrees (+ = toward master)

    # ---- sampling -------------------------------------------------------------
    def _sample(self, i, img, d):
        L = self.lenses[i]
        dl = d @ L.R.T
        x, y, z = dl.unbind(-1)
        th = torch.atan2(torch.hypot(x, y), z)
        ph = torch.atan2(y, x)
        t2 = th * th
        k = L.k
        r = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * (k[3] + t2 * k[4])))))
        cx, cy, rlim = self.limits[i]
        rpx = L.fx * r * L.s
        rl = _circ_interp(rlim, (ph % (2 * math.pi)) / (2 * math.pi) * rlim.shape[0])
        over = torch.relu(rpx - rl) + 1e3 * (th >= math.radians(TH_CUT))
        rc = torch.minimum(rpx, rl)          # pull rays outside the usable area back onto its edge
        u = cx + rc * torch.cos(ph)
        v = cy + rc * torch.sin(ph) * (L.fy / L.fx)
        g = torch.stack([u / L.W * 2 - 1, v / L.W * 2 - 1], -1)[None]
        col = F.grid_sample(img, g, align_corners=False, padding_mode="border")[0]
        gm = torch.stack([(cx + rpx * torch.cos(ph)) / L.W * 2 - 1, (cy + rpx * torch.sin(ph)) / L.W * 2 - 1], -1)[None]
        m = F.grid_sample(self.masks[i], gm, align_corners=False, padding_mode="zeros")[0, 0]
        m = m * (th < math.radians(TH_CUT))
        self._tls.over = over
        return col.clamp(0, 1) ** GAMMA, m

    # ---- per-frame analysis ---------------------------------------------------
    def update(self, imgs):
        cA, mA = self._sample(0, imgs[0], self.strip)
        cB, mB = self._sample(1, imgs[1], self.strip)
        both = ((mA > 0.5) & (mB > 0.5)).float()
        # exposure / colour: per-azimuth-bin means in linear light
        cpb = self.Wa // self.nb
        Wu = cpb * self.nb
        bw = both[:, :Wu].reshape(self.Hb, self.nb, cpb)
        cnt = bw.sum((0, 2))
        sa = (cA[:, :, :Wu].reshape(3, self.Hb, self.nb, cpb) * bw).sum((1, 3))
        sb = (cB[:, :, :Wu].reshape(3, self.Hb, self.nb, cpb) * bw).sum((1, 3))
        ok = (cnt > 0.03 * self.Hb * cpb).cpu().numpy()   # narrow overlaps (handle side) still count
        lr = torch.log((sa + 1e-3 * cnt + 1e-6) / (sb + 1e-3 * cnt + 1e-6)).cpu().numpy()
        lr = np.stack([_circ_smooth(_fill_circular(lr[c], ok), 1.5) for c in range(3)])
        if self.logr is None or not ok.any():
            self.logr = lr if self.logr is None else self.logr
        else:
            self.logr = self.gain_ema * self.logr + (1 - self.gain_ema) * lr
        if not getattr(self, "do_dp", True):
            return
        logr = torch.tensor(self.logr, dtype=torch.float32, device=self.dev)
        pos = (torch.arange(self.Wa, device=self.dev) * self.ares) / 360 * self.nb - 0.5
        lr_cols = _circ_interp(logr, pos % self.nb)[:, None, :]          # (3,1,Wa)
        a = (cA * torch.exp(-lr_cols / 2)) ** (1 / GAMMA)
        b = (cB * torch.exp(lr_cols / 2)) ** (1 / GAMMA)
        diff = (a - b).abs().sum(0)
        # gradient term helps the seam avoid crossing edges
        gx = lambda t: torch.nn.functional.pad(t[:, :, 1:] - t[:, :, :-1], (0, 1))
        gy = lambda t: torch.nn.functional.pad(t[:, 1:, :] - t[:, :-1, :], (0, 0, 0, 1))
        grad = (gx(a) - gx(b)).abs().sum(0) + (gy(a) - gy(b)).abs().sum(0)
        rows = torch.arange(self.Hb, device=self.dev, dtype=torch.float32)[:, None]
        lat = self.band - rows * self.bres
        cost = diff + 0.5 * grad + 3.0 * (1 - both) + 0.02 * lat.abs() / self.band
        if self.path is not None:
            prev = torch.tensor(self.path, dtype=torch.float32, device=self.dev)[None]
            cost = cost + self.lam_t * (lat - prev).abs() / self.band
        path_rows = _dp_closed_path(cost.cpu().numpy())
        new = self.band - path_rows * self.bres
        new = _circ_smooth(new, 2.0)
        self.path = new if self.path is None else self.seam_ema * self.path + (1 - self.seam_ema) * new

    # ---- composition ----------------------------------------------------------
    def render(self, imgs, d, parts=False):
        """d: (H, W, 3) body-frame directions -> (3, H, W) gamma-encoded image in [0, 1].
        parts=True also returns (slave contribution, master contribution, slave weight, master weight)."""
        if self.path is None:
            self.update(imgs)
        x, y, z = d.unbind(-1)
        alpha = torch.rad2deg(torch.atan2(z, x)) % 360
        beta = torch.rad2deg(torch.asin(y.clamp(-1, 1)))
        path = torch.tensor(self.path, dtype=torch.float32, device=self.dev)
        bstar = _circ_interp(path, (alpha / self.ares) % self.Wa)
        t = ((beta - bstar) / self.blend + 0.5).clamp(0, 1)
        sB = t * t * (3 - 2 * t)                                   # weight of master (+Y)
        logr = torch.tensor(self.logr, dtype=torch.float32, device=self.dev)
        g = logr.mean(1)[:, None, None]
        local = _circ_interp(logr, (alpha / 360 * self.nb - 0.5) % self.nb) - g
        fade = (1 - beta.abs() / 45).clamp(0, 1)
        lr = g + local * fade
        cA, mA = self._sample(0, imgs[0], d)
        oA = self._tls.over
        cB, mB = self._sample(1, imgs[1], d)
        oB = self._tls.over
        cA = cA * torch.exp(-lr / 2)
        cB = cB * torch.exp(lr / 2)
        # blind zone (neither lens usable): fall back to the lens whose usable edge is nearer
        px_deg = self.lenses[0].fx * self.lenses[0].s * math.pi / 180
        wA = (1 - sB) * mA + 1e-4 * mA + 1e-6 * torch.exp(-(oA / px_deg).clamp(max=60))
        wB = sB * mB + 1e-4 * mB + 1e-6 * torch.exp(-(oB / px_deg).clamp(max=60))
        out = (cA * wA + cB * wB) / (wA + wB)
        out = out.clamp(0, 1) ** (1 / GAMMA)
        if parts:
            n = wA + wB
            return out, cA.clamp(0, 1) ** (1 / GAMMA), cB.clamp(0, 1) ** (1 / GAMMA), wA / n, wB / n
        return out

    def gains_summary(self):
        lr = np.array(self.logr)
        return dict(global_ratio=np.exp(lr.mean(1)).round(3).tolist(),
                    min_ratio=float(np.exp(lr.mean(0)).min()), max_ratio=float(np.exp(lr.mean(0)).max()))


class FlowStitch(SmartStitch):
    """Parallax-aware blending: dense optical flow between the two lenses in the overlap
    band, then both images are morphed toward each other across a transition zone so that
    near objects stay single and continuous instead of doubling or being cut."""

    def __init__(self, cal, lenses, device=None, fband=4.5, fres=0.2, ramp=1.5, flow_ema=0.5,
                 max_flow=4.0, center="dp", ss=1, bands=1, wide_deg=6.0, **kw):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(cal, lenses, device=device, **kw)
        self.do_dp = False
        self.center = center          # "overlap": ramp centred in the overlap; "dp": on the min-disagreement path
        self.bands, self.wide_deg = int(bands), float(wide_deg)   # 2 = two-band blend (low freq. over wide_deg)
        self.fpath = None
        self.fband, self.fres, self.ramp, self.flow_ema, self.max_flow = fband, fres, ramp, flow_ema, max_flow
        a = torch.arange(0, 360, fres, device=device) * math.pi / 180
        b = torch.arange(fband, -fband - 1e-6, -fres, device=device) * math.pi / 180
        B, A = torch.meshgrid(b, a, indexing="ij")
        self.fstrip = torch.stack([torch.cos(B) * torch.cos(A), torch.sin(B), torch.cos(B) * torch.sin(A)], -1)
        self.fH, self.fW = self.fstrip.shape[:2]
        self.ss = int(ss)              # supersampling of the flow strip (anti-aliasing), 1 = off
        if self.ss > 1:
            r2 = fres / self.ss
            a2 = (torch.arange(0, 360, r2, device=device) + r2 * (self.ss - 1) / 2) * math.pi / 180
            b2 = (torch.arange(self.fH * self.ss, device=device) * -r2 + fband + r2 * (self.ss - 1) / 2) * math.pi / 180
            B2, A2 = torch.meshgrid(b2, a2, indexing="ij")
            self.fstrip_ss = torch.stack([torch.cos(B2) * torch.cos(A2), torch.sin(B2), torch.cos(B2) * torch.sin(A2)], -1)
        self.flow = None      # (2, fH, fW) in degrees: (d_alpha, d_beta), A(p) ~ B(p + F)
        self.last_conf = None

    def _gray(self, lin, gain):
        g = (lin * gain).clamp(0, 1) ** (1 / GAMMA)
        y = (0.299 * g[0] + 0.587 * g[1] + 0.114 * g[2])
        return (y * 255).byte().cpu().numpy()

    def update(self, imgs):
        super().update(imgs)                       # gains (+ DP path, kept for diagnostics)
        if self.ss > 1:
            pool = lambda x: F.avg_pool2d(x[None] if x.dim() == 3 else x[None, None], self.ss)[0] if x.dim() == 3                 else F.avg_pool2d(x[None, None], self.ss)[0, 0]
            cA, mA = self._sample(0, imgs[0], self.fstrip_ss)
            cB, mB = self._sample(1, imgs[1], self.fstrip_ss)
            cA, cB, mA, mB = pool(cA), pool(cB), pool(mA), pool(mB)
        else:
            cA, mA = self._sample(0, imgs[0], self.fstrip)
            cB, mB = self._sample(1, imgs[1], self.fstrip)
        logr = torch.tensor(self.logr, dtype=torch.float32, device=self.dev)
        pos = (torch.arange(self.fW, device=self.dev) * self.fres) / 360 * self.nb - 0.5
        lr = _circ_interp(logr, pos % self.nb)[:, None, :]
        gA, gB = self._gray(cA, torch.exp(-lr / 2)), self._gray(cB, torch.exp(lr / 2))
        both = ((mA > 0.5) & (mB > 0.5)).float().cpu().numpy()
        # usable overlap per azimuth column: [lo, hi] in degrees of beta
        betas = self.fband - np.arange(self.fH) * self.fres
        has = both.any(0)
        hi = np.where(has, np.max(np.where(both > 0, betas[:, None], -99), 0), 0.0)
        lo = np.where(has, np.min(np.where(both > 0, betas[:, None], 99), 0), 0.0)
        hi = np.minimum(hi, 6.0); lo = np.maximum(lo, -6.0)
        # the strip only spans +-fband: if the overlap reaches the strip edge, assume it continues
        hi = np.where(hi >= self.fband - 0.15, 6.0, hi); lo = np.where(lo <= -self.fband + 0.15, -6.0, lo)
        cen = _circ_smooth(np.clip((hi + lo) / 2, -3, 3), 10)
        half = _circ_smooth(np.clip((hi - lo) / 2 - 0.2, 0.15, self.ramp), 10)
        self.ov_c = torch.tensor(cen, dtype=torch.float32, device=self.dev)
        self.ov_w = torch.tensor(half, dtype=torch.float32, device=self.dev)
        pad = int(20 / self.fres)                   # 20 deg circular padding in alpha
        wrap = lambda x: np.concatenate([x[:, -pad:], x, x[:, :pad]], 1)
        fb = lambda a, b: cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 11, 4, 5, 1.1, 0)
        F_ = fb(wrap(gA), wrap(gB))[:, pad:-pad]
        # forward-backward consistency -> confidence
        Fb = fb(wrap(gB), wrap(gA))[:, pad:-pad]
        H, W = gA.shape
        xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        bx = cv2.remap(Fb[..., 0], (xs + F_[..., 0]) % W, ys + F_[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        by = cv2.remap(Fb[..., 1], (xs + F_[..., 0]) % W, ys + F_[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fb_err = np.hypot(F_[..., 0] + bx, F_[..., 1] + by)
        conf = both * np.exp(-(fb_err / 1.5) ** 2)
        # normalised convolution: fill unreliable areas from reliable neighbours, decay to 0
        sig = 1.5 / self.fres
        def blur(x):
            xx = np.concatenate([x[:, -int(3 * sig):], x, x[:, :int(3 * sig)]], 1)
            return cv2.GaussianBlur(xx, (0, 0), sig)[:, int(3 * sig):-int(3 * sig)]
        den = blur(conf) + 1e-3
        fx, fy = blur(F_[..., 0] * conf) / den, blur(F_[..., 1] * conf) / den
        fill = np.clip(den / 0.3, 0, 1)
        flow = np.stack([fx * self.fres, -fy * self.fres]) * fill   # px -> degrees; image y down = beta down
        mag = np.hypot(flow[0], flow[1])
        flow *= np.minimum(1, self.max_flow / (mag + 1e-6))
        if self.center == "dp":
            # half-way alignment of both lenses with the flow, then a min-disagreement closed path
            Fx, Fy = flow[0] / self.fres, -flow[1] / self.fres
            Ah = cv2.remap(gA.astype(np.float32), (xs - 0.5 * Fx) % W, ys - 0.5 * Fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            Bh = cv2.remap(gB.astype(np.float32), (xs + 0.5 * Fx) % W, ys + 0.5 * Fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            Ah, Bh = cv2.GaussianBlur(Ah, (0, 0), 1.0), cv2.GaussianBlur(Bh, (0, 0), 1.0)
            gxA, gxB = cv2.Sobel(Ah, cv2.CV_32F, 1, 0), cv2.Sobel(Bh, cv2.CV_32F, 1, 0)
            gyA, gyB = cv2.Sobel(Ah, cv2.CV_32F, 0, 1), cv2.Sobel(Bh, cv2.CV_32F, 0, 1)
            cost = (np.abs(Ah - Bh) + 0.25 * (np.abs(gxA - gxB) + np.abs(gyA - gyB))) / 255.0
            cost += 3.0 * (1 - both)
            # stay inside the overlap with room for the ramp; mild pull to the overlap centre
            room = np.minimum(betas[:, None] - lo[None, :], hi[None, :] - betas[:, None])
            cost += 2.0 * (room < 0.6) + 0.02 * np.abs(betas[:, None] - cen[None, :])
            if self.fpath is not None:
                cost += self.lam_t * np.abs(betas[:, None] - self.fpath[None, :]) / self.fband
            p = self.fband - _dp_closed_path(cost) * self.fres
            p = _circ_smooth(p, 3.0)
            self.fpath = p if self.fpath is None else self.seam_ema * self.fpath + (1 - self.seam_ema) * p
            w = np.clip(np.minimum(self.fpath - lo, hi - self.fpath) - 0.2, 0.15, self.ramp)
            self.ov_c = torch.tensor(self.fpath, dtype=torch.float32, device=self.dev)
            self.ov_w = torch.tensor(_circ_smooth(w, 5), dtype=torch.float32, device=self.dev)
            half = w
        flow *= np.minimum(1.0, half / self.ramp)[None, None, :]      # narrow transition -> gentler warp
        flow = torch.tensor(flow, dtype=torch.float32, device=self.dev)
        self.flow = flow if self.flow is None else self.flow_ema * self.flow + (1 - self.flow_ema) * flow
        self.last_conf = conf
        self.last_raw = F_          # raw Farneback flow (strip pixels), for diagnostics

    def _dirs(self, alpha_deg, beta_deg):
        a, b = torch.deg2rad(alpha_deg), torch.deg2rad(beta_deg)
        return torch.stack([torch.cos(b) * torch.cos(a), torch.sin(b), torch.cos(b) * torch.sin(a)], -1)

    def render(self, imgs, d, parts=False):
        if self.flow is None:
            self.update(imgs)
        x, y, z = d.unbind(-1)
        alpha = torch.rad2deg(torch.atan2(z, x)) % 360
        beta = torch.rad2deg(torch.asin(y.clamp(-1, 1)))
        col = (alpha / self.fres) % self.fW
        c = _circ_interp(self.ov_c, col)
        w = _circ_interp(self.ov_w, col)
        t = ((beta - c) / (2 * w) + 0.5).clamp(0, 1)
        t = t * t * (3 - 2 * t)                                     # 0 = slave (A), 1 = master (B)
        inb = (beta.abs() < self.fband - 0.2)
        gx = (alpha / 360) * 2 - 1
        gy = ((self.fband - beta) / (2 * self.fband)) * 2 - 1
        Fs = F.grid_sample(self.flow[None], torch.stack([gx, gy], -1)[None], align_corners=False,
                           padding_mode="zeros")[0] * inb
        fa, fb = Fs[0], Fs[1]
        dA = torch.where(inb[..., None], self._dirs(alpha - t * fa, beta - t * fb), d)
        dB = torch.where(inb[..., None], self._dirs(alpha + (1 - t) * fa, beta + (1 - t) * fb), d)
        logr = torch.tensor(self.logr, dtype=torch.float32, device=self.dev)
        g = logr.mean(1)[:, None, None]
        local = _circ_interp(logr, (alpha / 360 * self.nb - 0.5) % self.nb) - g
        fade = (1 - beta.abs() / 45).clamp(0, 1)
        lr = g + local * fade
        cA, mA = self._sample(0, imgs[0], dA)
        oA = self._tls.over
        cB, mB = self._sample(1, imgs[1], dB)
        oB = self._tls.over
        cA = cA * torch.exp(-lr / 2)
        cB = cB * torch.exp(lr / 2)
        px_deg = self.lenses[0].fx * self.lenses[0].s * math.pi / 180
        wA = (1 - t) * mA + 1e-4 * mA + 1e-6 * torch.exp(-(oA / px_deg).clamp(max=60))
        wB = t * mB + 1e-4 * mB + 1e-6 * torch.exp(-(oB / px_deg).clamp(max=60))
        out = (cA * wA + cB * wB) / (wA + wB)
        if getattr(self, "bands", 1) == 2:
            # two-band blend: low frequencies cross over a wide ramp (no brightness step),
            # high frequencies keep the narrow ramp (no ghosting)
            out = self._two_band(cA, cB, wA, wB, mA, mB, beta, c, w, d, out)
        out = out.clamp(0, 1) ** (1 / GAMMA)
        if parts:
            n = wA + wB
            return out, cA.clamp(0, 1) ** (1 / GAMMA), cB.clamp(0, 1) ** (1 / GAMMA), wA / n, wB / n
        return out

    def _two_band(self, cA, cB, wA, wB, mA, mB, beta, c, w, d, narrow):
        # output pixels per degree, from neighbouring view rays
        dd = (d[:, 1:] * d[:, :-1]).sum(-1).clamp(-1, 1)
        deg_px = float(torch.rad2deg(torch.acos(dd)).median())
        sig_full = max(1.0, (self.wide_deg / 2) / max(deg_px, 1e-4))
        fct = max(1, int(sig_full / 3))                    # blur on a downsampled copy for speed
        sig = max(1.0, sig_full / fct)
        k = int(3 * sig) * 2 + 1
        x = torch.arange(k, device=self.dev, dtype=torch.float32) - k // 2
        g = torch.exp(-0.5 * (x / sig) ** 2); g = (g / g.sum())
        H0, W0 = cA.shape[-2:]
        def blur(img):                                     # (C,H,W), normalised by the validity mask
            im = img[None]
            if fct > 1:
                im = F.avg_pool2d(im, fct, ceil_mode=True)
            im = F.conv2d(F.pad(im, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k).repeat(im.shape[1], 1, 1, 1), groups=im.shape[1])
            im = F.conv2d(F.pad(im, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1).repeat(im.shape[1], 1, 1, 1), groups=im.shape[1])
            if fct > 1:
                im = F.interpolate(im, size=(H0, W0), mode="bilinear", align_corners=False)
            return im[0]
        lowA = blur(cA * mA) / (blur(mA[None])[0] + 1e-4)
        lowB = blur(cB * mB) / (blur(mB[None])[0] + 1e-4)
        tw = ((beta - c) / (2 * self.wide_deg) + 0.5).clamp(0, 1)
        tw = tw * tw * (3 - 2 * tw)
        vA, vB = (mA > 0.5).float(), (mB > 0.5).float()
        uA = (1 - tw) * vA + 1e-4 * vA + 1e-6 * (beta < c)
        uB = tw * vB + 1e-4 * vB + 1e-6 * (beta >= c)
        low_wide = (lowA * uA + lowB * uB) / (uA + uB)
        low_narrow = (lowA * wA + lowB * wB) / (wA + wB)
        return (narrow - low_narrow + low_wide).clamp(0, 1)
