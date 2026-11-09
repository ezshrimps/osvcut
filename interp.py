"""Optical-flow frame interpolation for slow motion beyond the native frame rate.

    mids = Interpolator().between(a, b, [0.25, 0.5, 0.75])

a, b: (3,H,W) float tensors in 0..1 (rendered output frames, GPU or CPU).  Flow in both directions is estimated
once per pair (Farneback on a downscaled grey image); each intermediate frame warps both ends to time p with the
quadratic flow approximation of Super SloMo (Jiang et al. 2018),
    F_p->0 = -(1-p) p F_0->1 + p^2 F_1->0,   F_p->1 = (1-p)^2 F_0->1 - p (1-p) F_1->0,
and blends them with weights (1-p, p) times a forward-backward consistency term, so content that is visible in only
one end comes from that end.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class Interpolator:
    def __init__(self, scale=0.5, fb=(0.5, 4, 21, 4, 7, 1.5)):
        self.scale = scale
        self.fb = fb

    def _grey(self, x):
        g = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).clamp(0, 1)
        g = (g * 255).byte().cpu().numpy()
        if self.scale != 1:
            g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return g

    def _flow(self, g0, g1, H, W, dev):
        pyr, lev, win, it, n, sig = self.fb
        f = cv2.calcOpticalFlowFarneback(g0, g1, None, pyr, lev, win, it, n, sig, 0)
        f = torch.from_numpy(f).to(dev).permute(2, 0, 1)[None]          # (1,2,h,w) in low-res pixels
        f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False) / self.scale
        return f[0]

    @staticmethod
    def _warp(img, flow):
        """Sample img at x + flow(x) (backward warp)."""
        _, H, W = img.shape
        ys, xs = torch.meshgrid(torch.arange(H, device=img.device, dtype=torch.float32),
                                torch.arange(W, device=img.device, dtype=torch.float32), indexing="ij")
        gx = (xs + flow[0]) / (W - 1) * 2 - 1
        gy = (ys + flow[1]) / (H - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1)[None]
        return F.grid_sample(img[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0]

    def between(self, a, b, phases):
        if not phases:
            return []
        _, H, W = a.shape
        g0, g1 = self._grey(a), self._grey(b)
        f01 = self._flow(g0, g1, H, W, a.device)
        f10 = self._flow(g1, g0, H, W, a.device)
        # forward-backward consistency: where following F_0->1 then F_1->0 does not come back, the pixel is
        # occluded in the other frame (soft weight)
        c0 = torch.exp(-(f01 + self._warp(f10, f01)).pow(2).sum(0) / 2.0)
        c1 = torch.exp(-(f10 + self._warp(f01, f10)).pow(2).sum(0) / 2.0)
        out = []
        for p in phases:
            ft0 = -(1 - p) * p * f01 + p * p * f10
            ft1 = (1 - p) ** 2 * f01 - p * (1 - p) * f10
            w0 = (1 - p) * (self._warp(c0[None], ft0)[0] + 1e-3)
            w1 = p * (self._warp(c1[None], ft1)[0] + 1e-3)
            o = (w0 * self._warp(a, ft0) + w1 * self._warp(b, ft1)) / (w0 + w1)
            out.append(o.clamp(0, 1))
        return out
