"""Dense pyramidal Lucas-Kanade optical flow on the GPU (PyTorch), for short wide seam strips.

flow_lk(A, B) returns F (H, W, 2) in pixels with A(p) ~ B(p + F), plus a confidence map from the
smallest eigenvalue of the structure tensor and the residual after warping.  The strip wraps
around in x (azimuth), so x is padded circularly.
"""
import torch
import torch.nn.functional as F


def _gauss(sig, dev):
    k = int(3 * sig) * 2 + 1
    x = torch.arange(k, device=dev, dtype=torch.float32) - k // 2
    g = torch.exp(-0.5 * (x / sig) ** 2)
    return g / g.sum()


def _blur(x, g):
    k = g.numel()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k).expand(x.shape[1], 1, 1, k), groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1).expand(x.shape[1], 1, k, 1), groups=x.shape[1])


def _warp(img, flow):
    """Sample img (1,1,H,W) at p + flow (1,2,H,W) in pixels."""
    _, _, H, W = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=img.device, dtype=torch.float32),
                            torch.arange(W, device=img.device, dtype=torch.float32), indexing="ij")
    gx = (xs + flow[0, 0]) / (W - 1) * 2 - 1
    gy = (ys + flow[0, 1]) / (H - 1) * 2 - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1)[None], mode="bilinear", padding_mode="border", align_corners=True)


def flow_lk(A, B, levels=3, iters=4, win=2.5, pad=64, damp=50.0):
    """A, B: (H, W) float tensors in [0, 255]. Returns (flow (H, W, 2), confidence (H, W))."""
    dev = A.device
    H, W = A.shape
    a = torch.cat([A[:, -pad:], A, A[:, :pad]], 1)[None, None]
    b = torch.cat([B[:, -pad:], B, B[:, :pad]], 1)[None, None]
    g_pre, g_win = _gauss(0.8, dev), _gauss(win, dev)
    pa, pb = [_blur(a, g_pre)], [_blur(b, g_pre)]
    for _ in range(levels - 1):
        if pa[-1].shape[-2] < 12:
            break
        pa.append(F.avg_pool2d(pa[-1], 2, ceil_mode=True))
        pb.append(F.avg_pool2d(pb[-1], 2, ceil_mode=True))
    flow = torch.zeros(1, 2, *pa[-1].shape[-2:], device=dev)
    sx = torch.tensor([[-0.5, 0, 0.5]], device=dev).view(1, 1, 1, 3)
    sy = sx.view(1, 1, 3, 1)
    for la, lb in zip(reversed(pa), reversed(pb)):
        if flow.shape[-2:] != la.shape[-2:]:
            flow = F.interpolate(flow, size=la.shape[-2:], mode="bilinear", align_corners=False) * 2
        Ix = F.conv2d(F.pad(la, (1, 1, 0, 0), mode="replicate"), sx)
        Iy = F.conv2d(F.pad(la, (0, 0, 1, 1), mode="replicate"), sy)
        Sxx, Sxy, Syy = _blur(Ix * Ix, g_win), _blur(Ix * Iy, g_win), _blur(Iy * Iy, g_win)
        Sxx_d, Syy_d = Sxx + damp, Syy + damp            # Tikhonov damping: no runaway in flat areas
        det = Sxx_d * Syy_d - Sxy * Sxy
        for _ in range(iters):
            bw = _warp(lb, flow)
            It = bw - la
            Sxt, Syt = _blur(Ix * It, g_win), _blur(Iy * It, g_win)
            # inverse-compositional style step on A's gradients: solve S d = -[Sxt, Syt]
            du = -(Syy_d * Sxt - Sxy * Syt) / det
            dv = -(Sxx_d * Syt - Sxy * Sxt) / det
            flow = flow + torch.cat([du, dv], 1).clamp(-2, 2)
    # confidence: texture (min eigenvalue) and residual after the final warp
    tr = Sxx + Syy
    lam_min = 0.5 * (tr - torch.sqrt((Sxx - Syy) ** 2 + 4 * Sxy * Sxy + 1e-6))
    res = _blur((_warp(pb[0], flow) - pa[0]).abs(), g_win)
    conf = (1 - torch.exp(-lam_min / 20.0)) * torch.exp(-(res / 30.0) ** 2)
    fl = flow[0, :, :, pad:-pad].permute(1, 2, 0)
    return fl, conf[0, 0, :, pad:-pad]
