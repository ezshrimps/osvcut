"""How well does the flow used for morphing actually align the two lenses?
Residual |A(p) - B(p+F)| on the seam strip for: no warp, raw Farneback flow, the processed flow."""
import sys, math, numpy as np, torch, cv2
from osvcut import Lens, dev
from stitch import load_calib
from seam import FlowStitch, _circ_interp
from eval_seams import grab_pair
f,t=sys.argv[1],float(sys.argv[2])
cal=load_calib(f); W=cal["W"]; lenses=[Lens(cal,L) for L in cal["lenses"]]
imgs=grab_pair(f,t,W)
e=FlowStitch(cal,lenses,flow_ema=0); e.update(imgs)
cA,mA=e._sample(0,imgs[0],e.fstrip); cB,mB=e._sample(1,imgs[1],e.fstrip)
logr=torch.tensor(e.logr,dtype=torch.float32,device=dev)
pos=(torch.arange(e.fW,device=dev)*e.fres)/360*e.nb-0.5
lr=_circ_interp(logr,pos%e.nb)[:,None,:]
gA=e._gray(cA,torch.exp(-lr/2)).astype(np.float32); gB=e._gray(cB,torch.exp(lr/2)).astype(np.float32)
both=((mA>0.5)&(mB>0.5)).cpu().numpy()
H,Wd=gA.shape
core=both.copy(); core[:int(H*0.2)]=False; core[int(H*0.8):]=False
xs,ys=np.meshgrid(np.arange(Wd,dtype=np.float32),np.arange(H,dtype=np.float32))
def warpB(Fx,Fy): return cv2.remap(gB,(xs+Fx)%Wd,ys+Fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
pad=int(20/e.fres); wrap=lambda x:np.concatenate([x[:,-pad:],x,x[:,:pad]],1)
raw=cv2.calcOpticalFlowFarneback(wrap(gA.astype(np.uint8)),wrap(gB.astype(np.uint8)),None,0.5,3,11,4,5,1.1,0)[:,pad:-pad]
fl=e.flow.cpu().numpy()   # degrees (d_alpha, d_beta); pixels: x = dα/res, y = -dβ/res
Fx,Fy=fl[0]/e.fres,-fl[1]/e.fres
tex=(cv2.Sobel(gA,cv2.CV_32F,1,0)**2+cv2.Sobel(gA,cv2.CV_32F,0,1)**2)>200
m=core&tex
r=lambda X:np.abs(gA-X)[m].mean()
print(f"{f.split('/')[-1]} t={t}: textured core px={m.sum()}")
print(f"  no warp        : {r(gB):6.2f}")
print(f"  raw flow       : {r(warpB(raw[...,0],raw[...,1])):6.2f}   |F| med {np.median(np.hypot(raw[...,0],raw[...,1])[m])*e.fres:.2f} deg")
print(f"  processed flow : {r(warpB(Fx,Fy)):6.2f}   |F| med {np.median(np.hypot(Fx,Fy)[m])*e.fres:.2f} deg")
print(f"  sign-flipped   : {r(warpB(-Fx,-Fy)):6.2f}")

# ---- anti-aliased strips: sample at 0.05 deg, area-downsample to 0.2 deg ----
import math as _m
def strip(res,band=4.5):
    a=torch.arange(0,360,res,device=dev)*_m.pi/180; b=torch.arange(band,-band-1e-6,-res,device=dev)*_m.pi/180
    B,A=torch.meshgrid(b,a,indexing="ij"); return torch.stack([torch.cos(B)*torch.cos(A),torch.sin(B),torch.cos(B)*torch.sin(A)],-1)
def gray_at(res, lr_scale=True):
    D=strip(res)
    cA,mA=e._sample(0,imgs[0],D); cB,mB=e._sample(1,imgs[1],D)
    pos=(torch.arange(D.shape[1],device=dev)*res)/360*e.nb-0.5
    lr=_circ_interp(logr,pos%e.nb)[:,None,:]
    return e._gray(cA,torch.exp(-lr/2)).astype(np.float32), e._gray(cB,torch.exp(lr/2)).astype(np.float32)
hA,hB=gray_at(0.05)
k=4
aA=cv2.resize(hA,(hA.shape[1]//k,gA.shape[0]),interpolation=cv2.INTER_AREA)
aB=cv2.resize(hB,(hB.shape[1]//k,gA.shape[0]),interpolation=cv2.INTER_AREA)
raw2=cv2.calcOpticalFlowFarneback(wrap(aA.astype(np.uint8)),wrap(aB.astype(np.uint8)),None,0.5,3,11,4,5,1.1,0)[:,pad:-pad]
wB2=lambda Fx,Fy: cv2.remap(aB,(xs+Fx)%Wd,ys+Fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
r2=lambda X:np.abs(aA-X)[m].mean()
print(f"  [anti-aliased] no warp {r2(aB):6.2f} | raw flow {r2(wB2(raw2[...,0],raw2[...,1])):6.2f}  |F| med {np.median(np.hypot(raw2[...,0],raw2[...,1])[m])*e.fres:.2f} deg")
