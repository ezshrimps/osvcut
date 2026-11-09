"""Seam diagnostics: unwrap both lenses along the seam band and measure misalignment / exposure mismatch."""
import sys, math, numpy as np, cv2, torch
from osvcut import Lens, dev
from stitch import load_calib, grab
RES=0.1  # deg per px
def strips(f,t,band=8.0):
    cal=load_calib(f); lenses=[Lens(cal,L) for L in cal["lenses"]]
    imgs=[torch.from_numpy(grab(f,t,s,cal["W"])[:,:,::-1].copy()).to(dev).permute(2,0,1)[None].float()/255 for s in (0,1)]
    a=torch.arange(0,360,RES,device=dev)*math.pi/180
    b=torch.arange(band,-band,-RES,device=dev)*math.pi/180   # +b = toward master (+Y)
    B,A=torch.meshgrid(b,a,indexing="ij")
    d=torch.stack([torch.cos(B)*torch.cos(A),torch.sin(B),torch.cos(B)*torch.sin(A)],-1)
    out=[]
    for L,img in zip(lenses,imgs):
        c,w=L.sample(img,d); out.append((c.clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy()[:,:,::-1].copy())
    return out
def analyse(f,t,tag):
    s0,s1=strips(f,t)
    g0,g1=[cv2.cvtColor(s,cv2.COLOR_BGR2GRAY) for s in (s0,s1)]
    flow=cv2.calcOpticalFlowFarneback(g0,g1,None,0.5,4,21,5,7,1.5,0)
    H,W=g0.shape; core=slice(H//2-30,H//2+30)   # +-3 deg around seam
    tex=cv2.Sobel(g0,cv2.CV_32F,1,0)**2+cv2.Sobel(g0,cv2.CV_32F,0,1)**2
    rows=[]
    for x0 in range(0,W,150):   # 15 deg blocks
        sl=(core,slice(x0,x0+150))
        m=tex[sl]>np.percentile(tex,60)
        if m.sum()<200: rows.append((x0*RES,None,None,None)); continue
        fx,fy=flow[sl][...,0][m],flow[sl][...,1][m]
        br=(g0[sl].mean()+1)/(g1[sl].mean()+1)
        rows.append((x0*RES,np.median(np.abs(fx))*RES,np.median(np.abs(fy))*RES,br))
    good=[r for r in rows if r[1] is not None]
    print(f"== {tag} t={t}")
    print(" 角度段   沿缝错位°  跨缝错位°  亮度比(从/主)")
    for a,fx,fy,br in rows:
        print(f" {a:5.0f}°  " + ("   (无纹理)" if fx is None else f"{fx:8.2f}  {fy:9.2f}   {br:6.2f}"))
    tot=np.hypot([r[1] for r in good],[r[2] for r in good])
    print(f" 中位错位 {np.median(tot):.2f}°  最大 {tot.max():.2f}°  (1°≈{3840/360:.1f}px @3840 全景)")
    # visual: lens A / lens B / 50-50 blend / diff, 0..360 split in 2 rows
    blend=((s0.astype(np.float32)+s1)/2).astype(np.uint8); diff=cv2.absdiff(s0,s1)*3
    vis=[]
    for half in (slice(0,W//2),slice(W//2,W)):
        for im,lab in ((s0,"slave"),(s1,"master"),(blend,"50/50"),(diff,"|diff|x3")):
            p=im[:,half].copy(); cv2.putText(p,lab,(6,22),0,.7,(0,255,255),2); vis.append(p)
        vis.append(np.zeros((6,W//2,3),np.uint8))
    cv2.imwrite(f"seam_{tag}.jpg",np.vstack(vis))
if __name__=="__main__":
    analyse(sys.argv[1],float(sys.argv[2]),sys.argv[3])
