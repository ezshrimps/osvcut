"""Before/after along the seam: legacy wide feather vs SmartStitch."""
import sys, math, numpy as np, cv2, torch, time
from osvcut import Lens, dev, body_frames, attitudes, view_dirs
from stitch import load_calib, grab
from seam import SmartStitch, FlowStitch
def strip_dirs(band=8, res=0.1):
    a=torch.arange(0,360,res,device=dev)*math.pi/180; b=torch.arange(band,-band,-res,device=dev)*math.pi/180
    B,A=torch.meshgrid(b,a,indexing="ij")
    return torch.stack([torch.cos(B)*torch.cos(A),torch.sin(B),torch.cos(B)*torch.sin(A)],-1)
def to_img(t): return (t.clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy()[:,:,::-1].copy()
def run(f,t,tag):
    cal=load_calib(f); lenses=[Lens(cal,L) for L in cal["lenses"]]
    imgs=[torch.from_numpy(grab(f,t,s,cal["W"])[:,:,::-1].copy()).to(dev).permute(2,0,1)[None].float()/255 for s in (0,1)]
    d=strip_dirs()
    out=0; ws=0
    for L,img in zip(lenses,imgs):
        c,w=L.sample(img,d); out=out+c*w; ws=ws+w
    old=to_img(out/(ws+1e-6))
    eng=SmartStitch(cal,lenses)
    t0=time.time()
    for _ in range(4): eng.update(imgs)
    torch.cuda.synchronize(); t_upd=(time.time()-t0)/4
    new=to_img(eng.render(imgs,d))
    feng=FlowStitch(cal,lenses)
    for _ in range(4): feng.update(imgs)
    flow=to_img(feng.render(imgs,d))
    # draw seam path on a copy of new
    path=eng.path; H,W=new.shape[:2]; band=8; res=0.1
    marked=new.copy()
    for j,b in enumerate(path):
        x=int(j*eng.ares/res); y=int((band-b)/res)
        cv2.circle(marked,(x,y),1,(0,255,255),-1)
    print(tag,"update ms %.1f"%(t_upd*1000),eng.gains_summary())
    rows=[]
    for half in (slice(0,W//2),slice(W//2,W)):
        for im,lab in ((old,"legacy feather"),(new,"smart cut"),(flow,"flow morph")):
            p=im[:,half].copy(); cv2.putText(p,lab,(6,22),0,.7,(0,255,255),2); rows.append(p)
        rows.append(np.zeros((6,W//2,3),np.uint8))
    cv2.imwrite(f"samples/seamfix_{tag}.jpg",np.vstack(rows))
if __name__=="__main__":
    run(sys.argv[1],float(sys.argv[2]),sys.argv[3])
