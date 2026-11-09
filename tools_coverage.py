"""Coverage map: which lens supplies each equirect pixel (red = slave only, blue = master only,
green = both / overlap, white = blind zone), overlaid on the stitched frame."""
import sys,numpy as np,cv2,torch
from osvcut import Lens, dev, attitudes, body_frames, view_dirs
from stitch import load_calib, grab
from seam import FlowStitch
f,t,out=sys.argv[1],float(sys.argv[2]),sys.argv[3]
crop=[int(v) for v in sys.argv[4:8]] if len(sys.argv)>7 else None
cal=load_calib(f); lenses=[Lens(cal,L) for L in cal["lenses"]]
ts,qs=attitudes(f,max(0,t-1),t+1); Ms=body_frames(ts,qs,1.0,"level"); M=Ms[min(np.searchsorted(ts,t),len(Ms)-1)]
imgs=[torch.from_numpy(grab(f,t,s,cal["W"])[:,:,::-1].copy()).to(dev).permute(2,0,1)[None].float()/255 for s in (0,1)]
d=view_dirs("equirect",3840,1920)@torch.tensor(M,dtype=torch.float32,device=dev).T
e=FlowStitch(cal,lenses)
for _ in range(2): e.update(imgs)
im=(e.render(imgs,d)*255).byte().permute(1,2,0).cpu().numpy()[:,:,::-1].astype(np.float32)
_,mA=e._sample(0,imgs[0],d); _,mB=e._sample(1,imgs[1],d)
a=(mA>0.5).cpu().numpy(); b=(mB>0.5).cpu().numpy()
col=np.zeros_like(im); col[a&~b]=(60,60,255); col[b&~a]=(255,120,60); col[a&b]=(60,255,60); col[~a&~b]=(255,255,255)
ov=(im*0.55+col*0.45).astype(np.uint8)
if crop: x0,y0,w,h=crop; ov=ov[y0:y0+h,x0:x0+w]
cv2.imwrite(out,ov)
print("blind-zone fraction of sphere: %.4f%%"%(100*((~a&~b)*np.cos(np.linspace(np.pi/2,-np.pi/2,1920))[:,None]).sum()/(np.cos(np.linspace(np.pi/2,-np.pi/2,1920)).sum()*3840)))
