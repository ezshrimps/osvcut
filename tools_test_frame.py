import sys,numpy as np,cv2,torch,subprocess
from osvcut import Lens, dev, attitudes, body_frames, view_dirs
from stitch import load_calib, grab
from seam import SmartStitch, FlowStitch
f,t,x0,y0,out=sys.argv[1],float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
E=FlowStitch if (len(sys.argv)>6 and sys.argv[6]=='flow') else SmartStitch
cal=load_calib(f); lenses=[Lens(cal,L) for L in cal["lenses"]]
ts,qs=attitudes(f,max(0,t-1),t+1); Ms=body_frames(ts,qs,1.0,"level"); M=Ms[min(np.searchsorted(ts,t),len(Ms)-1)]
imgs=[torch.from_numpy(grab(f,t,s,cal["W"])[:,:,::-1].copy()).to(dev).permute(2,0,1)[None].float()/255 for s in (0,1)]
d=view_dirs("equirect",3840,1920)@torch.tensor(M,dtype=torch.float32,device=dev).T
eng=E(cal,lenses)
for _ in range(3): eng.update(imgs)
im=(eng.render(imgs,d)*255).byte().permute(1,2,0).cpu().numpy()[:,:,::-1].copy()
cv2.imwrite(out,im[y0:y0+520,x0:x0+900])
