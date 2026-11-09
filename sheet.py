"""AI-readable overview: levelled equirect thumbnails with yaw ticks + timestamps."""
import sys, numpy as np, cv2, torch, subprocess, json
from seam import FlowStitch
from osvcut import *
from stitch import grab
def frame_eq(f,t,cal,lenses,M,Wo=1024,Ho=512):
    imgs=[torch.from_numpy(grab(f,t,s,cal["W"])[:,:,::-1].copy()).to(dev).permute(2,0,1)[None].float()/255 for s in (0,1)]
    vs=[i for i,s in enumerate(json.loads(subprocess.run(["ffprobe","-v","error","-show_entries","stream=index,codec_name","-of","json",f],capture_output=True,text=True).stdout)["streams"]) if s.get("codec_name")=="hevc"]
    d=view_dirs("equirect",Wo,Ho)@torch.tensor(M,dtype=torch.float32,device=dev).T
    eng=FlowStitch(cal,lenses)
    for _ in range(3): eng.update(imgs)   # single frame: settle the gain/seam estimates
    im=(eng.render(imgs,d)*255).byte().permute(1,2,0).cpu().numpy()[:,:,::-1].copy()
    for yaw in range(-180,181,45):
        x=int((yaw/360+.5)*Wo); cv2.line(im,(x,0),(x,12),(0,255,255),2); cv2.putText(im,str(yaw),(max(x-14,0),28),0,.5,(0,255,255),1)
    return im
def make_sheet(f,n,out):
    dur=float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",f],capture_output=True,text=True).stdout)
    cal=load_calib(f); lenses=[Lens(cal,L) for L in cal["lenses"]]
    times=[dur*(i+.5)/n for i in range(n)]; tiles=[]
    ts,qs=attitudes(f,0,dur) if dur<120 else (None,None)
    for t in times:
        if ts is None: tt,qq=attitudes(f,t-1,t+1)
        else: tt,qq=ts,qs
        Ms=body_frames(tt,qq,1.0,"level"); M=Ms[min(np.searchsorted(tt,t),len(Ms)-1)]
        im=frame_eq(f,t,cal,lenses,M); cv2.putText(im,f"t={t:.1f}s",(8,500),0,.8,(0,0,255),2); tiles.append(im)
    cols=2; rows=[np.hstack(tiles[i:i+cols]+[np.zeros_like(tiles[0])]*(cols-len(tiles[i:i+cols]))) for i in range(0,len(tiles),cols)]
    cv2.imwrite(out,np.vstack(rows)); print(dur, times)
if __name__=="__main__": make_sheet(sys.argv[1],int(sys.argv[2]),sys.argv[3])
