"""Are the two lens streams time-aligned?  For a frame, find the offset (in source frames) of lens 1
that best matches lens 0 on the seam strip; also report per-stream start times."""
import sys, math, subprocess, json, numpy as np, torch
from osvcut import Lens, dev
from stitch import load_calib
f=sys.argv[1]; times=[float(x) for x in sys.argv[2:]] or [12.0]
if len(f)==12 and all(ch in "0123456789abcdef" for ch in f):
    import library as _lib; f=_lib.Library([r"E:\Videos"]).get(f)["path"]
j=json.loads(subprocess.run(["ffprobe","-v","error","-show_entries","stream=index,codec_name,start_time,start_pts,r_frame_rate,nb_frames","-of","json",f],capture_output=True,text=True).stdout)
print([ (s["index"],s.get("codec_name"),s.get("start_time"),s.get("nb_frames")) for s in j["streams"] if s.get("codec_name")=="hevc"])
cal=load_calib(f); W=cal["W"]; L=[Lens(cal,x) for x in cal["lenses"]]
fps=float(eval(j["streams"][0]["r_frame_rate"]))
a=torch.arange(0,360,0.25,device=dev)*math.pi/180; b=torch.arange(3,-3,-0.25,device=dev)*math.pi/180
B,A=torch.meshgrid(b,a,indexing="ij"); D=torch.stack([torch.cos(B)*torch.cos(A),torch.sin(B),torch.cos(B)*torch.sin(A)],-1)
def frames(stream,t,n):
    raw=subprocess.run(["ffmpeg","-v","error","-ss",str(t),"-i",f,"-map",f"0:v:{stream}","-frames:v",str(n),"-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True).stdout
    arr=np.frombuffer(raw,np.uint8).reshape(-1,W,W,3)
    return [torch.from_numpy(x.copy()).to(dev).permute(2,0,1)[None].float()/255 for x in arr]
for t in times:
    K=12
    A0=frames(0,t,1)[0]; B1=frames(1,t-K/fps,2*K+1)
    ca=L[0].sample(A0,D)[0].mean(0)
    errs=[]
    for k,img in enumerate(B1):
        cb=L[1].sample(img,D)[0].mean(0)
        errs.append(float((ca-cb).abs().mean()))
    best=int(np.argmin(errs))-K
    print(f"t={t}: best lens-1 offset {best:+d} frames ({best/fps*1000:+.0f} ms); err at 0: {errs[K]:.4f}, at best: {min(errs):.4f}")
