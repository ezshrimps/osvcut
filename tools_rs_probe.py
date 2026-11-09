"""Does seam misalignment grow with camera rotation speed? (rolling-shutter probe)
For frames across a clip: median |flow| in the overlap (confident pixels) vs IMU angular speed."""
import sys, numpy as np, torch, subprocess
from osvcut import Lens, dev, attitudes
from stitch import load_calib
from seam import FlowStitch
import analysis
f=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 40
cal=load_calib(f); W=cal["W"]; lenses=[Lens(cal,L) for L in cal["lenses"]]
ts,qs=attitudes(f,-1,1e12)
q=qs/np.linalg.norm(qs,axis=1,keepdims=True); ang=np.degrees(2*np.arccos(np.abs((q[1:]*q[:-1]).sum(1)).clip(0,1)))/np.diff(ts).clip(1e-3)
dur=ts[-1]; rows=[]
for k in range(n):
    t=dur*(k+.5)/n
    raw=subprocess.run(["ffmpeg","-v","error","-ss",str(t),"-i",f,"-filter_complex","[0:v:0][0:v:1]hstack","-frames:v","1","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True).stdout
    im=torch.frombuffer(bytearray(raw),dtype=torch.uint8).to(dev).view(W,2*W,3).permute(2,0,1).float()/255
    imgs=[im[None,:,:,:W],im[None,:,:,W:]]
    e=FlowStitch(cal,lenses,flow_ema=0); e.update(imgs)
    fl=e.flow.cpu().numpy(); conf=e.last_conf
    m=conf>0.5
    if m.sum()<500: continue
    mag=np.hypot(fl[0],fl[1])[m]
    i=np.searchsorted(ts,t); w=np.median(ang[max(0,i-5):i+5])
    rows.append((t,w,np.median(mag),np.percentile(mag,90)))
r=np.array(rows)
print("t  speed(deg/s)  flow_med(deg)  flow_p90")
for x in r: print("%6.1f %8.1f %8.2f %8.2f"%tuple(x))
print("corr(speed, median flow) = %.2f"%np.corrcoef(r[:,1],r[:,2])[0,1])
print("slope: %.4f deg flow per deg/s"%np.polyfit(r[:,1],r[:,2],1)[0])
