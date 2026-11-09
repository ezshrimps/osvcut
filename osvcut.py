"""osvcut: OSV -> stitched / levelled / reframed video, GPU (torch).  Prototype."""
import json, subprocess, sys, math, numpy as np, torch, torch.nn.functional as F
from djmd import packets, tree, get, djmd_index
from stitch import load_calib, qmat
from seam import SmartStitch, FlowStitch
dev="cuda"
P=np.array([[0,-1,0],[-1,0,0],[0,0,1.]])   # imu axes -> body axes (empirical)

def attitudes(f,t0,t1):
    pk=packets(f,None,None,f"{max(t0-0.05,0)}%{t1+0.05}")
    ts=[];qs=[]
    for t,b in pk:
        a=get(tree(b),3,2,9)
        if a: ts.append(t); qs.append([dict(a).get(i,0.) for i in (1,2,3,4)])
    return np.array(ts),np.array(qs)

def body_frames(ts,qs,smooth_s=1.0,mode="level"):
    """per-sample 3x3 M (columns = out x,y,z expressed in body)"""
    n=len(ts); up=np.zeros((n,3)); fw_w=np.zeros((n,3)); Rs=[]
    for i,q in enumerate(qs):
        R=qmat(q); Rs.append(R)
        up[i]=P@R.T@np.array([0,0,1.])
        f=R@P@np.array([0,1,0.]); f[2]=0; fw_w[i]=f/np.linalg.norm(f)
    dt=np.median(np.diff(ts)) if n>1 else .01; k=max(1,int(smooth_s/dt))
    ker=np.ones(k)/k
    if mode=="lock": sm=np.tile(fw_w[0],(n,1))
    else: sm=np.stack([np.convolve(np.pad(fw_w[:,j],(k//2,k-1-k//2),mode="edge"),ker,"valid") for j in range(3)],1)
    Ms=[]
    for i in range(n):
        z=up[i]
        if mode=="none": Ms.append(np.eye(3)); continue
        y=P@Rs[i].T@sm[i]; y=y-z*(y@z); y/=np.linalg.norm(y); x=np.cross(y,z)
        Ms.append(np.stack([x,y,z],1))
    return np.array(Ms)

class Lens:
    def __init__(self,cal,L):
        self.R=torch.tensor(qmat(L["q"]),dtype=torch.float32,device=dev)
        self.k=L["k"]; self.fx,self.fy,self.cx,self.cy=L["fx"],L["fy"],L["cx"],L["cy"]
        self.s=cal["dfl"]/L["fx"]; self.W=cal["W"]
    def sample(self,img,d):   # img 1x3xWxW, d HxWx3 body dirs
        d=d@self.R.T; x,y,z=d.unbind(-1)
        th=torch.atan2(torch.hypot(x,y),z); ph=torch.atan2(y,x); t2=th*th; k=self.k
        r=th*(1+t2*(k[0]+t2*(k[1]+t2*(k[2]+t2*(k[3]+t2*k[4])))))
        u=self.W/2+(self.fx*r*torch.cos(ph)+self.cx-1920)*self.s
        v=self.W/2+(self.fy*r*torch.sin(ph)+self.cy-1920)*self.s
        g=torch.stack([u/self.W*2-1,v/self.W*2-1],-1)[None]
        col=F.grid_sample(img,g,align_corners=False,padding_mode="zeros")[0]
        a,b=math.radians(85),math.radians(97)
        w=((b-th)/(b-a)).clamp(0,1); w=w*w*(3-2*w)
        return col,w

def view_dirs(mode,Wo,Ho,fov=None):
    if mode=="equirect":
        lon=((torch.arange(Wo,device=dev)+.5)/Wo-.5)*2*math.pi; lat=(.5-(torch.arange(Ho,device=dev)+.5)/Ho)*math.pi
        lat,lon=torch.meshgrid(lat,lon,indexing="ij")
        return torch.stack([torch.sin(lon)*torch.cos(lat),torch.cos(lon)*torch.cos(lat),torch.sin(lat)],-1)
    f=(Wo/2)/math.tan(math.radians(fov)/2)
    xs=(torch.arange(Wo,device=dev)+.5-Wo/2)/f; zs=-(torch.arange(Ho,device=dev)+.5-Ho/2)/f
    zs,xs=torch.meshgrid(zs,xs,indexing="ij")
    d=torch.stack([xs,torch.ones_like(xs),zs],-1); return d/d.norm(dim=-1,keepdim=True)

def cam_rot(yaw,pitch):
    y,p=math.radians(yaw),math.radians(pitch)
    Rz=np.array([[math.cos(y),math.sin(y),0],[-math.sin(y),math.cos(y),0],[0,0,1]])  # +yaw = turn right
    Rx=np.array([[1,0,0],[0,math.cos(p),-math.sin(p)],[0,math.sin(p),math.cos(p)]])
    return Rz@Rx

def interp_kf(kfs,t):
    kfs=sorted(kfs,key=lambda k:k["t"])
    if t<=kfs[0]["t"]: return kfs[0]
    for a,b in zip(kfs,kfs[1:]):
        if t<=b["t"]:
            s=(t-a["t"])/(b["t"]-a["t"]); s=s*s*(3-2*s)
            dy=((b["yaw"]-a["yaw"]+180)%360)-180
            return dict(yaw=a["yaw"]+dy*s,pitch=a["pitch"]+(b["pitch"]-a["pitch"])*s,fov=a["fov"]+(b["fov"]-a["fov"])*s)
    return kfs[-1]

def render(spec):
    f=spec["src"]; t0,t1=spec["in"],spec["out"]; mode=spec.get("proj","flat")
    Wo,Ho=spec.get("size",[1920,1080]); fps=spec.get("fps",50)
    cal=load_calib(f); W=cal["W"]; lenses=[Lens(cal,L) for L in cal["lenses"]]
    mode_seam=spec.get("seam","flow")
    smart=mode_seam!="legacy"
    eng=(FlowStitch if mode_seam=="flow" else SmartStitch)(cal,lenses,**spec.get("seam_opts",{})) if smart else None
    ts,qs=attitudes(f,t0,t1); Ms=body_frames(ts,qs,spec.get("smooth",1.0),spec.get("stabilize","level"))
    kfs=spec.get("keys",[{"t":0,"yaw":0,"pitch":0,"fov":100}])
    srcfps=float(eval(subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=r_frame_rate","-of","csv=p=0",f],capture_output=True,text=True).stdout.strip().split("\n")[0]))
    step=max(1,round(srcfps/fps)); fps_out=srcfps/step
    dec=subprocess.Popen(["ffmpeg","-v","error","-ss",str(t0),"-to",str(t1),"-i",f,"-filter_complex",
        f"[0:v:0][0:v:1]hstack,select='not(mod(n\,{step}))'","-fps_mode","passthrough","-f","rawvideo","-pix_fmt","rgb24","-"],stdout=subprocess.PIPE)
    enc=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{Wo}x{Ho}","-r",str(fps_out),"-i","-",
        "-ss",str(t0),"-to",str(t1),"-i",f,"-map","0:v","-map","1:a?","-c:v","h264_nvenc","-preset","p5","-cq",str(spec.get("cq",20)),"-pix_fmt","yuv420p","-c:a","aac","-shortest",spec["dst"]],stdin=subprocess.PIPE)
    fb=W*2*W*3; i=0
    total=max(1,int((t1-t0)*fps_out)); eq_base=None
    while True:
        raw=dec.stdout.read(fb)
        if len(raw)<fb: break
        t=t0+i*step/srcfps; i+=1
        im=torch.frombuffer(bytearray(raw),dtype=torch.uint8).to(dev).view(W,2*W,3).permute(2,0,1).float()/255
        imgs=[im[None,:,:,:W],im[None,:,:,W:]]
        M=Ms[min(np.searchsorted(ts,t),len(Ms)-1)]
        if i%10==0: print(f"progress {i}/{total}",flush=True)
        if mode=="equirect":
            if eq_base is None: eq_base=view_dirs("equirect",Wo,Ho)
            base=eq_base; C=np.eye(3)
        else:
            k=interp_kf(kfs,t-t0); base=view_dirs("flat",Wo,Ho,k["fov"]); C=cam_rot(k["yaw"],k["pitch"])
        Mt=torch.tensor(M@C,dtype=torch.float32,device=dev)
        d=base@Mt.T
        if smart:
            eng.update(imgs); out=eng.render(imgs,d)
        else:
            out=0; ws=0
            pairs=list(zip(lenses,imgs))
            if mode_seam=="single0": pairs=pairs[:1]          # reference: slave lens only, no seam
            for L,img in pairs:
                c,w=L.sample(img,d) if mode_seam!="single0" else (L.sample(img,d)[0],1.0)
                out=out+c*w; ws=ws+w
            out=(out/(ws+1e-6)).clamp(0,1)
        enc.stdin.write((out*255).byte().permute(1,2,0).contiguous().cpu().numpy().tobytes())
    enc.stdin.close(); enc.wait(); dec.wait()
    print(f"progress {total}/{total}",flush=True)
    print("rendered",i,"frames ->",spec["dst"],flush=True)

if __name__=="__main__":
    render(json.load(open(sys.argv[1],encoding="utf8")))
