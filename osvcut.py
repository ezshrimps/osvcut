"""osvcut: OSV -> stitched / levelled / reframed video, GPU (torch).  Prototype."""
import json, os, subprocess, sys, math, numpy as np, torch, torch.nn.functional as F
from djmd import packets, tree, get, djmd_index
from stitch import load_calib, qmat, lens_scale
from seam import SmartStitch, FlowStitch
dev="cuda" if torch.cuda.is_available() else "cpu"
P=np.array([[0,-1,0],[-1,0,0],[0,0,1.]])   # imu axes -> body axes (empirical)

_ATT_CACHE={}
def attitudes(f,t0,t1):
    """(seconds, quaternions) for frames in [t0, t1] from the primary djmd track."""
    from djmd import frame_attitudes
    if f not in _ATT_CACHE: _ATT_CACHE[f]=frame_attitudes(f)
    ts,qs=_ATT_CACHE[f]
    sel=(ts>=t0-0.05)&(ts<=t1+0.05)
    return ts[sel],qs[sel]

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

_LVL={}
def level_frames(f,stabilize="level",smooth=1.0):
    """Stabilisation frames for the WHOLE clip (cached), so a time maps to the same camera
    orientation in previews, exports and tracking regardless of the requested range."""
    key=(f,stabilize,float(smooth))
    if key not in _LVL:
        ts,qs=attitudes(f,-1,1e12); _LVL[key]=(ts,body_frames(ts,qs,smooth,stabilize))
    return _LVL[key]

class Lens:
    def __init__(self,cal,L):
        self.R=torch.tensor(qmat(L["q"]),dtype=torch.float32,device=dev)
        self.k=L["k"]; self.fx,self.fy,self.cx,self.cy=L["fx"],L["fy"],L["cx"],L["cy"]
        self.s=lens_scale(cal,L); self.W=cal["W"]
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

def view_dirs_eye(Wo,Ho,fov,dist):
    """Eye-offset ("generalised stereographic") virtual camera, same maths as the viewer shader:
    dist=0 rectilinear, dist=1 stereographic (little planet / crystal ball).  Returns (dirs, valid)."""
    dist=float(max(0.0,min(1.0,dist)))
    fmax=2*math.degrees(math.acos(-dist))-1
    h=math.radians(min(fov,fmax))/2
    f=(Wo/2)*(dist+math.cos(h))/((1+dist)*math.sin(h))
    px=torch.arange(Wo,device=dev)+.5-Wo/2; py=-(torch.arange(Ho,device=dev)+.5-Ho/2)
    py,px=torch.meshgrid(py,px,indexing="ij")
    k=torch.hypot(px,py)/(f*(1+dist)); s=k*dist/torch.sqrt(1+k*k)
    valid=s<1
    th=torch.atan(k)+torch.asin(s.clamp(max=1)); ph=torch.atan2(py,px)
    d=torch.stack([torch.sin(th)*torch.cos(ph),torch.cos(th),torch.sin(th)*torch.sin(ph)],-1)
    return d,valid

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
            s=(t-a["t"])/(b["t"]-a["t"])
            if a.get("ease")!="linear": s=s*s*(3-2*s)
            dy=((b["yaw"]-a["yaw"]+180)%360)-180
            return dict(yaw=a["yaw"]+dy*s,pitch=a["pitch"]+(b["pitch"]-a["pitch"])*s,fov=a["fov"]+(b["fov"]-a["fov"])*s,
                        dist=a.get("dist",0)+(b.get("dist",0)-a.get("dist",0))*s)
    return kfs[-1]


import threading, queue

class FrameReader:
    """Decode both lens streams side by side as yuv420p in a background thread (NVDEC when available)."""
    def __init__(self,f,t0,t1,step,W,hw=True):
        self.args=(f,t0,t1,step,W); self.fb=W*2*W*3//2; self.q=queue.Queue(maxsize=4)
        self.p=self._start(hw)
        first=self.p.stdout.read(self.fb)
        if len(first)<self.fb and hw:           # NVDEC unavailable / failed: fall back to CPU decoding
            self.p.kill(); self.p=self._start(False); first=self.p.stdout.read(self.fb)
        self.first=first if len(first)==self.fb else None
        self.th=threading.Thread(target=self._run,daemon=True); self.th.start()
    def _start(self,hw):
        f,t0,t1,step,W=self.args
        return subprocess.Popen(["ffmpeg","-v","error"]+(["-hwaccel","cuda"] if hw else [])+["-ss",str(t0),"-to",str(t1),"-i",f,
            "-filter_complex",(f"[0:v:0][0:v:1]hstack,select='not(mod(n\\,{step}))'" if step<=2 else
                f"[0:v:0]select='not(mod(n\\,{step}))'[a];[0:v:1]select='not(mod(n\\,{step}))'[b];[a][b]hstack"),"-fps_mode","passthrough",
            "-f","rawvideo","-pix_fmt","yuv420p","-"],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    def _run(self):
        if self.first is not None:
            self.q.put(self.first)
            while True:
                raw=self.p.stdout.read(self.fb)
                if len(raw)<self.fb: break
                self.q.put(raw)
        self.q.put(None)
    def __iter__(self):
        while True:
            raw=self.q.get()
            if raw is None: return
            yield raw
    def close(self):
        try: self.p.kill()
        except Exception: pass

class FrameWriter:
    """Feed the encoder from a background thread so GPU work overlaps encoding."""
    def __init__(self,enc):
        self.enc=enc; self.q=queue.Queue(maxsize=4)
        self.th=threading.Thread(target=self._run,daemon=True); self.th.start()
    def _run(self):
        while True:
            b=self.q.get()
            if b is None: break
            try: self.enc.stdin.write(b)
            except (BrokenPipeError, OSError): self.dead=True
        try: self.enc.stdin.close()
        except (BrokenPipeError, OSError): pass
    def put(self,b): self.q.put(b)
    def close(self): self.q.put(None); self.th.join()

def yuv420_to_rgb(raw,H,Wd):
    """yuv420p (limited range, BT.709) bytes -> (3,H,Wd) float RGB on the GPU."""
    b=torch.frombuffer(bytearray(raw),dtype=torch.uint8).to(dev)
    n=H*Wd
    Y=b[:n].view(1,1,H,Wd).float()
    UV=b[n:].view(2,1,H//2,Wd//2).float()
    UV=F.interpolate(UV,size=(H,Wd),mode="bilinear",align_corners=False)
    y=(Y[0,0]-16)*(1/219); u=(UV[0,0]-128)*(1/224); v=(UV[1,0]-128)*(1/224)
    kr,kb=0.2126,0.0722; kg=1-kr-kb
    R=y+2*(1-kr)*v; B=y+2*(1-kb)*u; G=(y-kr*R-kb*B)/kg
    return torch.stack([R,G,B]).clamp(0,1)

def nadir_patch(img,d_out,radius_deg):
    """Cover the area straight below the camera (hand, stick, shadow) with a smooth fill: the colour of a
    ring just outside the cap, per azimuth, blended in with a soft edge.  d_out: levelled view directions."""
    z=d_out[...,2]; el=torch.rad2deg(torch.asin((-z).clamp(-1,1)))       # elevation below the horizon
    az=torch.atan2(d_out[...,0],d_out[...,1])                              # -pi..pi
    lim=90-radius_deg
    ring=(el>lim-6)&(el<lim-1)
    nb=36; b=((az/(2*math.pi)+0.5)*nb).long().clamp(0,nb-1)
    cols=torch.zeros(3,nb,device=img.device); cnt=torch.zeros(nb,device=img.device)
    rb=b[ring]; cols.index_add_(1,rb,img[:,ring]); cnt.index_add_(0,rb,torch.ones_like(rb,dtype=torch.float32))
    have=cnt>0
    if not have.any(): return img
    mean=cols.sum(1,keepdim=True)/cnt.sum()
    cols=torch.where(have[None],cols/cnt.clamp(min=1)[None],mean)
    k=torch.tensor([1,2,3,2,1.],device=img.device); k=k/k.sum()            # circular smoothing of the ring colours
    cols=F.conv1d(F.pad(cols[:,None],(2,2),mode="circular"),k.view(1,1,5))[:,0]
    pos=(az/(2*math.pi)+0.5)*nb-0.5; i0=torch.floor(pos).long()%nb; fr=pos-torch.floor(pos)
    fill=cols[:,i0]*(1-fr)+cols[:,(i0+1)%nb]*fr
    # towards the pole the fill fades to the mean ring colour (no azimuthal pinch)
    tp=((el-lim)/radius_deg).clamp(0,1); fill=fill*(1-tp)+mean[:,:,None]*tp if fill.dim()==3 else fill
    w=((el-(lim-1))/2).clamp(0,1)
    return img*(1-w)+fill*w

def grade(img,g):
    """Basic colour grade on a gamma-encoded (3,H,W) image, same maths as the studio shader.
    g: {exposure: EV, contrast: -1..1, saturation: -1..1, temperature: -1..1 (warm > 0)}."""
    ev=float(g.get("exposure",0)); ct=float(g.get("contrast",0)); sat=float(g.get("saturation",0)); tmp=float(g.get("temperature",0))
    if not (ev or ct or sat or tmp): return img
    lin=img.clamp(0,1)**2.2*(2.0**ev)
    wb=torch.tensor([1+0.1*tmp,1.0,1-0.1*tmp],device=img.device).view(3,1,1)
    x=(lin*wb).clamp(0,1)**(1/2.2)
    x=(x-0.5)*(1+ct)+0.5
    y=(0.2126*x[0]+0.7152*x[1]+0.0722*x[2])[None]
    return (y+(x-y)*(1+sat)).clamp(0,1)

def _srt_time(x):
    ms=int(round(max(0.0,x)*1000)); h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); s_,ms=divmod(ms,1000)
    return f"{h:02d}:{m:02d}:{s_:02d},{ms:03d}"

def add_captions(dst,captions,t0,t1,speed=1.0,mode="burn",size=(1920,1080)):
    """captions: [{start, end, text}] in source seconds.  Writes <dst>.srt next to the video and either burns the
    lines into the picture (libass, NVENC re-encode) or muxes them as a soft subtitle track."""
    lines=[]
    for c in captions:
        a,b=max(float(c["start"]),t0),min(float(c["end"]),t1)
        if b-a<0.15 or not str(c.get("text","")).strip(): continue
        lines.append(((a-t0)/speed,(b-t0)/speed,str(c["text"]).strip()))
    if not lines: return 0
    srt=dst[:-4]+".srt"
    with open(srt,"w",encoding="utf8") as fh:
        for k,(a,b,txt) in enumerate(lines,1):
            fh.write(f"{k}\n{_srt_time(a)} --> {_srt_time(b)}\n{txt}\n\n")
    tmp=dst[:-4]+"_cap.mp4"
    if mode=="soft":
        cmd=["ffmpeg","-v","error","-y","-i",dst,"-i",srt,"-map","0","-map","1","-c","copy","-c:s","mov_text",
             "-metadata:s:s:0","language=und",tmp]
    else:
        # SRT is rendered on libass's default 384x288 canvas and scaled to the frame; sizes are in canvas units.
        # Measured: 14.4 gives ~5 % of the height in landscape; portrait frames scale by width and need ~12.5
        fs=14.4 if size[0]>=size[1] else 12.5
        # libass wants forward slashes and an escaped drive colon inside the filter argument
        esc=os.path.abspath(srt).replace("\\","/").replace(":","\\:")
        style=(f"FontName=Microsoft YaHei,FontSize={fs},PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=1,"
               f"Outline={max(0.6,fs/12):.1f},Shadow=0,MarginV={int(fs*1.3)},MarginL={int(fs)},MarginR={int(fs)}")
        cmd=["ffmpeg","-v","error","-y","-i",dst,"-vf",f"subtitles='{esc}':force_style='{style}'",
             "-c:v","h264_nvenc","-preset","p5","-cq","19","-c:a","copy","-movflags","+faststart",tmp]
        if subprocess.run(cmd).returncode!=0:
            cmd[cmd.index("h264_nvenc")]="libx264"; i=cmd.index("-preset"); cmd[i+1]="medium"; i=cmd.index("-cq"); cmd[i:i+2]=["-crf","19"]
    subprocess.run(cmd,check=True)
    os.replace(tmp,dst)
    return len(lines)

def render(spec):
    f=spec["src"]; t0,t1=spec["in"],spec["out"]; mode=spec.get("proj","flat")
    Wo,Ho=spec.get("size",[1920,1080]); fps=spec.get("fps",50)
    cal=load_calib(f); W=cal["W"]; lenses=[Lens(cal,L) for L in cal["lenses"]]
    mode_seam=spec.get("seam","flow")
    smart=mode_seam!="legacy"
    eng=(FlowStitch if mode_seam=="flow" else SmartStitch)(cal,lenses,**spec.get("seam_opts",{})) if smart else None
    ts,Ms=level_frames(f,spec.get("stabilize","level"),spec.get("smooth",1.0))
    kfs=spec.get("keys",[{"t":0,"yaw":0,"pitch":0,"fov":100}])
    srcfps=float(eval(subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=r_frame_rate","-of","csv=p=0",f],capture_output=True,text=True).stdout.strip().split("\n")[0]))
    speed=float(spec.get("speed",1.0))
    # local slow motion: {"from": a, "to": b, "speed": s} in source seconds, normal speed elsewhere
    sl=spec.get("slow")
    if sl:
        if abs(speed-1)>1e-6 or spec.get("freeze"): raise ValueError("slow cannot be combined with speed or freeze")
        sl_a,sl_b,sl_s=max(t0,float(sl["from"])),min(t1,float(sl["to"])),float(sl.get("speed",0.25))
        if not (sl_b>sl_a and 0.05<=sl_s<1): sl=None
    if sl:
        step=1; fps_out=float(fps)
    elif abs(speed-1)<1e-6:
        step=max(1,round(srcfps/fps)); fps_out=srcfps/step
    else:                                   # slow motion / fast forward: pick source frames by output time
        step=1; fps_out=float(fps)
    reader=FrameReader(f,t0,t1,step,W,hw=spec.get("decode","nvdec")!="cpu")
    # freeze orbit ("bullet time"): at source time `at` the picture stops for `dur` s while the camera turns
    fz=spec.get("freeze")
    if fz and abs(speed-1)>1e-6: raise ValueError("freeze cannot be combined with speed")
    fz_at=float(fz["at"]) if fz else None; fz_n=int(round(float(fz.get("dur",2.0))*(srcfps/step))) if fz else 0
    fz_turn=float(fz.get("turn",360.0)) if fz else 0.0
    def timed_frames():
        if sl:                                # variable rate: advance source time by 1/fps_out x local speed
            tout=t0
            for j,raw in enumerate(reader):
                tsrc=t0+j/srcfps
                while tout<=tsrc+0.5/srcfps and tout<t1:
                    yield tsrc,raw,None
                    tout+=(sl_s if sl_a<=tout<sl_b else 1.0)/fps_out
            return
        if abs(speed-1)<1e-6:
            done=False
            for j,raw in enumerate(reader):
                t=t0+j*step/srcfps
                yield t,raw,None
                if fz and not done and t>=fz_at:
                    done=True
                    for q in range(fz_n): yield t,raw,(q+1)/fz_n
        else:
            n=0
            for j,raw in enumerate(reader):
                tsrc=t0+j/srcfps
                while t0+n*speed/fps_out<=tsrc+0.5/srcfps and t0+n*speed/fps_out<t1:
                    yield tsrc,raw,None; n+=1
    atempo=[]
    if abs(speed-1)>1e-6:
        sp=speed
        while sp<0.5: atempo.append(0.5); sp/=0.5
        while sp>2.0: atempo.append(2.0); sp/=2.0
        atempo.append(sp)
    amap=["-map","1:a?"]
    if sl and bool(subprocess.run(["ffprobe","-v","error","-select_streams","a","-show_entries","stream=index","-of","csv=p=0",f],capture_output=True,text=True).stdout.strip()):
        af="aformat=sample_rates=48000:channel_layouts=stereo"; r1,r2=sl_a-t0,sl_b-t0
        tempo=[]; x=sl_s
        while x<0.5: tempo.append(0.5); x/=0.5
        tempo.append(x)
        amap=["-filter_complex",f"[1:a]{af},asplit=3[x][y][z];[x]atrim=0:{r1:.3f},asetpts=PTS-STARTPTS[a1];"
              f"[y]atrim={r1:.3f}:{r2:.3f},asetpts=PTS-STARTPTS,"+",".join(f"atempo={v:.6f}" for v in tempo)+"[a2];"
              f"[z]atrim={r2:.3f},asetpts=PTS-STARTPTS[a3];[a1][a2][a3]concat=n=3:v=0:a=1[aout]","-map","[aout]"]
    if fz:
        has_a=bool(subprocess.run(["ffprobe","-v","error","-select_streams","a","-show_entries","stream=index","-of","csv=p=0",f],capture_output=True,text=True).stdout.strip())
        if has_a:
            rel=max(0.0,fz_at-t0); fdur=fz_n/(srcfps/step); af="aformat=sample_rates=48000:channel_layouts=stereo"
            amap=["-filter_complex",f"[1:a]{af},asplit=2[x][y];[x]atrim=0:{rel:.3f},asetpts=PTS-STARTPTS[a1];[y]atrim={rel:.3f},asetpts=PTS-STARTPTS[a2];"
                  f"anullsrc=r=48000:cl=stereo,atrim=0:{fdur:.3f}[sl];[a1][sl][a2]concat=n=3:v=0:a=1[aout]","-map","[aout]"]
    enc=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{Wo}x{Ho}","-r",str(fps_out),"-i","-",
        "-ss",str(t0),"-to",str(t1),"-i",f,"-map","0:v"]+amap+(["-af",",".join(f"atempo={a:.6f}" for a in atempo)] if atempo else [])+["-c:v","h264_nvenc","-preset","p5","-cq",str(spec.get("cq",20)),"-pix_fmt","yuv420p","-c:a","aac"]+(["-shortest"] if not atempo else [])+(["-r",str(fps)] if spec.get("force_fps") else [])+[spec["dst"]],stdin=subprocess.PIPE)
    writer=FrameWriter(enc)
    from concurrent.futures import ThreadPoolExecutor
    seam_async=bool(spec.get("seam_async",True)) and smart
    pool=ThreadPoolExecutor(1) if seam_async else None; pending=None; eng_ready=True
    every=max(1,int(spec.get("seam_every",1))); i=0
    total=max(1,int((t1-t0)/speed*fps_out))+fz_n+(int((sl_b-sl_a)*(1/sl_s-1)*fps_out) if sl else 0); eq_base=None
    for t,raw,fzp in timed_frames():
        i+=1
        im=yuv420_to_rgb(raw,W,2*W)
        imgs=[im[None,:,:,:W],im[None,:,:,W:]]
        M=Ms[min(np.searchsorted(ts,t),len(Ms)-1)]
        if i%10==0: print(f"progress {i}/{total}",flush=True)
        vmask=None
        dyaw=0.0
        if fzp is not None:
            u=fzp*fzp*(3-2*fzp); dyaw=fz_turn*u
        if mode=="equirect":
            if eq_base is None: eq_base=view_dirs("equirect",Wo,Ho)
            base=eq_base; C=cam_rot(dyaw,0) if dyaw else np.eye(3)
        else:
            k=interp_kf(kfs,t-t0); C=cam_rot(k["yaw"]+dyaw,k["pitch"])
            if k.get("dist",0)>1e-3: base,vmask=view_dirs_eye(Wo,Ho,k["fov"],k["dist"])
            else: base=view_dirs("flat",Wo,Ho,k["fov"])
        Mt=torch.tensor(M@C,dtype=torch.float32,device=dev)
        d=base@Mt.T
        if smart:
            if seam_async:
                # analyse in a worker thread; render with the latest finished analysis (<= 1 frame behind)
                if pending is None or pending.done():
                    if pending is not None: pending.result()
                    if eng_ready or (i-1)%every==0: pending=pool.submit(eng.update,imgs); eng_ready=False
                if not getattr(eng,"_has_state",False):
                    pending.result(); eng._has_state=True
            elif (i-1)%every==0: eng.update(imgs)
            out=eng.render(imgs,d)
        else:
            out=0; ws=0
            pairs=list(zip(lenses,imgs))
            if mode_seam=="single0": pairs=pairs[:1]          # reference: slave lens only, no seam
            for L,img in pairs:
                c,w=L.sample(img,d) if mode_seam!="single0" else (L.sample(img,d)[0],1.0)
                out=out+c*w; ws=ws+w
            out=(out/(ws+1e-6)).clamp(0,1)
        if vmask is not None: out=out*vmask
        if spec.get("nadir_patch",0)>0:
            out=nadir_patch(out,base@torch.tensor(C,dtype=torch.float32,device=dev).T,float(spec["nadir_patch"]))
        if spec.get("grade"): out=grade(out,spec["grade"])
        writer.put((out*255).byte().permute(1,2,0).contiguous().cpu().numpy().tobytes())
    if pending is not None: pending.result()
    if pool is not None: pool.shutdown()
    writer.close(); enc.wait(); reader.close()
    if spec.get("captions"):
        n=add_captions(spec["dst"],spec["captions"],t0,t1,speed,spec.get("captions_mode","burn"),(Wo,Ho))
        print("captions:",n,"lines",flush=True)
    if spec.get("spherical") and mode=="equirect":
        import spherical; spherical.inject(spec["dst"])
    print(f"progress {total}/{total}",flush=True)
    print("rendered",i,"frames ->",spec["dst"],flush=True)

if __name__=="__main__":
    render(json.load(open(sys.argv[1],encoding="utf8")))
