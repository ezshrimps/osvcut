import numpy as np, cv2, sys, subprocess
from djmd import packets, tree, get

def load_calib(f):
    t=tree(packets(f,None,1,"%+#1")[0][1])
    clip=get(t,1); stream=get(t,2)
    dfl=get(clip,8,1); W=get(stream,3,1)
    lenses=[]
    for slot in (1,2):  # native_refine slave(stream0), master(stream1)
        d=dict((k,v) for k,v in get(stream,6,slot) if not isinstance(v,list) or k in(21,22,23))
        q=[v for k,v in get(get(stream,6,slot),28)] if False else [dict(get(stream,6,slot,28))[i] for i in (1,2,3,4)]
        lenses.append(dict(fx=d[1],fy=d[2],cx=d[3],cy=d[4],k=[d[5],d[6],d[7],d[8],d.get(15,0)],q=q,
                           occ=list(zip(d[22],d[23])) if 22 in d and 23 in d else None))
    return dict(dfl=dfl,W=W,lenses=lenses)

def qmat(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def equirect_dirs(Wo,Ho):
    px=(np.arange(Wo)+0.5)/Wo; py=(np.arange(Ho)+0.5)/Ho
    lon=(px-0.5)*2*np.pi; lat=(0.5-py)*np.pi
    lon,lat=np.meshgrid(lon,lat)
    return np.stack([np.sin(lon)*np.cos(lat),np.cos(lon)*np.cos(lat),np.sin(lat)],-1)

def lens_maps(dirs,cal,L,feather_deg=(85,97)):
    R=qmat(L["q"]); d=dirs@R.T
    x,y,z=d[...,0],d[...,1],d[...,2]
    th=np.arctan2(np.hypot(x,y),z); ph=np.arctan2(y,x)
    k=L["k"]; t2=th*th
    r=th*(1+k[0]*t2+k[1]*t2**2+k[2]*t2**3+k[3]*t2**4+k[4]*t2**5)
    s=cal["dfl"]/L["fx"]; Ws=cal["W"]
    u=Ws/2+(L["fx"]*r*np.cos(ph)+L["cx"]-1920)*s
    v=Ws/2+(L["fy"]*r*np.sin(ph)+L["cy"]-1920)*s
    a,b=np.radians(feather_deg)
    w=np.clip((b-th)/(b-a),0,1); w=w*w*(3-2*w)
    return u.astype(np.float32),v.astype(np.float32),w.astype(np.float32)

class Stitcher:
    def __init__(self,cal,Wo=3840,Ho=1920):
        dirs=equirect_dirs(Wo,Ho)
        self.m=[lens_maps(dirs,cal,L) for L in cal["lenses"]]
        ws=self.m[0][2]+self.m[1][2]+1e-6
        self.w=[(m[2]/ws)[...,None] for m in self.m]
    def __call__(self,f0,f1):
        out=0
        for img,(u,v,_),w in zip((f0,f1),self.m,self.w):
            out=out+cv2.remap(img,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(np.float32)*w
        return np.clip(out,0,255).astype(np.uint8)

def grab(f,t,stream,W):
    raw=subprocess.run(["ffmpeg","-v","error","-ss",str(t),"-i",f,"-map",f"0:{stream}","-frames:v","1","-f","rawvideo","-pix_fmt","bgr24","-"],capture_output=True).stdout
    return np.frombuffer(raw,np.uint8).reshape(W,W,3)

if __name__=="__main__":
    f,t,out=sys.argv[1],float(sys.argv[2]),sys.argv[3]
    cal=load_calib(f); print(cal)
    st=Stitcher(cal)
    eq=st(grab(f,t,0,cal["W"]),grab(f,t,1,cal["W"]))
    cv2.imwrite(out,eq)

def level_rot(q):
    """returns M with d_body = M @ d_out : horizon-levelled, heading follows camera +Y"""
    R=qmat(q)                       # body -> world
    fwd=R@np.array([0,1,0.]); yaw=np.arctan2(fwd[0],fwd[1])
    c,s=np.cos(yaw),np.sin(yaw)
    Rz=np.array([[c,s,0],[-s,c,0],[0,0,1]])   # rotates out +Y onto world heading
    return R.T@Rz

class LevelStitcher(Stitcher):
    def __init__(self,cal,M,Wo=3840,Ho=1920):
        self.cal=cal; dirs=equirect_dirs(Wo,Ho)@M.T
        self.m=[lens_maps(dirs,cal,L) for L in cal["lenses"]]
        ws=self.m[0][2]+self.m[1][2]+1e-6
        self.w=[(m[2]/ws)[...,None] for m in self.m]
