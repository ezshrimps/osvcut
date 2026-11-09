"""Detect each lens's usable image circle (inside the lens bezel) from the clip itself."""
import numpy as np, cv2, subprocess
from stitch import grab
def mean_frames(f,stream,W,n=12,dur=None):
    if dur is None:
        dur=float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",f],capture_output=True,text=True).stdout)
    acc=np.zeros((W,W),np.float64)
    ts=[dur*(i+.5)/n for i in range(n)]
    for t in ts: acc+=cv2.cvtColor(grab(f,t,stream,W),cv2.COLOR_BGR2GRAY)
    return (acc/n).astype(np.float32)
def fit_circle(x,y):
    A=np.stack([x,y,np.ones_like(x)],1); b=x*x+y*y
    c,_,_,_=np.linalg.lstsq(A,b,rcond=None); cx,cy=c[0]/2,c[1]/2
    return cx,cy,np.sqrt(c[2]+cx*cx+cy*cy)
def detect(f,stream,W,cx0,cy0,n=12):
    M=cv2.GaussianBlur(mean_frames(f,stream,W,n),(0,0),1.0)
    R=W/2; phis=np.linspace(0,2*np.pi,720,endpoint=False); rs=np.arange(0.86*R,1.0*R,0.5)
    X=cx0+np.cos(phis)[:,None]*rs[None]; Y=cy0+np.sin(phis)[:,None]*rs[None]
    P=cv2.remap(M,X.astype(np.float32),Y.astype(np.float32),cv2.INTER_LINEAR,borderValue=0)
    dP=np.diff(P,axis=1)             # most negative = content -> bezel
    idx=np.argmin(dP,1); r=rs[idx]; strength=-dP[np.arange(len(idx)),idx]
    ok=strength>np.percentile(strength,30)
    xs,ys=X[np.arange(len(idx)),idx][ok],Y[np.arange(len(idx)),idx][ok]
    for _ in range(3):                # robust refit
        cx,cy,rr=fit_circle(xs,ys); res=np.abs(np.hypot(xs-cx,ys-cy)-rr)
        keep=res<max(2.0,np.percentile(res,70)); xs,ys=xs[keep],ys[keep]
    return dict(cx=cx,cy=cy,r=rr,M=M,pts=(X[np.arange(len(idx)),idx],Y[np.arange(len(idx)),idx]))
