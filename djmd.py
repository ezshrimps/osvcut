import subprocess, json, struct, sys
def djmd_index(f):
    j=json.loads(subprocess.run(["ffprobe","-v","error","-show_entries","stream=index,codec_tag_string","-of","json",f],capture_output=True,text=True).stdout)
    return [s["index"] for s in j["streams"] if s.get("codec_tag_string")=="djmd"][0]
def packets(f, stream=None, n=None, interval=None):
    if stream is None: stream=djmd_index(f)
    cmd=["ffprobe","-v","error","-select_streams",str(stream),"-show_packets","-show_data","-of","json"]
    if interval: cmd+=["-read_intervals",interval]
    cmd.append(f)
    j=json.loads(subprocess.run(cmd,capture_output=True,text=True).stdout)
    out=[]
    for p in j["packets"][:n]:
        hexs="".join(l[10:50].replace(" ","") for l in p["data"].strip().split("\n"))
        out.append((float(p.get("pts_time",0)),bytes.fromhex(hexs)))
    return out
def varint(b,i):
    r=s=0
    while True:
        c=b[i]; i+=1; r|=(c&0x7f)<<s; s+=7
        if c<0x80: return r,i
def parse(b):
    i=0; out=[]
    while i<len(b):
        k,i=varint(b,i); fn,wt=k>>3,k&7
        if wt==0: v,i=varint(b,i)
        elif wt==1: v=struct.unpack_from("<d",b,i)[0]; i+=8
        elif wt==5: v=struct.unpack_from("<f",b,i)[0]; i+=4
        elif wt==2:
            L,i=varint(b,i); v=b[i:i+L]; i+=L
        else: raise ValueError(wt)
        out.append((fn,wt,v))
    return out
def tree(b,depth=0,maxdepth=8):
    try: fields=parse(b)
    except Exception: return None
    res=[]
    for fn,wt,v in fields:
        if wt==2:
            sub=tree(v,depth+1) if depth<maxdepth and len(v)>0 else None
            if sub is None:
                try: v=v.decode("ascii") if all(32<=c<127 for c in v) else (list(struct.unpack(f"<{len(v)//4}f",v)) if len(v)%4==0 else v.hex()[:40])
                except Exception: v=v.hex()[:40]
                res.append((fn,v))
            else: res.append((fn,sub))
        else: res.append((fn,v))
    return res
def get(t,*path):
    for p in path:
        t=[v for fn,v in t if fn==p]
        if not t: return None
        t=t[0]
    return t
if __name__=="__main__":
    f=sys.argv[1]
    pk=packets(f,None,2,"%+#2")
    t=tree(pk[0][1])
    import pprint; pprint.pprint(t,width=200,compact=True)


def djmd_raw(f, stream=None):
    """The whole primary djmd track as bytes (fast: stream copy, no hex dump)."""
    if stream is None:
        stream = djmd_index(f)
    return subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-map", f"0:{stream}", "-c", "copy", "-f", "data", "-"],
                          capture_output=True).stdout


def frame_attitudes(f):
    """Per-frame (seconds_from_first_frame, quaternion w,x,y,z) for the whole clip.

    The track is a concatenation of ProductMeta messages; parsed as one message it is a flat list of
    top-level fields in which every frame contributes exactly one field 3 (FrameMeta)."""
    import numpy as np
    ts, qs = [], []
    for fn, wt, v in parse(djmd_raw(f)):
        if fn != 3 or wt != 2:
            continue
        t_us = None
        q = None
        for fn2, wt2, v2 in parse(v):
            if fn2 == 1 and wt2 == 2:
                for fn3, wt3, v3 in parse(v2):
                    if fn3 == 2 and wt3 == 0:
                        t_us = v3
            elif fn2 == 2 and wt2 == 2:
                for fn3, wt3, v3 in parse(v2):
                    if fn3 == 9 and wt3 == 2:
                        d = {a: b for a, _, b in parse(v3)}
                        q = [d.get(1, 0.0), d.get(2, 0.0), d.get(3, 0.0), d.get(4, 0.0)]
        if t_us is not None and q is not None:
            ts.append(t_us)
            qs.append(q)
    ts = np.array(ts, dtype=np.float64)
    return (ts - ts[0]) / 1e6 if len(ts) else ts, np.array(qs)
