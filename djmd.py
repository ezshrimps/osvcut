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
