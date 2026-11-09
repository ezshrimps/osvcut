"""Tag an equirectangular MP4 as monoscopic 360 video (Google Spherical Video V1, XMP in a uuid box).

    python spherical.py in.mp4 [out.mp4]       # in place when out is omitted

The uuid box is appended to the first video `trak`.  Sizes of `trak` and `moov` are updated, and if
`moov` comes before `mdat` (faststart) every chunk offset in `stco` / `co64` is shifted by the number
of inserted bytes.
"""
import os
import struct
import sys

SPHERICAL_UUID = bytes.fromhex("ffcc8263f8554a938814587a02521fdd")
XMP = ('<?xml version="1.0"?><rdf:SphericalVideo xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
       'xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">'
       '<GSpherical:Spherical>true</GSpherical:Spherical><GSpherical:Stitched>true</GSpherical:Stitched>'
       '<GSpherical:StitchingSoftware>osvcut</GSpherical:StitchingSoftware>'
       '<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType></rdf:SphericalVideo>').encode()

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}


def boxes(buf, start, end):
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[i + 8:i + 16])[0]; hdr = 16
        elif size == 0:
            size = end - i
        yield i, size, typ, hdr
        i += size


def find(buf, start, end, path):
    for off, size, typ, hdr in boxes(buf, start, end):
        if typ == path[0]:
            if len(path) == 1:
                return off, size, hdr
            r = find(buf, off + hdr, off + size, path[1:])
            if r:
                return r
    return None


def is_video_trak(buf, off, size, hdr):
    h = find(buf, off + hdr, off + size, [b"mdia", b"hdlr"])
    return bool(h) and buf[h[0] + h[2] + 8:h[0] + h[2] + 12] == b"vide"


def shift_offsets(buf, start, end, delta):
    for off, size, typ, hdr in boxes(buf, start, end):
        if typ in CONTAINERS:
            shift_offsets(buf, off + hdr, off + size, delta)
        elif typ in (b"stco", b"co64"):
            n = struct.unpack(">I", buf[off + hdr + 4:off + hdr + 8])[0]
            p = off + hdr + 8
            fmt, w = (">I", 4) if typ == b"stco" else (">Q", 8)
            for k in range(n):
                v = struct.unpack(fmt, buf[p + k * w:p + (k + 1) * w])[0]
                buf[p + k * w:p + (k + 1) * w] = struct.pack(fmt, v + delta)


def inject(src, dst=None):
    data = bytearray(open(src, "rb").read())
    top = list(boxes(data, 0, len(data)))
    moov = next((b for b in top if b[2] == b"moov"), None)
    mdat = next((b for b in top if b[2] == b"mdat"), None)
    if not moov:
        raise ValueError("no moov box")
    m_off, m_size, _, m_hdr = moov
    trak = next(((o, s, h) for o, s, t, h in boxes(data, m_off + m_hdr, m_off + m_size)
                 if t == b"trak" and is_video_trak(data, o, s, h)), None)
    if not trak:
        raise ValueError("no video track")
    if SPHERICAL_UUID in data[trak[0]:trak[0] + trak[1]]:
        return dst or src                                 # already tagged
    box = struct.pack(">I4s", 8 + 16 + len(XMP), b"uuid") + SPHERICAL_UUID + XMP
    t_off, t_size, t_hdr = trak
    ins = t_off + t_size
    if mdat and m_off < mdat[0]:                          # chunk offsets point past moov: shift them
        shift_offsets(data, m_off + m_hdr, m_off + m_size, len(box))
    for off, hdr in ((t_off, t_hdr), (m_off, m_hdr)):     # grow trak and moov
        if hdr == 8:
            size = struct.unpack(">I", data[off:off + 4])[0]
            data[off:off + 4] = struct.pack(">I", size + len(box))
        else:
            size = struct.unpack(">Q", data[off + 8:off + 16])[0]
            data[off + 8:off + 16] = struct.pack(">Q", size + len(box))
    data[ins:ins] = box
    out = dst or src
    tmp = out + ".tmp"
    open(tmp, "wb").write(data)
    os.replace(tmp, out)
    return out


if __name__ == "__main__":
    print(inject(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
