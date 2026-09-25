"""Probe whether the GelSight UVC firmware provides a device (sensor) clock
timestamp via the UVC metadata node (UVCH). Reads struct uvc_meta_buf per
frame and extracts the host ns timestamp + the UVC payload-header PTS/SCR
(device clock). If PTS/SCR advance, a true device-capture timestamp IS
available; if they're zero/static, the firmware doesn't provide it.

The paired MJPG capture node must stream for metadata to flow, so we open it
with OpenCV in the background.

    python camera_stream/probe_uvc_metadata.py --cap 1 --meta 2
"""
from __future__ import annotations

import argparse, ctypes, fcntl, mmap, struct, threading, time

import cv2

# ---- V4L2 constants ----
V4L2_BUF_TYPE_META_CAPTURE = 13
V4L2_MEMORY_MMAP = 1
def v4l2_fourcc(a, b, c, d): return (ord(a) | ord(b) << 8 | ord(c) << 16 | ord(d) << 24)
V4L2_META_FMT_UVC = v4l2_fourcc('U', 'V', 'C', 'H')

# ioctl numbers (_IOWR('V', n, struct) ; dir=3 read|write)
_IOC_WRITE, _IOC_READ = 1, 2
def _IOC(d, t, nr, size): return (d << 30) | (size << 16) | (ord(t) << 8) | nr
def _IOWR(t, nr, size): return _IOC(_IOC_READ | _IOC_WRITE, t, nr, size)
def _IOW(t, nr, size): return _IOC(_IOC_WRITE, t, nr, size)


class v4l2_meta_format(ctypes.Structure):
    _fields_ = [("dataformat", ctypes.c_uint32), ("buffersize", ctypes.c_uint32)]


class v4l2_format(ctypes.Structure):
    # type(4) + pad(4, union is 8-aligned due to pointers in v4l2_window)
    # + 200-byte union -> 208 bytes total. meta format sits at union start.
    _fields_ = [("type", ctypes.c_uint32), ("_pad", ctypes.c_uint32),
                ("fmt", ctypes.c_uint8 * 200)]


class v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("capabilities", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 1)]


class v4l2_timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class v4l2_buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32), ("timestamp", v4l2_timeval),
        ("timecode", ctypes.c_uint8 * 16), ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32), ("offset_or_userptr", ctypes.c_ulong),
        ("length", ctypes.c_uint32), ("reserved2", ctypes.c_uint32),
        ("request_fd_or_reserved", ctypes.c_uint32),
    ]


# Hardcoded standard 64-bit ioctl values (size fields must match the kernel
# structs exactly, so we use the known-correct constants).
VIDIOC_S_FMT     = 0xc0d05605
VIDIOC_REQBUFS   = 0xc0145608
VIDIOC_QUERYBUF  = 0xc0585609
VIDIOC_QBUF      = 0xc058560f
VIDIOC_DQBUF     = 0xc0585611
VIDIOC_STREAMON  = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
assert ctypes.sizeof(v4l2_format) == 208, ctypes.sizeof(v4l2_format)
assert ctypes.sizeof(v4l2_buffer) == 88, ctypes.sizeof(v4l2_buffer)
assert ctypes.sizeof(v4l2_requestbuffers) == 20, ctypes.sizeof(v4l2_requestbuffers)
# meta format lives at union start (offset 8 within v4l2_format)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, required=True, help="MJPG capture video node")
    ap.add_argument("--meta", type=int, required=True, help="UVCH metadata video node")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    import os
    # 1) Configure the META node FIRST — S_FMT must happen before the UVC
    #    stream starts, else it returns EBUSY.
    fd = os.open(f"/dev/video{args.meta}", os.O_RDWR)
    fmt = v4l2_format(); fmt.type = V4L2_BUF_TYPE_META_CAPTURE
    mf = v4l2_meta_format(dataformat=V4L2_META_FMT_UVC, buffersize=4096)
    ctypes.memmove(fmt.fmt, ctypes.byref(mf), ctypes.sizeof(mf))
    fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)
    req = v4l2_requestbuffers(count=4, type=V4L2_BUF_TYPE_META_CAPTURE, memory=V4L2_MEMORY_MMAP)
    fcntl.ioctl(fd, VIDIOC_REQBUFS, req)
    bufs = []
    for i in range(req.count):
        b = v4l2_buffer(index=i, type=V4L2_BUF_TYPE_META_CAPTURE, memory=V4L2_MEMORY_MMAP)
        fcntl.ioctl(fd, VIDIOC_QUERYBUF, b)
        mm = mmap.mmap(fd, b.length, mmap.MAP_SHARED, mmap.PROT_READ, offset=b.offset_or_userptr)
        bufs.append(mm)
        fcntl.ioctl(fd, VIDIOC_QBUF, b)
    bt = ctypes.c_int(V4L2_BUF_TYPE_META_CAPTURE)
    fcntl.ioctl(fd, VIDIOC_STREAMON, bt)

    # 2) NOW start the capture node streaming so UVC payload metadata flows.
    stop = threading.Event()
    def _cap():
        c = cv2.VideoCapture(args.cap, cv2.CAP_V4L2)
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        while not stop.is_set():
            c.grab()
        c.release()
    th = threading.Thread(target=_cap, daemon=True); th.start()
    time.sleep(1.0)

    import select
    print(f"reading up to {args.n} metadata frames from /dev/video{args.meta} ...")
    pts_seen, scr_seen, rows = 0, 0, []
    deadline = time.time() + 8.0      # give up after 8s if nothing flows
    while len(rows) < args.n and time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 1.0)
        if not r:
            continue                  # no metadata available yet
        b = v4l2_buffer(type=V4L2_BUF_TYPE_META_CAPTURE, memory=V4L2_MEMORY_MMAP)
        try:
            fcntl.ioctl(fd, VIDIOC_DQBUF, b)
        except OSError:
            time.sleep(0.01); continue
        data = bufs[b.index][:b.bytesused]
        # parse uvc_meta_buf: ns(u64) sof(u16) length(u8) flags(u8) buf[]
        if len(data) >= 12:
            ns, sof, length, flags = struct.unpack_from("<QHBB", data, 0)
            body = data[12:12 + max(0, length - 2)]   # length includes its own 2 header bytes
            pts = scr_stc = scr_sof = None
            off = 0
            if flags & 0x04 and len(body) >= off + 4:
                pts = struct.unpack_from("<I", body, off)[0]; off += 4; pts_seen += 1
            if flags & 0x08 and len(body) >= off + 6:
                scr_stc, scr_sof = struct.unpack_from("<IH", body, off); scr_seen += 1
            rows.append((ns, sof, flags, pts, scr_stc))
        fcntl.ioctl(fd, VIDIOC_QBUF, b)
    fcntl.ioctl(fd, VIDIOC_STREAMOFF, bt)
    stop.set(); th.join(timeout=1)

    print(f"frames parsed={len(rows)}  PTS present={pts_seen}  SCR present={scr_seen}")
    for r in rows[:6]:
        print(f"  ns={r[0]}  sof={r[1]}  flags=0x{r[2]:02x}  PTS={r[3]}  SCR_STC={r[4]}")
    if pts_seen >= 2:
        ptss = [r[3] for r in rows if r[3] is not None]
        adv = len(set(ptss)) > 1
        print(f"  PTS advancing across frames? {adv}  (range {min(ptss)}..{max(ptss)})")
        print("  => DEVICE timestamp IS available via UVC metadata." if adv
              else "  => PTS present but static -> not a usable device clock.")
    else:
        print("  => firmware does NOT provide PTS in UVC metadata.")
    if scr_seen >= 2:
        scrs = [r[4] for r in rows if r[4] is not None]
        print(f"  SCR_STC advancing? {len(set(scrs))>1}  (range {min(scrs)}..{max(scrs)})")


if __name__ == "__main__":
    main()
