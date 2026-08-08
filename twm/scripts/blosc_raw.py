"""Decompress a blosc chunk taken straight out of a file.

The HDF5 blosc filter (id 32001) writes self-describing frames: a 16-byte
header carrying the uncompressed size, the block size and the compressed
size. That is what makes a chunk recoverable from a raw byte offset without
the library. Verified byte-identical against h5py on a healthy file.

A chunk whose B-tree filter mask has bit 0 set was stored WITHOUT the
filter and must not be passed here — it has no header.
"""
import ctypes, struct, numpy as np
LIB="/home/yxma/miniconda3/lib/python3.9/site-packages/hdf5plugin/plugins/libh5blosc.so"
_l=ctypes.CDLL(LIB)
_l.blosc_decompress.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_l.blosc_decompress.restype=ctypes.c_int
try:
    _l.blosc_init()
except Exception: pass

def blosc_header(buf):
    ver,verlz,flags,typesize = buf[0],buf[1],buf[2],buf[3]
    nbytes,blocksize,cbytes = struct.unpack("<III", buf[4:16])
    return dict(ver=ver,verlz=verlz,flags=flags,typesize=typesize,
                nbytes=nbytes,blocksize=blocksize,cbytes=cbytes)

def decompress(buf):
    h=blosc_header(buf)
    out=ctypes.create_string_buffer(h["nbytes"])
    n=_l.blosc_decompress(bytes(buf), out, h["nbytes"])
    if n<=0: raise RuntimeError(f"blosc_decompress returned {n}")
    return out.raw[:n], h
