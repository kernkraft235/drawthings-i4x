"""Native Draw Things rowwise codecs built from pinned upstream ccv code."""
import ctypes as C
from pathlib import Path

import numpy as np

from .build import native_library

DT_F16 = 0x20000
DT_F32 = 0x04000
DT_F64 = 0x10000
DT_BF16 = 0x80000
Q4K = 1
H256 = 0x100


class Codec:
    def __init__(self):
        self.lib = C.CDLL(str(native_library()))
        self.size = self.lib.ccv_nnc_8i_rowwise_x_data_size
        self.size.argtypes = [C.c_int, C.c_int, C.c_size_t, C.c_size_t]
        self.size.restype = C.c_size_t
        self.quant = self.lib.ccv_nnc_quantize_8i_rowwise_x
        self.quant.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_size_t, C.c_size_t,
                              C.c_int, C.c_void_p, C.c_size_t, C.c_void_p,
                              C.c_size_t, C.c_void_p, C.c_size_t]
        self.quant.restype = C.c_size_t
        self.dequant8 = self.lib.ccv_nnc_dequantize_8i_rowwise
        self.dequant8.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_size_t,
                                 C.c_size_t, C.c_void_p, C.c_size_t]
        self.dequant8.restype = None
        self.dequant4 = self.lib.ccv_nnc_dequantize_8i_rowwise_x
        self.dequant4.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_size_t,
                                 C.c_size_t, C.c_int, C.c_void_p, C.c_size_t]
        self.dequant4.restype = None

    def decode(self, data, shape, datatype=DT_F16, packed=None):
        dtype = {DT_F16: np.float16, DT_F32: np.float32,
                 DT_F64: np.float64, DT_BF16: np.uint16}[datatype]
        out = np.empty(shape, dtype=dtype)
        if packed is None:
            self.dequant8(C.c_char_p(data), datatype, 1, len(data), shape[-1],
                          out.ctypes.data, out.size)
        else:
            self.dequant4(C.c_char_p(data), datatype, 1, len(data), shape[-1],
                          packed, out.ctypes.data, out.size)
        return out

    def encode(self, values, datatype=DT_F16, packed=Q4K):
        assert values.flags.c_contiguous
        size = self.size(packed, datatype, values.size, values.shape[-1])
        if not size:
            raise ValueError("Unsupported shape for packed codec")
        out = np.empty(size, dtype=np.uint8)
        actual = self.quant(values.ctypes.data, datatype, 1, values.size,
                            values.shape[-1], packed, None, 0, None, 0,
                            out.ctypes.data, size)
        if actual != size:
            raise RuntimeError(f"Quantizer wrote {actual}, expected {size}")
        return out.tobytes()
