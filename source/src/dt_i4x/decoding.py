"""Read supported native DT source payloads without changing their weights."""
import math
import struct
import zlib

import numpy as np

from .codec import DT_F16, DT_F32, DT_F64, DT_BF16

RAW = 0
EZM7 = 0x511
I8X = 0x8A1E9B
EXTERNAL = 0x10000000
H256_ID = 0x01000000
FLOAT_DTYPES = {DT_F16, DT_F32, DT_F64, DT_BF16}
PALETTES = {0x8A1E4B: 4, 0x8A1E5B: 5, 0x8A1E6B: 6, 0x8A1E7B: 7, 0x8A1E8B: 8}
SUPPORTED = {RAW, EZM7, I8X, *PALETTES}


def source_codec(identifier):
    return identifier & ~(EXTERNAL | H256_ID)


def codec_name(identifier):
    return {RAW: 'raw', EZM7: 'ezm7', I8X: 'i8x', 0x8A1EAB: 'i4x',
            0x8A1E4B: 'q4p', 0x8A1E5B: 'q5p', 0x8A1E6B: 'q6p',
            0x8A1E7B: 'q7p', 0x8A1E8B: 'q8p'}.get(source_codec(identifier), hex(identifier))


def logical(values, datatype):
    if datatype == DT_BF16:
        return (values.astype(np.uint32) << 16).view(np.float32)
    return values


def decode(data, shape, datatype, identifier, native):
    base = source_codec(identifier)
    count = math.prod(shape)
    if base == I8X:
        if identifier & H256_ID:
            raise ValueError('Unexpected H256 flag on unpacked i8x input')
        size = {DT_F16: 2, DT_BF16: 2, DT_F32: 4, DT_F64: 8}[datatype]
        expected = ((count + 127) & -128) + count // shape[-1] * size
        if len(data) != expected:
            raise ValueError(f'Incorrect i8x payload length: {len(data)} != {expected}')
        return native.decode(data, shape, datatype)
    dtype = {DT_F16: '<f2', DT_F32: '<f4', DT_F64: '<f8', DT_BF16: '<u2'}[datatype]
    if base == RAW:
        if len(data) != count * np.dtype(dtype).itemsize:
            raise ValueError('Incorrect raw floating-point payload length')
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    if base in PALETTES:
        if len(data) < 4:
            raise ValueError('Truncated palette header')
        block = struct.unpack_from('<I', data)[0]
        bits = PALETTES[base]
        if not block or block % 8:
            raise ValueError('Invalid palette block size')
        blocks = (count + block - 1) // block
        element = np.dtype(dtype).itemsize
        expected = blocks * (1 << bits) * element + (count * bits + 7) // 8
        if len(data) != 4 + expected:
            raise ValueError('Incorrect palette payload size')
        output = np.empty(count, dtype=dtype)
        offset = 4
        for start in range(0, count, block):
            palette = np.frombuffer(data, dtype=dtype, count=1 << bits, offset=offset)
            offset += (1 << bits) * element
            n = min(block, count - start)
            size = (n * bits + 7) // 8
            packed = np.frombuffer(data, dtype=np.uint8, count=size, offset=offset)
            unpacked = np.unpackbits(packed, bitorder='big')[:n * bits].reshape(n, bits)
            indices = unpacked.astype(np.uint16) @ (1 << np.arange(bits - 1, -1, -1, dtype=np.uint16))
            output[start:start + n] = palette[indices]
            offset += size
        return output.reshape(shape)
    if base == EZM7:
        if datatype != DT_F16 or len(data) < 4:
            raise ValueError('EZM7 input requires FP16 source data')
        exponent_size = struct.unpack_from('<i', data)[0]
        if exponent_size < 0 or len(data) != 4 + exponent_size + count:
            raise ValueError('Invalid EZM7 payload')
        compressed = data[4:4 + exponent_size]
        try:
            exponent_bytes = zlib.decompress(compressed, -15)
        except zlib.error:
            exponent_bytes = zlib.decompress(compressed)
        if len(exponent_bytes) != count:
            raise ValueError('Incorrect decompressed EZM7 exponent count')
        exponents = np.frombuffer(exponent_bytes, dtype=np.uint8).astype(np.uint16)
        if (exponents > 31).any():
            raise ValueError('Invalid EZM7 exponent')
        mantissas = np.frombuffer(data, dtype=np.uint8, offset=4 + exponent_size).astype(np.uint16)
        bits = ((mantissas >> 7) << 15) | (exponents << 10) | ((mantissas & 127) << 3)
        return bits.view(np.float16).reshape(shape)
    raise ValueError(f'Unsupported source codec: {codec_name(identifier)}')
