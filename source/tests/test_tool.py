import contextlib
import io
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
import zlib

import numpy as np

from dt_i4x.cli import main, output_path
from dt_i4x.codec import Codec, DT_F16, DT_F32, DT_BF16, Q4K, H256
from dt_i4x.converter import Store, convert, inspect, shape_of
from dt_i4x.decoding import decode, logical


def fixture(path, datatype=DT_F16, unsupported=False, zero=False):
    values = np.zeros((32, 256), dtype=np.float32) if zero else np.random.default_rng(5).normal(0, .025, (32, 256)).astype(np.float32)
    if datatype == DT_F16:
        raw = values.astype(np.float16).tobytes()
    elif datatype == DT_F32:
        raw = values.tobytes()
    else:
        raw = (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE tensors (name TEXT PRIMARY KEY,type INTEGER,format INTEGER,datatype INTEGER,dim BLOB,data BLOB)')
    dims = struct.pack('<12i', 32, 256, *([0] * 10))
    typ = (0xDEAD << 32) | 1 if unsupported else 1
    db.execute('INSERT INTO tensors VALUES(?,?,?,?,?,?)', ('__text_model__[t-q_proj-0-0]', typ, 1, datatype, dims, raw))
    db.execute('INSERT INTO tensors VALUES(?,?,?,?,?,?)', ('__vision_model__[t-untouched-0-0]', 1, 1, datatype, dims, raw))
    db.commit()
    db.close()
    return values


class ToolTests(unittest.TestCase):
    def test_output_folder_and_filename(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'encoder_i8x.ckpt'
            self.assertEqual(output_path(source), Path(root).resolve() / 'encoder_i4x.ckpt')
            self.assertEqual(output_path(source, Path(root) / 'out'), Path(root).resolve() / 'out/encoder_i4x.ckpt')
            self.assertEqual(output_path(Path(root) / 'encoder_f16.ckpt').name, 'encoder_i4x.ckpt')

    def test_unsupported_source_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as root:
            source, dest = Path(root) / 'source.ckpt', Path(root) / 'dest.ckpt'
            fixture(source, unsupported=True)
            with self.assertRaisesRegex(ValueError, 'Unsupported selected source tensor'):
                convert(source, dest, quiet=True)
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name(dest.name + '.work').exists())

    def test_end_to_end_primitive_sources(self):
        native = Codec()
        for dtype in (DT_F16, DT_F32, DT_BF16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as root:
                source, dest = Path(root) / 'source.ckpt', Path(root) / 'out/converted.ckpt'
                fixture(source, dtype)
                original = source.read_bytes()
                result = convert(source, dest, quiet=True)
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(result['converted_tensors'], 2)
                before, after = Store(source), Store(dest)
                original_rows = {r[0]: r for r in before.iter_records()}
                output_rows = {r[0]: r for r in after.iter_records()}
                self.assertEqual(original_rows.keys(), output_rows.keys())
                for name, row in output_rows.items():
                    _, typ, fmt, dt, dim, data = row
                    payload, _ = after.payload(typ, data)
                    rotation = not name.startswith('__vision_model__')
                    self.assertEqual((typ >> 32) & 0x0fffffff, 0x8A1EAB | (0x1000000 if rotation else 0))
                    shape = shape_of(dim)
                    x = decode(original_rows[name][-1], shape, dt, 0, native)
                    y = native.decode(payload, shape, dt, Q4K | (H256 if rotation else 0))
                    error = np.linalg.norm(logical(y, dt) - logical(x, dt)) / np.linalg.norm(logical(x, dt))
                    self.assertLess(error, .1)

    def test_zero_tensor_is_valid(self):
        with tempfile.TemporaryDirectory() as root:
            source, dest = Path(root) / 'source.ckpt', Path(root) / 'out.ckpt'
            fixture(source, zero=True)
            result = convert(source, dest, quiet=True)
            self.assertEqual(result['tensors'][0]['relative_rmse'], 0)

    def test_inspect_and_existing_output_are_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'encoder_f16.ckpt'
            fixture(source)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(source), '--inspect']), 0)
            self.assertEqual(len(list(Path(root).iterdir())), 1)
            destination = output_path(source)
            destination.write_text('keep me')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(source)]), 1)
            self.assertEqual(destination.read_text(), 'keep me')

    def test_ezm7_decode(self):
        bits = np.array([0, 0x3c00, 0xbc00, 0x0008, 0x7bF8, 0x3550], dtype=np.uint16)
        exponent = ((bits >> 10) & 31).astype(np.uint8).tobytes()
        compressor = zlib.compressobj(wbits=-15)
        compressed = compressor.compress(exponent) + compressor.flush()
        mantissa = (((bits >> 15) << 7) | ((bits & 1023) >> 3)).astype(np.uint8).tobytes()
        payload = struct.pack('<i', len(compressed)) + compressed + mantissa
        values = decode(payload, (2, 3), DT_F16, 0x511, None)
        np.testing.assert_array_equal(values.view(np.uint16).ravel(), bits)

    def test_truncated_i8x_is_rejected_before_native_decode(self):
        with self.assertRaisesRegex(ValueError, 'Incorrect i8x payload length'):
            decode(b'\0' * 16, (32, 256), DT_F16, 0x8A1E9B, None)


if __name__ == '__main__':
    unittest.main()
