import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import struct
import zlib
import numpy as np

from test_tool import fixture
from dt_i4x.codec import DT_F16, DT_F32, DT_BF16
from dt_i4x.converter import Store, convert

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'dist/dt-i4x'


@unittest.skipUnless(BINARY.exists(), 'Build native binary before running native CLI tests')
class NativeCLITests(unittest.TestCase):
    def test_i8x_ezm7_and_all_palette_sources_match_reference(self):
        rng = np.random.default_rng(15)
        for source_codec in ('i8x', 'ezm7', 4, 5, 6, 7, 8):
            with self.subTest(source_codec=source_codec), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / 'encoder_source.ckpt'
                fixture(source)
                rows, width = 32, 256
                count = rows * width
                if source_codec == 'i8x':
                    q = rng.integers(-127, 128, (rows, width), dtype=np.int8)
                    scale = rng.uniform(.0001, .001, rows).astype(np.float16)
                    payload = q.tobytes() + scale.tobytes()
                    identifier = 0x8A1E9B
                elif source_codec == 'ezm7':
                    values = rng.normal(0, .02, count).astype(np.float16)
                    bits = values.view(np.uint16)
                    exponents = ((bits >> 10) & 31).astype(np.uint8).tobytes()
                    compressor = zlib.compressobj(wbits=-15)
                    zipped = compressor.compress(exponents) + compressor.flush()
                    rest = (((bits >> 15) << 7) | ((bits & 1023) >> 3)).astype(np.uint8).tobytes()
                    payload = struct.pack('<i', len(zipped)) + zipped + rest
                    identifier = 0x511
                else:
                    bits = source_codec
                    palette = np.linspace(-.05, .05, 1 << bits, dtype=np.float16)
                    pieces = [struct.pack('<I', width)]
                    for _ in range(rows):
                        indices = rng.integers(0, 1 << bits, width, dtype=np.uint16)
                        unpacked = ((indices[:, None] >> np.arange(bits-1,-1,-1,dtype=np.uint16)) & 1).astype(np.uint8)
                        pieces.extend((palette.tobytes(), np.packbits(unpacked.ravel(), bitorder='big').tobytes()))
                    payload = b''.join(pieces)
                    identifier = 0x8A1E0B + (bits << 4)
                db = __import__('sqlite3').connect(source)
                db.execute('UPDATE tensors SET type=?,data=? WHERE name LIKE ?', ((identifier << 32) | 1, payload, '__text_model__%'))
                db.commit(); db.close()
                r = subprocess.run([BINARY, source, '-q'], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                native_path = root / 'encoder_source_i4x.ckpt'
                reference_path = root / 'reference/out.ckpt'
                convert(source, reference_path, quiet=True)
                native, reference = Store(native_path), Store(reference_path)
                nr = next(x for x in native.iter_records() if x[0].startswith('__text_model__'))
                rr = next(x for x in reference.iter_records() if x[0].startswith('__text_model__'))
                self.assertEqual(native.payload(nr[1],nr[5])[0],reference.payload(rr[1],rr[5])[0])

    def test_inspect_and_reject_unsupported_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'encoder_i8x.ckpt'
            fixture(source, unsupported=True)
            r = subprocess.run([BINARY, source, '--inspect'], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            details = json.loads(r.stdout)
            self.assertEqual(details['selected_matrices'], 2)
            self.assertEqual(details['hadamard_matrices'], 1)
            self.assertEqual(details['plain_matrices'], 1)
            self.assertEqual(details['unsupported'][0]['codec'], '0xdead')
            r = subprocess.run([BINARY, source], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn('Unsupported selected source codec', r.stderr)
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_default_output_and_exact_reference_payloads(self):
        for dtype in (DT_F16, DT_F32, DT_BF16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / 'encoder_f16.ckpt'
                fixture(source, dtype)
                original = source.read_bytes()
                r = subprocess.run([BINARY, source, '-q'], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                native_path = root / 'encoder_i4x.ckpt'
                self.assertEqual(Path(r.stdout.strip()).resolve(), native_path.resolve())
                self.assertTrue(native_path.exists())
                self.assertFalse(Path(str(native_path)+'.work').exists())
                self.assertEqual(source.read_bytes(), original)
                reference_path = root / 'reference/encoder_i4x.ckpt'
                convert(source, reference_path, quiet=True)
                native, reference = Store(native_path), Store(reference_path)
                native_rows = {x[0]: x for x in native.iter_records()}
                reference_rows = {x[0]: x for x in reference.iter_records()}
                self.assertEqual(native_rows.keys(), reference_rows.keys())
                for name, row in native_rows.items():
                    ref = reference_rows[name]
                    self.assertEqual(row[1:5], ref[1:5])
                    self.assertEqual(native.payload(row[1],row[5])[0],reference.payload(ref[1],ref[5])[0])

    def test_explicit_output_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'encoder_f16.ckpt'
            fixture(source)
            directory=root/'export'
            r=subprocess.run([BINARY,source,'-o',directory,'--no-hadamard','-q'],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            dest=directory/'encoder_i4x.ckpt'
            before=dest.read_bytes()
            r=subprocess.run([BINARY,source,'-o',directory,'-q'],capture_output=True,text=True)
            self.assertEqual(r.returncode,1)
            self.assertEqual(dest.read_bytes(),before)
            native=Store(dest)
            row=next(x for x in native.iter_records() if x[0].startswith('__text_model__'))
            self.assertEqual((row[1]>>32)&0xfffffff,0x8a1eab)

    def test_split_and_sidecar_inputs(self):
        for layout in ('split', 'sidecar'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                source=root/'encoder_f16.ckpt'
                fixture(source)
                db=__import__('sqlite3').connect(source)
                name,payload=db.execute('SELECT name,data FROM tensors WHERE name LIKE ?',('__text_model__%',)).fetchone()
                if layout=='split':
                    db.execute('CREATE TABLE tensor_splits(name TEXT,part INTEGER,data BLOB,PRIMARY KEY(name,part))')
                    db.execute('UPDATE tensors SET format=?,data=? WHERE name=?',((1<<32)|1,payload[:4096],name))
                    for part, start in enumerate(range(4096,len(payload),4096)):
                        db.execute('INSERT INTO tensor_splits VALUES(?,?,?)',(name,part,payload[start:start+4096]))
                else:
                    sidecar=Path(str(source)+'-tensordata')
                    sidecar.write_bytes(b'\0'*16384+payload)
                    db.execute('UPDATE tensors SET type=?,data=? WHERE name=?',((0x10000000<<32)|1,struct.pack('<QQ',16384,len(payload)),name))
                db.commit();db.close()
                before=source.read_bytes()
                r=subprocess.run([BINARY,source,'-q'],capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
                self.assertTrue((root/'encoder_i4x.ckpt').exists())
                self.assertEqual(source.read_bytes(),before)
                if layout=='sidecar':self.assertEqual(sidecar.read_bytes(),b'\0'*16384+payload)

    def test_default_includes_vision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'encoder_f16.ckpt'
            fixture(source)
            r=subprocess.run([BINARY,source,'-q'],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            output=Store(root/'encoder_i4x.ckpt')
            self.assertEqual(len(output.records),2)
            self.assertTrue(all(((row[1]>>32)&0xeffffff)==0x8a1eab for row in output.records))


if __name__ == '__main__':
    unittest.main()
