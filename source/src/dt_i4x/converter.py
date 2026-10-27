#!/usr/bin/env python3
"""Convert Qwen3-VL encoder text matrices to native Draw Things i4x.

Input: completed native i8x checkpoint, including its trailer or sidecar.
Output: single trailer-store checkpoint. Original input is never modified.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import time
import re

import numpy as np

from .codec import Codec, DT_F16, Q4K, H256
from .decoding import SUPPORTED, FLOAT_DTYPES, codec_name, decode, logical, source_codec

EXTERNAL = 0x10000000
HADAMARD_ID = 0x01000000
I8X_ID = 0x8A1E9B
I4X_ID = 0x8A1EAB
PALETTES = set(range(0x8A1E4B, 0x8A1E8C, 0x10))


def shape_of(dim):
    dims = struct.unpack('<' + 'i' * (len(dim) // 4), dim)
    return tuple(d for d in dims if d > 0)


class Store:
    def close(self):
        external = getattr(self, 'external', None)
        file = getattr(self, 'file', None)
        if external is not None and external is not file:
            external.close()
        if file is not None:
            file.close()
        db = getattr(self, 'db', None)
        if db is not None:
            db.close()

    def __del__(self):
        self.close()

    def __init__(self, path):
        self.path = Path(path).resolve()
        if self.path.name.endswith(('.partial', '.part')):
            raise ValueError('Input download is incomplete')
        self.file = self.path.open('rb')
        header = self.file.read(100)
        if header[:16] != b'SQLite format 3\0':
            raise ValueError('Input is not a native SQLite checkpoint')
        self.trailer = int.from_bytes(header[60:64], 'big')
        if self.trailer and not 100 <= self.trailer <= self.path.stat().st_size:
            raise ValueError('Invalid trailer offset')
        self.external = self.file if self.trailer else None
        if not self.trailer and Path(str(self.path) + '-tensordata').exists():
            self.external = Path(str(self.path) + '-tensordata').open('rb')
        self.db = sqlite3.connect(self.path.as_uri() + '?mode=ro&immutable=1', uri=True)
        self.records = self.db.execute('SELECT name,type,format,datatype,dim,length(data) FROM tensors ORDER BY name').fetchall()

    def iter_records(self):
        for name, typ, fmt, dtype, dim, length in self.records:
            data = self.db.execute('SELECT data FROM tensors WHERE name=?', (name,)).fetchone()[0]
            if fmt & (1 << 32):
                pieces = [data]
                parts = self.db.execute('SELECT part,data FROM tensor_splits WHERE name=? ORDER BY part', (name,))
                for expected_part, (part, piece) in enumerate(parts):
                    if expected_part != part:
                        raise ValueError(f'Missing tensor split: {name}')
                    pieces.append(piece)
                data = b''.join(pieces)
                fmt &= 0xffffffff
            yield name, typ, fmt, dtype, dim, data

    def payload(self, typ, data):
        identifier = typ >> 32
        if not identifier & EXTERNAL:
            return data, b''
        base = identifier & ~(EXTERNAL | HADAMARD_ID)
        prefix = data[:8] if base in PALETTES else b''
        offset, length = struct.unpack_from('<QQ', data, len(prefix))
        if self.external is None:
            raise ValueError('Missing external tensor data')
        if offset + length + self.trailer > os.fstat(self.external.fileno()).st_size:
            raise ValueError('External tensor exceeds input file size')
        self.external.seek(offset + self.trailer)
        result = self.external.read(length)
        if len(result) != length:
            raise ValueError('Truncated tensor')
        return result, prefix


def align(file, alignment=16384):
    at = file.tell()
    offset = (at + alignment - 1) // alignment * alignment
    file.write(b'\0' * (offset - at))
    return offset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_matrix(name, shape, all_matrices=False):
    text = re.match(r'^__text_model(?:_?\d+)?__', name) is not None
    return (text or all_matrices) and len(shape) == 2 and min(shape) > 1


def rotate_matrix(name, shape, hadamard=True):
    # Rotate only known Qwen linear operators. Embeddings use index selection.
    return hadamard and shape[-1] % 256 == 0 and bool(re.search(
        r'(?:t-(?:q_proj|k_proj|v_proj|out_proj)-|mlp_(?:gate|up|down)_proj-)', name))


def inspect(source, all_matrices=False):
    store = source if isinstance(source, Store) else Store(source)
    candidates, unsupported = [], []
    for name, typ, fmt, dtype, dim, length in store.records:
        shape = shape_of(dim)
        if not eligible_matrix(name, shape, all_matrices):
            continue
        identifier = typ >> 32
        item = {'name': name, 'shape': shape, 'codec': codec_name(identifier),
                'parameters': math.prod(shape), 'datatype': dtype}
        candidates.append(item)
        if source_codec(identifier) not in SUPPORTED or dtype not in FLOAT_DTYPES:
            unsupported.append(item)
    return {'input': str(store.path), 'total_tensors': len(store.records),
            'selected_matrices': len(candidates),
            'selected_parameters': sum(c['parameters'] for c in candidates),
            'source_codecs': sorted({c['codec'] for c in candidates}),
            'unsupported': unsupported}


def convert(source, destination, limit=None, all_matrices=False, hadamard=True, quiet=False):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    source = Store(source)
    details = inspect(source, all_matrices)
    if not details['selected_matrices']:
        raise ValueError('No matching text matrices found. Use --all-matrices for a different checkpoint layout.')
    if details['unsupported']:
        first = details['unsupported'][0]
        raise ValueError(f"Unsupported selected source tensor: {first['name']} ({first['codec']}, datatype {first['datatype']}). Supported sources: raw FP16/FP32/FP64/BF16, EZM7 FP16, i8x, q4p/q5p/q6p/q7p/q8p. No output was created.")
    codec = Codec()
    stage = destination.with_name(destination.name + '.work')
    stage.mkdir(parents=True, exist_ok=False)
    db_path, data_path = stage / 'metadata.sqlite', stage / 'tensordata'
    db = sqlite3.connect(db_path)
    db.execute('CREATE TABLE tensors (name TEXT, type INTEGER, format INTEGER, datatype INTEGER, dim BLOB, data BLOB, PRIMARY KEY (name))')
    manifest = {'input': str(source.path), 'input_bytes': source.path.stat().st_size,
                'upstream_ccv': '810fb2992bd4b5463052a8f94338c8bffc273cc6',
                'policy': 'Native Q4K; H256 only for recognized Qwen linear operators; unselected tensors preserved',
                'tensors': []}
    start = time.monotonic()
    converted = 0
    converted_elements = 0
    with data_path.open('w+b') as external:
        for index, (name, typ, fmt, datatype, dim, data) in enumerate(source.iter_records()):
            shape = shape_of(dim)
            identifier = typ >> 32
            base = identifier & ~(EXTERNAL | HADAMARD_ID)
            eligible = eligible_matrix(name, shape, all_matrices)
            payload, prefix = source.payload(typ, data)
            entry = {'name': name, 'shape': shape, 'input_codec': hex(identifier), 'input_payload_bytes': len(payload)}
            if eligible:
                source_bytes = prefix[:4] + payload if prefix else payload
                values = decode(source_bytes, shape, datatype, identifier, codec)
                if not np.isfinite(logical(values, datatype)).all():
                    raise ValueError(f'Nonfinite input: {name}')
                rotation = rotate_matrix(name, shape, hadamard)
                packed = Q4K | (H256 if rotation else 0)
                tensor_start = time.monotonic()
                payload = codec.encode(values, datatype, packed)
                output_id = I4X_ID | (HADAMARD_ID if rotation else 0)
                # Check several rows with the upstream decoder, in the original logical basis.
                # Decoding a full matrix here is bounded to one matrix, including the embedding.
                decoded = codec.decode(payload, shape, datatype, packed)
                rows = np.unique(np.linspace(0, shape[0] - 1, min(shape[0], 64), dtype=int))
                x = logical(values[rows], datatype).astype(np.float64)
                y = logical(decoded[rows], datatype).astype(np.float64)
                if not np.isfinite(y).all():
                    raise ValueError(f'Nonfinite output: {name}')
                relative_rmse = float(np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-30))
                if relative_rmse > 0.30:
                    raise ValueError(f'Excessive quantization error: {name}: {relative_rmse}')
                entry.update({'output_codec': hex(output_id | EXTERNAL), 'relative_rmse': relative_rmse,
                              'quantization_seconds': time.monotonic() - tensor_start})
                converted += 1
                converted_elements += math.prod(shape)
                elapsed = time.monotonic() - start
                if not quiet:
                    print(f'{converted:3d} {name} {shape} -> {len(payload)/1e6:.1f} MB; relRMSE={relative_rmse:.4f}; elapsed={elapsed:.0f}s; rate={converted_elements/max(elapsed,1)/1e6:.1f}M/s', flush=True)
                del values, decoded, x, y
                prefix = b''
                identifier = output_id
            else:
                entry['output_codec'] = hex(identifier)
            if identifier & EXTERNAL or eligible or (identifier == 0 and len(payload) >= 65536):
                offset = align(external)
                external.write(payload)
                metadata = prefix + struct.pack('<QQ', offset, len(payload))
                identifier |= EXTERNAL
                typ = (identifier << 32) | (typ & 0xffffffff)
            else:
                metadata = payload
            entry['output_payload_bytes'] = len(payload)
            entry['payload_sha256'] = hashlib.sha256(payload).hexdigest()
            db.execute('INSERT INTO tensors VALUES (?,?,?,?,?,?)', (name, typ, fmt, datatype, dim, metadata))
            manifest['tensors'].append(entry)
            db.commit()
            (stage / 'progress.json').write_text(json.dumps(manifest, indent=2))
            if limit and converted >= limit:
                break
        external.flush()
        os.fsync(external.fileno())
    db.execute('VACUUM')
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    db.close()
    partial = stage / 'assembled.ckpt'
    with partial.open('w+b') as output, db_path.open('rb') as database, data_path.open('rb') as tensors:
        shutil.copyfileobj(database, output, 8 << 20)
        trailer = align(output)
        if trailer > 0xffffffff:
            raise ValueError('Metadata exceeds trailer-store offset limit')
        shutil.copyfileobj(tensors, output, 8 << 20)
        output.seek(60)
        output.write(struct.pack('>I', trailer))
        output.flush()
        os.fsync(output.fileno())
    check = Store(partial)
    if len(check.records) != len(manifest['tensors']):
        raise ValueError('Final tensor count mismatch')
    expected = {entry['name']: entry for entry in manifest['tensors']}
    for name, typ, fmt, dtype, dim, data in check.iter_records():
        payload, _ = check.payload(typ, data)
        if hashlib.sha256(payload).hexdigest() != expected[name]['payload_sha256']:
            raise ValueError(f'Final payload hash mismatch: {name}')
    manifest.update({'output_bytes': partial.stat().st_size, 'converted_tensors': converted,
                     'converted_elements': converted_elements, 'elapsed_seconds': time.monotonic() - start,
                     'sha256': sha256(partial)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Hard-link creation fails rather than replacing an existing destination.
    os.link(partial, destination)
    destination.with_suffix('.validation.json').write_text(json.dumps(manifest, indent=2))
    if not quiet:
        print(json.dumps({k: v for k, v in manifest.items() if k != 'tensors'}, indent=2), flush=True)
    return manifest
