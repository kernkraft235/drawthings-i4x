#!/usr/bin/env python3
"""Quantize native Draw Things checkpoint matrices.

Input: completed native checkpoint, including its trailer or sidecar.
Output: single trailer-store checkpoint. Original input is never modified.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import sys
import time
import re

import numpy as np

from .codec import Codec, H256
from .decoding import SUPPORTED, FLOAT_DTYPES, codec_name, decode, logical, source_codec
from .formats import FORMATS, format_name

EXTERNAL = 0x10000000
HADAMARD_ID = 0x01000000
I8X_ID = 0x8A1E9B
I4X_ID = 0x8A1EAB
PALETTES = set(range(0x8A1E4B, 0x8A1E8C, 0x10))


def shape_of(dim):
    if len(dim) % 4:
        raise ValueError('Invalid dimension metadata')
    dims = struct.unpack('<' + 'i' * (len(dim) // 4), dim)
    shape = []
    for d in dims:
        # Match ccv: slots after the shape terminator are not dimensions.
        if d == 0:
            break
        if d < 0:
            raise ValueError('Invalid tensor dimensions')
        shape.append(d)
    return tuple(shape)


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


class Progress:
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.started = time.monotonic()
        self.last_update = -math.inf

    def update(self, phase, completed=0, total=0, detail='', *, force=False):
        now = time.monotonic()
        if self.quiet or (not force and now - self.last_update < 1):
            return
        self.last_update = now
        percent = f' {100 * completed / total:5.1f}%' if total else ''
        suffix = f' | {detail}' if detail else ''
        print(f'{phase}{percent} | elapsed {now - self.started:.0f}s{suffix}', file=sys.stderr, flush=True)


def copy_file(source, output, progress):
    total = os.fstat(source.fileno()).st_size
    completed = 0
    for chunk in iter(lambda: source.read(8 << 20), b''):
        output.write(chunk)
        completed += len(chunk)
        progress.update('Assembling checkpoint', completed, total)


def sha256(path, progress=None):
    digest = hashlib.sha256()
    total = Path(path).stat().st_size
    completed = 0
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
            completed += len(chunk)
            if progress:
                progress.update('Hashing checkpoint', completed, total)
    return digest.hexdigest()


def eligible_matrix(shape):
    return len(shape) == 2 and min(shape) > 1


def rotate_matrix(name, shape, hadamard=True):
    # Keep the established text-projection allowlist; dimensions alone do not identify an operator.
    return hadamard and bool(re.match(r'^__text_model(?:_?\d+)?__', name)) and shape[-1] % 256 == 0 and bool(re.search(
        r'(?:t-(?:q_proj|k_proj|v_proj|out_proj)-|mlp_(?:gate|up|down)_proj-)', name))


def inspect(source, hadamard=True):
    store = source if isinstance(source, Store) else Store(source)
    candidates, unsupported = [], []
    for name, typ, fmt, dtype, dim, length in store.records:
        shape = shape_of(dim)
        if not eligible_matrix(shape):
            continue
        identifier = typ >> 32
        item = {'name': name, 'shape': shape, 'codec': codec_name(identifier),
                'parameters': math.prod(shape), 'datatype': dtype,
                'hadamard': rotate_matrix(name, shape, hadamard)}
        candidates.append(item)
        if source_codec(identifier) not in SUPPORTED or dtype not in FLOAT_DTYPES:
            unsupported.append(item)
    rotations = sum(c['hadamard'] for c in candidates)
    return {'input': str(store.path), 'total_tensors': len(store.records),
            'selected_matrices': len(candidates),
            'selected_parameters': sum(c['parameters'] for c in candidates),
            'source_codecs': sorted({c['codec'] for c in candidates}),
            'unsupported': unsupported, 'hadamard_policy': 'auto' if hadamard else 'disabled',
            'hadamard_matrices': rotations, 'plain_matrices': len(candidates) - rotations}


def convert(source, destination, limit=None, hadamard=True, quiet=False, output_format='q4k'):
    output_format = format_name(output_format)
    target = FORMATS[output_format]
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.with_suffix('.validation.json').exists():
        raise FileExistsError(destination.with_suffix('.validation.json'))
    source = Store(source)
    details = inspect(source, hadamard)
    if not details['selected_matrices']:
        raise ValueError('No eligible 2D matrices found for the selected scope')
    if details['unsupported']:
        first = details['unsupported'][0]
        raise ValueError(f"Unsupported selected source tensor: {first['name']} ({first['codec']}, datatype {first['datatype']}). Run --help for supported sources. No output was created.")
    progress = Progress(quiet)
    matrix_total, parameter_total = details['selected_matrices'], details['selected_parameters']
    progress.update('Preparing', detail=f"{matrix_total} matrices; format: {target.suffix}; H256: {details['hadamard_matrices']} auto-selected; unrotated: {details['plain_matrices']}", force=True)
    codec = Codec()
    stage = destination.with_name(destination.name + '.work')
    stage.mkdir(parents=True, exist_ok=False)
    db_path, data_path = stage / 'metadata.sqlite', stage / 'tensordata'
    db = sqlite3.connect(db_path)
    db.execute('CREATE TABLE tensors (name TEXT, type INTEGER, format INTEGER, datatype INTEGER, dim BLOB, data BLOB, PRIMARY KEY (name))')
    manifest = {'input': str(source.path), 'input_bytes': source.path.stat().st_size,
                'upstream_ccv': '810fb2992bd4b5463052a8f94338c8bffc273cc6',
                'policy': 'Native packed rowwise; automatic H256 only for recognized compatible text projections; unselected tensors preserved',
                'output_format': output_format,
                'hadamard_policy': details['hadamard_policy'],
                'hadamard_matrices': details['hadamard_matrices'], 'plain_matrices': details['plain_matrices'],
                'tensors': []}
    start = time.monotonic()
    converted = 0
    converted_elements = 0
    with data_path.open('w+b') as external:
        for index, (name, typ, fmt, datatype, dim, data) in enumerate(source.iter_records()):
            shape = shape_of(dim)
            identifier = typ >> 32
            base = identifier & ~(EXTERNAL | HADAMARD_ID)
            eligible = eligible_matrix(shape)
            payload, prefix = source.payload(typ, data)
            entry = {'name': name, 'shape': shape, 'input_codec': hex(identifier), 'input_payload_bytes': len(payload)}
            if eligible:
                rotation = rotate_matrix(name, shape, hadamard)
                progress.update('Quantizing', converted_elements, parameter_total,
                                f'[{converted}/{matrix_total} complete] {target.suffix} {"H256" if rotation else "unrotated"} {name}', force=True)
                source_bytes = prefix[:4] + payload if prefix else payload
                values = decode(source_bytes, shape, datatype, identifier, codec)
                if not np.isfinite(logical(values, datatype)).all():
                    raise ValueError(f'Nonfinite input: {name}')
                packed = target.packed | (H256 if rotation else 0)
                tensor_start = time.monotonic()
                payload = codec.encode(values, datatype, packed)
                output_id = target.identifier | (HADAMARD_ID if rotation else 0)
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
                              'hadamard': rotation,
                              'quantization_seconds': time.monotonic() - tensor_start})
                converted += 1
                converted_elements += math.prod(shape)
                progress.update('Quantizing', converted_elements, parameter_total,
                                f'[{converted}/{matrix_total} complete] {name} -> {len(payload)/1e6:.1f} MB; relRMSE={relative_rmse:.4f}')
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
    progress.update('Quantizing', converted_elements, parameter_total,
                    f'[{converted}/{matrix_total} complete]', force=True)
    progress.update('Checking output SQLite', force=True)
    db.execute('VACUUM')
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    db.close()
    partial = stage / 'assembled.ckpt'
    progress.update('Assembling checkpoint', force=True)
    with partial.open('w+b') as output, db_path.open('rb') as database, data_path.open('rb') as tensors:
        copy_file(database, output, progress)
        trailer = align(output)
        if trailer > 0xffffffff:
            raise ValueError('Metadata exceeds trailer-store offset limit')
        copy_file(tensors, output, progress)
        output.seek(60)
        output.write(struct.pack('>I', trailer))
        output.flush()
        os.fsync(output.fileno())
    progress.update('Checking assembled checkpoint', force=True)
    check = Store(partial)
    if len(check.records) != len(manifest['tensors']):
        raise ValueError('Final tensor count mismatch')
    expected = {entry['name']: entry for entry in manifest['tensors']}
    progress.update('Verifying tensors', 0, len(check.records), force=True)
    for index, (name, typ, fmt, dtype, dim, data) in enumerate(check.iter_records(), 1):
        payload, _ = check.payload(typ, data)
        if hashlib.sha256(payload).hexdigest() != expected[name]['payload_sha256']:
            raise ValueError(f'Final payload hash mismatch: {name}')
        progress.update('Verifying tensors', index, len(check.records))
    progress.update('Hashing checkpoint', force=True)
    manifest.update({'output_bytes': partial.stat().st_size, 'converted_tensors': converted,
                     'converted_elements': converted_elements, 'elapsed_seconds': time.monotonic() - start,
                     'sha256': sha256(partial, progress)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Hard-link creation fails rather than replacing an existing destination.
    os.link(partial, destination)
    destination.with_suffix('.validation.json').write_text(json.dumps(manifest, indent=2))
    progress.update('Finished', 1, 1, force=True)
    return manifest
