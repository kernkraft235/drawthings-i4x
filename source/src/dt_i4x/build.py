"""Compile the bundled CPU codec into the user's cache on first use."""
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


def native_library():
    if sys.platform != 'darwin':
        raise RuntimeError('This release supports macOS; its native codec uses libdispatch.')
    native = Path(__file__).parent / 'native'
    sources = sorted(p for p in native.iterdir() if p.suffix in ('.c', '.h', '.inc'))
    digest = hashlib.sha256(platform.machine().encode())
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'dt-i4x' / digest.hexdigest()[:16]
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / 'libdtquant.dylib'
    with (cache / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            return destination
        clang = shutil.which('clang')
        if not clang:
            raise RuntimeError('Install Apple Command Line Tools with: xcode-select --install')
        with tempfile.TemporaryDirectory(prefix='compile-', dir=cache) as temporary:
            output = Path(temporary) / destination.name
            environment = dict(os.environ, TMPDIR=temporary,
                               CLANG_MODULE_CACHE_PATH=str(Path(temporary) / 'module-cache'))
            command = [clang, '-O3', '-fno-math-errno', '-fno-trapping-math',
                       '-ffp-contract=off', '-fblocks', '-DUSE_DISPATCH', '-dynamiclib',
                       str(native / 'ccv_nnc_8i_rowwise.c'), '-o', str(output)]
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('Native codec build failed:\n' + result.stderr.strip())
            os.replace(output, destination)
    return destination
