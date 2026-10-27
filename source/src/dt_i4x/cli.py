"""dt-i4x input.ckpt [-o output-directory]."""
import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys

from . import __version__
from .converter import convert, inspect


def output_path(input_path, output_dir=None):
    source = Path(input_path).expanduser().resolve()
    stem = re.sub(r'_(?:i[2-8]x|q[4-8]p|f16|fp16|bf16|f32|fp32)$', '', source.stem, flags=re.I)
    name = stem + '_i4x.ckpt'
    directory = Path(output_dir).expanduser().resolve() if output_dir else source.parent
    return directory / name


def main(argv=None):
    parser = argparse.ArgumentParser(description='Convert native Draw Things text matrices to 4-bit S (i4x).')
    parser.add_argument('input', type=Path, help='Completed native DT .ckpt file')
    parser.add_argument('-o', '--output-dir', type=Path, help='Output folder; defaults to the input folder')
    parser.add_argument('--all-matrices', action='store_true', help='Convert every floating-point 2D matrix, including diffusion/vision matrices')
    parser.add_argument('--no-hadamard', action='store_true', help='Use plain native Q4K for all matrices')
    parser.add_argument('--inspect', action='store_true', help='Show selected tensors and source codec support without writing output')
    parser.add_argument('-q', '--quiet', action='store_true', help='Hide per-tensor progress; print only the finished path')
    parser.add_argument('--version', action='version', version=__version__)
    args = parser.parse_args(argv)
    try:
        source = args.input.expanduser().resolve()
        if args.inspect:
            print(json.dumps(inspect(source, args.all_matrices), indent=2))
            return 0
        destination = output_path(source, args.output_dir)
        if destination == source:
            raise ValueError('The input already has the target filename; choose another output directory.')
        result = convert(source, destination, all_matrices=args.all_matrices,
                         hadamard=not args.no_hadamard, quiet=args.quiet)
        print(destination)
        if not args.quiet:
            print(f"{result['output_bytes'] / 2**30:.2f} GiB; {result['converted_tensors']} matrices converted; all payload hashes verified.")
        return 0
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
        print(f'dt-i4x: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('dt-i4x: interrupted; the input is unchanged. Intermediate output remains in the .work directory.', file=sys.stderr)
        return 130
