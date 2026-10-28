"""dt-i4x input.ckpt [-o output-directory]."""
import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys

from . import __version__
from .converter import convert, inspect
from .formats import FORMATS, format_name


def output_path(input_path, output_dir=None, output_format='q4k'):
    source = Path(input_path).expanduser().resolve()
    stem = re.sub(r'_(?:i[2-8]x|iq[23](?:s|xs|xxs)|q[2-6]k|q[4-8]p|f16|fp16|bf16|f32|fp32)$', '', source.stem, flags=re.I)
    name = stem + '_' + FORMATS[format_name(output_format)].suffix + '.ckpt'
    directory = Path(output_dir).expanduser().resolve() if output_dir else source.parent
    return directory / name


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Quantize a Draw Things checkpoint or safetensors model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Formats:
  Standard:  i2x, i3x, i4x, i5x, i6x
  IQ:        iq2xxs, iq2xs, iq2s, iq3xxs, iq3s

Defaults:
  The input defines what gets quantized. Convolutions and vectors are preserved.
  H256 is automatic for supported operators. Progress appears during conversion.

Safetensors:
  Requires the official draw-things-cli importer; import runs offline.

Examples:
  dt-i4x model.ckpt --format i3x
  dt-i4x model.safetensors -o converted''')
    parser.add_argument('input', type=Path, metavar='INPUT', help='Native .ckpt or single-file .safetensors model')
    parser.add_argument('--format', type=format_name, choices=FORMATS, metavar='FORMAT', default='q4k', help='Output quantization (default: i4x)')
    parser.add_argument('--draw-things-cli', type=Path, metavar='PATH', help='Safetensors importer (default: found on PATH)')
    parser.add_argument('-o', '--output-dir', type=Path, help='Output folder; defaults to the input folder')
    parser.add_argument('--no-hadamard', action='store_true', help='Disable automatic H256 rotation')
    parser.add_argument('--inspect', action='store_true', help='Inspect the input without converting')
    parser.add_argument('-q', '--quiet', action='store_true', help='Print only the finished output path')
    parser.add_argument('--version', action='version', version=__version__)
    args = parser.parse_args(argv)
    try:
        source = args.input.expanduser().resolve()
        if source.suffix.lower() == '.safetensors':
            from .importing import convert_safetensors, inspect_safetensors
            if args.inspect:
                inspect_safetensors(source, args.draw_things_cli)
                return 0
            destination = output_path(source, args.output_dir, args.format)
            result = convert_safetensors(source, destination, args.draw_things_cli,
                                         hadamard=not args.no_hadamard,
                                         quiet=args.quiet, output_format=args.format)
        elif args.inspect:
            print(json.dumps(inspect(source, not args.no_hadamard), indent=2))
            return 0
        else:
            destination = output_path(source, args.output_dir, args.format)
            if destination == source:
                raise ValueError('The input already has the target filename; choose another output directory.')
            result = convert(source, destination, hadamard=not args.no_hadamard,
                             quiet=args.quiet, output_format=args.format)
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
