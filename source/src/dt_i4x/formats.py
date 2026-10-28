"""Packed formats and checkpoint identifiers from the pinned ccv/s4nnc versions."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    packed: int
    identifier: int
    suffix: str


FORMATS = {
    'q4k': Format(1, 0x8A1EAB, 'i4x'),
    'q3k': Format(2, 0x8A1EAC, 'i3x'),
    'q2k': Format(3, 0x8A1EAD, 'i2x'),
    'iq2s': Format(4, 0x8A1EAE, 'iq2s'),
    'iq2xs': Format(5, 0x8A1EAF, 'iq2xs'),
    'iq3s': Format(6, 0x8A1EB0, 'iq3s'),
    'iq3xxs': Format(7, 0x8A1EB1, 'iq3xxs'),
    'iq2xxs': Format(8, 0x8A1EB2, 'iq2xxs'),
    'q5k': Format(9, 0x8A1EB3, 'i5x'),
    'q6k': Format(10, 0x8A1EB4, 'i6x'),
}
BY_ID = {format.identifier: format for format in FORMATS.values()}


def format_name(value):
    name = value.lower().replace('_', '')
    name = {format.suffix: key for key, format in FORMATS.items()}.get(name, name)
    if name not in FORMATS:
        raise ValueError(f'Unknown format {value!r}; choose from {", ".join(format.suffix for format in FORMATS.values())}')
    return name
