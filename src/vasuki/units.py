"""Unit conversion with exact rational factors.

Currency is deliberately absent: exchange rates are data, not constants, so a
plan must multiply by an explicit rate instead.
"""

from __future__ import annotations

import math
from fractions import Fraction as F
from typing import NamedTuple

from .errors import PlanError, is_number, unknown


class Unit(NamedTuple):
    dimension: str
    # base = (value + offset) * scale
    scale: F
    offset: F = F(0)


_SCALES = {
    "length": {
        "mm": F(1, 1000), "cm": F(1, 100), "m": F(1), "km": F(1000),
        "in": F(254, 10_000), "ft": F(3048, 10_000), "yd": F(9144, 10_000), "mi": F(1_609_344, 1000),
    },
    "mass": {
        "mg": F(1, 10**6), "g": F(1, 1000), "kg": F(1), "t": F(1000),
        "oz": F(28_349_523_125, 10**12), "lb": F(45_359_237, 10**8),
    },
    "time": {"ms": F(1, 1000), "s": F(1), "min": F(60), "h": F(3600), "day": F(86_400)},
    "data": {
        "B": F(1), "KB": F(10**3), "MB": F(10**6), "GB": F(10**9), "TB": F(10**12),
        "KiB": F(2**10), "MiB": F(2**20), "GiB": F(2**30), "TiB": F(2**40),
    },
    "ratio": {"fraction": F(1), "percent": F(1, 100), "bps": F(1, 10_000)},
}

UNITS: dict[str, Unit] = {
    name: Unit(dimension, scale) for dimension, scales in _SCALES.items() for name, scale in scales.items()
}
UNITS.update({
    "K": Unit("temperature", F(1)),
    "C": Unit("temperature", F(1), F(27_315, 100)),
    "F": Unit("temperature", F(5, 9), F(45_967, 100)),
})

DIMENSIONS: dict[str, list[str]] = {}
for _name, _unit in UNITS.items():
    DIMENSIONS.setdefault(_unit.dimension, []).append(_name)

_FOLDED: dict[str, list[str]] = {}
for _name in UNITS:
    _FOLDED.setdefault(_name.casefold(), []).append(_name)


def _lookup(name: str) -> Unit:
    if name in UNITS:
        return UNITS[name]
    matches = _FOLDED.get(name.casefold(), [])
    if len(matches) == 1:
        return UNITS[matches[0]]
    error = unknown("unit", name, UNITS)
    raise PlanError("unit_error", error.message)


def check_units(src: str, dst: str) -> tuple[Unit, Unit]:
    source, target = _lookup(src), _lookup(dst)
    if source.dimension != target.dimension:
        raise PlanError(
            "unit_error",
            f"cannot convert {src!r} ({source.dimension}) to {dst!r} ({target.dimension})",
        )
    return source, target


def convert(value, src: str, dst: str):
    """Convert `value` between units; ints stay ints when the result is whole."""
    source, target = check_units(src, dst)
    if value is None:
        return None
    if not is_number(value) or not math.isfinite(value):
        raise PlanError("type_error", f"cannot convert {value!r} from {src!r}: not a finite number")
    # str() keeps the decimal the user wrote, so 0.3 s is exactly 300 ms.
    exact = (F(str(value)) + source.offset) * source.scale / target.scale - target.offset
    if isinstance(value, int) and exact.denominator == 1:
        return int(exact)
    return float(exact)
