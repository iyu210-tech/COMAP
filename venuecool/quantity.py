"""Quantities with explicit units, provenance and optional uncertainty bounds.

Every influential parameter in a configuration is written as::

    {value: 0.8, unit: "-", source: "assumption: rigid media pad", bounds: [0.7, 0.9]}

The schema declares the canonical internal unit for each field. Accepted alternative units are
converted on load (e.g. kW -> W, m3/h -> m3/s, % -> fraction). Unknown or mismatched units are
rejected rather than guessed.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Tuple

from pydantic import AfterValidator, BaseModel, ConfigDict, field_validator, model_validator

# (from_unit, canonical_unit) -> (scale, offset): canonical = value*scale + offset
_CONVERSIONS = {
    ("kW", "W"): (1000.0, 0.0),
    ("W", "W"): (1.0, 0.0),
    ("m3/h", "m3/s"): (1.0 / 3600.0, 0.0),
    ("L/s", "m3/s"): (1.0e-3, 0.0),
    ("%", "-"): (0.01, 0.0),
    ("K", "degC"): (1.0, -273.15),
    ("mm", "m"): (1.0e-3, 0.0),
    ("cm", "m"): (1.0e-2, 0.0),
    ("kJ/(kg*K)", "J/(kg*K)"): (1000.0, 0.0),
    ("hPa", "Pa"): (100.0, 0.0),
    ("kPa", "Pa"): (1000.0, 0.0),
    ("USD/MWh", "USD/kWh"): (1.0e-3, 0.0),
    ("USD/L", "USD/m3"): (1000.0, 0.0),
}


class Quantity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str
    source: str
    bounds: Optional[Tuple[float, float]] = None
    distribution: Literal["uniform", "triangular"] = "uniform"

    @field_validator("source")
    @classmethod
    def _source_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("every quantity needs a non-empty provenance 'source'")
        return v

    @model_validator(mode="after")
    def _bounds_ok(self):
        if self.bounds is not None:
            lo, hi = self.bounds
            if lo > hi:
                raise ValueError(f"bounds {self.bounds} are reversed")
            if not (lo <= self.value <= hi):
                raise ValueError(f"nominal value {self.value} lies outside bounds {self.bounds}")
        return self

    def to(self, canonical: str) -> "Quantity":
        if self.unit == canonical:
            return self
        key = (self.unit, canonical)
        if key not in _CONVERSIONS:
            raise ValueError(
                f"unit '{self.unit}' is not accepted here; expected '{canonical}'"
                + (f" (convertible: {[k[0] for k in _CONVERSIONS if k[1] == canonical]})")
            )
        s, o = _CONVERSIONS[key]
        b = None if self.bounds is None else (self.bounds[0] * s + o, self.bounds[1] * s + o)
        return Quantity(value=self.value * s + o, unit=canonical,
                        source=self.source + f" [converted from {self.unit}]",
                        bounds=b, distribution=self.distribution)

    @property
    def v(self) -> float:
        return self.value


def _unit_checker(canonical: str):
    def check(q: Quantity) -> Quantity:
        return q.to(canonical)
    return check


def Q(canonical: str):
    """Annotated Quantity type whose unit is validated/converted to ``canonical``."""
    return Annotated[Quantity, AfterValidator(_unit_checker(canonical))]


def q(value, unit, source, bounds=None) -> dict:
    """Convenience constructor for raw config dicts (used by tests and generators)."""
    d = {"value": value, "unit": unit, "source": source}
    if bounds is not None:
        d["bounds"] = list(bounds)
    return d
