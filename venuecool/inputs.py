"""CSV weather and occupancy schedules.

Weather CSV (hourly or finer), '#' comment lines may carry ``key: value`` metadata such as
``data_status: SYNTHETIC_DEMONSTRATION``. Required columns::

    time_h, temp_C, rh_pct, pressure_Pa, wind_ms, ghi_Wm2

``dni_Wm2`` and ``dhi_Wm2`` are required unless the configuration explicitly selects
``solar_split: erbs`` (then they are *derived* and labelled as such). Missing columns, gaps,
NaNs or out-of-range values raise :class:`InputDataError` -- nothing is silently fabricated.

Occupancy CSV: ``time_h`` plus one column of people counts per section. Each row holds the mean
number of people present from that row's time until the next row (step function).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import psychro, solar


class InputDataError(ValueError):
    pass


def read_meta(path) -> Dict[str, str]:
    meta = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") and ":" in line:
            k, v = line[1:].split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def _read_csv_with_meta(path: Path):
    if not Path(path).exists():
        raise InputDataError(f"input file not found: {path} (no substitute data will be generated)")
    meta = read_meta(path)
    try:
        df = pd.read_csv(path, comment="#")
    except Exception as e:  # noqa: BLE001
        raise InputDataError(f"failed to read {path}: {e}") from e
    meta.setdefault("data_status", "UNSPECIFIED")
    return df, meta


@dataclass
class Weather:
    t_s: np.ndarray  # seconds from day-0 midnight
    temp_C: np.ndarray
    w: np.ndarray  # humidity ratio kg/kg_da (derived from RH, interpolated as conserved quantity)
    pressure_Pa: np.ndarray
    wind_ms: np.ndarray
    ghi: np.ndarray
    dni: np.ndarray
    dhi: np.ndarray
    meta: Dict[str, str] = field(default_factory=dict)

    def at(self, t_s):
        """Linear interpolation of all channels at time(s) t_s (seconds)."""
        f = lambda a: np.interp(t_s, self.t_s, a)  # noqa: E731
        return f(self.temp_C), f(self.w), f(self.pressure_Pa), f(self.wind_ms), f(self.ghi), f(self.dni), f(self.dhi)


def load_weather(path, site, solar_split="provided") -> Weather:
    df, meta = _read_csv_with_meta(path)
    req = ["time_h", "temp_C", "rh_pct", "pressure_Pa", "wind_ms", "ghi_Wm2"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise InputDataError(f"{path}: missing required weather columns {missing}")
    has_split = "dni_Wm2" in df.columns and "dhi_Wm2" in df.columns
    if not has_split and solar_split != "erbs":
        raise InputDataError(
            f"{path}: dni_Wm2/dhi_Wm2 missing. Provide them or set weather.solar_split: erbs "
            "explicitly to derive them (derived values are labelled).")
    cols = req + (["dni_Wm2", "dhi_Wm2"] if has_split and solar_split == "provided" else [])
    if df[cols].isna().any().any():
        raise InputDataError(f"{path}: NaN/blank values in weather data; fill or remove the period explicitly")
    t = df["time_h"].to_numpy(float)
    if np.any(np.diff(t) <= 0):
        raise InputDataError(f"{path}: time_h must be strictly increasing")
    if np.max(np.diff(t)) > 3.0 + 1e-9:
        raise InputDataError(f"{path}: gaps longer than 3 h in weather data")
    rh = df["rh_pct"].to_numpy(float)
    if np.any((rh < 0) | (rh > 100)):
        raise InputDataError(f"{path}: rh_pct outside [0, 100]")
    for c, lo, hi in [("temp_C", -40, 60), ("pressure_Pa", 5e4, 1.1e5), ("wind_ms", 0, 60), ("ghi_Wm2", 0, 1500)]:
        a = df[c].to_numpy(float)
        if np.any((a < lo) | (a > hi)):
            raise InputDataError(f"{path}: {c} outside plausible range [{lo}, {hi}]")
    temp = df["temp_C"].to_numpy(float)
    p = df["pressure_Pa"].to_numpy(float)
    w = psychro.w_from_rh(temp, rh / 100.0, p)
    ghi = df["ghi_Wm2"].to_numpy(float)
    if solar_split == "provided":
        dni, dhi = df["dni_Wm2"].to_numpy(float), df["dhi_Wm2"].to_numpy(float)
    else:
        alt, _ = solar.solar_position(t, site.day_of_year, site.latitude_deg, site.longitude_deg, site.utc_offset_h)
        dni, dhi = solar.erbs_split(ghi, alt, site.day_of_year)
        meta["solar_split"] = "DERIVED via Erbs (1982) from GHI"
    return Weather(t * 3600.0, temp, w, p, df["wind_ms"].to_numpy(float), ghi, dni, dhi, meta)


@dataclass
class OccupancySchedule:
    t_s: np.ndarray  # row start times (s)
    counts: Dict[str, np.ndarray]
    meta: Dict[str, str]

    def at(self, col: str, t_s):
        """People present (step function, value of the most recent row)."""
        idx = np.searchsorted(self.t_s, t_s, side="right") - 1
        idx = np.clip(idx, 0, len(self.t_s) - 1)
        return self.counts[col][idx]

    def person_seconds(self, col: str, t0: float, t1: float) -> float:
        """Exact integral of the step schedule over [t0, t1]."""
        edges = np.concatenate([self.t_s, [np.inf]])
        tot = 0.0
        for i in range(len(self.t_s)):
            a, b = max(t0, edges[i]), min(t1, edges[i + 1])
            if b > a:
                tot += self.counts[col][i] * (b - a)
        return tot


def load_occupancy(path, columns) -> OccupancySchedule:
    df, meta = _read_csv_with_meta(path)
    if "time_h" not in df.columns:
        raise InputDataError(f"{path}: missing time_h column")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise InputDataError(f"{path}: missing occupancy columns {missing}")
    if df[list(columns)].isna().any().any():
        raise InputDataError(f"{path}: blank occupancy values")
    t = df["time_h"].to_numpy(float)
    if np.any(np.diff(t) <= 0):
        raise InputDataError(f"{path}: time_h must be strictly increasing")
    counts = {c: df[c].to_numpy(float) for c in columns}
    for c, a in counts.items():
        if np.any(a < 0):
            raise InputDataError(f"{path}: negative people count in column {c}")
    return OccupancySchedule(t * 3600.0, counts, meta)
