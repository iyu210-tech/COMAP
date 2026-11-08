"""Airflow providers: outdoor-air exchange and inter-section exchange flows.

The default :class:`ParametricAirflow` uses Q = Q_min + C_w * A_open * U_10 for every opening
(balanced two-way exchange: equal dry-air mass in and out). Later measurements (tracer gas) or
CFD can replace it with :class:`TableAirflow` (or any object with the same ``flows`` method)
without touching the thermal model.
"""

from __future__ import annotations

from typing import Dict, Protocol, Tuple

import numpy as np
import pandas as pd


class AirflowProvider(Protocol):
    def flows(self, wind_ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (Q_outdoor[n_t, n_sections], Q_connection[n_t, n_connections]) in m3/s."""


class ParametricAirflow:
    def __init__(self, cfg):
        self.q0_o = np.array([s.outdoor.min_flow.value for s in cfg.sections])
        self.ca_o = np.array([s.outdoor.wind_coefficient.value * s.outdoor.opening_area_m2 for s in cfg.sections])
        self.q0_c = np.array([c.min_flow.value for c in cfg.connections])
        self.ca_c = np.array([c.wind_coefficient.value * c.opening_area_m2 for c in cfg.connections])

    def flows(self, wind_ms):
        u = np.asarray(wind_ms, float)[:, None]
        return self.q0_o[None, :] + self.ca_o[None, :] * u, self.q0_c[None, :] + self.ca_c[None, :] * u


class TableAirflow:
    """Interpolate measured / CFD-derived flows vs wind speed from a CSV with columns
    ``wind_ms, out:<section_id>..., conn:<connection_id>...`` (m3/s)."""

    def __init__(self, cfg, csv_path):
        from .inputs import InputDataError
        df = pd.read_csv(csv_path, comment="#")
        need = ["wind_ms"] + [f"out:{s.id}" for s in cfg.sections] + [f"conn:{c.id}" for c in cfg.connections]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise InputDataError(f"airflow table {csv_path}: missing columns {missing}")
        vals = df[need].to_numpy(float)
        if not np.all(np.isfinite(vals)) or np.any(vals < 0):
            raise InputDataError(f"airflow table {csv_path}: values must be finite and non-negative")
        if np.any(np.diff(df["wind_ms"].to_numpy(float)) <= 0):
            raise InputDataError(f"airflow table {csv_path}: wind_ms must be strictly increasing")
        self.u = df["wind_ms"].to_numpy(float)
        self.out = [df[f"out:{s.id}"].to_numpy(float) for s in cfg.sections]
        self.conn = [df[f"conn:{c.id}"].to_numpy(float) for c in cfg.connections]

    def flows(self, wind_ms):
        u = np.asarray(wind_ms, float)
        qo = np.stack([np.interp(u, self.u, a) for a in self.out], axis=1) if self.out else np.zeros((len(u), 0))
        qc = np.stack([np.interp(u, self.u, a) for a in self.conn], axis=1) if self.conn else np.zeros((len(u), 0))
        return qo, qc


PROVIDERS: Dict[str, type] = {"parametric": ParametricAirflow, "table": TableAirflow}


def make_airflow(cfg):
    spec = cfg.airflow
    if spec.provider == "table":
        return TableAirflow(cfg, cfg.path(spec.file))
    return ParametricAirflow(cfg)
