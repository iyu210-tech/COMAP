"""Batch Plotly report figures for the single-zone Problem B slice.

Writes standalone HTML files (no Streamlit). Cumulative kWh comes from the timeseries export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ORDER = ["baseline", "A_fan", "B_evap", "C_large_evap"]
COLORS = {
    "baseline": "#4C566A",
    "A_fan": "#5E81AC",
    "B_evap": "#A3BE8C",
    "C_large_evap": "#BF616A",
}


def _scenario_order(df: pd.DataFrame, col: str = "scenario") -> pd.DataFrame:
    cats = [s for s in ORDER if s in set(df[col])]
    extra = [s for s in df[col].unique() if s not in cats]
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=cats + extra, ordered=True)
    return out.sort_values([col, "t_h"] if "t_h" in out.columns else [col])


def people_weighted_set(comfort: pd.DataFrame) -> pd.DataFrame:
    """People-weighted mean equivalent temperature (SET) per scenario and time."""
    df = comfort.copy()
    if "et" not in df.columns:
        df["et"] = df["set_extended"] if "set_extended" in df.columns else df["set_strict"]
    rows = []
    for (sc, t), g in df.groupby(["scenario", "t_h"], sort=True):
        w = g["people"].to_numpy(float)
        et = g["et"].to_numpy(float)
        m = np.isfinite(et) & np.isfinite(w) & (w >= 0)
        if not m.any() or w[m].sum() <= 0:
            val = float("nan")
        else:
            val = float((et[m] * w[m]).sum() / w[m].sum())
        rows.append(dict(scenario=sc, t_h=t, SET_C=val, people=float(w[m].sum()) if m.any() else 0.0))
    return pd.DataFrame(rows)


def write_report(outdir, timeseries: pd.DataFrame, comfort: pd.DataFrame,
                 comparison: Optional[pd.DataFrame] = None) -> Dict[str, Path]:
    """Write the four Problem B report figures. Returns {name: path}."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = _scenario_order(timeseries)
    air = (ts.groupby(["scenario", "t_h"], as_index=False)
             .agg(T_air_C=("T_air_C", "mean"), RH_pct=("RH_pct", "mean"),
                  T_out_C=("T_out_C", "mean"), electricity_kWh_cum=("electricity_kWh_cum", "max")))
    air = _scenario_order(air)

    paths: Dict[str, Path] = {}

    fig = go.Figure()
    for sc in air["scenario"].cat.categories:
        g = air[air["scenario"] == sc]
        fig.add_trace(go.Scatter(x=g["t_h"], y=g["T_air_C"], mode="lines", name=str(sc),
                                 line=dict(color=COLORS.get(str(sc)))))
    base = air[air["scenario"] == air["scenario"].cat.categories[0]]
    fig.add_trace(go.Scatter(x=base["t_h"], y=base["T_out_C"], mode="lines", name="outdoor",
                             line=dict(color="#D08770", dash="dash")))
    fig.update_layout(title="Shelter air temperature (24 h synthetic day)",
                      xaxis_title="Hour of day", yaxis_title="Air temperature [C]",
                      template="plotly_white", legend_title="Scenario")
    p = outdir / "air_temperature.html"
    fig.write_html(str(p), include_plotlyjs="cdn")
    paths["air_temperature"] = p

    fig = px.line(air, x="t_h", y="RH_pct", color="scenario",
                  color_discrete_map=COLORS,
                  title="Shelter relative humidity",
                  labels={"t_h": "Hour of day", "RH_pct": "Relative humidity [%]", "scenario": "Scenario"})
    fig.update_layout(template="plotly_white")
    p = outdir / "relative_humidity.html"
    fig.write_html(str(p), include_plotlyjs="cdn")
    paths["relative_humidity"] = p

    set_df = _scenario_order(people_weighted_set(comfort))
    fig = px.line(set_df, x="t_h", y="SET_C", color="scenario",
                  color_discrete_map=COLORS,
                  title="People-weighted mean SET (thermal comfort index)",
                  labels={"t_h": "Hour of day", "SET_C": "SET [C]", "scenario": "Scenario"})
    fig.update_layout(template="plotly_white")
    p = outdir / "thermal_comfort.html"
    fig.write_html(str(p), include_plotlyjs="cdn")
    paths["thermal_comfort"] = p

    fig = px.line(air, x="t_h", y="electricity_kWh_cum", color="scenario",
                  color_discrete_map=COLORS,
                  title="Cumulative electricity use",
                  labels={"t_h": "Hour of day", "electricity_kWh_cum": "Cumulative electricity [kWh]",
                          "scenario": "Scenario"})
    fig.update_layout(template="plotly_white")
    p = outdir / "cumulative_kWh.html"
    fig.write_html(str(p), include_plotlyjs="cdn")
    paths["cumulative_kWh"] = p

    if comparison is not None and len(comparison):
        comparison.to_csv(outdir / "comparison.csv", index=False)
    return paths
