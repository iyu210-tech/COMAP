"""Aggregation of local comfort results into section and whole-venue metrics.

Definitions (per population-interval, computed locally *before* aggregation):
  discomfort D = max(ET - target, 0)            [K]   ET = SET (decision basis)
  burden       B = D * people * dt              [K * person-hours]  ("modeled person-equivalent-degree-hours")
  person-hours = people * dt (NOT unique visitors; never multiplied by dwell time)

Relief = baseline burden - intervention burden, evaluated on section-intervals assessed in
*both* runs (matched). Negative relief is preserved. Ratios are formed only after summing
numerators and denominators over the venue (no averaging of section ratios).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def crf(rate: float, years: float) -> float:
    if years <= 0:
        raise ValueError("lifetime must be positive")
    if rate == 0:
        return 1.0 / years
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def section_intervals(df: pd.DataFrame, target: float) -> pd.DataFrame:
    d = df.copy()
    d["discomfort"] = np.maximum(d["et"] - target, 0.0)
    d["burden"] = np.where(d["assessed"], d["discomfort"] * d["person_hours"], 0.0)
    d["meet_ph"] = np.where(d["assessed"] & (d["et"] <= target), d["person_hours"], 0.0)
    d["assessed_ph"] = np.where(d["assessed"], d["person_hours"], 0.0)
    d["supported_ph"] = np.where(d["cls"] == "supported", d["person_hours"], 0.0)
    d["extrapolated_ph"] = np.where(d["cls"] == "extrapolated", d["person_hours"], 0.0)
    d["served_ph"] = np.where(d["served"], d["person_hours"], 0.0)
    d["unassessed_pop"] = (~d["assessed"]) & (d["person_hours"] > 0)
    g = d.groupby(["section", "k"], sort=False).agg(
        t_h=("t_h", "first"), person_hours=("person_hours", "sum"), burden=("burden", "sum"),
        meet_ph=("meet_ph", "sum"), assessed_ph=("assessed_ph", "sum"), supported_ph=("supported_ph", "sum"),
        extrapolated_ph=("extrapolated_ph", "sum"), served_ph=("served_ph", "sum"),
        any_unassessed=("unassessed_pop", "any"))
    g["si_assessed"] = ~g["any_unassessed"]
    return g.reset_index()


def summarize(df: pd.DataFrame, cfg, target: float | None = None) -> dict:
    target = cfg.comfort.target.value if target is None else float(target)
    si = section_intervals(df, target)
    out = {"sections": {}}

    def agg(x):
        occ = float(x["person_hours"].sum())
        ass = float(x["assessed_ph"].sum())
        meet = float(x["meet_ph"].sum())
        return dict(occupied_person_hours=occ, assessed_person_hours=ass, unassessed_person_hours=occ - ass,
                    supported_person_hours=float(x["supported_ph"].sum()),
                    extrapolated_person_hours=float(x["extrapolated_ph"].sum()),
                    served_person_hours=float(x["served_ph"].sum()),
                    burden_Kph=float(x["burden"].sum()), meeting_person_hours=meet,
                    attainment=(meet / ass) if ass > 0 else float("nan"),
                    unassessed_fraction=((occ - ass) / occ) if occ > 0 else 0.0,
                    burden_per_person_hour=(float(x["burden"].sum()) / ass) if ass > 0 else float("nan"))
    for sid, x in si.groupby("section", sort=False):
        out["sections"][sid] = agg(x)
    out["venue"] = agg(si)
    thr = cfg.comfort.poorly_served_attainment
    out["poorly_served"] = [s for s, v in out["sections"].items()
                            if v["occupied_person_hours"] > 0 and (np.isnan(v["attainment"]) or v["attainment"] < thr)]
    out["_si"] = si
    return out


def ledger_totals(ledgers, passive, cfg, window_days: float) -> dict:
    t = cfg.tariffs
    scale = t.season_days.value / window_days
    r = t.discount_rate.value
    kwh = sum(x["electricity_kWh"] for x in ledgers)
    water = sum(x["water_m3"] for x in ledgers)
    e_cost = sum(x["electricity_cost_usd"] for x in ledgers)
    w_cost = sum(x["water_cost_usd"] for x in ledgers)
    cap = sum(x["capital_usd"] for x in ledgers) + sum(p.capital_cost.value for p in passive)
    ann = sum(crf(r, x["lifetime_y"]) * x["capital_usd"] + x["maintenance_usd_y"] for x in ledgers)
    ann += sum(crf(r, p.lifetime.value) * p.capital_cost.value + p.maintenance.value for p in passive)
    peak_kw = sum(x["electric_power_kW"] for x in ledgers)
    return dict(electricity_kWh=kwh, water_m3=water, energy_cost_usd=e_cost, water_cost_usd=w_cost,
                investment_usd=cap, installed_kW=peak_kw,
                season_scale=scale, season_electricity_kWh=kwh * scale, season_water_m3=water * scale,
                season_operating_usd=(e_cost + w_cost) * scale, season_capital_and_maintenance_usd=ann,
                season_total_cost_usd=(e_cost + w_cost) * scale + ann, n_devices=len(ledgers))


def compare(base: dict, inter: dict, base_cost: dict, int_cost: dict) -> dict:
    """Matched comparison of an intervention against the baseline (same weather/occupancy/draw)."""
    a = base["_si"].set_index(["section", "k"]).add_suffix("_b")
    b = inter["_si"].set_index(["section", "k"]).add_suffix("_i")
    j = a.join(b, how="inner")
    if not np.allclose(j["person_hours_b"], j["person_hours_i"], rtol=1e-9, atol=1e-9):
        raise RuntimeError("baseline and intervention occupancy differ: comparison is not matched")
    matched = j["si_assessed_b"] & j["si_assessed_i"]
    relief_si = np.where(matched, j["burden_b"] - j["burden_i"], 0.0)
    j = j.assign(relief=relief_si, matched=matched)
    per = {}
    for sid, x in j.groupby(level=0, sort=False):
        per[sid] = dict(relief_Kph=float(x["relief"].sum()),
                        baseline_burden_matched_Kph=float(np.where(x["matched"], x["burden_b"], 0).sum()),
                        residual_burden_matched_Kph=float(np.where(x["matched"], x["burden_i"], 0).sum()),
                        unmatched_person_hours=float(np.where(~x["matched"], x["person_hours_b"], 0).sum()))
    relief = float(j["relief"].sum())
    d_kwh = int_cost["electricity_kWh"] - base_cost["electricity_kWh"]
    d_cost = int_cost["season_total_cost_usd"] - base_cost["season_total_cost_usd"]
    scale = int_cost["season_scale"]
    return dict(
        relief_Kph=relief,
        baseline_burden_matched_Kph=float(np.where(j["matched"], j["burden_b"], 0).sum()),
        residual_burden_matched_Kph=float(np.where(j["matched"], j["burden_i"], 0).sum()),
        unmatched_person_hours=float(np.where(~j["matched"], j["person_hours_b"], 0).sum()),
        incremental_kWh=d_kwh, incremental_season_cost_usd=d_cost,
        relief_per_kWh=(relief / d_kwh) if d_kwh > 1e-12 else float("nan"),
        season_relief_Kph=relief * scale,
        relief_per_usd=(relief * scale / d_cost) if d_cost > 1e-9 else float("nan"),
        sections=per, worsened_sections=[s for s, v in per.items() if v["relief_Kph"] < -1e-9])
