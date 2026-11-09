"""Sensitivity of the single-zone bus shelter to a few labelled assumptions.

Each case is a full re-simulation (outdoor airflow, solar opening, device coverage). Comfort
targets of 28, 30 and 32 C SET are applied afterwards to the same comfort trajectories.

Three rankings, reported separately:

* greatest total relief
* greatest relief per kWh
* lowest electricity among systems that meet the comfort target (attainment >= 0.60),
  including the uncooled baseline; otherwise the case is infeasible

Relief per kWh considers only interventions with positive relief. If none has positive
relief, that ranking selects the baseline. A system is on the Pareto front when no other
system has both less electricity and more relief.

Supported, extrapolated and unassessed person-hours are reported separately. Strict-only
relief and attainment use the person-intervals that are inside the standard SET limits in
every alternative, so the eligible hours do not depend on which device is being scored.
``extrapolation_heavy`` is true when the greatest-relief claim shrinks by more than half
once those shared hours are the only ones counted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .plots import write_robustness_figures

SCENARIOS = ["A_fan", "B_evap", "C_large_evap"]
TARGETS = (28.0, 30.0, 32.0)
# Fraction of assessed person-hours that must be at or below the SET target.
ATTAINMENT_BAR = 0.60


def assumption_cases():
    """Nominal values and the documented bounds in examples/common (DEMO-ASSUMPTIONS)."""
    return [
        ("nominal", {}),
        ("airflow_low", {
            "sections[shelter].outdoor.wind_coefficient.value": 0.04,
            "sections[shelter].outdoor.min_flow.value": 0.2,
        }),
        ("airflow_high", {
            "sections[shelter].outdoor.wind_coefficient.value": 0.20,
            "sections[shelter].outdoor.min_flow.value": 1.0,
        }),
        ("solar_low", {"sections[shelter].sun.open_side_fraction.value": 0.3}),
        ("solar_high", {"sections[shelter].sun.open_side_fraction.value": 0.7}),
        ("coverage_low", {
            "device_presets.fan_system.covered_air_speed.value": 0.8,
            "device_presets.evaporative_cooler.plume_fraction.value": 0.10,
            "device_presets.evaporative_cooler.covered_air_speed.value": 0.6,
            "device_presets.large_evaporative_cooler.plume_fraction.value": 0.15,
            "device_presets.large_evaporative_cooler.covered_air_speed.value": 0.8,
        }),
        ("coverage_high", {
            "device_presets.fan_system.covered_air_speed.value": 1.8,
            "device_presets.evaporative_cooler.plume_fraction.value": 0.45,
            "device_presets.evaporative_cooler.covered_air_speed.value": 1.5,
            "device_presets.large_evaporative_cooler.plume_fraction.value": 0.55,
            "device_presets.large_evaporative_cooler.covered_air_speed.value": 1.8,
        }),
    ]


def _row(case, target, scenario, summary, relief, costs, preferred):
    v = summary["venue"]
    return dict(
        case=case, target_C=target, scenario=scenario,
        relief_Kph=relief, burden_Kph=v["burden_Kph"], attainment=v["attainment"],
        supported_person_hours=v["supported_person_hours"],
        extrapolated_person_hours=v["extrapolated_person_hours"],
        assessed_person_hours=v["assessed_person_hours"],
        unassessed_person_hours=v["unassessed_person_hours"],
        electricity_kWh=costs["electricity_kWh"],
        preferred=preferred, preferred_scenario="")


def preferred_scenario(relief_by_scenario: dict) -> str:
    best = max(relief_by_scenario, key=lambda s: (relief_by_scenario[s],))
    if relief_by_scenario[best] <= 0:
        return "baseline"
    return best


def evidence_class(source: str) -> str:
    """Classify a quantity's provenance from its source label.

    An explicit assumption wins over a parenthetical note such as
    "replace with CFD/measurement". Nothing in the examples is measured.
    """
    s = (source or "").lower()
    if "demo-assumption" in s or "assumption" in s:
        return "assumed"
    if any(w in s for w in ("measured", "measurement", "metered", "observed")):
        return "measured"
    if any(w in s for w in ("supplied", "published", "ashrae", "problem statement")):
        return "published"
    return "assumed"


def evidence_table(cfg) -> list:
    rows = []
    for name, preset in cfg.device_presets.items():
        for field in ("electric_power", "airflow", "covered_air_speed", "saturation_effectiveness",
                      "plume_fraction", "bleed_fraction"):
            q = getattr(preset, field)
            if q is None:
                continue
            rows.append(dict(preset=name, field=field, value=q.value, unit=q.unit,
                             evidence=evidence_class(q.source), source=q.source))
    return rows


def as_strict(df: pd.DataFrame) -> pd.DataFrame:
    """One scenario, restricted to its own supported hours.

    Comparisons across alternatives must use ``metrics.common_supported_frames`` so the
    eligible person-hours are the intersection, not each device's own supported set.
    """
    return metrics.common_supported_frames({"_": df})["_"]


def outside_applicability_fraction(df: pd.DataFrame) -> float:
    """Share of person-hours outside the standard (strict) SET input limits."""
    ph = df["person_hours"].to_numpy(float)
    total = float(ph.sum())
    if total <= 0:
        return 0.0
    outside = ph[df["cls"].to_numpy() != "supported"].sum()
    return float(outside / total)


def classify_criteria(records, attainment_bar=ATTAINMENT_BAR):
    """Return greatest-relief, relief-per-kWh, min-energy, and Pareto names.

    ``records`` include the uncooled baseline (relief 0, electricity 0). Min-energy is the
    lowest electricity among every record that reaches ``attainment_bar``, baseline
    included, and ``infeasible`` when none does. Relief per kWh ignores nonpositive
    relief; if every intervention has nonpositive relief the ranking is baseline.
    """
    inter = [r for r in records if r["scenario"] != "baseline"]
    greatest = preferred_scenario({r["scenario"]: r["relief_Kph"] for r in inter}) if inter else "infeasible"
    rated = [(r["relief_Kph"] / r["electricity_kWh"], r["scenario"])
             for r in inter
             if r["electricity_kWh"] > 1e-12 and np.isfinite(r["relief_Kph"]) and r["relief_Kph"] > 0]
    per_kwh = max(rated)[1] if rated else "baseline"
    feas = [r for r in records if np.isfinite(r["attainment"]) and r["attainment"] >= attainment_bar]
    if not feas:
        min_energy = "infeasible"
    else:
        min_energy = min(feas, key=lambda r: (r["electricity_kWh"], -r["relief_Kph"]))["scenario"]
    pareto = []
    for r in records:
        dominated = False
        for o in records:
            if o["scenario"] == r["scenario"]:
                continue
            cheaper = o["electricity_kWh"] <= r["electricity_kWh"] + 1e-9
            better = o["relief_Kph"] >= r["relief_Kph"] - 1e-9
            strict = (o["electricity_kWh"] < r["electricity_kWh"] - 1e-9
                      or o["relief_Kph"] > r["relief_Kph"] + 1e-9)
            if cheaper and better and strict:
                dominated = True
                break
        if not dominated:
            pareto.append(r["scenario"])
    return dict(greatest_relief=greatest, relief_per_kWh=per_kwh,
                min_energy_meeting_target=min_energy, pareto=pareto)


def run_robustness(study, outdir, cases=None, targets=None, scenarios=None) -> pd.DataFrame:
    cases = cases if cases is not None else assumption_cases()
    targets = tuple(targets) if targets is not None else TARGETS
    scenarios = scenarios or SCENARIOS
    rows = []
    for name, draw in cases:
        outs, _comps = study.compare_all(scenarios, draw=draw)
        base = outs["baseline"]
        strict_frames = metrics.common_supported_frames({sid: o.comfort_df for sid, o in outs.items()})
        for tgt in targets:
            base_sum = metrics.summarize(base.comfort_df, base.cfg, target=tgt)
            base_strict = metrics.summarize(strict_frames["baseline"], base.cfg, target=tgt)
            packed = []
            for sid, o in outs.items():
                summary = base_sum if sid == "baseline" else metrics.summarize(o.comfort_df, o.cfg, target=tgt)
                strict_sum = base_strict if sid == "baseline" else metrics.summarize(strict_frames[sid], o.cfg, target=tgt)
                if sid == "baseline":
                    rel = 0.0
                    strict_rel = 0.0
                else:
                    rel = metrics.compare(base_sum, summary, base.costs, o.costs)["relief_Kph"]
                    strict_rel = metrics.compare(base_strict, strict_sum, base.costs, o.costs)["relief_Kph"]
                packed.append(dict(
                    scenario=sid, summary=summary, strict=strict_sum, relief_Kph=rel,
                    strict_relief_Kph=strict_rel, electricity_kWh=o.costs["electricity_kWh"],
                    attainment=summary["venue"]["attainment"],
                    strict_attainment=strict_sum["venue"]["attainment"],
                    outside=outside_applicability_fraction(o.comfort_df)))
            labels = classify_criteria(packed)
            strict_labels = classify_criteria([
                dict(scenario=p["scenario"], relief_Kph=p["strict_relief_Kph"],
                     electricity_kWh=p["electricity_kWh"], attainment=p["strict_attainment"])
                for p in packed])
            strict_ranked = sorted((p["strict_relief_Kph"], p["scenario"])
                                   for p in packed if p["scenario"] != "baseline")
            gap = strict_ranked[-1][0] - strict_ranked[-2][0] if len(strict_ranked) >= 2 else 0.0
            # A label change with almost equal strict relief is a tie, not a new ranking.
            changes = gap > 1.0 and strict_labels["greatest_relief"] != labels["greatest_relief"]
            common_hours = [p["strict"]["venue"]["assessed_person_hours"] for p in packed]
            if not np.allclose(common_hours, common_hours[0], rtol=1e-6, atol=1e-6):
                raise RuntimeError(f"{name}: strict eligible person-hours differ across scenarios")
            pref = next(p for p in packed if p["scenario"] == labels["greatest_relief"])
            # The relief claim depends on extrapolated hours when the shared supported
            # hours retain less than half of it.
            heavy = (pref["scenario"] != "baseline" and pref["relief_Kph"] > 1.0
                     and pref["strict_relief_Kph"] < 0.5 * pref["relief_Kph"])
            for p in packed:
                rec = _row(name, tgt, p["scenario"], p["summary"], p["relief_Kph"],
                           {"electricity_kWh": p["electricity_kWh"]}, p["scenario"] == labels["greatest_relief"])
                rec.update(
                    preferred_scenario=labels["greatest_relief"],
                    relief_per_kWh=(p["relief_Kph"] / p["electricity_kWh"] if p["electricity_kWh"] > 1e-12 else np.nan),
                    best_per_kwh=p["scenario"] == labels["relief_per_kWh"],
                    per_kwh_scenario=labels["relief_per_kWh"],
                    min_energy=p["scenario"] == labels["min_energy_meeting_target"],
                    min_energy_scenario=labels["min_energy_meeting_target"],
                    on_pareto=p["scenario"] in labels["pareto"],
                    pareto=";".join(labels["pareto"]),
                    outside_applicability_fraction=p["outside"],
                    strict_relief_Kph=p["strict_relief_Kph"],
                    strict_attainment=p["strict_attainment"],
                    strict_preferred_scenario=strict_labels["greatest_relief"],
                    strict_per_kwh_scenario=strict_labels["relief_per_kWh"],
                    strict_min_energy_scenario=strict_labels["min_energy_meeting_target"],
                    strict_common_person_hours=common_hours[0],
                    strict_relief_gap_Kph=gap,
                    conclusion_changes_if_strict=changes,
                    extrapolation_heavy=heavy)
                rows.append(rec)
    df = pd.DataFrame(rows)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "robustness_results.csv", index=False)
    evid = pd.DataFrame(evidence_table(study.cfg0))
    evid.to_csv(outdir / "equipment_evidence.csv", index=False)
    winners = (df[df["preferred"]]
               .groupby(["case", "target_C"], as_index=False)
               [["preferred_scenario", "per_kwh_scenario", "min_energy_scenario",
                 "strict_preferred_scenario", "conclusion_changes_if_strict",
                 "extrapolation_heavy"]].first())
    summary = dict(
        attainment_bar=ATTAINMENT_BAR,
        rankings=["greatest_relief", "relief_per_kWh", "min_energy_meeting_target"],
        targets_C=list(targets),
        greatest_relief_changes=bool(winners["preferred_scenario"].nunique() > 1),
        winners=winners.to_dict(orient="records"),
        evidence_counts=evid["evidence"].value_counts().to_dict() if len(evid) else {},
        any_measured_equipment_input=bool((evid["evidence"] == "measured").any()) if len(evid) else False,
        note=("Electric power is published (problem statement). Airflow, effectiveness, plume fraction "
              "and coverage speed are assumed. No equipment input is measured. "
              "supported, extrapolated and unassessed person-hours are separate. "
              "strict_* rankings use only person-intervals supported in every alternative. "
              "extrapolation_heavy means the greatest-relief claim keeps less than half of its "
              "relief on those shared hours. Min-energy includes the uncooled baseline. "
              "Relief per kWh ignores nonpositive relief. Weather and occupancy remain SYNTHETIC_DEMONSTRATION."))
    (outdir / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
    write_robustness_figures(outdir, df)
    return df
