"""Sensitivity of the single-zone bus shelter to a few labelled assumptions.

Each case is a full re-simulation (outdoor airflow, solar opening, device coverage). Comfort
targets of 28, 30 and 32 C SET are applied afterwards to the same comfort trajectories, so
supported vs extrapolated person-hours — which depend on the SET input envelope, not the
target — are identical across targets.

Preferred system: the intervention with the largest matched relief. Baseline is preferred
when every intervention has relief <= 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import metrics
from .plots import write_robustness_figures

SCENARIOS = ["A_fan", "B_evap", "C_large_evap"]
TARGETS = (28.0, 30.0, 32.0)


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


def run_robustness(study, outdir, cases=None, targets=None, scenarios=None) -> pd.DataFrame:
    cases = cases if cases is not None else assumption_cases()
    targets = tuple(targets) if targets is not None else TARGETS
    scenarios = scenarios or SCENARIOS
    rows = []
    for name, draw in cases:
        outs, _comps = study.compare_all(scenarios, draw=draw)
        base = outs["baseline"]
        for tgt in targets:
            base_sum = metrics.summarize(base.comfort_df, base.cfg, target=tgt)
            relief = {}
            packed = []
            for sid, o in outs.items():
                summary = base_sum if sid == "baseline" else metrics.summarize(o.comfort_df, o.cfg, target=tgt)
                if sid == "baseline":
                    rel = 0.0
                else:
                    rel = metrics.compare(base_sum, summary, base.costs, o.costs)["relief_Kph"]
                    relief[sid] = rel
                packed.append((sid, summary, rel, o.costs))
            winner = preferred_scenario(relief)
            for sid, summary, rel, costs in packed:
                rec = _row(name, tgt, sid, summary, rel, costs, sid == winner)
                rec["preferred_scenario"] = winner
                rows.append(rec)
    df = pd.DataFrame(rows)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "robustness_results.csv", index=False)
    winners = (df[df["preferred"]]
               .groupby(["case", "target_C"], as_index=False)["preferred_scenario"].first())
    changed = int(winners["preferred_scenario"].nunique() > 1)
    summary = dict(
        definition="preferred = intervention with the largest matched relief; baseline if all relief <= 0",
        targets_C=list(targets),
        preferred_changes=bool(changed),
        winners=winners.to_dict(orient="records"),
        note=("supported_person_hours and extrapolated_person_hours do not depend on the SET target. "
              "They count comfort evaluations inside strict ASHRAE 55 input limits vs the declared "
              "extended envelope. Weather and occupancy remain SYNTHETIC_DEMONSTRATION."))
    (outdir / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
    write_robustness_figures(outdir, df)
    return df
