"""Bounded configuration search over actual choices.

Decision variables: for each placement slot, a device option (none / preset / preset*N), a
common operating schedule, and any subset of the passive-option catalogue. Air temperature and
humidity are never decision variables -- they are simulation outputs.

Each candidate is simulated against the same weather and occupancy as the baseline. Outcomes:
  meets_target            service target and all constraints satisfied
  physical_shortfall      evaluable, but attainment / section attainment below target
  unsupported_evaluation  too many unassessed person-hours, or the solver failed
  constraint_violation    water (or other post-simulation) constraint exceeded
Configurations violating pre-simulation constraints (device count, investment) are not run.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import metrics

_W = {}


def enumerate_configs(cfg) -> List[dict]:
    sp = cfg.search
    slot_choices = []
    for sl in sp.slots:
        ch = []
        for o in sl.options:
            name, _, n = o.partition("*")
            ch.append((name, int(n) if n else (0 if name == "none" else 1)))
        slot_choices.append(ch)
    passive_sets = [()]
    for r in range(1, len(sp.passive_options) + 1):
        passive_sets += list(itertools.combinations(sp.passive_options, r))
    popts = {p.id: p for p in cfg.passive_options}
    out = []
    for choice in itertools.product(*slot_choices):
        n_dev = sum(n for _, n in choice)
        scheds = sp.schedules if n_dev > 0 else [None]
        for sch in scheds:
            for ps in passive_sets:
                inv = sum(cfg.device_presets[p].capital_cost.value * n for p, n in choice if p != "none")
                inv += sum(popts[p].capital_cost.value for p in ps)
                cand = dict(slots={sl.id: (p, n) for sl, (p, n) in zip(sp.slots, choice)}, schedule=sch,
                            passive=list(ps), n_devices=n_dev, investment_usd=inv)
                if n_dev > sp.constraints.max_devices or inv > sp.constraints.max_investment_usd:
                    continue
                out.append(cand)
    out = out[: sp.max_configurations]
    for i, c in enumerate(out):
        c["config_id"] = i
    return out


def candidate_devices(cfg, cand) -> List[dict]:
    devs = []
    slots = {s.id: s for s in cfg.search.slots}
    for sid, (preset, n) in cand["slots"].items():
        if preset == "none":
            continue
        sl = slots[sid]
        for j in range(n):
            d = dict(id=f"{sid}__{preset}__{j}", preset=preset, schedule=cand["schedule"],
                     serves=[s.model_dump(exclude_none=True) for s in sl.serves])
            if sl.intake:
                d["intake"] = sl.intake
            devs.append(d)
    return devs


def label(cand) -> str:
    parts = [f"{k}={p}x{n}" for k, (p, n) in cand["slots"].items() if p != "none"]
    s = ",".join(parts) or "no-devices"
    if cand["schedule"]:
        s += f"@{cand['schedule']}"
    if cand["passive"]:
        s += "+" + "+".join(cand["passive"])
    return s


def evaluate_candidate(study, cand, base, sim_overrides=None, draw=None) -> dict:
    cfg = study.cfg0
    sp = cfg.search
    devs = candidate_devices(cfg, cand)
    t0 = time.perf_counter()
    o = study.run([d["id"] for d in devs], cand["passive"], draw=draw, extra_devices=devs, label=label(cand),
                  sim_overrides=sim_overrides)
    row = dict(config_id=cand["config_id"], label=label(cand), schedule=cand["schedule"] or "",
               passive=";".join(cand["passive"]), n_devices=cand["n_devices"], investment_usd=cand["investment_usd"],
               wall_s=time.perf_counter() - t0)
    if not o.ok:
        row.update(status="unsupported_evaluation", reason="solver: " + o.sim.message)
        return row
    v = o.summary["venue"]
    comp = metrics.compare(base.summary, o.summary, base.costs, o.costs)
    secs = {k: s["attainment"] for k, s in o.summary["sections"].items() if s["occupied_person_hours"] > 0}
    min_sec = min((x for x in secs.values() if not np.isnan(x)), default=float("nan"))
    row.update(attainment=v["attainment"], min_section_attainment=min_sec, unassessed_fraction=v["unassessed_fraction"],
               burden_Kph=v["burden_Kph"], relief_Kph=comp["relief_Kph"], electricity_kWh=o.costs["electricity_kWh"],
               water_m3=o.costs["water_m3"], season_total_cost_usd=o.costs["season_total_cost_usd"],
               relief_per_kWh=comp["relief_per_kWh"], relief_per_usd=comp["relief_per_usd"],
               poorly_served=";".join(o.summary["poorly_served"]), worsened=";".join(comp["worsened_sections"]),
               energy_residual_rel=o.sim.conservation["energy_residual_rel"], invalid_states=sum(o.sim.invalid.values()))
    tg = sp.target
    if v["unassessed_fraction"] > tg.max_unassessed_fraction:
        row.update(status="unsupported_evaluation", reason=f"unassessed fraction {v['unassessed_fraction']:.3f}")
    elif o.costs["water_m3"] > sp.constraints.max_water_m3_per_day * (cfg.simulation.end_h - cfg.simulation.start_h) / 24:
        row.update(status="constraint_violation", reason=f"water {o.costs['water_m3']:.2f} m3")
    elif v["attainment"] < tg.min_attainment or (not np.isnan(min_sec) and min_sec < tg.min_section_attainment):
        row.update(status="physical_shortfall", reason=f"attainment {v['attainment']:.3f}, worst section {min_sec:.3f}")
    else:
        row.update(status="meets_target", reason="")
    return row


def _init(raw, sim_overrides):
    from .runner import Study
    _W["study"] = Study(raw, sim_overrides=sim_overrides)
    _W["base"] = _W["study"].run([], [], label="baseline")
    _W["sim_overrides"] = sim_overrides


def _work(cand):
    return evaluate_candidate(_W["study"], cand, _W["base"])


FIELDS = ["config_id", "label", "status", "reason", "schedule", "passive", "n_devices", "investment_usd", "attainment",
          "min_section_attainment", "unassessed_fraction", "burden_Kph", "relief_Kph", "electricity_kWh", "water_m3",
          "season_total_cost_usd", "relief_per_kWh", "relief_per_usd", "poorly_served", "worsened",
          "energy_residual_rel", "invalid_states", "wall_s"]


def run_search(study, outdir, workers: Optional[int] = None, limit: Optional[int] = None,
               sim_overrides=None, verify=True, progress=print) -> dict:
    cfg = study.cfg0
    if cfg.search is None:
        raise ValueError("configuration has no 'search' block")
    cands = enumerate_configs(cfg)
    if limit:
        cands = cands[:limit]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "search_results.csv"
    progress(f"search: {len(cands)} candidate configurations, {workers} workers -> {path}")
    t0 = time.perf_counter()
    rows = []
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        if workers == 1:
            _init(study.raw, sim_overrides)
            it = (_work(c) for c in cands)
            for i, r in enumerate(it):
                rows.append(r)
                w.writerow(r)
                fh.flush()
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(study.raw, sim_overrides)) as ex:
                futs = [ex.submit(_work, c) for c in cands]
                for i, f in enumerate(as_completed(futs)):
                    r = f.result()
                    rows.append(r)
                    w.writerow(r)
                    fh.flush()
                    if (i + 1) % max(1, len(cands) // 10) == 0:
                        progress(f"  {i + 1}/{len(cands)} done ({time.perf_counter() - t0:.1f} s)")
    wall = time.perf_counter() - t0
    key = "electricity_kWh" if cfg.search.objective == "energy" else "season_total_cost_usd"
    feas = [r for r in rows if r["status"] == "meets_target"]
    feas.sort(key=lambda r: (r[key], r["season_total_cost_usd"], r["investment_usd"]))
    counts = {s: sum(r["status"] == s for r in rows) for s in
              ("meets_target", "physical_shortfall", "unsupported_evaluation", "constraint_violation")}
    result = dict(n_candidates=len(cands), workers=workers, wall_s=wall,
                  mean_candidate_s=float(np.mean([r["wall_s"] for r in rows])) if rows else None,
                  objective=cfg.search.objective, target=cfg.search.target.model_dump(),
                  constraints=cfg.search.constraints.model_dump(), status_counts=counts, results_csv=str(path))
    if not feas:
        result["selected"] = None
        result["message"] = ("NO TESTED CONFIGURATION MEETS THE TARGET. "
                             f"physical shortfalls: {counts['physical_shortfall']}, unsupported evaluations: "
                             f"{counts['unsupported_evaluation']}, constraint violations: {counts['constraint_violation']}.")
        best_short = sorted([r for r in rows if r["status"] == "physical_shortfall"],
                            key=lambda r: -r.get("attainment", 0))[:5]
        result["closest_physical_shortfalls"] = best_short
    else:
        sel = feas[0]
        result["selected"] = sel
        result["runner_up"] = feas[1:6]
        if verify:
            cand = next(c for c in cands if c["config_id"] == sel["config_id"])
            from .runner import Study
            fresh = Study(study.raw, sim_overrides=dict(rtol=1e-8, output_step_s=150.0, **(sim_overrides or {})))
            base = fresh.run([], [], label="baseline")
            v = evaluate_candidate(fresh, cand, base)
            result["verification"] = dict(
                method="independent re-simulation in a fresh process state with rtol=1e-8 and 150 s output step",
                status=v["status"], attainment=v.get("attainment"), electricity_kWh=v.get("electricity_kWh"),
                attainment_difference=(v.get("attainment", np.nan) - sel["attainment"]),
                confirmed=v["status"] == "meets_target")
    (outdir / "search_summary.json").write_text(json.dumps(_jsonable(result), indent=2))
    return result


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o
