"""Seeded uncertainty propagation and sensitivity.

* Latin-hypercube sampling over every quantity that declares ``bounds`` (or a selected list).
* Common random numbers: within one draw, the baseline and every intervention are simulated with
  the *same* parameter values, weather and occupancy, so relief differences are matched.
* Sensitivity: Spearman rank correlation of each parameter with each scenario's relief, and an
  optional one-at-a-time low/high sweep (tornado).
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import qmc, spearmanr

from . import metrics
from .config import find_uncertain

_W = {}


def uncertain_parameters(study, selection="all"):
    found = find_uncertain(study.raw)
    if selection != "all":
        sel = set(selection)
        found = [f for f in found if f[0] in sel or f[0].rsplit(".value", 1)[0] in sel]
    return found


def unused_by(study, sid, path):
    """True if ``path`` cannot affect scenario ``sid`` or its baseline (preset, device or passive
    option the scenario does not use). Such correlations are pure sampling noise."""
    sc = next((s for s in study.cfg0.scenarios if s.id == sid), None)
    if sc is None:
        return False
    used_presets = {d.preset for d in study.cfg0.devices if d.id in sc.devices}
    for prefix, used in (("passive_options[", set(sc.passive)), ("devices[", set(sc.devices))):
        if path.startswith(prefix):
            return path[len(prefix):].split("]", 1)[0] not in used
    if path.startswith("device_presets."):
        return path.split(".")[1] not in used_presets
    return False


def lhs_draws(params, n, seed):
    sampler = qmc.LatinHypercube(d=len(params), seed=seed)
    u = sampler.random(n)
    draws = []
    for row in u:
        d = {}
        for (path, q), x in zip(params, row):
            lo, hi = q["bounds"]
            if q.get("distribution", "uniform") == "triangular":
                c = (q["value"] - lo) / (hi - lo) if hi > lo else 0.5
                x = np.sqrt(x * c) if x < c else 1 - np.sqrt((1 - x) * (1 - c))
            d[path] = float(lo + x * (hi - lo))
        draws.append(d)
    return draws


def _init(raw):
    from .runner import Study
    _W["study"] = Study(raw)


def evaluate_draw(study, draw, scenarios):
    base = study.run_scenario("baseline", draw=draw)
    row = {}
    if not base.ok:
        return dict(status="baseline_solver_failed")
    row["baseline_burden_Kph"] = base.summary["venue"]["burden_Kph"]
    row["baseline_attainment"] = base.summary["venue"]["attainment"]
    for sid in scenarios:
        o = study.run_scenario(sid, draw=draw)
        if not o.ok:
            row[f"{sid}:status"] = "solver_failed"
            continue
        c = metrics.compare(base.summary, o.summary, base.costs, o.costs)
        row[f"{sid}:relief_Kph"] = c["relief_Kph"]
        row[f"{sid}:attainment"] = o.summary["venue"]["attainment"]
        row[f"{sid}:electricity_kWh"] = o.costs["electricity_kWh"]
        row[f"{sid}:water_m3"] = o.costs["water_m3"]
        row[f"{sid}:relief_per_kWh"] = c["relief_per_kWh"]
        row[f"{sid}:unassessed_ph"] = o.summary["venue"]["unassessed_person_hours"]
    row["status"] = "ok"
    return row


def _work(args):
    i, draw, scenarios = args
    t = time.perf_counter()
    r = evaluate_draw(_W["study"], draw, scenarios)
    r["draw"] = i
    r["wall_s"] = time.perf_counter() - t
    return r


def run_uncertainty(study, outdir, samples=None, seed=None, workers=None, scenarios=None, progress=print):
    spec = study.cfg0.uncertainty
    samples = samples or (spec.samples if spec else 32)
    seed = seed if seed is not None else (spec.seed if spec else 0)
    scenarios = scenarios or (spec.scenarios if spec and spec.scenarios else
                              [s for s in study.scenario_ids() if s != "baseline"])
    params = uncertain_parameters(study, spec.parameters if spec else "all")
    if not params:
        raise ValueError("no quantities with bounds found")
    draws = lhs_draws(params, samples, seed)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    progress(f"uncertainty: {len(params)} parameters, {samples} LHS draws (seed {seed}), "
             f"{len(scenarios)} scenarios + matched baseline per draw, {workers} workers")
    t0 = time.perf_counter()
    jobs = [(i, d, scenarios) for i, d in enumerate(draws)]
    if workers == 1:
        _init(study.raw)
        rows = [_work(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(study.raw,)) as ex:
            rows = list(ex.map(_work, jobs))
    wall = time.perf_counter() - t0
    names = [p for p, _ in params]
    path = outdir / "uncertainty_draws.csv"
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["draw"] + names + [k for k in keys if k != "draw"])
        w.writeheader()
        for r, d in zip(sorted(rows, key=lambda r: r["draw"]), draws):
            w.writerow({**d, **r})
    ok = [r for r in sorted(rows, key=lambda r: r["draw"]) if r.get("status") == "ok"]
    # |rho| below ~2/sqrt(n) is indistinguishable from zero at ~95% for independent inputs
    noise = 2.0 / np.sqrt(max(len(ok) - 1, 1))
    summary = dict(samples=samples, seed=seed, workers=workers, wall_s=wall, parameters=names,
                   scenarios=scenarios, results_csv=str(path), n_ok=len(ok),
                   spearman_noise_level=float(noise), outputs={}, sensitivity={})
    X = np.array([[draws[r["draw"]][p] for p in names] for r in ok])
    for sid in scenarios:
        for m in ("relief_Kph", "attainment", "electricity_kWh", "relief_per_kWh"):
            v = np.array([r.get(f"{sid}:{m}", np.nan) for r in ok], float)
            fin = v[np.isfinite(v)]
            if len(fin):
                summary["outputs"][f"{sid}:{m}"] = dict(
                    mean=float(fin.mean()), p05=float(np.percentile(fin, 5)), p50=float(np.percentile(fin, 50)),
                    p95=float(np.percentile(fin, 95)), n=int(len(fin)),
                    prob_negative=float(np.mean(fin < 0)) if m == "relief_Kph" else None)
        y = np.array([r.get(f"{sid}:relief_Kph", np.nan) for r in ok], float)
        sens = []
        for j, p in enumerate(names):
            if np.ptp(X[:, j]) > 0 and np.all(np.isfinite(y)) and np.ptp(y) > 0:
                rho = spearmanr(X[:, j], y).statistic
                sens.append((p, float(rho)))
        sens.sort(key=lambda t: -abs(t[1]))
        summary["sensitivity"][sid] = [
            dict(parameter=p, spearman_rho=r, unused_by_scenario=unused_by(study, sid, p),
                 above_noise=bool(abs(r) > noise and not unused_by(study, sid, p)))
            for p, r in sens]
    (outdir / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def oat_sweep(study, scenario, outdir, params=None, workers=None):
    """One-at-a-time low/high sweep of each bounded parameter for one scenario's relief."""
    found = uncertain_parameters(study)
    if params:
        found = [f for f in found if f[0] in params]
    jobs = []
    for path, q in found:
        for side, val in (("low", q["bounds"][0]), ("high", q["bounds"][1])):
            jobs.append((len(jobs), {path: float(val)}, [scenario]))
    jobs.append((len(jobs), {}, [scenario]))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(study.raw,)) as ex:
        rows = list(ex.map(_work, jobs))
    rows.sort(key=lambda r: r["draw"])
    nominal = rows[-1].get(f"{scenario}:relief_Kph")
    out = []
    for k, (path, q) in enumerate(found):
        lo, hi = rows[2 * k], rows[2 * k + 1]
        out.append(dict(parameter=path, low_value=q["bounds"][0], high_value=q["bounds"][1], nominal=q["value"],
                        relief_low=lo.get(f"{scenario}:relief_Kph"), relief_high=hi.get(f"{scenario}:relief_Kph"),
                        relief_nominal=nominal))
    out.sort(key=lambda r: -abs((r["relief_high"] or 0) - (r["relief_low"] or 0)))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(out).to_csv(outdir / f"oat_sweep_{scenario}.csv", index=False)
    return out
