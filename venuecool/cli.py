"""Command-line interface for batch simulations.

    venuecool validate   CONFIG
    venuecool compare    CONFIG [--weather CSV] [--scenarios a,b] [--out DIR]
    venuecool search     CONFIG [--workers N] [--limit N] [--out DIR]
    venuecool uncertainty CONFIG [--samples N] [--seed S] [--workers N] [--out DIR]
    venuecool sweep      CONFIG --scenario ID [--workers N]
    venuecool benchmark  CONFIG [--runs N] [--workers 1,4,11]
    venuecool calibrate  CONFIG [--out DIR]
    venuecool refine     CONFIG [--scenario ID]
    venuecool report     CONFIG [--weather CSV] [--scenarios a,b] [--out DIR]
    venuecool robust     CONFIG [--weather CSV] [--out DIR]
    venuecool examples   [--out DIR]          # every example x both synthetic climates
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
CLIMATES = {"hot_dry": EXAMPLES / "data" / "weather_hot_dry.csv",
            "warm_humid": EXAMPLES / "data" / "weather_warm_humid.csv"}
EXAMPLE_CONFIGS = ["bus_stop.yaml", "rail_station.yaml", "venue3.yaml"]


def _study(args):
    from .runner import Study
    return Study(args.config, weather_file=getattr(args, "weather", None))


def _default_out(args, kind):
    if getattr(args, "out", None):
        return Path(args.out)
    wfile = getattr(args, "weather", None)
    if not wfile:
        from .config import load_raw
        wfile = load_raw(args.config)["weather"]["file"]
    w = Path(wfile).stem.replace("weather_", "")
    return ROOT / "outputs" / Path(args.config).stem / w / kind


def cmd_validate(args):
    from .runner import Study
    s = Study(args.config, weather_file=args.weather)
    c = s.cfg0
    print(f"OK  {c.meta.name}  [{c.meta.data_status}]")
    print(f"  sections: {[x.id for x in c.sections]}  connections: {[x.id for x in c.connections]}")
    print(f"  devices: {[d.id for d in c.devices]}  scenarios: {[x.id for x in c.scenarios]}")
    print(f"  weather: {c.weather.file} status={s.weather.meta.get('data_status')}  site from {s.site_source}")
    from .config import find_uncertain
    print(f"  uncertain parameters (with bounds): {len(find_uncertain(s.raw))}")


def cmd_compare(args):
    from .runner import comparison_rows, export
    s = _study(args)
    scen = args.scenarios.split(",") if args.scenarios else None
    t = time.perf_counter()
    outs, comps = s.compare_all(scen)
    out = export(outs, comps, s, _default_out(args, "compare"))
    df = comparison_rows(outs, comps)
    cols = ["scenario", "attainment", "burden_Kph", "relief_Kph", "electricity_kWh", "water_m3",
            "season_total_cost_usd", "relief_per_kWh", "relief_per_usd", "poorly_served", "worsened_sections"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[[c for c in cols if c in df]].round(3).to_string(index=False))
    print(f"[{s.cfg0.meta.data_status}] wrote {out}  ({time.perf_counter() - t:.1f} s)")


def cmd_report(args):
    """Run the four Problem B scenarios and write Tair / RH / SET / cumulative-kWh graphs."""
    from .plots import write_report
    from .runner import comparison_rows, export
    s = _study(args)
    scen = args.scenarios.split(",") if args.scenarios else ["A_fan", "B_evap", "C_large_evap"]
    t = time.perf_counter()
    outs, comps = s.compare_all(scen)
    report_override = args.out
    args.out = None  # CSVs stay under the standard compare/ folder
    compare_dir = _default_out(args, "compare")
    export(outs, comps, s, compare_dir)
    df = comparison_rows(outs, comps)
    args.out = report_override
    report_dir = Path(report_override) if report_override else _default_out(args, "report")
    # reuse exported frames so graphs match the CSV archive
    ts = pd.read_csv(compare_dir / "timeseries.csv")
    cf = pd.read_csv(compare_dir / "comfort_populations.csv")
    paths = write_report(report_dir, ts, cf, df)
    cols = ["scenario", "attainment", "burden_Kph", "relief_Kph", "electricity_kWh", "water_m3"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[[c for c in cols if c in df]].round(3).to_string(index=False))
    print(f"[{s.cfg0.meta.data_status}] compare -> {compare_dir}")
    print(f"report graphs -> {report_dir}")
    for name, p in paths.items():
        print(f"  {name}: {p}")
    print(f"({time.perf_counter() - t:.1f} s)")


def cmd_robust(args):
    """Re-simulate the single-zone shelter under airflow, solar and coverage assumptions."""
    from .robustness import run_robustness
    s = _study(args)
    out = _default_out(args, "robustness")
    df = run_robustness(s, out)
    winners = df[df["preferred"]][["case", "target_C", "preferred_scenario", "relief_Kph",
                                   "supported_person_hours", "extrapolated_person_hours"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(winners.round(3).to_string(index=False))
    print(f"preferred system changes across the table: {df.loc[df.preferred, 'preferred_scenario'].nunique() > 1}")
    print(f"wrote {out}")


def cmd_search(args):
    from .search import run_search
    s = _study(args)
    r = run_search(s, _default_out(args, "search"), workers=args.workers, limit=args.limit)
    print(json.dumps({k: r[k] for k in ("n_candidates", "workers", "wall_s", "mean_candidate_s", "status_counts")}, indent=1))
    if r["selected"] is None:
        print(r["message"])
    else:
        sel = r["selected"]
        print(f"SELECTED: {sel['label']}  kWh/day={sel['electricity_kWh']:.2f}  season cost=${sel['season_total_cost_usd']:.0f}"
              f"  attainment={sel['attainment']:.3f}")
        print("verification:", r.get("verification"))


def cmd_uncertainty(args):
    from .uncertainty import run_uncertainty
    s = _study(args)
    r = run_uncertainty(s, _default_out(args, "uncertainty"), samples=args.samples, seed=args.seed, workers=args.workers)
    print(f"{r['samples']} draws in {r['wall_s']:.1f} s with {r['workers']} workers")
    for k, v in r["outputs"].items():
        if k.endswith("relief_Kph"):
            print(f"  {k}: mean {v['mean']:.1f}  5-95% [{v['p05']:.1f}, {v['p95']:.1f}]  P(relief<0)={v['prob_negative']:.2f}")
    print(f"  rank correlations with |rho| <= {r['spearman_noise_level']:.2f} are within sampling noise")
    for sid, lst in r["sensitivity"].items():
        sig = [d for d in lst if d["above_noise"]][:4]
        print(f"  drivers of relief for {sid}: " +
              (", ".join(f"{d['parameter']} ({d['spearman_rho']:+.2f})" for d in sig) or "none above noise"))


def cmd_sweep(args):
    from .uncertainty import oat_sweep
    s = _study(args)
    rows = oat_sweep(s, args.scenario, _default_out(args, "sweep"), workers=args.workers)
    for r in rows[:12]:
        print(f"{r['parameter']:<75s} low {r['relief_low']:.1f}  high {r['relief_high']:.1f}  (nominal {r['relief_nominal']:.1f})")


def cmd_benchmark(args):
    from .runner import Study
    from .search import enumerate_configs
    s = _study(args)
    ids = [x for x in s.scenario_ids() if x != "baseline"][:1] or ["baseline"]
    ts = []
    for _ in range(args.runs):
        t = time.perf_counter()
        o = s.run_scenario(ids[0])
        ts.append((time.perf_counter() - t, o.sim.runtime_s))
    per = sum(x[0] for x in ts) / len(ts)
    ode = sum(x[1] for x in ts) / len(ts)
    print(f"serial: {per:.2f} s per scenario run (ODE {ode:.2f} s, rest = comfort/metrics), {args.runs} runs")
    res = {"serial_s_per_run": per, "ode_s_per_run": ode, "parallel": {}}
    if s.cfg0.search:
        n = len(enumerate_configs(s.cfg0))
        res["search_candidates"] = n
    for w in [int(x) for x in args.workers.split(",")]:
        if w <= 1:
            continue
        from concurrent.futures import ProcessPoolExecutor
        from . import uncertainty as U
        jobs = [(i, {}, ids) for i in range(w * 2)]
        t = time.perf_counter()
        with ProcessPoolExecutor(max_workers=w, initializer=U._init, initargs=(s.raw,)) as ex:
            list(ex.map(U._work, jobs))
        wall = time.perf_counter() - t
        thr = len(jobs) * 2 / wall  # each job = baseline + one scenario
        res["parallel"][w] = dict(wall_s=wall, runs=len(jobs) * 2, runs_per_s=thr)
        print(f"{w} workers: {len(jobs) * 2} runs in {wall:.1f} s (incl. worker start-up) -> {thr:.2f} runs/s")
    if "search_candidates" in res:
        best = max([v["runs_per_s"] for v in res["parallel"].values()] + [1 / per])
        print(f"estimated full search ({res['search_candidates']} candidates): {res['search_candidates'] / best / 60:.1f} min "
              f"at the best measured throughput")
    out = _default_out(args, "benchmark")
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(res, indent=2))


def cmd_calibrate(args):
    from .calibrate import preset_checks, simulated_checks
    s = _study(args)
    out = _default_out(args, "calibrate")
    out.mkdir(parents=True, exist_ok=True)
    pc = pd.DataFrame(preset_checks(s.cfg0))
    pc.to_csv(out / "preset_reference_checks.csv", index=False)
    outs, _ = s.compare_all()
    sc = pd.DataFrame(simulated_checks(s, outs))
    sc.to_csv(out / "simulated_claim_checks.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(pc.drop(columns=["claim", "note"]).round(2).to_string(index=False))
        print(sc.drop(columns=["claim"]).round(2).to_string(index=False))
    print(f"wrote {out}")


def cmd_refine(args):
    """Numerical refinement: tolerance and output-step halving must not change the metrics materially."""
    from .runner import Study
    from . import metrics
    rows = []
    for rtol, step in [(1e-4, 600.0), (1e-6, 300.0), (1e-8, 150.0), (1e-9, 60.0)]:
        s = Study(args.config, weather_file=args.weather, sim_overrides=dict(rtol=rtol, output_step_s=step))
        b = s.run_scenario("baseline")
        o = s.run_scenario(args.scenario) if args.scenario else b
        c = metrics.compare(b.summary, o.summary, b.costs, o.costs)
        rows.append(dict(rtol=rtol, output_step_s=step, baseline_burden=b.summary["venue"]["burden_Kph"],
                         relief=c["relief_Kph"], attainment=o.summary["venue"]["attainment"],
                         peak_T_air=float(o.sim.T_air.max()), energy_residual_rel=o.sim.conservation["energy_residual_rel"],
                         ode_s=o.sim.runtime_s))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = _default_out(args, "refine")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "refinement.csv", index=False)


def cmd_examples(args):
    from .runner import Study, comparison_rows, export
    root = Path(args.out) if args.out else ROOT / "outputs"
    allrows = []
    for cfgname in EXAMPLE_CONFIGS:
        for clim, wf in CLIMATES.items():
            t = time.perf_counter()
            s = Study(EXAMPLES / cfgname, weather_file=str(wf))
            outs, comps = s.compare_all()
            od = export(outs, comps, s, root / Path(cfgname).stem / clim / "compare")
            df = comparison_rows(outs, comps).assign(example=Path(cfgname).stem, climate=clim)
            allrows.append(df)
            print(f"{cfgname:<20s} {clim:<11s} {len(outs)} runs in {time.perf_counter() - t:5.1f} s -> {od}")
    big = pd.concat(allrows, ignore_index=True)
    big.to_csv(root / "examples_overview.csv", index=False)
    cols = ["example", "climate", "scenario", "attainment", "relief_Kph", "electricity_kWh", "water_m3",
            "season_total_cost_usd", "relief_per_kWh", "poorly_served"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(big[cols].round(3).to_string(index=False))


def main(argv=None):
    p = argparse.ArgumentParser(prog="venuecool", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, config=True):
        sp = sub.add_parser(name)
        if config:
            sp.add_argument("config")
            sp.add_argument("--weather", default=None, help="override weather CSV")
        sp.add_argument("--out", default=None)
        sp.set_defaults(fn=fn)
        return sp
    add("validate", cmd_validate)
    sp = add("compare", cmd_compare)
    sp.add_argument("--scenarios", default=None)
    sp = add("report", cmd_report)
    sp.add_argument("--scenarios", default=None,
                    help="comma-separated interventions (baseline always included); "
                         "default A_fan,B_evap,C_large_evap")
    add("robust", cmd_robust)
    sp = add("search", cmd_search)
    sp.add_argument("--workers", type=int, default=None)
    sp.add_argument("--limit", type=int, default=None)
    sp = add("uncertainty", cmd_uncertainty)
    sp.add_argument("--samples", type=int, default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--workers", type=int, default=None)
    sp = add("sweep", cmd_sweep)
    sp.add_argument("--scenario", required=True)
    sp.add_argument("--workers", type=int, default=None)
    sp = add("benchmark", cmd_benchmark)
    sp.add_argument("--runs", type=int, default=3)
    sp.add_argument("--workers", default=f"1,{max(1, (os.cpu_count() or 2) - 1)}")
    add("calibrate", cmd_calibrate)
    sp = add("refine", cmd_refine)
    sp.add_argument("--scenario", default=None)
    add("examples", cmd_examples, config=False)
    args = p.parse_args(argv)
    try:
        args.fn(args)
    except Exception as e:  # noqa: BLE001
        from .inputs import InputDataError
        if isinstance(e, (InputDataError, ValueError, KeyError)):
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
