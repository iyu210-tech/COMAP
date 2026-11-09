"""Scenario execution, matched baseline comparisons and exports."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import comfort, metrics
from .config import (VenueConfig, apply_passive, expand_templates, load_raw, set_path, VenueConfig as _VC)
from .inputs import load_occupancy, load_weather, read_meta
from .model import Model


@dataclass
class Outcome:
    label: str
    device_ids: List[str]
    passive: List[str]
    cfg: VenueConfig
    model: Model
    sim: object
    comfort_df: Optional[pd.DataFrame]
    summary: Optional[dict]
    costs: Optional[dict]

    @property
    def ok(self):
        return bool(self.sim.ok)


class Study:
    """Holds an expanded raw configuration and the shared weather/occupancy inputs."""

    def __init__(self, config, weather_file: Optional[str] = None, occupancy_file: Optional[str] = None,
                 sim_overrides: Optional[dict] = None):
        raw = load_raw(config) if isinstance(config, (str, Path)) else copy.deepcopy(config)
        if weather_file:
            raw["weather"]["file"] = str(Path(weather_file).resolve())
        if occupancy_file:
            raw["occupancy"]["file"] = str(Path(occupancy_file).resolve())
        for k, v in (sim_overrides or {}).items():
            raw.setdefault("simulation", {})[k] = v
        raw = expand_templates(raw)
        wpath = Path(raw["weather"]["file"])
        if not wpath.is_absolute():
            wpath = Path(raw.get("base_dir", ".")) / wpath
        self.site_source = "configuration"
        if wpath.exists():
            meta = read_meta(wpath)
            keys = ("latitude_deg", "longitude_deg", "utc_offset_h", "day_of_year")
            if all(k in meta for k in keys):
                raw["site"] = {k: (int(float(meta[k])) if k == "day_of_year" else float(meta[k])) for k in keys}
                self.site_source = f"weather file metadata ({wpath.name})"
        self.raw = raw
        self.cfg0 = _VC.model_validate(self.raw)
        self.weather = load_weather(self.cfg0.path(self.cfg0.weather.file), self.cfg0.site, self.cfg0.weather.solar_split)
        cols = sorted({s.occupancy.column or s.id for s in self.cfg0.sections})
        self.occupancy = load_occupancy(self.cfg0.path(self.cfg0.occupancy.file), cols)

    # ------------------------------------------------------------------
    def raw_for(self, passive=(), draw: Optional[Dict[str, float]] = None, extra_devices=None, sim_overrides=None):
        r = copy.deepcopy(self.raw)
        for path, val in (draw or {}).items():
            set_path(r, path, val)
        opts = {p["id"]: p for p in r.get("passive_options", [])}
        for pid in passive:
            apply_passive(r, opts[pid])
        if extra_devices:
            r["devices"] = list(r.get("devices", [])) + copy.deepcopy(extra_devices)
        for k, v in (sim_overrides or {}).items():
            r["simulation"][k] = v
        return r

    def run(self, device_ids=(), passive=(), draw=None, extra_devices=None, label="custom",
            sim_overrides=None, evaluate=True) -> Outcome:
        cfg = _VC.model_validate(self.raw_for(passive, draw, extra_devices, sim_overrides))
        model = Model(cfg, self.weather, self.occupancy, list(device_ids))
        sim = model.simulate()
        if not sim.ok or not evaluate:
            return Outcome(label, list(device_ids), list(passive), cfg, model, sim, None, None, None)
        df = comfort.evaluate(cfg, model, sim)
        summ = metrics.summarize(df, cfg)
        window_days = (cfg.simulation.end_h - cfg.simulation.start_h) / 24.0
        popts = [p for p in cfg.passive_options if p.id in passive]
        costs = metrics.ledger_totals(sim.ledgers, popts, cfg, window_days)
        return Outcome(label, list(device_ids), list(passive), cfg, model, sim, df, summ, costs)

    def scenario_ids(self):
        return [s.id for s in self.cfg0.scenarios]

    def run_scenario(self, sid, draw=None, sim_overrides=None) -> Outcome:
        if sid == "baseline" and "baseline" not in self.scenario_ids():
            return self.run([], [], draw, label="baseline", sim_overrides=sim_overrides)
        sc = next(s for s in self.cfg0.scenarios if s.id == sid)
        return self.run(sc.devices, sc.passive, draw, label=sid, sim_overrides=sim_overrides)

    def compare_all(self, scenarios=None, draw=None, sim_overrides=None):
        ids = scenarios or [s for s in self.scenario_ids() if s != "baseline"]
        base = self.run_scenario("baseline", draw, sim_overrides)
        outs = {"baseline": base}
        comps = {}
        for sid in ids:
            o = self.run_scenario(sid, draw, sim_overrides)
            outs[sid] = o
            if base.ok and o.ok:
                comps[sid] = metrics.compare(base.summary, o.summary, base.costs, o.costs)
        return outs, comps


# ---------------------------------------------------------------------- reporting
def provenance_table(raw) -> pd.DataFrame:
    rows = []

    def walk(node, path):
        if isinstance(node, dict):
            if {"value", "unit", "source"} <= set(node):
                rows.append(dict(path=path, value=node["value"], unit=node["unit"], source=node["source"],
                                 bounds=json.dumps(node.get("bounds"))))
                return
            for k, v in node.items():
                if k != "templates":
                    walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for x in node:
                if isinstance(x, dict) and "id" in x:
                    walk(x, f"{path}[{x['id']}]")
    walk(raw, "")
    return pd.DataFrame(rows)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def cumulative_electricity_kWh(o: Outcome) -> np.ndarray:
    """Cumulative electricity [kWh] at each reporting time, using exact segment on-intervals.

    A device that switches between two reporting samples contributes only its real on-time.
    The value at the last reporting time matches the device ledger.
    """
    s = o.sim
    cum = np.zeros(len(s.t))
    if not o.model.devs:
        return cum
    pieces = []
    t0, t1 = o.model.t0, o.model.t1
    for a, b, _N, on in o.model.segments:
        if b <= t0 or a >= t1 or not np.any(on):
            continue
        kw = sum(dev.power_W / 1000.0 for d, dev in enumerate(o.model.devs) if on[d])
        pieces.append((max(a, t0), min(b, t1), kw))
    for i, ti in enumerate(s.t):
        e = 0.0
        for lo, hi, kw in pieces:
            overlap = min(hi, float(ti)) - lo
            if overlap > 0:
                e += kw * overlap / 3600.0
        cum[i] = e
    return cum


def timeseries_frame(o: Outcome) -> pd.DataFrame:
    s = o.sim
    cum_kwh = cumulative_electricity_kWh(o)
    rows = []
    for i, sid in enumerate(s.section_ids):
        rows.append(pd.DataFrame(dict(scenario=o.label, t_h=s.t / 3600, section=sid, T_air_C=s.T_air[:, i],
                                      RH_pct=s.RH[:, i] * 100, W_kg_kg=s.W[:, i], T_out_C=s.drivers["To"],
                                      RH_out_pct=None, wind_ms=s.drivers["U"], sun_fraction=s.drivers["fsun"][:, i],
                                      floor_surface_C=s.T_nodes[:, o.model.floor_node[i]],
                                      roof_underside_C=(s.T_nodes[:, o.model.roof_node[i]] if i in o.model.roof_node else np.nan),
                                      electricity_kWh_cum=cum_kwh,
                                      invalid_state=s.invalid_mask[:, i])))
    df = pd.concat(rows, ignore_index=True)
    from .psychro import rh_from_w
    df["RH_out_pct"] = np.tile(rh_from_w(s.drivers["To"], s.drivers["Wo"], s.drivers["p"]) * 100, len(s.section_ids))
    return df


def comparison_rows(outs, comps) -> pd.DataFrame:
    rows = []
    for sid, o in outs.items():
        if not o.ok:
            rows.append(dict(scenario=sid, status="SOLVER_FAILED: " + o.sim.message))
            continue
        v = o.summary["venue"]
        c = comps.get(sid, {})
        rows.append(dict(
            scenario=sid, status="ok", devices=";".join(o.device_ids), passive=";".join(o.passive),
            occupied_person_hours=v["occupied_person_hours"], assessed_person_hours=v["assessed_person_hours"],
            unassessed_person_hours=v["unassessed_person_hours"], extrapolated_person_hours=v["extrapolated_person_hours"],
            served_person_hours=v["served_person_hours"], burden_Kph=v["burden_Kph"], attainment=v["attainment"],
            relief_Kph=c.get("relief_Kph", 0.0 if sid == "baseline" else np.nan),
            residual_matched_Kph=c.get("residual_burden_matched_Kph", np.nan),
            electricity_kWh=o.costs["electricity_kWh"], water_m3=o.costs["water_m3"],
            investment_usd=o.costs["investment_usd"], season_total_cost_usd=o.costs["season_total_cost_usd"],
            relief_per_kWh=c.get("relief_per_kWh", np.nan), relief_per_usd=c.get("relief_per_usd", np.nan),
            poorly_served=";".join(o.summary["poorly_served"]),
            worsened_sections=";".join(c.get("worsened_sections", [])),
            energy_residual_rel=o.sim.conservation["energy_residual_rel"],
            water_residual_rel=o.sim.conservation["water_residual_rel"],
            invalid_states=sum(o.sim.invalid.values()), runtime_s=o.sim.runtime_s))
    return pd.DataFrame(rows)


def export(outs, comps, study: Study, outdir) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = study.cfg0
    comparison_rows(outs, comps).to_csv(outdir / "comparison.csv", index=False)
    ts, cf, led, brk = [], [], [], []
    summary = dict(venue=cfg.meta.name, data_status=cfg.meta.data_status, description=cfg.meta.description,
                   notes=cfg.meta.notes, weather_file=str(cfg.path(cfg.weather.file)),
                   weather_metadata=study.weather.meta, occupancy_metadata=study.occupancy.meta,
                   comfort_index=cfg.comfort.index, target_C=cfg.comfort.target.value,
                   decision_basis=cfg.comfort.decision_basis,
                   horizon=dict(simulated_hours=cfg.simulation.end_h - cfg.simulation.start_h,
                                season_days=cfg.tariffs.season_days.value, note=cfg.tariffs.horizon_note),
                   disclaimer=("Modeled comfort burden (SET-based) for comparing configurations; not a measured "
                               "satisfaction rate and not a medical heat-safety guarantee. Person-hours are not visitors."),
                   scenarios={})
    for sid, o in outs.items():
        if not o.ok:
            summary["scenarios"][sid] = dict(status="SOLVER_FAILED", message=o.sim.message)
            continue
        ts.append(timeseries_frame(o))
        cf.append(o.comfort_df.assign(scenario=sid))
        led += [dict(scenario=sid, **x) for x in o.sim.ledgers]
        for sec, d in o.sim.breakdown_J.items():
            brk.append(dict(scenario=sid, section=sec, **{k + "_kWh": v / 3.6e6 for k, v in d.items()}))
        summary["scenarios"][sid] = dict(
            devices=o.device_ids, passive=o.passive, metrics=o.summary, costs=o.costs, ledgers=o.sim.ledgers,
            conservation=o.sim.conservation, invalid_states=o.sim.invalid, runtime_s=o.sim.runtime_s,
            rhs_evaluations=o.sim.n_rhs, comparison=comps.get(sid))
    pd.concat(ts).to_csv(outdir / "timeseries.csv", index=False)
    pd.concat(cf).to_csv(outdir / "comfort_populations.csv", index=False)
    pd.DataFrame(led).to_csv(outdir / "device_ledgers.csv", index=False)
    pd.DataFrame(brk).to_csv(outdir / "enthalpy_flow_breakdown.csv", index=False)
    provenance_table(study.raw).to_csv(outdir / "provenance.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(_clean(summary), indent=2))
    return outdir
