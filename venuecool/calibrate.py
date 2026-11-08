"""Check device presets against their quoted (conditional) performance claims.

The supplied claims are not model inputs. This module reports what the configured presets
actually do at reference conditions and inside a simulated venue, so a reader can see whether
the assumed airflow / effectiveness / coverage are consistent with the claims.
"""

from __future__ import annotations

import numpy as np

from . import comfort, psychro
from .devices import evaporative_pad

REFERENCE = [
    dict(name="warm_moderate", tdb=32.0, rh=0.40, p=101325.0),
    dict(name="hot_dry", tdb=38.0, rh=0.20, p=101325.0),
    dict(name="very_hot_dry", tdb=41.0, rh=0.11, p=97000.0),
    dict(name="warm_humid", tdb=33.0, rh=0.60, p=101325.0),
]


def preset_checks(cfg, met=1.1, clo=0.5, v_still=0.2):
    rows = []
    for pid, p in cfg.device_presets.items():
        for ref in REFERENCE:
            t, rh, pa = ref["tdb"], ref["rh"], ref["p"]
            w = psychro.w_from_rh(t, rh, pa)
            row = dict(preset=pid, claim=p.claim, reference=ref["name"], tdb_C=t, rh_pct=rh * 100)
            v = np.sqrt(v_still ** 2 + p.covered_air_speed.value ** 2)
            s0 = comfort._set(np.array([t]), np.array([t]), np.array([v_still]), np.array([rh * 100]), met, clo,
                              np.array([pa]), "sitting", False)[0]
            if p.kind == "fan":
                s1 = comfort._set(np.array([t]), np.array([t]), np.array([v]), np.array([rh * 100]), met, clo,
                                  np.array([pa]), "sitting", False)[0]
                row.update(covered_air_speed=v, set_still=s0, set_with_device=s1, set_reduction_K=s0 - s1,
                           note="air at MRT=tdb, motor heat excluded (local effect only)")
            else:
                t_pad, w_pad, t_wb = evaporative_pad(np.array([t]), np.array([w]), pa, p.saturation_effectiveness.value,
                                                    fast=False)
                rho = psychro.rho_da(t, w, pa)
                water_kg_h = rho * p.airflow.value * (w_pad[0] - w) * 3600 * (1 + p.bleed_fraction.value)
                b = p.plume_fraction.value
                h_loc = psychro.enthalpy(t, w) + b * (psychro.enthalpy(t_pad[0], w_pad[0]) - psychro.enthalpy(t, w))
                w_loc = w + b * (w_pad[0] - w)
                t_loc = psychro.t_from_hw(h_loc, w_loc)
                rh_loc = psychro.rh_from_w(t_loc, w_loc, pa) * 100
                s1 = comfort._set(np.array([t_loc]), np.array([t]), np.array([v]), np.array([rh_loc]), met, clo,
                                  np.array([pa]), "sitting", False)[0]
                row.update(wet_bulb_C=float(t_wb[0]), outlet_C=float(t_pad[0]), outlet_drop_K=float(t - t_pad[0]),
                           wet_bulb_limit_K=float(t - t_wb[0]), plume_local_drop_K=float(t - t_loc),
                           water_L_per_h=float(water_kg_h), set_still=s0, set_with_device=s1, set_reduction_K=s0 - s1,
                           note="plume-local drop excludes whole-section cooling; see simulated table")
            rows.append(row)
    return rows


def simulated_checks(study, outs):
    """For single-device scenarios: max occupied-hour reduction of section-mean and covered-zone air temp."""
    base = outs["baseline"]
    rows = []
    for sid, o in outs.items():
        if sid == "baseline" or not o.ok or len(o.device_ids) != 1 or o.passive:
            continue
        dev = o.model.devs[0]
        for si, frac, groups in dev.serves:
            sec = o.cfg.sections[si].id
            occ = base.sim.mid["people"][:, si] > 0
            on = o.sim.device_on[:, 0] & occ
            if not on.any():
                continue
            dT = (base.sim.mid["T_air"][:, si] - o.sim.mid["T_air"][:, si])[on]
            cb, ci = base.comfort_df, o.comfort_df
            sel = lambda d: d[(d.section == sec) & (d.group == groups[0]) & (d.sun_state == "shade")].set_index("k")  # noqa: E731
            b, i = sel(cb), sel(ci)
            k = np.where(on)[0]
            dloc = (b.loc[k, "tdb"] - i.loc[k, "tdb"]).to_numpy()
            dset = (b.loc[k, "set_extended"] - i.loc[k, "set_extended"]).to_numpy()
            rows.append(dict(scenario=sid, device=dev.id, preset=dev.preset, claim=o.cfg.device_presets[dev.preset].claim,
                             section=sec, covered_group=groups[0],
                             max_section_mean_air_drop_K=float(np.max(dT)), mean_section_mean_air_drop_K=float(np.mean(dT)),
                             max_covered_zone_air_drop_K=float(np.max(dloc)),
                             max_covered_zone_SET_drop_K=float(np.max(dset)), min_covered_zone_SET_drop_K=float(np.min(dset))))
    return rows
