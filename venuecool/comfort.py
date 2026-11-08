"""Local human response: mean radiant temperature, local air, SET via pythermalcomfort.

Populations: each section is split into exposure groups (device coverage) and, within each
group, into sunlit and shaded people. The sunlit share equals the section's sunlit floor
fraction f_sun(t) -- the *same* number used for floor solar absorption, so sunlight is treated
consistently for surfaces and occupants.

MRT (ASHRAE 55-2023 Appendix C 'SolarCal' structure):
  T_lw^4 = sum_k F_k T_k^4  (floor, roof underside, walls, sky, surroundings at outdoor air temp)
  dMRT_dir = SolarCal direct-beam term (pythermalcomfort.solar_gain, f_svv=0, f_bes=1),
             averaged over body orientations, scaled linearly with DNI (the term is linear)
  dMRT_dif = 0.5 f_eff f_svv (DHI + I_floor R_floor) (a_sw/a_lw) / (f_eff h_r)
  MRT = T_lw + f_bes * dMRT_dir + dMRT_dif
with I_floor and R_floor = 1 - floor absorptance taken from the floor surface model.
Prescribed mode: T_lw = T_air + prescribed offset (solar optional).

Local air in a covered group: h_loc = h_sec + sum_d w_d (h_out,d - h_sec) (same for W), with
total weight 1 - prod(1 - plume_d) split in proportion to plume_d (bounded by 1, no double
counting). Local speed v = sqrt(v_base^2 + sum_d v_d^2).

Classification of every population-interval:
  supported        SET finite with pythermalcomfort limit_inputs=True (ASHRAE 55 ranges)
  extrapolated     outside those ranges but inside the declared extended envelope (flagged)
  unassessed       invalid air state or outside the extended envelope
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

from . import psychro
from .constants import T0K

F_EFF = {"standing": 0.725, "sitting": 0.696}
H_R = 6.012
A_LW = 0.95


@lru_cache(maxsize=32)
def _direct_table(posture: str, asw: float, sharps: tuple):
    from pythermalcomfort.models import solar_gain
    alts = np.linspace(0.5, 89.5, 90)
    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in sharps:
            r = solar_gain(sol_altitude=alts, sharp=np.full_like(alts, s), sol_radiation_dir=np.full_like(alts, 1000.0),
                           sol_transmittance=np.ones_like(alts), f_svv=np.zeros_like(alts), f_bes=np.ones_like(alts),
                           asw=asw, posture=posture, round_output=False)
            vals.append(np.asarray(r.delta_mrt, float) / 1000.0)
    return alts, np.mean(vals, axis=0)


def direct_dmrt(alt_deg, dni, posture, asw, sharps):
    alts, per = _direct_table(posture, float(asw), tuple(float(s) for s in sharps))
    alt = np.asarray(alt_deg, float)
    return np.where(alt > 0, np.interp(alt, alts, per) * np.asarray(dni, float), 0.0)


def diffuse_dmrt(f_svv, dhi, i_floor, r_floor, posture, asw):
    fe = F_EFF[posture]
    erf = 0.5 * fe * f_svv * (np.asarray(dhi) + np.asarray(i_floor) * r_floor) * asw / A_LW
    return erf / (fe * H_R)


def _set(tdb, tr, v, rh_pct, met, clo, p, posture, strict):
    from pythermalcomfort.models import set_tmp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = set_tmp(tdb=tdb, tr=tr, v=v, rh=rh_pct, met=met, clo=clo, p_atm=p, position=posture,
                        limit_inputs=strict, round_output=False)
            return np.asarray(r.set, float)
        except Exception:  # noqa: BLE001  pythermalcomfort raises on some invalid inputs
            out = np.full(np.shape(tdb), np.nan)
            for i in range(len(out)):
                try:
                    out[i] = float(set_tmp(tdb=tdb[i], tr=tr[i], v=v[i], rh=rh_pct[i], met=met, clo=clo, p_atm=p[i],
                                           position=posture, limit_inputs=strict, round_output=False).set)
                except Exception:  # noqa: BLE001
                    out[i] = np.nan
            return out


def evaluate(cfg, model, res) -> pd.DataFrame:
    """Long-form DataFrame of population-interval comfort results."""
    mid = res.mid
    nk = len(mid["t"])
    comfort = cfg.comfort
    lim = comfort.extended_limits
    asw = comfort.solar_absorptance_body.value
    rows = []
    for i, sec in enumerate(cfg.sections):
        T = mid["T_air"][:, i]
        hsec, Wsec = mid["h"][:, i], mid["W"][:, i]
        p = mid["p"]
        Tn = mid["T_nodes"]
        ov = sec.radiation.occupant_view
        fl = model.floor_node[i]
        Tf = Tn[:, fl]
        Tr = Tn[:, model.roof_node[i]] if i in model.roof_node else Tf
        wn = model.wall_nodes.get(i, [])
        if wn:
            A = np.array([model.node_info[j]["A"] for j in wn])
            Tw4 = ((Tn[:, wn] + T0K) ** 4 * A).sum(axis=1) / A.sum()
        else:
            Tw4 = (T + T0K) ** 4
        f_sur = 1 - ov.floor - ov.roof - ov.walls - ov.sky
        if sec.radiation.mode == "computed":
            T_lw = (ov.floor * (Tf + T0K) ** 4 + ov.roof * (Tr + T0K) ** 4 + ov.walls * Tw4
                    + ov.sky * (mid["Tsky"] + T0K) ** 4 + f_sur * (mid["To"] + T0K) ** 4) ** 0.25 - T0K
            add_solar = True
        else:
            T_lw = T + sec.radiation.prescribed_mrt_minus_air.value
            add_solar = sec.radiation.prescribed_add_solar
        posture = sec.occupancy.posture
        floor_su = next(s for s in sec.surfaces if s.kind == "floor")
        r_floor = 1 - floor_su.solar_absorptance_interior.value
        fsun = mid["fsun"][:, i]
        sin_alt = np.maximum(np.sin(np.radians(mid["alt"])), 0)
        i_floor = fsun * mid["dni"] * sin_alt + sec.radiation.floor_view.sky * mid["dhi"]
        d_dir = direct_dmrt(mid["alt"], mid["dni"], posture, asw, comfort.occupant_sharp_deg) if add_solar else 0 * T
        d_dif = diffuse_dmrt(ov.sky, mid["dhi"], i_floor, r_floor, posture, asw) if add_solar else 0 * T
        v_base = np.maximum(cfg.physics.still_air_speed.value, sec.outdoor.local_wind_factor.value * mid["U"])
        people = mid["people"][:, i]
        met, clo = sec.occupancy.met.value, sec.occupancy.clo.value
        for g in sec.exposure_groups:
            covering = [d for d, dev in enumerate(model.devs) if any(s == i and g.id in grp for s, _, grp in dev.serves)]
            h_loc, W_loc, v2 = hsec.copy(), Wsec.copy(), v_base ** 2
            served = np.zeros(nk, bool)
            if covering:
                on = res.device_on[:, covering]
                beta = np.array([model.devs[d].plume for d in covering])[None, :] * on
                tot = 1 - np.prod(1 - beta, axis=1)
                ssum = beta.sum(axis=1)
                w = np.where(ssum[:, None] > 0, beta * (tot / np.where(ssum > 0, ssum, 1))[:, None], 0)
                h_loc = hsec + (w * (res.device_outlet_h[:, covering] - hsec[:, None])).sum(axis=1)
                W_loc = Wsec + (w * (res.device_outlet_W[:, covering] - Wsec[:, None])).sum(axis=1)
                vc = np.array([model.devs[d].covered_air_speed for d in covering])[None, :] * on
                v2 = v2 + (vc ** 2).sum(axis=1)
                served = on.any(axis=1)
            T_loc = psychro.t_from_hw(h_loc, W_loc)
            rh_loc = psychro.rh_from_w(T_loc, W_loc, p) * 100
            v_loc = np.sqrt(v2)
            for state, share in (("sun", fsun), ("shade", 1 - fsun)):
                tr = T_lw + d_dif + (d_dir if state == "sun" else 0)
                strict = _set(T_loc, tr, v_loc, rh_loc, met, clo, p, posture, True)
                ext = _set(T_loc, tr, v_loc, rh_loc, met, clo, p, posture, False)
                in_ext = ((T_loc >= lim.tdb[0]) & (T_loc <= lim.tdb[1]) & (tr >= lim.tr[0]) & (tr <= lim.tr[1])
                          & (v_loc >= lim.v[0]) & (v_loc <= lim.v[1]) & (rh_loc >= lim.rh[0]) & (rh_loc <= lim.rh[1]))
                invalid = mid["invalid"][:, i] | ~np.isfinite(T_loc) | (rh_loc > 100.1)
                cls = np.where(invalid, "unassessed_invalid_state",
                               np.where(np.isfinite(strict), "supported",
                                        np.where(np.isfinite(ext) & in_ext, "extrapolated", "unassessed_out_of_range")))
                n_people = people * g.occupant_share * share
                rows.append(pd.DataFrame(dict(
                    k=np.arange(nk), t_h=mid["t"] / 3600.0, dt_h=mid["dt"] / 3600.0, section=sec.id, group=g.id,
                    sun_state=state, people=n_people, person_hours=n_people * mid["dt"] / 3600.0,
                    tdb=T_loc, rh_pct=rh_loc, v=v_loc, mrt=tr, set_strict=strict, set_extended=ext,
                    cls=cls, served=served, met=met, clo=clo)))
    df = pd.concat(rows, ignore_index=True)
    basis = cfg.comfort.decision_basis
    ok = (df["cls"] == "supported") | ((basis == "extended") & (df["cls"] == "extrapolated"))
    df["et"] = np.where(ok, np.where(df["cls"] == "supported", df["set_strict"], df["set_extended"]), np.nan)
    df["assessed"] = ok
    return df
