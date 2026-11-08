"""Transient coupled section model.

State vector y (SI):
  h[i]     moist-air enthalpy of section i air            [J/kg_da]
  W[i]     humidity ratio of section i air                [kg_w/kg_da]
  T[n]     surface node temperatures                      [degC]
  accumulators (do not feed back): boundary energy [J], boundary water [kg], gross |energy| [J],
  gross |water| [kg], per-device evaporated water [kg], per-section enthalpy-flow breakdown [J].

Air balances (dry-air mass M_i constant, all exchanges balanced in dry-air mass):
  M_i dh_i/dt = sum_src m_src (h_src - h_i) + Q_conv,i + Q_occ,i
  M_i dW_i/dt = sum_src m_src (W_src - W_i) + G_occ,i
Sources: outdoor exchange, inter-section exchange (equal and opposite), device delivery.
Surfaces: lumped RC nodes with solar absorption, convection, conduction, ground coupling and
grey-body longwave exchange (pairwise floor-roof / floor-wall with reciprocity, plus sky and
surroundings).
"""

from __future__ import annotations

import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

from . import psychro, solar
from .airflow import make_airflow
from .config import Schedule, VenueConfig
from .constants import MET_W_M2, SIGMA, T0K
from .devices import outlet_state
from .inputs import OccupancySchedule, Weather

DAY = 86400.0
DT_FINE = 60.0
N_BRK = 5
BREAKDOWN_NAMES = ["outdoor_exchange", "internal_exchange", "surface_convection", "occupants", "devices"]


# --------------------------------------------------------------------------- devices
@dataclass
class ResolvedDevice:
    id: str
    preset: str
    kind: str
    label: str
    power_W: float
    airflow_m3s: float
    covered_air_speed: float
    eff: float
    bleed: float
    plume: float
    motor_heat: float
    capital_usd: float
    lifetime_y: float
    maintenance_usd_y: float
    intake: str
    serves: list  # [(section_index, fraction, [group ids])]
    schedule: Schedule
    schedule_name: str


def resolve_devices(cfg: VenueConfig, device_ids: List[str]) -> List[ResolvedDevice]:
    by_id = {d.id: d for d in cfg.devices}
    sidx = {s.id: i for i, s in enumerate(cfg.sections)}
    out = []
    for did in device_ids:
        d = by_id[did]
        p = cfg.device_presets[d.preset].model_copy()
        vals = {k: getattr(p, k) for k in type(p).model_fields}
        for k, q in d.overrides.items():
            vals[k] = q
        g = lambda k: (vals[k].value if vals[k] is not None else 0.0)  # noqa: E731
        serves = []
        for s in d.serves:
            groups = s.groups or [x.id for x in cfg.section(s.section).exposure_groups]
            serves.append((sidx[s.section], s.airflow_fraction, groups))
        out.append(ResolvedDevice(
            id=d.id, preset=d.preset, kind=p.kind, label=p.label, power_W=g("electric_power"),
            airflow_m3s=g("airflow"), covered_air_speed=g("covered_air_speed"),
            eff=g("saturation_effectiveness") if p.kind == "evaporative" else 0.0,
            bleed=g("bleed_fraction"), plume=g("plume_fraction"), motor_heat=g("motor_heat_to_air"),
            capital_usd=g("capital_cost"), lifetime_y=g("lifetime"), maintenance_usd_y=g("maintenance"),
            intake=d.intake or p.default_intake, serves=serves, schedule=cfg.schedules[d.schedule],
            schedule_name=d.schedule))
    return out


# --------------------------------------------------------------------------- helpers
def sun_fraction(sec, hour_of_day, alt_deg):
    s = sec.sun
    mult = s.shade_multiplier.value
    alt = np.asarray(alt_deg, float)
    if s.mode == "constant":
        f = np.full_like(alt, s.value.value)
    elif s.mode == "schedule":
        pts = np.array(s.points, float)
        f = np.interp(hour_of_day, pts[:, 0], pts[:, 1])
    else:
        tan_alt = np.tan(np.radians(np.maximum(alt, 0.1)))
        f = s.open_side_fraction.value * np.minimum(1.0, s.canopy_height_m / (s.canopy_depth_m * tan_alt))
    return np.where(alt > 0, f * mult, 0.0)


def sky_temperature(t_c, w, p):
    tdp = psychro.dew_point(w, p)
    eps = 0.711 + 0.56 * (tdp / 100.0) + 0.73 * (tdp / 100.0) ** 2  # Berdahl & Martin (1984), clear sky
    return eps ** 0.25 * (np.asarray(t_c) + T0K) - T0K


@dataclass
class SimResult:
    ok: bool
    message: str
    section_ids: List[str]
    t: np.ndarray  # output endpoint times [s] (reporting window)
    h: np.ndarray  # [nt, n_sec]
    W: np.ndarray
    T_air: np.ndarray
    RH: np.ndarray
    T_nodes: np.ndarray  # [nt, n_nodes]
    node_info: List[dict]
    drivers: Dict[str, np.ndarray]  # outdoor conditions at endpoints
    mid: Dict[str, np.ndarray]  # midpoint quantities for interval metrics
    device_on: np.ndarray  # [n_int, n_dev] bool
    device_outlet_h: np.ndarray  # [n_int, n_dev] at interval midpoints
    device_outlet_W: np.ndarray
    ledgers: List[dict]
    breakdown_J: Dict[str, Dict[str, float]]
    conservation: Dict[str, float]
    invalid: Dict[str, int]
    invalid_mask: np.ndarray  # [nt, n_sec] bool
    runtime_s: float
    n_rhs: int


class Model:
    def __init__(self, cfg: VenueConfig, weather: Weather, occ: OccupancySchedule,
                 device_ids: Optional[List[str]] = None, airflow=None):
        self.cfg = cfg
        self.wx = weather
        self.occ = occ
        self.devs = resolve_devices(cfg, device_ids or [])
        self.airflow = airflow or make_airflow(cfg)
        sim = cfg.simulation
        self.t0 = sim.start_h * 3600.0
        self.t1 = sim.end_h * 3600.0
        if self.t1 <= self.t0:
            raise ValueError("simulation.end_h must exceed start_h")
        self.ts = self.t0 - sim.warmup_days * DAY
        need_end = max(self.t1, self.t0 + DAY if sim.warmup_days > 0 else self.t1)
        if weather.t_s[0] > self.t0 + 1e-6 or weather.t_s[-1] < need_end - 1e-6:
            raise ValueError(
                f"weather covers {weather.t_s[0]/3600:.1f}-{weather.t_s[-1]/3600:.1f} h but the run needs "
                f"{self.t0/3600:.1f}-{need_end/3600:.1f} h (warm-up cycles the first 24 h)")
        self._compile()

    # ------------------------------------------------------------------ compile
    def _wrap(self, t):
        t = np.asarray(t, float)
        return np.where(t >= self.t0, t, self.t0 + np.mod(t - self.t0, DAY))

    def _compile(self):
        cfg = self.cfg
        secs = cfg.sections
        n = self.n = len(secs)
        self.sec_ids = [s.id for s in secs]
        site = cfg.site
        ph = cfg.physics
        self.conv_a, self.conv_b = ph.convection_a.value, ph.convection_b.value
        self.still = ph.still_air_speed.value
        tab = np.array(ph.occupant_sensible_fraction, float)
        self.fs_T, self.fs_f = tab[:, 0], tab[:, 1]
        self.lwf = np.array([s.outdoor.local_wind_factor.value for s in secs])
        self.met_W = np.array([s.occupancy.met.value * MET_W_M2 * ph.body_surface_area.value for s in secs])
        self.occ_cols = [s.occupancy.column or s.id for s in secs]
        sidx = {s.id: i for i, s in enumerate(secs)}
        self.conn_a = np.array([sidx[c.a] for c in cfg.connections], int)
        self.conn_b = np.array([sidx[c.b] for c in cfg.connections], int)

        # ---- surfaces
        nodes = []  # dicts
        cond = []  # (i, j, K)
        ground = []  # (node, UA, Tg)
        pairs = []  # (ni, nj, coef)
        env = []  # (node, coef_sky, coef_sur)
        int_conv = []  # (node, sec, A)
        ext_conv = []  # (node, A)
        solar_spec = []  # (node, type, A, alpha, extra)
        self.floor_node, self.roof_node, self.wall_nodes = {}, {}, {}
        vol = []
        for i, s in enumerate(secs):
            A_sec = s.area_m2
            vol.append(A_sec * s.height_m * s.volume_factor.value)
            fv = s.radiation.floor_view
            surf_nodes = {}
            areas = {}
            for su in s.surfaces:
                m = cfg.materials[su.material]
                if su.kind == "wall" and su.area_m2 is None:
                    raise ValueError(f"section {s.id} wall {su.id}: area_m2 required")
                A = su.area_m2 if su.area_m2 is not None else A_sec * su.area_fraction
                areas[su.id] = A
                C = m.density.value * m.specific_heat.value * A * su.thickness.value
                ids = []
                for layer in range(su.nodes):
                    ids.append(len(nodes))
                    nodes.append(dict(section=s.id, sec=i, surface=su.id, kind=su.kind,
                                      layer=("interior" if layer == 0 else "exterior"), C=C / su.nodes, A=A))
                if su.nodes == 2:
                    cond.append((ids[0], ids[1], m.conductivity.value * A / su.thickness.value))
                surf_nodes[su.id] = (su, ids, A)
                nin, nex = ids[0], ids[-1]
                int_conv.append((nin, i, A))
                if su.kind == "floor":
                    self.floor_node[i] = nin
                    ground.append((nex, su.ground_conductance.value * A, su.deep_ground_temperature.value))
                    solar_spec.append((nin, "floor", A, su.solar_absorptance_interior.value, i))
                else:
                    ext_conv.append((nex, A))
                    fsky = su.exterior_sky_fraction
                    env.append((nex, SIGMA * su.emissivity.value * A * fsky,
                                SIGMA * su.emissivity.value * A * (1 - fsky)))
                    if su.kind == "roof":
                        self.roof_node[i] = nin
                        solar_spec.append((nex, "roof", A, su.solar_absorptance_exterior.value, None))
                    else:
                        self.wall_nodes.setdefault(i, []).append(nin)
                        solar_spec.append((nex, "wall", A, su.solar_absorptance_exterior.value, su.azimuth_deg))
            # interior longwave network
            floor = next(v for v in surf_nodes.values() if v[0].kind == "floor")
            roof = next((v for v in surf_nodes.values() if v[0].kind == "roof"), None)
            walls = [v for v in surf_nodes.values() if v[0].kind == "wall"]
            fsu, fids, fA = floor
            eps_f = fsu.emissivity.value
            if fv.roof > 0 and roof is None:
                raise ValueError(f"section {s.id}: floor_view.roof > 0 but no roof surface")
            if fv.walls > 0 and not walls:
                raise ValueError(f"section {s.id}: floor_view.walls > 0 but no wall surfaces")
            f_sur = 1.0 - fv.roof - fv.walls - fv.sky
            env.append((fids[0], SIGMA * eps_f * fA * fv.sky, SIGMA * eps_f * fA * f_sur))
            if roof is not None:
                rsu, rids, rA = roof
                F_rf = fA * fv.roof / rA
                if F_rf > 1 + 1e-9:
                    raise ValueError(f"section {s.id}: roof->floor view {F_rf:.2f} > 1 (reciprocity)")
                pairs.append((fids[0], rids[0], SIGMA * eps_f * rsu.emissivity.value * fA * fv.roof))
                env.append((rids[0], 0.0, SIGMA * rsu.emissivity.value * rA * (1 - F_rf)))
            if walls:
                wA_tot = sum(w[2] for w in walls)
                for wsu, wids, wA in walls:
                    F_fw = fv.walls * wA / wA_tot
                    F_wf = fA * F_fw / wA
                    if F_wf > 1 + 1e-9:
                        raise ValueError(f"section {s.id}: wall {wsu.id} -> floor view > 1")
                    pairs.append((fids[0], wids[0], SIGMA * eps_f * wsu.emissivity.value * fA * F_fw))
                    rem = 1 - F_wf
                    e = SIGMA * wsu.emissivity.value * wA
                    env.append((wids[0], e * rem * wsu.interior_sky_fraction, e * rem * (1 - wsu.interior_sky_fraction)))
        self.node_info = nodes
        self.nn = len(nodes)
        self.C = np.array([x["C"] for x in nodes])
        self.cond = np.array(cond, float).reshape(-1, 3)
        self.ground = np.array(ground, float).reshape(-1, 3)
        self.pairs = np.array(pairs, float).reshape(-1, 3)
        self.env = np.array(env, float).reshape(-1, 3)
        ic = np.array(int_conv, float).reshape(-1, 3)
        self.ic_node, self.ic_sec, self.ic_A = ic[:, 0].astype(int), ic[:, 1].astype(int), ic[:, 2]
        ec = np.array(ext_conv, float).reshape(-1, 2)
        self.ec_node, self.ec_A = ec[:, 0].astype(int), ec[:, 1]
        self.vol = np.array(vol)

        # ---- driver table on a fine grid (time-only inputs)
        tg = np.arange(self.ts, self.t1 + DT_FINE, DT_FINE)
        tw = self._wrap(tg)
        To, Wo, p, U, ghi, dni, dhi = self.wx.at(tw)
        hour = np.mod(tw / 3600.0, 24.0)
        alt, az = solar.solar_position(hour, site.day_of_year, site.latitude_deg, site.longitude_deg, site.utc_offset_h)
        sin_alt = np.maximum(np.sin(np.radians(alt)), 0.0)
        dni = np.where(alt > 0, dni, 0.0)
        fsun = np.stack([sun_fraction(s, hour, alt) for s in secs], axis=1)
        sol = np.zeros((len(tg), self.nn))
        for node, typ, A, alpha, extra in solar_spec:
            if typ == "roof":
                sol[:, node] += alpha * A * ghi
            elif typ == "wall":
                sol[:, node] += alpha * A * (dni * solar.vertical_beam_factor(alt, az, extra) + 0.5 * dhi + 0.5 * 0.2 * ghi)
            else:
                fsky = secs[extra].radiation.floor_view.sky
                sol[:, node] += alpha * A * (fsun[:, extra] * dni * sin_alt + fsky * dhi)
        Qo, Qc = self.airflow.flows(U)
        rho = psychro.rho_da(To, Wo, p)
        self.tg = tg
        self.drv = dict(To=To, Wo=Wo, p=p, U=U, ghi=ghi, dni=dni, dhi=dhi, alt=alt, az=az,
                        Tsky=sky_temperature(To, Wo, p), ho=psychro.enthalpy(To, Wo), rho=rho,
                        fsun=fsun, sol=sol, Qo=Qo, Qc=Qc, hour=hour)
        # dry-air masses from reference density at the run start (labelled: constant M_i)
        self.M = self.vol * rho[0]

        # ---- devices
        nd = self.nd = len(self.devs)
        self.F = np.zeros((nd, n))
        for d, dev in enumerate(self.devs):
            for si, frac, _ in dev.serves:
                self.F[d, si] = frac
        self.d_air = np.array([d.airflow_m3s for d in self.devs])
        self.d_evap = np.array([d.kind == "evaporative" for d in self.devs], bool)
        self.d_eff = np.array([d.eff for d in self.devs])
        self.d_heat = np.array([d.power_W * d.motor_heat for d in self.devs])
        self.d_out = np.array([d.intake == "outdoor" for d in self.devs], bool)

        # ---- segments
        self._build_segments()

        # ---- state layout
        self.i_h = slice(0, n)
        self.i_w = slice(n, 2 * n)
        self.i_T = slice(2 * n, 2 * n + self.nn)
        self.n_phys = 2 * n + self.nn
        # the state size must not depend on the number of devices: the solver's RMS error norm
        # would otherwise change step selection, and an idle device would perturb the baseline
        self.i_acc = self.n_phys
        self.i_evap = self.n_phys + 4
        self.i_brk = slice(self.n_phys + 5, self.n_phys + 5 + N_BRK * n)
        self.ny = self.n_phys + 5 + N_BRK * n

    def _schedule_on(self, sch: Schedule, t_mid, served_secs):
        tw = float(self._wrap(t_mid))
        hod = (tw / 3600.0) % 24.0
        if sch.mode == "always":
            return True
        if sch.mode == "never":
            return False
        if sch.mode == "hours":
            return any(a <= hod < b for a, b in sch.intervals)
        To = float(self.wx.at(tw)[0])
        people = sum(self.occ.at(self.occ_cols[i], tw) for i in served_secs)
        in_hours = (not sch.intervals) or any(a <= hod < b for a, b in sch.intervals)
        return bool(people > 0 and To >= sch.outdoor_threshold.value and in_hours)

    def _build_segments(self):
        day_pts = set()
        for tt in self.wx.t_s:
            day_pts.add(float(tt))
        for tt in self.occ.t_s:
            day_pts.add(float(tt))
        hrs = []
        for dev in self.devs:
            for a, b in dev.schedule.intervals:
                hrs += [a, b]
        for tou in self.cfg.tariffs.time_of_use:
            hrs += [tou.start_h, tou.end_h]
        hrs += list(range(25))
        span_days = int(np.ceil((self.t1 - self.t0) / DAY)) + 1
        for d0 in range(-1, span_days + 1):
            for hh in hrs:
                day_pts.add(d0 * DAY + hh * 3600.0)
        rep = sorted(x for x in day_pts if self.t0 <= x <= self.t1)
        first_day = sorted(x - self.t0 for x in day_pts if self.t0 <= x < self.t0 + DAY)
        warm = []
        for k in range(self.cfg.simulation.warmup_days, 0, -1):
            base = self.t0 - k * DAY
            warm += [base + x for x in first_day]
        pts = sorted(set(warm + rep + [self.ts, self.t0, self.t1]))
        pts = [x for x in pts if self.ts <= x <= self.t1]
        segs = []
        for a, b in zip(pts[:-1], pts[1:]):
            if b - a < 1e-6:
                continue
            mid = 0.5 * (a + b)
            tw = float(self._wrap(mid))
            N = np.array([self.occ.at(c, tw) for c in self.occ_cols], float)
            on = np.array([self._schedule_on(d.schedule, mid, [s[0] for s in d.serves]) for d in self.devs], bool)
            segs.append((a, b, N, on))
        self.segments = segs

    # ------------------------------------------------------------------ physics
    def _drivers(self, t):
        x = (t - self.ts) / DT_FINE
        i = int(min(max(np.floor(x), 0), len(self.tg) - 2))
        f = min(max(x - i, 0.0), 1.0)
        d = self.drv
        g = lambda a: a[i] + f * (a[i + 1] - a[i])  # noqa: E731
        return (g(d["To"]), g(d["Wo"]), g(d["p"]), g(d["U"]), g(d["Tsky"]), g(d["ho"]), g(d["rho"]),
                g(d["Qo"]), g(d["Qc"]), g(d["sol"]))

    def rhs_parts(self, t, y, N, on):
        """Return (dy, details). details holds boundary/internal terms for diagnostics."""
        n = self.n
        h = y[self.i_h]
        W = y[self.i_w]
        Ts = y[self.i_T]
        T = psychro.t_from_hw(h, W)
        To, Wo, p, U, Tsky, ho, rho, Qo, Qc, sol = self._drivers(t)
        dH = np.zeros(n)
        dM = np.zeros(n)
        brk = np.zeros((N_BRK, n))
        # outdoor exchange
        mo = rho * Qo
        q_out = mo * (ho - h)
        w_out = mo * (Wo - W)
        dH += q_out
        dM += w_out
        brk[0] = q_out
        # inter-section exchange (equal and opposite)
        if len(self.conn_a):
            mc = rho * Qc
            dq = mc * (h[self.conn_b] - h[self.conn_a])
            dw = mc * (W[self.conn_b] - W[self.conn_a])
            qi = np.bincount(self.conn_a, dq, n) - np.bincount(self.conn_b, dq, n)
            wi = np.bincount(self.conn_a, dw, n) - np.bincount(self.conn_b, dw, n)
            dH += qi
            dM += wi
            brk[1] = qi
        # surfaces
        dTs = np.zeros(self.nn)
        v_sec = np.maximum(self.still, self.lwf * U)
        hin = self.conv_a + self.conv_b * v_sec[self.ic_sec]
        qc = hin * self.ic_A * (Ts[self.ic_node] - T[self.ic_sec])
        qsec = np.bincount(self.ic_sec, qc, n)
        dH += qsec
        brk[2] = qsec
        dTs -= np.bincount(self.ic_node, qc, self.nn)
        if len(self.cond):
            i, j = self.cond[:, 0].astype(int), self.cond[:, 1].astype(int)
            q = self.cond[:, 2] * (Ts[i] - Ts[j])
            dTs -= np.bincount(i, q, self.nn)
            dTs += np.bincount(j, q, self.nn)
        TK = Ts + T0K
        if len(self.pairs):
            i, j = self.pairs[:, 0].astype(int), self.pairs[:, 1].astype(int)
            q = self.pairs[:, 2] * (TK[i] ** 4 - TK[j] ** 4)
            dTs -= np.bincount(i, q, self.nn)
            dTs += np.bincount(j, q, self.nn)
        en = self.env[:, 0].astype(int)
        q_env = self.env[:, 1] * (TK[en] ** 4 - (Tsky + T0K) ** 4) + self.env[:, 2] * (TK[en] ** 4 - (To + T0K) ** 4)
        dTs -= np.bincount(en, q_env, self.nn)
        hex_ = self.conv_a + self.conv_b * U
        q_ext = hex_ * self.ec_A * (Ts[self.ec_node] - To)
        dTs -= np.bincount(self.ec_node, q_ext, self.nn)
        gn = self.ground[:, 0].astype(int)
        q_g = self.ground[:, 1] * (Ts[gn] - self.ground[:, 2])
        dTs -= np.bincount(gn, q_g, self.nn)
        dTs += sol
        # occupants: total metabolic heat added once; latent part carried as vapour enthalpy
        Qtot = N * self.met_W
        fs = np.interp(T, self.fs_T, self.fs_f)
        hv = psychro.h_vapour(T)
        m_occ = Qtot * (1 - fs) / hv
        dH += Qtot * fs + m_occ * hv
        dM += m_occ
        brk[3] = Qtot
        # devices
        evap = np.zeros(self.nd)
        dev_bE = 0.0
        dev_bW = 0.0
        gross_dev = 0.0
        if self.nd:
            act = on
            if np.any(act):
                m = rho * self.d_air * act
                h_in = np.where(self.d_out, ho, self.F @ h)
                w_in = np.where(self.d_out, Wo, self.F @ W)
                h_o, w_o, ev = outlet_state(h_in, w_in, p, self.d_evap & act, self.d_eff, self.d_heat * act,
                                            np.where(act, m, 1.0))
                ev = ev * act
                Fm = self.F * m[:, None]
                qd = (Fm * (h_o[:, None] - h[None, :])).sum(axis=0)
                wd = (Fm * (w_o[:, None] - W[None, :])).sum(axis=0)
                dH += qd
                dM += wd
                brk[4] = qd
                evap = ev
                # boundary part: outdoor-intake devices exchange with outdoors; recirculating ones only
                # add their process enthalpy/water (their redistribution is internal and sums to zero)
                rowq = (Fm * (h_o[:, None] - h[None, :])).sum(axis=1)
                roww = (Fm * (w_o[:, None] - W[None, :])).sum(axis=1)
                dev_bE = float(np.sum(np.where(self.d_out, rowq, m * (h_o - h_in))))
                dev_bW = float(np.sum(np.where(self.d_out, roww, m * (w_o - w_in))))
                gross_dev = float(np.sum(np.abs(Fm * (h_o[:, None] - h[None, :]))))
        dh = dH / self.M
        dw = dM / self.M
        dT = dTs / self.C
        bE = (q_out.sum() + Qtot.sum() + sol.sum() - q_env.sum() - q_ext.sum() - q_g.sum() + dev_bE)
        bW = w_out.sum() + m_occ.sum() + dev_bW
        gE = (np.abs(q_out).sum() + Qtot.sum() + sol.sum() + np.abs(q_env).sum() + np.abs(q_ext).sum()
              + np.abs(q_g).sum() + abs(dev_bE) + gross_dev)
        gW = np.abs(w_out).sum() + m_occ.sum() + abs(dev_bW)
        dy = np.empty(self.ny)
        dy[self.i_h] = dh
        dy[self.i_w] = dw
        dy[self.i_T] = dT
        dy[self.i_acc:self.i_acc + 4] = (bE, bW, gE, gW)
        dy[self.i_evap] = float(np.sum(evap))
        dy[self.i_brk] = brk.ravel()
        return dy, dict(T=T, To=To, Wo=Wo, p=p, U=U, Tsky=Tsky, rho=rho, Qo=Qo, Qc=Qc)

    # ------------------------------------------------------------------ run
    def initial_state(self):
        To, Wo, p, *_ = self._drivers(self.ts)
        ini = [s.initial for s in self.cfg.sections]
        Tair = np.array([To if i.air == "outdoor" else i.air.value for i in ini])
        W = np.array([Wo if i.humidity_ratio == "outdoor" else i.humidity_ratio.value for i in ini])
        y = np.zeros(self.ny)
        y[self.i_h] = psychro.enthalpy(Tair, W)
        y[self.i_w] = W
        Tn = []
        for nd in self.node_info:
            s = ini[nd["sec"]].surfaces
            Tn.append(To if s == "outdoor_air" else s.value)
        y[self.i_T] = np.array(Tn)
        return y

    def simulate(self) -> SimResult:
        cfg = self.cfg
        sim = cfg.simulation
        t_start = _time.perf_counter()
        y = self.initial_state()
        step = sim.output_step_s
        grid = np.arange(self.t0, self.t1 + 1e-6, step)
        if abs(grid[-1] - self.t1) > 1e-6:
            grid = np.append(grid, self.t1)
        typ = np.concatenate([np.full(self.n, 1e5), np.full(self.n, 1e-2), np.full(self.nn, 1e2),
                              [1e9, 1e3, 1e9, 1e3], [1e2], np.full(N_BRK * self.n, 1e9)])
        atol = sim.atol_scale * typ
        sparsity = np.zeros((self.ny, self.ny))
        sparsity[:, :self.n_phys] = 1.0
        outs_t, outs_y = [], []
        n_rhs = [0]
        msg = "ok"
        ok = True
        y_t0 = None
        for (a, b, N, on) in self.segments:
            def fun(t, yy, N=N, on=on):
                n_rhs[0] += 1
                return self.rhs_parts(t, yy, N, on)[0]
            te = grid[(grid >= a - 1e-9) & (grid <= b + 1e-9)]
            kw = dict(method=sim.method, rtol=sim.rtol, atol=atol, max_step=sim.max_step_s,
                      t_eval=te if len(te) else None)
            if sim.method in ("BDF", "Radau"):
                kw["jac_sparsity"] = sparsity
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = solve_ivp(fun, (a, b), y, **kw)
            if not sol.success:
                ok, msg = False, f"solver failed in segment {a/3600:.2f}-{b/3600:.2f} h: {sol.message}"
                break
            if abs(a - self.t0) < 1e-6:
                y_t0 = y.copy()
            if len(te):
                for k, tt in enumerate(sol.t):
                    if not outs_t or tt > outs_t[-1] + 1e-6:
                        outs_t.append(tt)
                        outs_y.append(sol.y[:, k])
            y = sol.y[:, -1]
            if not np.all(np.isfinite(y)):
                ok, msg = False, f"non-finite state at {b/3600:.2f} h"
                break
        if y_t0 is None and outs_y:
            y_t0 = outs_y[0]
        runtime = _time.perf_counter() - t_start
        if not ok or len(outs_t) < 2:
            return self._failed(msg, runtime, n_rhs[0])
        return self._postprocess(np.array(outs_t), np.array(outs_y), y_t0, runtime, n_rhs[0])

    def _failed(self, msg, runtime, nrhs):
        e = np.zeros((0, self.n))
        return SimResult(False, msg, self.sec_ids, np.zeros(0), e, e, e, e, np.zeros((0, self.nn)), self.node_info,
                         {}, {}, np.zeros((0, self.nd), bool), np.zeros((0, self.nd)), np.zeros((0, self.nd)),
                         [], {}, {}, {}, np.zeros((0, self.n), bool), runtime, nrhs)

    def _seg_at(self, t):
        for a, b, N, on in self.segments:
            if a <= t < b:
                return N, on
        return self.segments[-1][2], self.segments[-1][3]

    def _postprocess(self, t, Y, y_t0, runtime, nrhs) -> SimResult:
        n, nd = self.n, self.nd
        h, W, Tn = Y[:, self.i_h], Y[:, self.i_w], Y[:, self.i_T]
        T = psychro.t_from_hw(h, W)
        drv = {k: np.interp(t, self.tg, self.drv[k]) for k in ("To", "Wo", "p", "U", "Tsky", "ghi", "dni", "dhi", "alt", "hour")}
        fsun_e = np.stack([np.interp(t, self.tg, self.drv["fsun"][:, i]) for i in range(n)], axis=1)
        drv["fsun"] = fsun_e
        RH = psychro.rh_from_w(T, W, drv["p"][:, None])
        # invalid states (flagged, never clipped)
        val = psychro.state_validity(T, W, drv["p"][:, None])
        node_bad = (Tn < -40) | (Tn > 95) | ~np.isfinite(Tn)
        inv_mask = np.zeros_like(T, bool)
        for v in val.values():
            inv_mask |= v
        invalid = {k: int(v.sum()) for k, v in val.items()}
        invalid["surface_temperature_out_of_range"] = int(node_bad.sum())
        if node_bad.any():
            for j in range(self.nn):
                inv_mask[:, self.node_info[j]["sec"]] |= node_bad[:, j]
        # interval midpoints
        tm = 0.5 * (t[:-1] + t[1:])
        hm, Wm = 0.5 * (h[:-1] + h[1:]), 0.5 * (W[:-1] + W[1:])
        Tnm = 0.5 * (Tn[:-1] + Tn[1:])
        mid = {k: np.interp(tm, self.tg, self.drv[k]) for k in ("To", "Wo", "p", "U", "Tsky", "ghi", "dni", "dhi", "alt", "az", "hour")}
        mid["fsun"] = np.stack([np.interp(tm, self.tg, self.drv["fsun"][:, i]) for i in range(n)], axis=1)
        mid.update(t=tm, dt=np.diff(t), h=hm, W=Wm, T_air=psychro.t_from_hw(hm, Wm), T_nodes=Tnm,
                   invalid=(inv_mask[:-1] | inv_mask[1:]))
        mid["people"] = np.array([[self.occ.person_seconds(c, float(self._wrap(a)), float(self._wrap(a)) + (b - a)) / (b - a)
                                   for c in self.occ_cols] for a, b in zip(t[:-1], t[1:])])
        # device status and outlet states per interval
        on = np.zeros((len(tm), nd), bool)
        dh_o = np.zeros((len(tm), nd))
        dw_o = np.zeros((len(tm), nd))
        evap_rate = np.zeros((len(tm), nd))
        for k, tt in enumerate(tm):
            _, o = self._seg_at(tt)
            on[k] = o
            if nd:
                rho = psychro.rho_da(mid["To"][k], mid["Wo"][k], mid["p"][k])
                ho = psychro.enthalpy(mid["To"][k], mid["Wo"][k])
                h_in = np.where(self.d_out, ho, self.F @ hm[k])
                w_in = np.where(self.d_out, mid["Wo"][k], self.F @ Wm[k])
                m = rho * self.d_air
                ho_, wo_, ev_ = outlet_state(h_in, w_in, mid["p"][k], self.d_evap, self.d_eff, self.d_heat,
                                             np.maximum(m, 1e-9))
                dh_o[k], dw_o[k] = ho_, wo_
                evap_rate[k] = ev_ * o
        # ledgers over the reporting window
        ledgers = []
        acc0 = y_t0
        acc1 = Y[-1]
        evap_total_ode = float(acc1[self.i_evap] - acc0[self.i_evap])
        evap_quad = (evap_rate * mid["dt"][:, None]).sum(axis=0) if nd else np.zeros(0)
        q_sum = float(evap_quad.sum())
        evap_ratio = evap_total_ode / q_sum if q_sum > 0 else 1.0
        # per-device split from midpoint quadrature, reconciled to the integrated (ODE) total
        evap_kg = evap_quad * evap_ratio
        for d, dev in enumerate(self.devs):
            on_s = 0.0
            e_cost = 0.0
            for a, b, N, o in self.segments:
                if b <= self.t0 or a >= self.t1 or not o[d]:
                    continue
                dt = min(b, self.t1) - max(a, self.t0)
                on_s += dt
                hod = (float(self._wrap(0.5 * (a + b))) / 3600.0) % 24
                price = self.cfg.tariffs.electricity.value
                for tou in self.cfg.tariffs.time_of_use:
                    if tou.start_h <= hod < tou.end_h:
                        price = tou.price.value
                e_cost += dev.power_W * dt / 3.6e6 * price
            water_m3 = evap_kg[d] * (1 + dev.bleed) / 1000.0
            ledgers.append(dict(device=dev.id, preset=dev.preset, kind=dev.kind, intake=dev.intake,
                                schedule=dev.schedule_name,
                                serves=";".join(f"{self.sec_ids[s]}:{f:.2f}" for s, f, _ in dev.serves),
                                on_hours=on_s / 3600.0, electric_power_kW=dev.power_W / 1000.0,
                                electricity_kWh=dev.power_W * on_s / 3.6e6, electricity_cost_usd=e_cost,
                                evaporated_kg=float(evap_kg[d]), water_m3=float(water_m3),
                                water_cost_usd=float(water_m3 * self.cfg.tariffs.water.value),
                                capital_usd=dev.capital_usd, lifetime_y=dev.lifetime_y,
                                maintenance_usd_y=dev.maintenance_usd_y))
        brk = (acc1[self.i_brk] - acc0[self.i_brk]).reshape(N_BRK, n)
        breakdown = {self.sec_ids[i]: {BREAKDOWN_NAMES[k]: float(brk[k, i]) for k in range(N_BRK)} for i in range(n)}
        # conservation
        def stored(yv):
            return float(np.sum(self.M * yv[self.i_h]) + np.sum(self.C * yv[self.i_T]))

        def water(yv):
            return float(np.sum(self.M * yv[self.i_w]))
        dE = stored(acc1) - stored(acc0)
        bE = acc1[self.i_acc] - acc0[self.i_acc]
        gE = acc1[self.i_acc + 2] - acc0[self.i_acc + 2]
        dW_ = water(acc1) - water(acc0)
        bW = acc1[self.i_acc + 1] - acc0[self.i_acc + 1]
        gW = acc1[self.i_acc + 3] - acc0[self.i_acc + 3]
        evap_total = evap_total_ode
        # instantaneous algebraic check: internal transfers cancel exactly
        inst = 0.0
        for k in range(0, len(t), max(1, len(t) // 48)):
            N, o = self._seg_at(min(t[k], self.t1 - 1e-6))
            dy, _ = self.rhs_parts(t[k], Y[k], N, o)
            lhs = np.sum(self.M * dy[self.i_h]) + np.sum(self.C * dy[self.i_T])
            inst = max(inst, abs(lhs - dy[self.i_acc]) / max(dy[self.i_acc + 2], 1.0))
        # dry-air balance per section (in minus out) from the flow structure
        dry = 0.0
        for k in range(0, len(tm), max(1, len(tm) // 48)):
            rho = psychro.rho_da(mid["To"][k], mid["Wo"][k], mid["p"][k])
            ko = int(np.clip(np.searchsorted(self.tg, tm[k]), 0, len(self.tg) - 1))
            m_o = rho * self.drv["Qo"][ko]
            m_c = rho * self.drv["Qc"][ko]
            m_dev = rho * self.d_air * on[k]
            inflow = m_o + np.bincount(self.conn_a, m_c, n) + np.bincount(self.conn_b, m_c, n) \
                + (self.F * m_dev[:, None]).sum(axis=0)
            # leaving: to outdoors (exchange), to neighbours, and displaced/drawn by devices
            outflow = m_o.copy() + np.bincount(self.conn_a, m_c, n) + np.bincount(self.conn_b, m_c, n)
            for d in range(nd):
                outflow += self.F[d] * m_dev[d]
            dry = max(dry, float(np.max(np.abs(inflow - outflow))) if n else 0.0)
        cons = dict(energy_residual_J=dE - bE, energy_gross_J=gE,
                    energy_residual_rel=(dE - bE) / max(gE, 1.0),
                    water_residual_kg=dW_ - bW, water_gross_kg=gW, water_residual_rel=(dW_ - bW) / max(gW, 1e-9),
                    internal_cancellation_rel_max=inst, dry_air_imbalance_kg_s=dry,
                    device_evaporated_kg=evap_total, evaporation_ode_to_quadrature_ratio=evap_ratio,
                    stored_energy_change_J=dE)
        return SimResult(True, "ok", self.sec_ids, t, h, W, T, RH, Tn, self.node_info, drv, mid, on, dh_o, dw_o,
                         ledgers, breakdown, cons, invalid, inv_mask, runtime, nrhs)
