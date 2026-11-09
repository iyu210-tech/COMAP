"""Coupled-model checks: conservation, mixing, fans, evaporative coolers, shared devices,
energy units, baseline equivalence, occupancy aggregation and numerical refinement.

These are software/physics consistency checks, NOT validation against measurements.
"""

import numpy as np
import pytest

from conftest import q, sealed, study_with
from venuecool import metrics


def test_conservation_residuals_small(mini_study):
    for sid in ("baseline", "fan", "evap", "shared"):
        o = mini_study.run_scenario(sid)
        c = o.sim.conservation
        assert o.ok
        assert abs(c["energy_residual_rel"]) < 1e-9
        assert abs(c["water_residual_rel"]) < 1e-9
        assert c["internal_cancellation_rel_max"] < 1e-10  # internal transfers equal and opposite
        assert c["dry_air_imbalance_kg_s"] == 0.0
        assert sum(o.sim.invalid.values()) == 0


def test_conservative_mixing_between_sealed_sections(mini_raw):
    def mut(r):
        sealed(r)
        r["sections"][0]["initial"] = {"humidity_ratio": q(0.004, "kg/kg")}
        r["sections"][1]["initial"] = {"humidity_ratio": q(0.016, "kg/kg")}
    s = study_with(mini_raw, mut)
    o = s.run_scenario("baseline")
    M = o.model.M
    total = (o.sim.W * M).sum(axis=1)
    # warm-up already mixed the sections; total moisture stays constant over the reporting window
    np.testing.assert_allclose(total, total[0], rtol=1e-8)
    assert np.max(np.abs(o.sim.W[:, 0] - o.sim.W[:, 1])) < 1e-6
    # mixed humidity ratio is the dry-air-mass-weighted mean of the initial values
    assert o.sim.W[-1, 0] == pytest.approx((0.004 * M[0] + 0.016 * M[1]) / M.sum(), rel=1e-6)


def test_fan_raises_air_speed_not_outdoor_exchange(mini_study, mini_base):
    o = mini_study.run_scenario("fan")
    # no refrigeration: section air never colder than baseline
    assert np.all(o.sim.T_air[:, 0] >= mini_base.sim.T_air[:, 0] - 1e-6)
    # outdoor exchange flow structure identical (recirculating fan adds no outside air)
    np.testing.assert_allclose(o.model.drv["Qo"], mini_base.model.drv["Qo"])
    # the only added enthalpy is motor heat: devices term = P * on_time
    on_s = o.sim.ledgers[0]["on_hours"] * 3600
    assert o.sim.breakdown_J["A"]["devices"] == pytest.approx(250.0 * on_s, rel=1e-6)
    c = o.comfort_df
    cov = c[(c.section == "A") & (c.group == "covered") & c.served]
    unc = c[(c.section == "A") & (c.group == "open")]
    assert cov.v.min() > unc.v.max() - 1e-9
    assert not c[(c.section == "A") & (c.group == "open")].served.any()


def test_table_airflow_provider_reproduces_parametric(tmp_path, mini_raw, mini_study, mini_base):
    import pandas as pd
    from venuecool.airflow import ParametricAirflow
    from venuecool.inputs import InputDataError
    cfg = mini_study.cfg0
    u = np.array([0.0, 30.0])  # flows are linear in wind, so two rows interpolate exactly
    qo, qc = ParametricAirflow(cfg).flows(u)
    df = pd.DataFrame({"wind_ms": u, **{f"out:{s.id}": qo[:, i] for i, s in enumerate(cfg.sections)},
                       **{f"conn:{c.id}": qc[:, j] for j, c in enumerate(cfg.connections)}})
    df.to_csv(tmp_path / "flows.csv", index=False)
    spec = {"provider": "table", "file": str(tmp_path / "flows.csv"), "source": "test copy of parametric flows"}
    o = study_with(mini_raw, lambda r: r.__setitem__("airflow", spec)).run_scenario("baseline")
    np.testing.assert_allclose(o.sim.T_air, mini_base.sim.T_air, rtol=1e-9, atol=1e-9)
    df.drop(columns=[f"out:{cfg.sections[0].id}"]).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(InputDataError, match="missing columns"):
        study_with(mini_raw, lambda r: r.__setitem__("airflow", {**spec, "file": str(tmp_path / "bad.csv")})
                   ).run_scenario("baseline")


def test_energy_units_fan_one_hour_is_quarter_kwh(mini_raw):
    s = study_with(mini_raw, lambda r: None)
    o = s.run(["fan_hour"], label="fan_hour")
    led = o.sim.ledgers[0]
    assert led["on_hours"] == pytest.approx(1.0)
    assert led["electricity_kWh"] == pytest.approx(0.25)  # 0.25 kW for 1 h
    assert o.costs["electricity_kWh"] == pytest.approx(0.25)
    assert o.costs["installed_kW"] == pytest.approx(0.25)  # power (kW) kept separate from energy (kWh)


def test_evaporative_water_ledger_matches_moisture_added(mini_study, mini_base):
    o = mini_study.run_scenario("evap")
    led = o.sim.ledgers[0]
    assert led["evaporated_kg"] > 0
    assert led["water_m3"] == pytest.approx(led["evaporated_kg"] * 1.15 / 1000)  # bleed 15%
    # total boundary water = baseline-like terms + device evaporation, checked by residual
    assert abs(o.sim.conservation["water_residual_rel"]) < 1e-9
    on = o.sim.device_on[:, 0]
    tA = o.sim.mid["T_air"][:, 0]
    assert np.all(tA[on] < mini_base.sim.mid["T_air"][on, 0])
    from venuecool import psychro
    t_wb = psychro.wet_bulb(o.sim.mid["To"][on], o.sim.mid["Wo"][on], o.sim.mid["p"][on])
    t_out = psychro.t_from_hw(o.sim.device_outlet_h[on, 0], o.sim.device_outlet_W[on, 0])
    assert np.all(t_out >= t_wb - 1e-6)  # wet-bulb limit (motor heat only raises it)


def test_recirculating_cooler_in_sealed_room_humidifies_and_weakens(mini_raw):
    def mut(r):
        sealed(r, zero_people=True)
    s = study_with(mini_raw, mut)
    rec = s.run(["evap_rec"], label="rec")
    out = s.run(["evap_out"], label="out")
    on = rec.sim.device_on[:, 0]
    assert rec.sim.mid["W"][on, 0].max() > out.sim.mid["W"][on, 0].max()
    # outdoor-intake unit creates through-flow; recirculating unit does not exchange outdoor air
    assert out.sim.breakdown_J["A"]["devices"] != rec.sim.breakdown_J["A"]["devices"]


def test_shared_device_counted_once(mini_study, mini_base):
    o = mini_study.run_scenario("shared")
    assert len(o.sim.ledgers) == 1
    led = o.sim.ledgers[0]
    assert led["electricity_kWh"] == pytest.approx(2.5 * led["on_hours"])
    assert led["serves"] == "A:0.60;B:0.40"
    # both served sections are cooler while the unit runs (direct evaporative cooling is ~isenthalpic,
    # so the sign of the delivered enthalpy is not a cooling indicator)
    on = o.sim.device_on[:, 0]
    for i in (0, 1):
        assert np.all(o.sim.mid["T_air"][on, i] < mini_base.sim.mid["T_air"][on, i])
    assert abs(o.sim.conservation["evaporation_ode_to_quadrature_ratio"] - 1) < 0.01
    v = o.summary["venue"]
    assert v["served_person_hours"] <= v["occupied_person_hours"] + 1e-9


def test_overlapping_devices_do_not_double_count_beneficiaries(mini_study):
    o = mini_study.run(["evap_rec", "evap_rec2"], label="two_on_same_group")
    one = mini_study.run(["evap_rec"], label="one")
    assert o.summary["venue"]["served_person_hours"] == pytest.approx(one.summary["venue"]["served_person_hours"])
    assert o.costs["electricity_kWh"] == pytest.approx(2 * one.costs["electricity_kWh"])


def test_baseline_equivalence(mini_study, mini_base):
    again = mini_study.run_scenario("baseline")
    np.testing.assert_array_equal(again.sim.T_air, mini_base.sim.T_air)
    never = mini_study.run(["fan_never"], label="never")
    c = metrics.compare(mini_base.summary, never.summary, mini_base.costs, never.costs)
    assert c["relief_Kph"] == pytest.approx(0.0, abs=1e-9)
    assert c["incremental_kWh"] == 0.0
    np.testing.assert_allclose(never.sim.T_air, mini_base.sim.T_air, rtol=1e-12)


def test_negative_relief_is_preserved(mini_study, mini_base):
    # a "fan" whose only effect is heat (covers no occupied group): must give relief <= 0
    o = mini_study.run(["fan1"], label="fan")
    c = metrics.compare(mini_base.summary, o.summary, mini_base.costs, o.costs)
    sec_b = c["sections"]["B"]["relief_Kph"]
    assert sec_b <= 1e-9  # B only receives the fan's motor heat through the connection
    assert c["relief_Kph"] == pytest.approx(sum(v["relief_Kph"] for v in c["sections"].values()))


def test_occupancy_aggregation(mini_study, mini_base):
    v = mini_base.summary["venue"]
    # A: 10 people x 10 h, B: 4 people x 4 h within the 24 h window -> 116 person-hours
    assert v["occupied_person_hours"] == pytest.approx(116.0, rel=1e-9)
    s = mini_base.summary["sections"]
    assert s["A"]["occupied_person_hours"] == pytest.approx(100.0)
    assert s["B"]["occupied_person_hours"] == pytest.approx(16.0)
    df = mini_base.comfort_df
    assert df.person_hours.sum() == pytest.approx(116.0)  # groups x sun states partition people exactly
    assert v["assessed_person_hours"] + v["unassessed_person_hours"] == pytest.approx(116.0)


def test_ratios_formed_after_summing(mini_study, mini_base):
    o = mini_study.run_scenario("shared")
    c = metrics.compare(mini_base.summary, o.summary, mini_base.costs, o.costs)
    assert c["relief_per_kWh"] == pytest.approx(c["relief_Kph"] / c["incremental_kWh"])


def test_offgrid_schedule_matches_fine_grid(mini_raw):
    """A schedule edge that is not on the reporting grid must still hand the true endpoint onward."""
    def prepare(r, step):
        r["simulation"]["warmup_days"] = 0
        r["simulation"]["end_h"] = 2
        r["simulation"]["output_step_s"] = step
        r["simulation"]["rtol"] = 1e-7
        r["schedules"]["offgrid"] = {"mode": "hours", "intervals": [[0.0, 0.35]]}
        for d in r["devices"]:
            if d["id"] == "evap_out":
                d["schedule"] = "offgrid"
    coarse = study_with(mini_raw, lambda r: prepare(r, 600.0)).run(["evap_out"], label="coarse")
    fine = study_with(mini_raw, lambda r: prepare(r, 60.0)).run(["evap_out"], label="fine")
    assert 1260.0 not in set(np.round(coarse.sim.t, 5))
    assert any(abs(t - 1260.0) < 1e-6 for t in fine.sim.t)

    def air_at(sim, t):
        i = int(np.argmin(np.abs(sim.t - t)))
        assert abs(sim.t[i] - t) < 1.0
        return sim.T_air[i, 0]
    # 1800 s and 3600 s lie on both grids, after the 1260 s switch
    assert air_at(coarse.sim, 1800) == pytest.approx(air_at(fine.sim, 1800), abs=0.15)
    assert air_at(coarse.sim, 3600) == pytest.approx(air_at(fine.sim, 3600), abs=0.15)
    # reporting grid itself stays on the requested step
    assert np.all(np.diff(coarse.sim.t) == pytest.approx(600.0))


def test_event_metrics_match_ledger_and_fine_comfort(mini_raw):
    """Off-grid switching: cumulative kWh matches the ledger, and comfort matches a fine grid."""
    from venuecool.runner import cumulative_electricity_kWh

    def prepare(r, step):
        r["simulation"]["warmup_days"] = 0
        r["simulation"]["end_h"] = 2
        r["simulation"]["output_step_s"] = step
        r["simulation"]["rtol"] = 1e-7
        r["schedules"]["offgrid"] = {"mode": "hours", "intervals": [[0.0, 0.35]]}
        for d in r["devices"]:
            if d["id"] == "evap_out":
                d["schedule"] = "offgrid"
    coarse = study_with(mini_raw, lambda r: prepare(r, 600.0)).run(["evap_out"], label="coarse")
    fine = study_with(mini_raw, lambda r: prepare(r, 60.0)).run(["evap_out"], label="fine")
    ledger = sum(x["electricity_kWh"] for x in coarse.sim.ledgers)
    cum = cumulative_electricity_kWh(coarse)
    assert cum[-1] == pytest.approx(ledger, rel=1e-9, abs=1e-9)
    assert cum[0] == pytest.approx(0.0)
    # 1260 s is not a reporting sample. Comfort splits 1200-1800 into 60 s on and 540 s off.
    assert not any(abs(tt - 1260.0) < 1e-6 for tt in coarse.sim.t)
    dt = coarse.sim.mid["dt"]
    on = coarse.sim.device_on[:, 0]
    assert np.any(np.isclose(dt, 60.0) & on)
    assert np.any(np.isclose(dt, 540.0) & ~on)
    assert np.all(np.diff(coarse.sim.t) == pytest.approx(600.0))
    assert coarse.summary["venue"]["occupied_person_hours"] == pytest.approx(
        fine.summary["venue"]["occupied_person_hours"], rel=1e-6)
    assert coarse.summary["venue"]["burden_Kph"] == pytest.approx(
        fine.summary["venue"]["burden_Kph"], rel=0.05, abs=0.5)


def test_boundary_state_converges_to_fine_grid(mini_raw):
    """The state at an off-grid switch is the ODE value, and it converges as the grid is refined."""
    def prepare(r, step):
        r["simulation"]["warmup_days"] = 0
        r["simulation"]["end_h"] = 2
        r["simulation"]["output_step_s"] = step
        r["simulation"]["rtol"] = 1e-7
        r["schedules"]["offgrid"] = {"mode": "hours", "intervals": [[0.0, 0.35]]}
        for d in r["devices"]:
            if d["id"] == "evap_out":
                d["schedule"] = "offgrid"

    coarse = study_with(mini_raw, lambda r: prepare(r, 600.0)).run(["evap_out"], label="coarse")
    mid = study_with(mini_raw, lambda r: prepare(r, 60.0)).run(["evap_out"], label="mid")
    fine = study_with(mini_raw, lambda r: prepare(r, 30.0)).run(["evap_out"], label="fine")
    assert np.all(np.diff(coarse.sim.t) == pytest.approx(600.0))
    assert not any(abs(tt - 1260.0) < 1e-6 for tt in coarse.sim.t)

    def knot_state(sim, t):
        i = int(np.argmin(np.abs(sim.mid["knot_t"] - t)))
        assert abs(float(sim.mid["knot_t"][i]) - t) < 1e-6
        return float(sim.mid["knot_T_air"][i, 0]), float(sim.mid["knot_W"][i, 0])

    def report_state(sim, t):
        i = int(np.argmin(np.abs(sim.t - t)))
        assert abs(float(sim.t[i]) - t) < 1.0
        return float(sim.T_air[i, 0]), float(sim.W[i, 0])

    t_k, w_k = knot_state(coarse.sim, 1260.0)
    t_fine, w_fine = report_state(fine.sim, 1260.0)
    t_mid, w_mid = report_state(mid.sim, 1260.0)
    t0, w0 = report_state(coarse.sim, 1200.0)
    t1, w1 = report_state(coarse.sim, 1800.0)
    frac = (1260.0 - 1200.0) / (1800.0 - 1200.0)
    t_lin = t0 + frac * (t1 - t0)
    w_lin = w0 + frac * (w1 - w0)
    # The stored boundary state matches the refined solution, and refining further does not move it.
    assert t_k == pytest.approx(t_fine, abs=0.05)
    assert w_k == pytest.approx(w_fine, abs=2e-5)
    assert t_mid == pytest.approx(t_fine, abs=0.02)
    assert abs(t_k - t_fine) < abs(t_lin - t_fine) or abs(w_k - w_fine) < abs(w_lin - w_fine)
    # Comfort on the coarse grid agrees with the fine grid once the boundary state is exact.
    assert coarse.summary["venue"]["burden_Kph"] == pytest.approx(
        fine.summary["venue"]["burden_Kph"], rel=0.05, abs=0.5)


def test_numerical_refinement(mini_raw):
    from venuecool.runner import Study
    res = {}
    for rtol, step in [(1e-5, 600.0), (1e-8, 150.0)]:
        s = Study(mini_raw, sim_overrides=dict(rtol=rtol, output_step_s=step))
        b = s.run_scenario("baseline")
        e = s.run_scenario("evap")
        res[rtol] = (b.summary["venue"]["burden_Kph"], metrics.compare(b.summary, e.summary, b.costs, e.costs)["relief_Kph"],
                     b.sim.T_air.max())
    (b1, r1, t1), (b2, r2, t2) = res[1e-5], res[1e-8]
    assert b1 == pytest.approx(b2, rel=0.01)
    assert r1 == pytest.approx(r2, rel=0.01)
    assert t1 == pytest.approx(t2, abs=0.02)


def test_warmup_reaches_periodic_state(mini_raw):
    from venuecool.runner import Study
    s = Study(mini_raw, sim_overrides=dict(warmup_days=3))
    o = s.run_scenario("baseline")
    # constant-forcing day repeated: start and end of the reporting day nearly coincide
    np.testing.assert_allclose(o.sim.T_nodes[0], o.sim.T_nodes[-1], atol=0.3)


def test_search_reports_statuses_and_no_feasible(tmp_path, mini_raw):
    from venuecool.runner import Study
    from venuecool.search import run_search
    s = Study(mini_raw)
    r = run_search(s, tmp_path, workers=1, progress=lambda *a: None)
    assert r["n_candidates"] == 3 and sum(r["status_counts"].values()) == 3
    assert (tmp_path / "search_results.csv").exists()
    raw2 = dict(mini_raw)
    raw2["search"] = dict(mini_raw["search"], target={"min_attainment": 1.0, "max_unassessed_fraction": 0.0})
    r2 = run_search(Study(raw2), tmp_path / "b", workers=1, verify=False, progress=lambda *a: None)
    assert r2["selected"] is None and "NO TESTED CONFIGURATION" in r2["message"]


def test_matched_uncertainty_draws(mini_study):
    from venuecool.uncertainty import lhs_draws, uncertain_parameters
    params = uncertain_parameters(mini_study)
    d1 = lhs_draws(params, 4, 7)
    d2 = lhs_draws(params, 4, 7)
    assert d1 == d2  # seeded
    for path, qd in params:
        lo, hi = qd["bounds"]
        assert all(lo <= d[path] <= hi for d in d1)
