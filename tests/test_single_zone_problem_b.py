"""Focused checks for the single-zone HiMCM Problem B bus-shelter slice.

Reuses the existing psychro / fan / wet-bulb suite; this file only covers the one-section
example, the four Problem B scenarios, and cumulative-energy bookkeeping.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from venuecool import metrics, psychro
from venuecool.quantity import Quantity
from venuecool.runner import Study, cumulative_electricity_kWh, timeseries_frame

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "examples" / "bus_shelter_single.yaml"


@pytest.fixture(scope="module")
def study():
    return Study(CFG)


@pytest.fixture(scope="module")
def runs(study):
    base = study.run_scenario("baseline")
    fan = study.run_scenario("A_fan")
    evap = study.run_scenario("B_evap")
    large = study.run_scenario("C_large_evap")
    assert all(o.ok for o in (base, fan, evap, large))
    return dict(baseline=base, A_fan=fan, B_evap=evap, C_large_evap=large)


def test_equipment_evidence_has_no_measured_inputs(study):
    from venuecool.robustness import evidence_table
    rows = evidence_table(study.cfg0)
    assert rows
    assert {r["evidence"] for r in rows} == {"published", "assumed"}
    published = [r for r in rows if r["evidence"] == "published"]
    assert {r["field"] for r in published} == {"electric_power"}
    plume = next(r for r in rows if r["preset"] == "evaporative_cooler" and r["field"] == "plume_fraction")
    assert plume["evidence"] == "assumed"


def test_single_zone_config_shape(study):
    cfg = study.cfg0
    assert len(cfg.sections) == 1
    assert cfg.sections[0].id == "shelter"
    assert cfg.connections == []
    assert {s.id for s in cfg.scenarios} == {"baseline", "A_fan", "B_evap", "C_large_evap"}
    assert study.weather.meta.get("data_status", "").startswith("SYNTHETIC")
    assert Quantity(value=0.25, unit="kW", source="t").to("W").value == pytest.approx(250.0)
    assert cfg.device_presets["fan_system"].electric_power.to("W").value == pytest.approx(250.0)
    assert cfg.device_presets["evaporative_cooler"].electric_power.to("W").value == pytest.approx(1200.0)
    assert cfg.device_presets["large_evaporative_cooler"].electric_power.to("W").value == pytest.approx(2500.0)


def test_fan_raises_speed_without_cooling_air(runs):
    base, fan = runs["baseline"], runs["A_fan"]
    # no artificial refrigeration: mean section air is not colder than baseline
    assert fan.sim.T_air.mean() >= base.sim.T_air.mean() - 1e-6
    on = fan.sim.device_on[:, 0]
    assert on.any()
    # covered occupants see higher air speed than the matched baseline population
    bf = base.comfort_df
    ff = fan.comfort_df
    assert ff.loc[ff.served, "v"].mean() > bf["v"].mean()
    led = fan.sim.ledgers[0]
    assert led["electricity_kWh"] == pytest.approx(0.25 * led["on_hours"], rel=1e-9)
    cum = cumulative_electricity_kWh(fan)
    assert cum[-1] == pytest.approx(led["electricity_kWh"], rel=1e-3, abs=1e-3)


def test_evaporative_cools_and_humidifies_within_wet_bulb(runs):
    base, evap = runs["baseline"], runs["B_evap"]
    on = evap.sim.device_on[:, 0]
    assert on.any()
    assert np.all(evap.sim.mid["T_air"][on, 0] < base.sim.mid["T_air"][on, 0])
    # RH rises while the cooler runs (moisture added to dry synthetic air)
    assert evap.sim.RH[1:][on].mean() > base.sim.RH[1:][on].mean()
    t_wb = psychro.wet_bulb(evap.sim.mid["To"][on], evap.sim.mid["Wo"][on], evap.sim.mid["p"][on])
    t_out = psychro.t_from_hw(evap.sim.device_outlet_h[on, 0], evap.sim.device_outlet_W[on, 0])
    assert np.all(t_out >= t_wb - 1e-6)
    led = evap.sim.ledgers[0]
    assert led["electricity_kWh"] == pytest.approx(1.2 * led["on_hours"], rel=1e-9)


def test_large_cooler_uses_more_energy_and_gives_more_relief(runs):
    base, b, c = runs["baseline"], runs["B_evap"], runs["C_large_evap"]
    assert c.costs["electricity_kWh"] > b.costs["electricity_kWh"]
    cb = metrics.compare(base.summary, b.summary, base.costs, b.costs)
    cc = metrics.compare(base.summary, c.summary, base.costs, c.costs)
    assert cc["relief_Kph"] > cb["relief_Kph"]


def test_set_target_changes_burden_not_applicability(runs):
    from venuecool import metrics
    base = runs["baseline"]
    low = metrics.summarize(base.comfort_df, base.cfg, target=28.0)
    high = metrics.summarize(base.comfort_df, base.cfg, target=32.0)
    assert low["venue"]["burden_Kph"] > high["venue"]["burden_Kph"]
    assert low["venue"]["supported_person_hours"] == pytest.approx(high["venue"]["supported_person_hours"])
    assert low["venue"]["extrapolated_person_hours"] == pytest.approx(high["venue"]["extrapolated_person_hours"])


def test_preferred_system_rule():
    from venuecool.robustness import classify_criteria, evidence_class, preferred_scenario
    assert preferred_scenario({"A_fan": 10.0, "B_evap": 40.0, "C_large_evap": 30.0}) == "B_evap"
    assert preferred_scenario({"A_fan": -1.0, "B_evap": 0.0}) == "baseline"
    rows = [
        dict(scenario="baseline", relief_Kph=0.0, electricity_kWh=0.0, attainment=0.2),
        dict(scenario="A_fan", relief_Kph=10.0, electricity_kWh=4.0, attainment=0.4),
        dict(scenario="B_evap", relief_Kph=30.0, electricity_kWh=19.0, attainment=0.7),
        dict(scenario="C_large_evap", relief_Kph=40.0, electricity_kWh=40.0, attainment=0.8),
    ]
    got = classify_criteria(rows, attainment_bar=0.6)
    assert got["greatest_relief"] == "C_large_evap"
    assert got["relief_per_kWh"] == "A_fan"  # 10/4 > 30/19 > 40/40
    assert got["min_energy_meeting_target"] == "B_evap"
    assert "C_large_evap" in got["pareto"] and "baseline" in got["pareto"]
    short = [dict(scenario="baseline", relief_Kph=0.0, electricity_kWh=0.0, attainment=0.2),
             dict(scenario="A_fan", relief_Kph=5.0, electricity_kWh=4.0, attainment=0.3)]
    assert classify_criteria(short, attainment_bar=0.6)["min_energy_meeting_target"] == "infeasible"
    # The uncooled shelter is the minimum-energy strategy when it already meets the target.
    comfortable = [
        dict(scenario="baseline", relief_Kph=0.0, electricity_kWh=0.0, attainment=0.8),
        dict(scenario="A_fan", relief_Kph=10.0, electricity_kWh=4.0, attainment=0.9),
    ]
    assert classify_criteria(comfortable, attainment_bar=0.6)["min_energy_meeting_target"] == "baseline"
    # Nonpositive relief is not an efficiency gain. Doing nothing ranks above spending energy to do worse.
    worse = [
        dict(scenario="baseline", relief_Kph=0.0, electricity_kWh=0.0, attainment=0.2),
        dict(scenario="A_fan", relief_Kph=-5.0, electricity_kWh=4.0, attainment=0.1),
        dict(scenario="B_evap", relief_Kph=-1.0, electricity_kWh=19.0, attainment=0.15),
    ]
    worse_got = classify_criteria(worse, attainment_bar=0.6)
    assert worse_got["relief_per_kWh"] == "baseline"
    assert worse_got["greatest_relief"] == "baseline"
    assert worse_got["min_energy_meeting_target"] == "infeasible"
    assert worse_got["pareto"] == ["baseline"]
    assert evidence_class("SUPPLIED: problem statement table") == "published"
    assert evidence_class("DEMO-ASSUMPTION: typical pad") == "assumed"
    assert evidence_class("measured on site") == "measured"
    # A future-work note must not reclassify an explicit assumption as measured.
    note = ("DEMO-ASSUMPTION: share of supply air in the air breathed in the covered zone "
            "(replace with CFD/measurement)")
    assert evidence_class(note) == "assumed"


def test_strict_hours_match_across_alternatives():
    """Strict relief uses only person-intervals supported for every cooling alternative."""
    from venuecool.metrics import applicability_hours, common_supported_frames, summarize

    def frame(flags, strict_et):
        rows = []
        for k, supported in enumerate(flags):
            rows.append(dict(
                section="shelter", group="all", sun_state="shade", k=k, t_h=k + 0.5,
                person_hours=10.0, cls="supported" if supported else "extrapolated",
                applicability="supported" if supported else "extrapolated",
                set_strict=strict_et if supported else np.nan, set_extended=strict_et,
                et=strict_et, assessed=True, served=False))
        return pd.DataFrame(rows)

    frames = {
        "baseline": frame([True, True], 35.0),
        "A_fan": frame([True, False], 31.0),
        "C_large_evap": frame([True, True], 29.0),
    }
    aligned = common_supported_frames(frames)
    for df in aligned.values():
        assert df["assessed"].tolist() == [True, False]
    # Class labels stay with the scenario. The fan's second hour is extrapolated;
    # the same hour on the baseline is still supported, and is excluded only from the shared set.
    assert aligned["A_fan"]["applicability"].tolist() == ["supported", "extrapolated"]
    assert aligned["baseline"]["applicability"].tolist() == ["supported", "supported"]

    class Cfg:
        class comfort:
            poorly_served_attainment = 0.5

    sums = {name: summarize(df, Cfg, target=30.0) for name, df in aligned.items()}
    hours = [s["venue"]["assessed_person_hours"] for s in sums.values()]
    assert hours == pytest.approx([10.0, 10.0, 10.0])
    # The fan's second hour stays visible as extrapolated, not as supported or unassessed.
    classes = applicability_hours(aligned["A_fan"])
    assert classes["supported"] == pytest.approx(10.0)
    assert classes["extrapolated"] == pytest.approx(10.0)
    assert classes["unassessed"] == pytest.approx(0.0)
    assert applicability_hours(aligned["baseline"]) == pytest.approx(
        {"supported": 20.0, "extrapolated": 0.0, "unassessed": 0.0})
    relief = metrics.compare(sums["baseline"], sums["A_fan"],
                             {"electricity_kWh": 0.0, "season_total_cost_usd": 0.0, "season_scale": 1.0},
                             {"electricity_kWh": 4.0, "season_total_cost_usd": 1.0, "season_scale": 1.0})
    # Shared hour only: baseline exceedance 5 K, fan exceedance 1 K, 10 person-hours.
    assert relief["relief_Kph"] == pytest.approx(40.0)
    assert relief["unmatched_person_hours"] == pytest.approx(10.0)


def test_applicability_partitions_person_hours(runs):
    from venuecool.metrics import applicability_hours
    for o in runs.values():
        hours = applicability_hours(o.comfort_df)
        occupied = o.summary["venue"]["occupied_person_hours"]
        assert sum(hours.values()) == pytest.approx(occupied)
        assert set(o.comfort_df["applicability"].unique()) <= {"supported", "extrapolated", "unassessed"}


def test_timeseries_includes_cumulative_kwh(runs):
    df = timeseries_frame(runs["A_fan"])
    assert "electricity_kWh_cum" in df.columns
    assert df["electricity_kWh_cum"].iloc[0] == pytest.approx(0.0)
    assert df["electricity_kWh_cum"].iloc[-1] > 0
