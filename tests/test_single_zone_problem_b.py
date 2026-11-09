"""Focused checks for the single-zone HiMCM Problem B bus-shelter slice.

Reuses the existing psychro / fan / wet-bulb suite; this file only covers the one-section
example, the four Problem B scenarios, and cumulative-energy bookkeeping.
"""

from pathlib import Path

import numpy as np
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
    from venuecool.robustness import preferred_scenario
    assert preferred_scenario({"A_fan": 10.0, "B_evap": 40.0, "C_large_evap": 30.0}) == "B_evap"
    assert preferred_scenario({"A_fan": -1.0, "B_evap": 0.0}) == "baseline"


def test_timeseries_includes_cumulative_kwh(runs):
    df = timeseries_frame(runs["A_fan"])
    assert "electricity_kWh_cum" in df.columns
    assert df["electricity_kWh_cum"].iloc[0] == pytest.approx(0.0)
    assert df["electricity_kWh_cum"].iloc[-1] > 0
