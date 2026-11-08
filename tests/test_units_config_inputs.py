"""Software checks: units, provenance, configuration validation, input-data handling."""

import copy

import numpy as np
import pytest
from pydantic import ValidationError

from conftest import MINI, q
from venuecool.config import VenueConfig, expand_templates, find_uncertain, load_raw, set_path
from venuecool.inputs import InputDataError, load_weather
from venuecool.quantity import Quantity


def test_unit_conversion_and_rejection():
    assert Quantity(value=0.25, unit="kW", source="x").to("W").value == pytest.approx(250.0)
    assert Quantity(value=3600, unit="m3/h", source="x").to("m3/s").value == pytest.approx(1.0)
    assert Quantity(value=50, unit="%", source="x").to("-").value == pytest.approx(0.5)
    with pytest.raises(ValueError):
        Quantity(value=1, unit="kWh", source="x").to("W")  # energy is not power
    with pytest.raises(ValidationError):
        Quantity(value=1, unit="W", source="  ")  # provenance required
    with pytest.raises(ValidationError):
        Quantity(value=5, unit="W", source="x", bounds=(0, 1))  # nominal outside bounds


def test_schema_rejects_wrong_units_and_unknown_fields(mini_raw):
    r = copy.deepcopy(mini_raw)
    r["device_presets"]["fan_system"]["electric_power"] = q(0.25, "kWh")
    with pytest.raises(ValidationError, match="kWh"):
        VenueConfig.model_validate(r)
    r = copy.deepcopy(mini_raw)
    r["sections"][0]["roof_colour"] = "white"
    with pytest.raises(ValidationError):
        VenueConfig.model_validate(r)


def test_device_airflow_fractions_must_reconcile(mini_raw):
    r = copy.deepcopy(mini_raw)
    dev = next(d for d in r["devices"] if d["id"] == "shared")
    dev["serves"][1]["airflow_fraction"] = 0.6  # 0.6 + 0.6
    with pytest.raises(ValidationError, match="reconcile"):
        VenueConfig.model_validate(r)
    r = copy.deepcopy(mini_raw)
    r["devices"].append(copy.deepcopy(r["devices"][0]))
    with pytest.raises(ValidationError, match="duplicate device"):
        VenueConfig.model_validate(r)


def test_template_sections_keep_their_own_identity(mini_raw):
    r = copy.deepcopy(mini_raw)
    r["sections"].append(dict(copy.deepcopy(r["sections"][0]), id="A2", name="second A"))
    r = expand_templates(r)
    set_path(r, "sections[A].surfaces[roof].solar_absorptance_exterior.value", 0.6)
    cfg = VenueConfig.model_validate(r)
    a, a2 = cfg.section("A"), cfg.section("A2")
    roof = lambda s: next(x for x in s.surfaces if x.id == "roof")  # noqa: E731
    assert roof(a).solar_absorptance_exterior.value == 0.6
    assert roof(a2).solar_absorptance_exterior.value == 0.7
    assert a.template == a2.template == "roofed_open_seating"


def test_value_outside_template_bounds_is_rejected(mini_raw):
    r = copy.deepcopy(mini_raw)
    set_path(r, "sections[A].surfaces[roof].solar_absorptance_exterior.value", 0.2)
    with pytest.raises(ValidationError, match="outside bounds"):
        VenueConfig.model_validate(r)
    # replacing the whole quantity (new provenance, no bounds) is allowed and is not re-bounded
    r = copy.deepcopy(mini_raw)
    set_path(r, "sections[A].surfaces[roof].solar_absorptance_exterior", q(0.2, "-", "white coating"))
    cfg = VenueConfig.model_validate(expand_templates(r))
    assert next(x for x in cfg.section("A").surfaces if x.id == "roof").solar_absorptance_exterior.bounds is None


def test_uncertain_parameters_have_units_and_sources(mini_raw):
    found = find_uncertain(mini_raw)
    assert len(found) > 10
    for path, qd in found:
        assert qd["unit"] and qd["source"].strip() and len(qd["bounds"]) == 2


def test_uncertain_passive_option_values_are_drawable():
    from venuecool.runner import Study
    from pathlib import Path
    study = Study(Path(__file__).resolve().parents[1] / "examples" / "bus_stop.yaml")
    paths = [p for p, _ in find_uncertain(study.raw) if p.startswith("passive_options[")]
    assert "passive_options[cool_roof].set::sections[shelter].surfaces[roof].solar_absorptance_exterior.value" in paths
    for p in paths:  # every reported path must be settable
        set_path(copy.deepcopy(study.raw), p, 0.3)
    r = study.raw_for(passive=["cool_roof"], draw={paths[0]: 0.31})
    roof = next(s for s in next(x for x in r["sections"] if x["id"] == "shelter")["surfaces"] if s["id"] == "roof")
    assert roof["solar_absorptance_exterior"]["value"] == 0.31
    from venuecool.uncertainty import unused_by
    assert unused_by(study, "A_fan", paths[0])                      # fan scenario has no cool roof
    assert unused_by(study, "A_fan", "device_presets.evaporative_cooler.airflow.value")
    assert not unused_by(study, "B_evap", "device_presets.evaporative_cooler.airflow.value")
    assert not unused_by(study, "A_fan", "sections[shelter].occupancy.clo.value")
    from venuecool.uncertainty import unused_by
    assert unused_by(study, "A_fan", paths[0])                      # fan scenario has no cool roof
    assert unused_by(study, "A_fan", "device_presets.evaporative_cooler.airflow.value")
    assert not unused_by(study, "B_evap", "device_presets.evaporative_cooler.airflow.value")
    assert not unused_by(study, "A_fan", "sections[shelter].occupancy.clo.value")


def test_weather_missing_columns_and_nans_raise(tmp_path, mini_study):
    site = mini_study.cfg0.site
    p = tmp_path / "w.csv"
    p.write_text("time_h,temp_C,rh_pct,pressure_Pa,wind_ms\n0,30,40,101325,1\n1,31,40,101325,1\n")
    with pytest.raises(InputDataError, match="missing required"):
        load_weather(p, site)
    p.write_text("time_h,temp_C,rh_pct,pressure_Pa,wind_ms,ghi_Wm2\n0,30,40,101325,1,0\n1,31,40,101325,1,10\n")
    with pytest.raises(InputDataError, match="dni_Wm2/dhi_Wm2 missing"):
        load_weather(p, site)  # no silent decomposition
    w = load_weather(p, site, solar_split="erbs")
    assert "DERIVED" in w.meta["solar_split"]
    p.write_text("time_h,temp_C,rh_pct,pressure_Pa,wind_ms,ghi_Wm2,dni_Wm2,dhi_Wm2\n0,30,,101325,1,0,0,0\n1,31,40,101325,1,0,0,0\n")
    with pytest.raises(InputDataError, match="NaN"):
        load_weather(p, site)
    with pytest.raises(InputDataError, match="not found"):
        load_weather(tmp_path / "missing.csv", site)


def test_weather_must_cover_run(tmp_path, mini_raw):
    from venuecool.runner import Study
    r = copy.deepcopy(mini_raw)
    p = tmp_path / "short.csv"
    p.write_text("time_h,temp_C,rh_pct,pressure_Pa,wind_ms,ghi_Wm2,dni_Wm2,dhi_Wm2\n" +
                 "\n".join(f"{h},30,40,101325,1,0,0,0" for h in range(12)) + "\n")
    r["weather"]["file"] = str(p)
    s = Study(r)
    with pytest.raises(ValueError, match="weather covers"):
        s.run_scenario("baseline")


def test_load_raw_include_merges():
    raw = load_raw(MINI)
    assert "fan_system" in raw["device_presets"] and "templates" in raw
