"""Physics checks: psychrometrics and evaporative / fan device thermodynamics."""

import numpy as np
import pytest

from venuecool import psychro
from venuecool.constants import CP_W
from venuecool.devices import evaporative_pad, outlet_state


def test_saturation_pressure_reference_values():
    # ASHRAE Fundamentals 2017 Table 3: 20 C -> 2.3392 kPa, 30 C -> 4.2467 kPa, -10 C (ice) -> 0.2601 kPa
    assert psychro.p_ws(20.0) == pytest.approx(2339.2, rel=1e-3)
    assert psychro.p_ws(30.0) == pytest.approx(4246.7, rel=1e-3)
    assert psychro.p_ws(-10.0) == pytest.approx(260.1, rel=2e-3)


def test_state_roundtrips():
    t = np.array([10.0, 25.0, 41.0])
    w = psychro.w_from_rh(t, np.array([0.9, 0.5, 0.1]))
    np.testing.assert_allclose(psychro.rh_from_w(t, w), [0.9, 0.5, 0.1], rtol=1e-12)
    h = psychro.enthalpy(t, w)
    np.testing.assert_allclose(psychro.t_from_hw(h, w), t, rtol=1e-12)


def test_wet_bulb_reference_and_consistency():
    # 30 C / 50% RH at sea level: wet bulb ~ 22.0 C (psychrometric chart)
    w = psychro.w_from_rh(30.0, 0.5)
    assert psychro.wet_bulb(30.0, w) == pytest.approx(22.0, abs=0.15)
    t = np.array([25.0, 33.0, 41.0])
    w = psychro.w_from_rh(t, np.array([0.7, 0.6, 0.1]))
    np.testing.assert_allclose(psychro.wet_bulb_fast(t, w), psychro.wet_bulb(t, w), atol=1e-4)


def test_supersaturation_is_flagged_not_clipped():
    w = 1.2 * psychro.w_sat(30.0)
    assert psychro.rh_from_w(30.0, w) > 1.0
    flags = psychro.state_validity(np.array([30.0]), np.array([w]))
    assert flags["supersaturated"][0]
    assert psychro.state_validity(np.array([30.0]), np.array([-1e-3]))["negative_humidity"][0]


@pytest.mark.parametrize("t,rh", [(41.0, 0.11), (33.0, 0.6), (28.0, 0.85)])
def test_evaporative_pad_respects_wet_bulb_and_energy(t, rh):
    w = psychro.w_from_rh(t, rh)
    for eff in (0.0, 0.5, 0.8, 1.0):
        t_pad, w_pad, t_wb = evaporative_pad(np.array([t]), np.array([w]), 101325.0, eff, fast=False)
        assert t_pad[0] >= t_wb[0] - 1e-9
        assert t_pad[0] == pytest.approx(t - eff * (t - t_wb[0]))
        assert w_pad[0] >= w - 1e-15
        assert psychro.rh_from_w(t_pad[0], w_pad[0]) <= 1.0 + 1e-6
        # pad energy balance: h_out = h_in + (W_out - W_in) * h_liquid(T_water = T_wb)
        lhs = psychro.enthalpy(t_pad[0], w_pad[0])
        rhs = psychro.enthalpy(t, w) + (w_pad[0] - w) * CP_W * t_wb[0]
        assert lhs == pytest.approx(rhs, rel=1e-12)
    # eff = 1 reaches saturation at the wet bulb
    t_pad, w_pad, t_wb = evaporative_pad(np.array([t]), np.array([w]), 101325.0, 1.0, fast=False)
    assert psychro.rh_from_w(t_pad[0], w_pad[0]) == pytest.approx(1.0, abs=1e-4)


def test_evaporative_cooling_weaker_in_humid_air():
    dry = 41.0 - evaporative_pad(np.array([41.0]), psychro.w_from_rh(np.array([41.0]), 0.11), 101325.0, 0.75)[0][0]
    humid = 33.0 - evaporative_pad(np.array([33.0]), psychro.w_from_rh(np.array([33.0]), 0.6), 101325.0, 0.75)[0][0]
    assert dry > 2.5 * humid > 0


def test_fan_outlet_adds_only_motor_heat():
    h = psychro.enthalpy(35.0, 0.01)
    h_o, w_o, ev = outlet_state(np.array([h]), np.array([0.01]), 101325.0, np.array([False]), np.array([0.0]),
                                np.array([250.0]), np.array([2.0]))
    assert h_o[0] - h == pytest.approx(125.0)  # 250 W / 2 kg/s
    assert w_o[0] == 0.01 and ev[0] == 0.0
    assert psychro.t_from_hw(h_o, w_o)[0] > 35.0  # never colder: no refrigeration term
