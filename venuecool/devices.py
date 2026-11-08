"""Cooling devices as 'air handlers'.

Every device moves a dry-air mass flow m [kg_da/s] from an intake (outdoor air, or a recirculated
mixture of the sections it serves) through an optional process to its outlet, then delivers the
outlet air to served sections in fixed airflow fractions f_k (sum f_k = 1, reconciled once).

* Fan (kind 'fan'): process = none. Electrical input P ends as heat (motor_heat_to_air * P) in the
  delivered air. There is NO refrigeration term; comfort benefit comes only through local air
  speed (covered_air_speed in covered exposure groups) and mixing.
* Direct evaporative cooler (kind 'evaporative'): adiabatic pad,
  T_pad = T_in - eps (T_in - T_wb,in); W_pad from the pad energy balance
  h_pad = h_in + (W_pad - W_in) h_liquid(T_water), with make-up water at the inlet wet-bulb
  temperature (labelled assumption). Outlet h_out = h_pad + motor heat / m.
  eps <= 1 guarantees T_pad >= T_wb and W_pad <= W_sat(T_pad).

The same function is used inside the ODE right-hand side and in post-processing, so the comfort
model sees exactly the outlet states that drove the energy balance.
"""

from __future__ import annotations

import numpy as np

from . import psychro
from .constants import CP_DA, CP_V, CP_W, H_FG0


def evaporative_pad(t_in, w_in, p, eff, fast=True):
    """Return (t_pad, w_pad, t_wb). Arrays broadcast."""
    t_wb = psychro.wet_bulb_fast(t_in, w_in, p) if fast else psychro.wet_bulb(t_in, w_in, p)
    t_pad = t_in - eff * (t_in - t_wb)
    h_in = psychro.enthalpy(t_in, w_in)
    hf = CP_W * t_wb
    w_pad = (h_in - w_in * hf - CP_DA * t_pad) / (H_FG0 + CP_V * t_pad - hf)
    return t_pad, w_pad, t_wb


def outlet_state(h_in, w_in, p, kind_evap, eff, heat_W, m_da, fast=True):
    """Outlet (h_out, w_out, evap_kg_per_s) for arrays of devices.

    kind_evap: boolean array, eff: saturation effectiveness (ignored for fans).
    """
    h_in = np.asarray(h_in, float)
    w_in = np.asarray(w_in, float)
    t_in = psychro.t_from_hw(h_in, w_in)
    h_out = h_in.copy()
    w_out = w_in.copy()
    if np.any(kind_evap):
        k = np.asarray(kind_evap, bool)
        pk = np.broadcast_to(p, h_in.shape)[k]
        t_pad, w_pad, _ = evaporative_pad(t_in[k], w_in[k], pk, np.asarray(eff)[k], fast=fast)
        h_out[k] = psychro.enthalpy(t_pad, w_pad)
        w_out[k] = w_pad
    h_out = h_out + np.asarray(heat_W) / np.maximum(m_da, 1e-12)
    evap = np.asarray(m_da) * (w_out - w_in)
    return h_out, w_out, evap
