"""Moist-air psychrometrics (ASHRAE Fundamentals 2017, Ch. 1).

Conserved air states are the humidity ratio W [kg_w/kg_da] and the moist-air specific
enthalpy h [J/kg_da]. Temperature and relative humidity are always *derived*.
Relative humidity (RH, 0-1 fraction internally) is never used as a state variable.

No function clips values: callers decide what to do with supersaturated or negative
states (see :func:`state_validity`).
"""

from __future__ import annotations

import numpy as np

from .constants import CP_DA, CP_V, CP_W, EPS_W, H_FG0, P_STD, R_DA, T0K


def p_ws(t_c):
    """Saturation vapour pressure [Pa] over liquid water (t>=0) or ice (t<0); Hyland-Wexler."""
    t = np.asarray(t_c, dtype=float) + T0K
    # over ice (ASHRAE eq. 5)
    c1, c2, c3, c4, c5, c6, c7 = (-5.6745359e03, 6.3925247e00, -9.6778430e-03, 6.2215701e-07,
                                  2.0747825e-09, -9.4840240e-13, 4.1635019e00)
    ln_ice = c1 / t + c2 + c3 * t + c4 * t**2 + c5 * t**3 + c6 * t**4 + c7 * np.log(t)
    # over water (ASHRAE eq. 6)
    c8, c9, c10, c11, c12, c13 = (-5.8002206e03, 1.3914993e00, -4.8640239e-02, 4.1764768e-05,
                                  -1.4452093e-08, 6.5459673e00)
    ln_w = c8 / t + c9 + c10 * t + c11 * t**2 + c12 * t**3 + c13 * np.log(t)
    return np.exp(np.where(t >= T0K, ln_w, ln_ice))


def w_from_pw(p_w, p=P_STD):
    return EPS_W * p_w / (p - p_w)


def pw_from_w(w, p=P_STD):
    w = np.asarray(w, dtype=float)
    return p * w / (EPS_W + w)


def w_sat(t_c, p=P_STD):
    return w_from_pw(p_ws(t_c), p)


def w_from_rh(t_c, rh, p=P_STD):
    """Humidity ratio from temperature [degC] and relative humidity *fraction* (0-1)."""
    return w_from_pw(np.asarray(rh) * p_ws(t_c), p)


def rh_from_w(t_c, w, p=P_STD):
    """Relative humidity fraction (may exceed 1 for supersaturated states; not clipped)."""
    return pw_from_w(w, p) / p_ws(t_c)


def enthalpy(t_c, w):
    """Moist-air enthalpy [J/kg_da], datum: dry air at 0 degC and liquid water at 0 degC."""
    t_c = np.asarray(t_c, dtype=float)
    return CP_DA * t_c + np.asarray(w) * (H_FG0 + CP_V * t_c)


def t_from_hw(h, w):
    """Dry-bulb temperature [degC] from enthalpy and humidity ratio (exact inverse of enthalpy)."""
    w = np.asarray(w, dtype=float)
    return (np.asarray(h, dtype=float) - H_FG0 * w) / (CP_DA + CP_V * w)


def h_vapour(t_c):
    """Specific enthalpy of water vapour [J/kg_w] at t_c (same datum as :func:`enthalpy`)."""
    return H_FG0 + CP_V * np.asarray(t_c, dtype=float)


def h_liquid(t_c):
    return CP_W * np.asarray(t_c, dtype=float)


def rho_da(t_c, w, p=P_STD):
    """Dry-air density [kg_da per m3 of moist air]."""
    p_da = p - pw_from_w(w, p)
    return p_da / (R_DA * (np.asarray(t_c, dtype=float) + T0K))


def _wb_resid(ts, h_in, w, p):
    """Adiabatic-saturation residual using the same enthalpy constants as :func:`enthalpy`:
    h(t*, Ws(t*)) - [h_in + (Ws(t*) - W) * h_liquid(t*)]  (increasing in t*)."""
    ws = w_sat(ts, p)
    return enthalpy(ts, ws) - h_in - (ws - w) * CP_W * ts


def wet_bulb(t_c, w, p=P_STD, tol=1e-7, max_iter=80):
    """Thermodynamic (adiabatic-saturation) wet-bulb temperature [degC], vectorised bisection."""
    t = np.atleast_1d(np.asarray(t_c, dtype=float))
    w = np.broadcast_to(np.asarray(w, dtype=float), t.shape)
    p = np.broadcast_to(np.asarray(p, dtype=float), t.shape)
    h_in = enthalpy(t, w)
    lo = np.full_like(t, -40.0)
    hi = t + 1e-9
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        r = _wb_resid(mid, h_in, w, p)
        hi = np.where(r > 0, mid, hi)
        lo = np.where(r > 0, lo, mid)
        if np.max(hi - lo) < tol:
            break
    out = 0.5 * (lo + hi)
    return out if np.ndim(t_c) else float(out[0])


def wet_bulb_fast(t_c, w, p=P_STD, n_iter=8):
    """Secant-method wet bulb for use inside the ODE right-hand side (same definition)."""
    t = np.asarray(t_c, dtype=float)
    w = np.asarray(w, dtype=float)
    h_in = enthalpy(t, w)
    a = t - 0.6 * np.maximum(t - dew_point_fast(w, p), 0.0)
    b = a + 0.3
    fa = _wb_resid(a, h_in, w, p)
    fb = _wb_resid(b, h_in, w, p)
    for _ in range(n_iter):
        d = fb - fa
        c = np.where(np.abs(d) > 1e-12, b - fb * (b - a) / np.where(np.abs(d) > 1e-12, d, 1.0), b)
        a, fa = b, fb
        b, fb = c, _wb_resid(c, h_in, w, p)
    return b


def dew_point_fast(w, p=P_STD):
    """Magnus-form dew point (initial guess only)."""
    pw = np.maximum(pw_from_w(w, p), 1e-3)
    g = np.log(pw / 610.94)
    return 243.04 * g / (17.625 - g)


def dew_point(w, p=P_STD):
    """Dew-point temperature [degC] by bisection on p_ws(t_dp) = p_w."""
    pw = np.atleast_1d(pw_from_w(w, p))
    lo = np.full_like(pw, -60.0)
    hi = np.full_like(pw, 80.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = p_ws(mid) > pw
        hi = np.where(f, mid, hi)
        lo = np.where(f, lo, mid)
    out = 0.5 * (lo + hi)
    return out if np.ndim(w) else float(out[0])


def state_validity(t_c, w, p=P_STD, t_min=-40.0, t_max=70.0, rh_tol=1e-3):
    """Return a dict of boolean arrays flagging physically invalid / unsupported air states."""
    t_c = np.asarray(t_c, dtype=float)
    w = np.asarray(w, dtype=float)
    rh = rh_from_w(t_c, w, p)
    return {
        "negative_humidity": w < 0,
        "supersaturated": rh > 1.0 + rh_tol,
        "temperature_out_of_range": (t_c < t_min) | (t_c > t_max),
        "non_finite": ~np.isfinite(t_c) | ~np.isfinite(w),
    }
