"""Solar geometry (NOAA/Spencer approximations) and optional Erbs diffuse split.

Angles in degrees; azimuth measured clockwise from north.
"""

from __future__ import annotations

import numpy as np


def solar_position(hour_local, day_of_year, lat_deg, lon_deg, utc_offset_h):
    """Return (altitude_deg, azimuth_deg) for local standard clock hours (array-like)."""
    h = np.asarray(hour_local, dtype=float)
    g = 2 * np.pi / 365.0 * (day_of_year - 1 + (h - 12) / 24.0)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))  # minutes
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = h * 60 + eot + 4 * lon_deg - 60 * utc_offset_h  # true solar time, minutes
    ha = np.radians(tst / 4.0 - 180.0)
    lat = np.radians(lat_deg)
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1, 1)
    zen = np.arccos(cosz)
    alt = 90 - np.degrees(zen)
    sinz = np.maximum(np.sin(zen), 1e-9)
    cos_az = (np.sin(decl) - np.sin(lat) * cosz) / (np.cos(lat) * sinz)
    az = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
    az = np.where(ha > 0, 360 - az, az)
    return alt, az


def extraterrestrial_normal(day_of_year):
    return 1361.0 * (1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365.0))


def erbs_split(ghi, alt_deg, day_of_year):
    """Erbs et al. (1982) decomposition GHI -> (DNI, DHI). Derived, not measured."""
    ghi = np.asarray(ghi, dtype=float)
    sin_alt = np.sin(np.radians(np.asarray(alt_deg, dtype=float)))
    i0h = extraterrestrial_normal(day_of_year) * np.maximum(sin_alt, 0)
    kt = np.where(i0h > 10, ghi / np.maximum(i0h, 1e-9), 0.0)
    kd = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
                           0.165))
    dhi = kd * ghi
    dni = np.where(sin_alt > 0.05, (ghi - dhi) / np.maximum(sin_alt, 0.05), 0.0)
    return np.maximum(dni, 0), dhi


def vertical_beam_factor(alt_deg, az_deg, surface_az_deg):
    """cos(incidence) for a vertical surface with outward normal azimuth (0 if sun behind)."""
    alt = np.radians(alt_deg)
    c = np.cos(alt) * np.cos(np.radians(np.asarray(az_deg) - surface_az_deg))
    return np.where(np.asarray(alt_deg) > 0, np.maximum(c, 0), 0.0)
