"""Generate the clearly labelled SYNTHETIC demonstration weather and occupancy CSVs.

These are NOT measurements. They are smooth design-day profiles chosen to represent a hot/dry
and a warm/humid summer day so the examples are runnable. Replace them with measured data
(same columns) for any real assessment.

    python examples/make_synthetic_inputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def _diurnal(h, tmin, tmax, h_min=5.0, h_max=15.0):
    """Piecewise-cosine diurnal temperature (min at h_min, max at h_max), periodic in 24 h."""
    h = np.mod(h, 24.0)
    out = np.empty_like(h)
    rise = (h >= h_min) & (h <= h_max)
    out[rise] = tmin + (tmax - tmin) * 0.5 * (1 - np.cos(np.pi * (h[rise] - h_min) / (h_max - h_min)))
    fall = ~rise
    dh = np.where(h[fall] > h_max, h[fall] - h_max, h[fall] + 24 - h_max)
    out[fall] = tmax - (tmax - tmin) * 0.5 * (1 - np.cos(np.pi * dh / (24 - (h_max - h_min))))
    return out


def _solar(h, lat, lon, utc, doy, beam_k, diffuse_frac):
    import sys
    sys.path.insert(0, str(HERE.parent))
    from venuecool.solar import solar_position, extraterrestrial_normal
    alt, _ = solar_position(h, doy, lat, lon, utc)
    s = np.maximum(np.sin(np.radians(alt)), 0)
    am = np.where(s > 0.01, 1 / np.maximum(s, 0.01), 0)
    dni = np.where(s > 0.01, extraterrestrial_normal(doy) * beam_k ** (am ** 0.678), 0.0)
    dhi = diffuse_frac * dni * s + np.where(s > 0.01, 40 * s, 0)
    ghi = dni * s + dhi
    return ghi, dni, dhi


def _pw(t):
    t = t + 273.15
    return np.exp(-5.8002206e03 / t + 1.3914993 - 4.8640239e-02 * t + 4.1764768e-05 * t**2
                  - 1.4452093e-08 * t**3 + 6.5459673 * np.log(t))


def weather(name, site, tmin, tmax, dew_c, wind_mean, beam_k, diffuse_frac, note):
    h = np.arange(0, 49, 1.0)  # 48 h + closing hour so 2-day windows are possible
    T = _diurnal(h, tmin, tmax)
    dew = dew_c + 0.8 * np.sin(2 * np.pi * (h - 9) / 24)
    rh = np.clip(100 * _pw(dew) / _pw(T), 1, 100)
    wind = wind_mean * (1 + 0.35 * np.sin(2 * np.pi * (h - 10) / 24))
    ghi, dni, dhi = _solar(np.mod(h, 24), *site, beam_k, diffuse_frac)
    lat, lon, utc, doy = site
    lines = [
        "# data_status: SYNTHETIC_DEMONSTRATION (not measured)",
        f"# description: {note}",
        f"# latitude_deg: {lat}", f"# longitude_deg: {lon}", f"# utc_offset_h: {utc}", f"# day_of_year: {doy}",
        "# generator: examples/make_synthetic_inputs.py (clear-sky Meinel beam model, cosine diurnal temperature)",
        "time_h,temp_C,rh_pct,pressure_Pa,wind_ms,ghi_Wm2,dni_Wm2,dhi_Wm2",
    ]
    for i in range(len(h)):
        lines.append(f"{h[i]:.1f},{T[i]:.2f},{rh[i]:.1f},{site_pressure(name):.0f},{wind[i]:.2f},"
                     f"{ghi[i]:.1f},{dni[i]:.1f},{dhi[i]:.1f}")
    (DATA / f"weather_{name}.csv").write_text("\n".join(lines) + "\n")


def site_pressure(name):
    return 97000.0 if name == "hot_dry" else 101200.0


def occupancy(name, columns, profiles, note):
    lines = ["# data_status: SYNTHETIC_DEMONSTRATION (not measured)",
             f"# description: {note}",
             "# meaning: mean number of people present during [time_h, next time_h) -- not visitor counts",
             "time_h," + ",".join(columns)]
    for hh in range(49):
        vals = [profiles[c](hh % 24) for c in columns]
        lines.append(f"{hh}," + ",".join(f"{v:.1f}" for v in vals))
    (DATA / f"occupancy_{name}.csv").write_text("\n".join(lines) + "\n")


def _peaks(base, peaks):
    def f(h):
        v = base(h)
        for c, w, a in peaks:
            v += a * np.exp(-0.5 * ((h - c) / w) ** 2)
        return max(v, 0.0)
    return f


def main():
    DATA.mkdir(exist_ok=True)
    weather("hot_dry", (33.45, -112.07, -7, 196), tmin=29.0, tmax=41.0, dew_c=4.0, wind_mean=2.6,
            beam_k=0.72, diffuse_frac=0.10,
            note="Synthetic hot/dry desert design day (Phoenix-like site geometry); Tmax 41 C, dew point ~4 C")
    weather("warm_humid", (29.76, -95.37, -5, 196), tmin=26.0, tmax=34.0, dew_c=24.0, wind_mean=2.0,
            beam_k=0.62, diffuse_frac=0.22,
            note="Synthetic warm/humid design day (Houston-like site geometry); Tmax 34 C, dew point ~24 C")
    service = lambda h: 0.0 if (h < 5 or h >= 23) else 1.0  # noqa: E731
    occupancy("bus_stop", ["shelter", "curb"], {
        "shelter": _peaks(lambda h: 1.5 * service(h), [(8, 1.0, 5.0), (17.5, 1.3, 6.0), (12.5, 2.0, 2.0)]),
        "curb": _peaks(lambda h: 0.5 * service(h), [(8, 1.0, 2.5), (17.5, 1.3, 3.0)]),
    }, "Open bus-stop shelter and adjacent curb queue; mean people waiting")
    occupancy("rail_station", ["platform_open", "platform_canopy", "waiting_room"], {
        "platform_open": _peaks(lambda h: 3.0 * service(h), [(8, 1.2, 15.0), (17.5, 1.5, 18.0)]),
        "platform_canopy": _peaks(lambda h: 8.0 * service(h), [(8, 1.2, 35.0), (17.5, 1.5, 40.0)]),
        "waiting_room": _peaks(lambda h: 6.0 * service(h), [(8, 1.2, 20.0), (13, 2.5, 8.0), (17.5, 1.5, 25.0)]),
    }, "Semi-open rail-station waiting area: open platform end, canopy platform, semi-enclosed waiting room")
    event = lambda h: 1.0 if 10 <= h < 22 else 0.0  # noqa: E731
    occupancy("venue3", ["seating", "plaza", "concourse"], {
        "seating": _peaks(lambda h: 40.0 * event(h), [(19.5, 1.5, 260.0), (14.5, 1.5, 120.0)]),
        "plaza": _peaks(lambda h: 15.0 * event(h), [(18.0, 1.0, 80.0), (13.5, 1.0, 50.0)]),
        "concourse": _peaks(lambda h: 25.0 * event(h), [(18.3, 1.0, 120.0), (21.8, 0.6, 100.0), (14, 1, 60)]),
    }, "Synthetic three-section open-air venue (roofed seating, sunny plaza, partially enclosed concourse)")
    print(f"wrote synthetic inputs to {DATA}")


if __name__ == "__main__":
    main()
