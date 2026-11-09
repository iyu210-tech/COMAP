# Single-zone bus shelter: labelled assumptions

This note lists the free parameters used by
[`examples/bus_shelter_single.yaml`](../examples/bus_shelter_single.yaml). Equations and the
full model are in [`MODEL.md`](MODEL.md). Electrical draws **0.25 / 1.2 / 2.5 kW** are supplied
by the problem statement; everything else below is a **DEMO-ASSUMPTION** unless marked otherwise.

## Weather and occupancy

| Item | Status |
|---|---|
| Hot/dry 24 h weather (`examples/data/weather_hot_dry.csv`) | `SYNTHETIC_DEMONSTRATION` — Phoenix-like geometry, not a measured day |
| Occupancy (`examples/data/occupancy_bus_shelter_single.csv`) | `SYNTHETIC_DEMONSTRATION` — mean people present per hour under the canopy |

## Geometry and surfaces

| Item | Assumption |
|---|---|
| Footprint | 4 m × 1.8 m open-sided shelter, height 2.6 m |
| Roof | Polycarbonate, absorbed share 0.5 (transmission neglected) |
| Rear panel | Glass wall, 8 m² |
| Floor / ground | Concrete slab with deep-ground temperature 28 °C |
| Sun | Canopy mode; open-side fraction 0.5 toward low sun |

## Outdoor air exchange

Parametric form \(Q = Q_{\min} + C_w A_{\mathrm{open}} U_{10}\) (no CFD):

| Parameter | Nominal | Notes |
|---|---|---|
| Opening area | 14 m² | Three open sides of the shelter |
| Wind coefficient \(C_w\) | 0.10 (template) | Open-side exchange; bounds 0.04–0.2 |
| Min flow | 0.5 m³/s | Calm-condition floor |
| Local wind factor | 0.35 | Occupant-level speed / 10 m wind under a roof |

## Equipment (airflow and coverage — not supplied by the problem)

The problem gives electrical draw and qualitative claims ("feels up to 3 °C", "up to 5 / 8 °C").
It does **not** fix airflow or coverage. VenueCool uses:

| Preset | Power (supplied) | Airflow | Coverage speed | Effectiveness / plume |
|---|---|---|---|---|
| Fan | 0.25 kW | 2.5 m³/s | 1.3 m/s | plume = 0 (local air = section air) |
| Evaporative | 1.2 kW | 3.0 m³/s | 1.0 m/s | ε_sat = 0.75, plume = 0.25 |
| Large evaporative | 2.5 kW | 6.0 m³/s | 1.3 m/s | ε_sat = 0.82, plume = 0.35 |

- **Fan:** no refrigeration term. Benefit is higher local air speed in SET; motor heat is added to the air.
- **Evaporative:** adiabatic pad approaching the wet bulb; outdoor intake by default; bleed water 15%.
- Schedule: service hours 06:00–22:00 for all three devices in this slice.

Check what these assumptions produce with `venuecool calibrate examples/bus_stop.yaml` (same presets).

## Comfort

- Index: SET (`pythermalcomfort`), target 30 °C (see `examples/common/physics_materials.yaml`).
- Decision basis: `extended` (hot-dry peaks exceed the strict ASHRAE 55 40 °C input limit).
- This is a **comfort comparison**, not a heat-safety guarantee and not a satisfaction percentage.

## What this slice deliberately omits

- Multi-zone coupling (no curb, no connections)
- Search / optimization / uncertainty
- CFD or measured airflow tables (the `airflow.provider: table` interface exists but is unused here)

## Verification vs validation

`pytest` and the single-zone tests check units, wet-bulb limits, fan behaviour and baseline energy
bookkeeping against the **stated model**. They are not validation against field measurements.
