# Three fixes (integration, device overrides, robustness)

These changes sit on top of the existing engine. The CLI, CSV exports, and the single-zone
report command behave as before.

## 1. Segment hand-off (`venuecool/model.py`)

`solve_ivp` stores only the times in `t_eval`. The reporting grid is still that series, but a
schedule edge (or any other segment boundary) is often not on the grid. The next segment used
to start from the last *stored* sample, which is earlier than the true endpoint, so a slice of
dynamics was dropped and the state jumped.

`simulate()` now appends the segment end to the solver's evaluation times and copies that
column into the initial state of the next segment. Points that are not on the reporting grid
are not written into the output series.

Regression: `test_offgrid_schedule_matches_fine_grid` runs an evaporative cooler that switches
off at 0.35 h (1260 s) with a 600 s output step, and compares air temperature at later shared
times with a 60 s reference.

## 2. Device overrides (`venuecool/config.py`, `venuecool/model.py`)

Overrides were plain `Quantity` objects. A value entered in kW kept the number 0.25 and was
then used as if it were watts. `apply_device_overrides` merges the override into the preset and
re-validates it, which:

* converts units with the same whitelist as the rest of the schema (kW to W, m³/h to m³/s, …);
* rejects unknown fields;
* rejects non-physical values: airflow ≤ 0, effectiveness outside [0, 1], coverage
  (`plume_fraction`) outside [0, 1], negative power or bleed, motor-heat fraction outside [0, 1].

This runs both when the configuration is loaded and when devices are resolved for a run.
Regression: `test_device_override_converts_kw_and_rejects_invalid`.

## 3. Assumption sensitivity (`venuecool robust`)

```bash
.venv/bin/python -m venuecool robust examples/bus_shelter_single.yaml
```

Re-simulates the single-zone shelter at the documented low/high bounds for outdoor airflow,
open-side solar fraction, and device coverage (plume fraction and covered air speed). For each
trajectory it scores SET targets of 28 °C, 30 °C and 32 °C without a new integration.

Outputs in `outputs/bus_shelter_single/hot_dry/robustness/`:

| file | contents |
|---|---|
| `robustness_results.csv` | burden, relief, attainment, **supported** and **extrapolated** person-hours |
| `robustness_summary.json` | which system is preferred in each cell |
| `preferred_system.html` | case × target, labelled with the preferred system |
| `relief_by_assumption.html` | relief of the three systems at 30 °C |
| `supported_vs_extrapolated.html` | applicability split (does not depend on the target) |

Preferred means the largest matched relief. Baseline is preferred when every intervention has
relief ≤ 0. Supported person-hours are inside the strict ASHRAE 55 input limits;
extrapolated person-hours are outside those limits but inside the declared extended envelope.
The decision basis in the examples still *uses* extrapolated values for burden when
`decision_basis` is `extended`; the two counts are reported separately either way.

Inputs remain `SYNTHETIC_DEMONSTRATION`. This is a sensitivity of the stated model, not a
field validation.

On the hot/dry synthetic day the preferred system (largest relief) is the 2.5 kW cooler in
every cell: airflow, solar, coverage, and all three SET targets. Relief itself moves a lot
(about 40 to 248 K·person-h). Less outdoor air and more sun increase the cooler's benefit;
more outdoor air and a tighter canopy reduce it. Supported and extrapolated person-hours
trade off with those same assumptions (more sun shifts hours into the extrapolated class)
and do not change when only the SET target changes.
