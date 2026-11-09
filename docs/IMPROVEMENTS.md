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

## 4. Event-aware comfort and electricity

The reporting series (air temperature, humidity) stays on `output_step_s`. Comfort intervals
and cumulative kWh are split at schedule, weather, occupancy and tariff edges, including
edges that fall between reporting samples. The moist-air state at an operating-schedule
boundary is the ODE solution at that time. States are interpolated only between samples that
do not contain a switch. Cumulative electricity at the end of the day matches the device
ledger. `test_event_metrics_match_ledger_and_fine_comfort` checks the ledger and a 60 s
comfort reference. `test_boundary_state_converges_to_fine_grid` checks the off-grid switch
state against 60 s and 30 s solutions.

## 5. Three rankings, Pareto, and SET applicability

`venuecool robust` now reports, for each assumption and each SET target:

* greatest total relief
* greatest relief per kWh
* lowest electricity among systems with attainment at least 0.60, including the uncooled
  baseline (otherwise `infeasible`)
* the Pareto set in the relief–kWh plane

Relief per kWh ignores interventions whose relief is not positive. Supported, extrapolated
and unassessed person-hours are counted separately. `outside_applicability_fraction` is the
share of person-hours outside the standard SET input limits. Strict rankings use only the
person-intervals that are supported in every alternative, so each device is scored on the
same hours. `conclusion_changes_if_strict` is true only when the strict-only relief winner
is a different system and leads the next system by more than 1 K·person-h. A near-tie is
not counted as a new ranking. `extrapolation_heavy` is true when that greatest-relief claim
keeps less than half of its relief on the shared supported hours.

On the hot/dry nominal day at a 30 °C target, about 52–68% of person-hours fall outside the
standard SET limits (68% for the shelter and the fan, 61% for the 1.2 kW cooler, 52% for
the 2.5 kW cooler). There are no unassessed person-hours. The 2.5 kW cooler has the most
relief (140 K·person-h) but the least relief per kWh (3.5, against 8.3 for the fan). The
least electricity that still reaches 60% attainment is the 1.2 kW cooler (19.2 kWh). All
four options, including the baseline, sit on the Pareto front. The hours supported in every
alternative are 21.9 person-hours, and the strict relief on those hours is 0.03 K·person-h
for every device. The extended-basis ranking is therefore extrapolation-heavy, and the
strict relief gap is 0.

That pattern holds for all 21 assumption-by-target cells: `extrapolation_heavy` is true and
`conclusion_changes_if_strict` is false. The earlier high-coverage, 28 °C result, in which
the 1.2 kW cooler appeared to lead on its own supported hours, does not survive once every
device is scored on the shared 11.7 person-hours. One cell changes the minimum-energy
choice once the baseline is included: low solar at 32 °C, where the uncooled shelter already
reaches about 70% attainment, so the minimum-energy strategy is to add no cooling. No cell
is infeasible at the 0.60 bar.

Equipment inputs are labelled in `equipment_evidence.csv`:

| evidence | meaning in this project |
|---|---|
| published | supplied by the problem statement (the three electrical draws) |
| assumed | DEMO-ASSUMPTION airflow, effectiveness, plume fraction, coverage speed |
| measured | not used; no metered equipment data were available |

## 6. Boundary states, shared strict hours, and the baseline

Three corrections, without new rankings:

* Off-grid schedule boundaries store the ODE state from the segment endpoint. The reporting
  grid is unchanged. On the mini venue, the state at a switch 1260 s into a 600 s grid matches
  a 30 s reference within 0.05 °C and 2×10⁻⁵ kg/kg.
* Strict relief and attainment use the person-intervals supported for every alternative.
  Each scenario still reports its own supported, extrapolated and unassessed person-hours.
  `extrapolation_heavy` marks a greatest-relief claim that loses more than half of its relief
  on that shared set.
* Minimum energy includes the uncooled baseline. Relief per kWh skips nonpositive relief and
  selects the baseline when every device fails to reduce the burden.

The single-zone hot/dry report is unchanged at the published precision: baseline attainment
0.278 and burden 161.8 K·person-h; fan relief 33.3 K·person-h and 4.0 kWh; 1.2 kW cooler
relief 111.2 and 19.2 kWh; 2.5 kW cooler relief 139.6 and 40.0 kWh. Service-hour edges fall
on the 300 s reporting grid, so the boundary-state correction does not move those totals.
