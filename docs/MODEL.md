# VenueCool model: equations, assumptions and limitations

This document states what the engine in `venuecool/` computes. Every equation below maps to a
specific function; file references are given in brackets.

## 1. Units and states

Internal units are SI. Temperatures are °C (differences in K), pressure in Pa, mass in kg,
power in **W**, and energy in **J**. Energies are converted to **kWh** only in reports
(`J_PER_KWH = 3.6e6`). The code keeps power (kW installed) and energy (kWh used) separate:
`installed_kW` and `electricity_kWh` are different fields.

Humidity is carried as the **humidity ratio** W [kg water / kg dry air]. Relative humidity (RH)
is only ever derived from it. Configuration quantities are `{value, unit, source, bounds}`. The
schema declares a canonical unit for each field. A small whitelist of conversions is accepted
(kW→W, m³/h→m³/s, %→fraction, mm→m, K→°C, …), and any other unit is rejected
(`quantity.py`).

State vector (`model.py`):

| state | meaning |
|---|---|
| `h_i` | moist-air enthalpy of section *i* [J/kg_da] |
| `W_i` | humidity ratio of section *i* |
| `T_n` | temperature of surface node *n* |
| accumulators | boundary energy, boundary water, gross flux magnitudes, total device evaporation, per-section enthalpy-flow breakdown |

The accumulators are integrated but never feed back into the physics. The state size is
independent of how many devices a scenario has, so an idle device cannot perturb the solver's
step selection; the `test_baseline_equivalence` test checks this.

## 2. Psychrometrics (`psychro.py`)

* Saturation pressure: Hyland–Wexler (ASHRAE Fundamentals 2017, Ch. 1, eqs. 5–6).
* Enthalpy: \(h = c_{p,a} T + W (h_{fg,0} + c_{p,v} T)\), with \(c_{p,a}=1006\),
  \(c_{p,v}=1860\) J/(kg K), \(h_{fg,0}=2.501\times10^6\) J/kg. The datum is dry air and liquid
  water at 0 °C. Temperature is the exact inverse: \(T = (h - h_{fg,0}W)/(c_{p,a}+c_{p,v}W)\).
* Wet bulb: the adiabatic-saturation temperature \(t^*\) solving
  \(h(t^*, W_s(t^*)) = h_{in} + (W_s(t^*) - W)\,c_w t^*\), using the *same* constants. A pad at
  100% effectiveness therefore lands exactly on saturation.
* Nothing is clipped. `state_validity` flags negative humidity, supersaturation (RH > 1),
  out-of-range or non-finite temperatures. Flagged section-intervals are excluded from comfort
  assessment as `unassessed_invalid_state` and counted.

## 3. Air balances

Dry-air mass \(M_i = \rho_{da} V_{eff,i}\) is fixed at the run-start density. This is an
approximation: thermal expansion is not tracked. Every exchange is balanced in dry-air mass.

\[
M_i \frac{dh_i}{dt} = \sum_{src} \dot m_{src}(h_{src}-h_i) + Q_{conv,i} + Q_{occ,i},\qquad
M_i \frac{dW_i}{dt} = \sum_{src} \dot m_{src}(W_{src}-W_i) + G_{occ,i}
\]

The sources are:

* **Outdoor exchange** (open sections always exchange):
  \(\dot m_{o,i} = \rho\,(Q_{min} + C_w A_{open} U_{10})\), two-way and balanced.
* **Inter-section exchange**: \(\dot m_{ij} = \rho(Q_{min} + C_w A U_{10})\). The term enters
  section *i* as \(\dot m (h_j-h_i)\) and section *j* as \(\dot m (h_i - h_j)\), so it is exactly
  equal and opposite.
* **Devices** (Section 6): \(f_k \dot m_d (h_{out,d} - h_k)\) for each served section *k*.

Airflows come from an `AirflowProvider` (`airflow.py`). The default is the parametric form
above. `TableAirflow` reads flows versus wind speed from measurements or CFD, so it can replace
the parametric model without code changes elsewhere.

## 4. Surfaces and radiation

Each surface is a 1- or 2-node RC element. For 2 nodes, each node holds half of
\(C=\rho c A L\), and the conductance between them is \(kA/L\). Node 0 is the face toward the
section.

* **Roof (exterior face)**: absorbs \(\alpha A\,\mathrm{GHI}\). It exchanges longwave with the
  sky (fraction `exterior_sky_fraction`) and surroundings at outdoor air temperature, and
  convects with \(h = a + bU_{10}\).
* **Wall (exterior face)**: absorbs
  \(\alpha A(\mathrm{DNI}\cos\theta + 0.5\,\mathrm{DHI} + 0.5\cdot0.2\,\mathrm{GHI})\) and has
  sky/surroundings longwave and convection.
* **Floor (interior face)**: absorbs
  \(\alpha A(f_{sun}\,\mathrm{DNI}\sin\beta + F_{f,sky}\,\mathrm{DHI})\). Its bottom node
  conducts to a deep-ground temperature through `ground_conductance`.
* **Interior faces**: convection to section air with \(h = a + b\,v_{local}\). Grey longwave
  exchange runs floor↔roof and floor↔walls,
  \(q=\sigma\varepsilon_i\varepsilon_j A_f F_{f\to j}(T_f^4-T_j^4)\). Reciprocity sets
  \(F_{j\to f} = A_f F_{f\to j}/A_j\), and a view factor above 1 is rejected. Each face's
  remaining view goes to sky and/or surroundings.
* **Sky temperature**: Berdahl–Martin clear-sky emissivity from the dew point. Clouds are not
  modelled.
* **Sun exposure** \(f_{sun}(t)\): `constant`, `schedule` (hour→fraction), or `canopy`, where
  \(f = f_{open}\min(1, H\cot\beta/D)\), times a `shade_multiplier` that passive shade options
  change. Materially different sun exposure should become a separate section.

## 5. Occupants

Total metabolic heat per person is \(Q_{tot} = M\cdot 58.15\cdot A_{body}\), with external work
taken as 0. A sensible fraction \(f_s(T_{air})\), from a labelled table in the config, splits it.
The moisture gain is \(G = Q_{tot}(1-f_s)/h_g(T_{air})\), and the air receives
\(Q_{tot}f_s + G\,h_g = Q_{tot}\) exactly, so **the same metabolic energy is never counted
twice**.

Crowding effects represented:

* heat and moisture gains scale with the number of people present;
* exposure groups divide people by device coverage.

Approximations (not represented):

* body shading of floor and neighbours, and person-to-person radiation;
* blockage of air jets by crowds;
* radiative/convective split of body heat (all sensible heat goes to air);
* solar energy absorbed by people, which affects comfort only and is not added to the air.

## 6. Devices (`devices.py`)

Every device is an air handler. It moves \(\dot m_d = \rho_o\,\dot V_d\) from an intake
(outdoor air, or the \(f_k\)-weighted mixture of its served sections when recirculating),
through a process, to outlet state \((h_{out}, W_{out})\). It delivers \(f_k\dot m_d\) to
section *k*, with \(\sum f_k = 1\) enforced at validation.

* **Fan**: no process. \(h_{out} = h_{in} + \eta P/\dot m\), where all electric power becomes
  heat in the air. There is no refrigeration term. A fan's benefit comes only from local air
  speed in the exposure groups it covers. A recirculating fan in one section adds no outdoor
  exchange; `test_fan_raises_air_speed_not_outdoor_exchange` checks this.
* **Direct evaporative cooler**: \(T_{pad}=T_{in}-\varepsilon(T_{in}-T_{wb})\), with \(W_{pad}\)
  from the pad energy balance \(h_{pad}=h_{in}+(W_{pad}-W_{in})c_wT_{wb}\). Make-up water is
  assumed at the wet bulb, a labelled assumption. Since \(\varepsilon\le1\), the outlet stays at
  or above the wet bulb and \(W_{pad}\le W_s\). Motor heat is added afterwards. Water use is
  \(\dot m(W_{out}-W_{in})(1+\text{bleed})\).
* **Outdoor intake vs recirculation**: outdoor intake creates a through-flow, with outdoor air
  in and displaced room air out. Recirculation only redistributes and humidifies the room air.
* **One ledger per physical device**: electricity \(=P\times\)on-time. It is counted once even
  when the device serves several sections. Total evaporation is integrated in the ODE; the
  per-device split uses midpoint quadrature reconciled to that total, and the ratio is reported.

### Local exposure (coverage) model

For a covered group, local air is
\(h_{loc}=h_{sec}+\sum_d w_d(h_{out,d}-h_{sec})\), and likewise for \(W\). Here
\(\sum_d w_d = 1-\prod_d(1-\beta_d)\), split in proportion to the plume fractions \(\beta_d\).
The weight is therefore bounded by 1 and overlapping devices are not double counted. Local
speed is \(v=\sqrt{v_{base}^2+\sum_d v_d^2}\) with \(v_{base}=\max(v_{still}, k_{loc}U_{10})\).
Coverage changes only the comfort inputs, not the section balance, so it is never applied
twice. A group is counted as *served* once, however many devices cover it.

### Supplied presets and claims

The electrical draws are supplied: 0.25, 1.2, and 2.5 kW. The quoted effects ("feels up to
3 °C cooler", "lowers air temperature up to 5 / 8 °C") are **not inputs**. `venuecool calibrate`
reports what the assumed airflow, effectiveness, coverage speed, and plume fraction produce:

* at four reference climates (`preset_reference_checks.csv`);
* inside the simulated bus stop (`simulated_claim_checks.csv`).

In the hot/dry bus-stop demonstration, the shelter-mean air-temperature drop is ≈4.6 K for
cooler B (claim ≤5 °C) and ≈7.9 K for cooler C (claim ≤8 °C).

The fan's SET reduction against still air is 2.5–3.2 K in the dry references and 3.7 K at
33 °C and 60% RH. The humid case exceeds the "up to 3 °C" wording, and the code reports this
rather than hiding it. Inside the simulated shelter the reduction is only 0.3–0.9 K. The
synthetic wind already gives ≈1 m/s locally, and extra speed has diminishing SET benefit.
The fan's motor heat slightly warms the section air.

In warm/humid air the evaporative drops are much smaller, as expected from the wet-bulb limit.

## 7. Mean radiant temperature and comfort (`comfort.py`)

* Longwave: \(T_{lw}^4=\sum_k F_k T_k^4\) over floor, roof underside, walls, sky, and
  surroundings, using the occupant view factors from the config.
* Solar: the ASHRAE 55-2023 Appendix C (SolarCal) structure.
  * The direct-beam ΔMRT comes from `pythermalcomfort.models.solar_gain` with `f_svv=0`,
    `f_bes=1`, averaged over body orientations (SHARP 0–180°), and scaled linearly with DNI
    (the term is linear in DNI).
  * Diffuse and floor-reflected:
    \(\Delta T = 0.5 f_{eff} f_{svv}(\mathrm{DHI}+I_{floor}R_{floor})(\alpha_{sw}/\alpha_{lw})/(f_{eff}h_r)\),
    with \(I_{floor}\) and \(R_{floor}=1-\alpha_{floor}\) taken from the floor model.
* **Consistent sunlight**: each group is split into *sunlit* people (share \(f_{sun}\), direct
  beam on the body) and *shaded* people (share \(1-f_{sun}\)). \(f_{sun}\) is the same value that
  heats the floor.
* **Prescribed radiant mode**: `radiation.mode: prescribed` sets
  \(T_{lw}=T_{air}+\Delta\), with solar added optionally.
* **Equivalent temperature**: SET (`pythermalcomfort.models.set_tmp`, Gagge two-node model),
  using local air temperature, RH, local speed, MRT, met, clo, and pressure.
* **Applicability**: each population-interval is classed as
  * `supported`: finite with `limit_inputs=True` (ASHRAE 55 ranges: 10–40 °C, v < 2 m/s, …);
  * `extrapolated`: outside those ranges but inside a declared `extended_limits` envelope,
    computed with `limit_inputs=False` and flagged;
  * `unassessed`: anything else, or an invalid air state.

  `comfort.decision_basis` chooses whether `extrapolated` counts as assessed. The examples use
  `extended` because hot-dry peaks exceed 40 °C. Extrapolated person-hours are always reported.

## 8. Metrics (`metrics.py`)

For every population-interval, discomfort is \(D=\max(ET-target,0)\) and burden is
\(B = D\times\text{people}\times\Delta t\) [K·person-h]. Interval values use the midpoint of
conserved states (h, W averaged), and people are the exact integral of the step occupancy
schedule.

* **Person-hours** \(=\int N\,dt\). These are not visitors and are never multiplied by dwell time.
* **Relief** = baseline burden − intervention burden, summed only over section-intervals
  assessed in *both* runs (matched). Negative values are kept, and worsened sections are listed.
* **Attainment** = person-hours with ET ≤ target ÷ assessed person-hours.
* **Poorly served**: sections with attainment below `poorly_served_attainment`.
* **Ratios** (relief/kWh, relief/USD) are formed after summing across the venue.
* **Costs over the horizon**: operating electricity (with time-of-use) and water scale by
  `season_days / simulated_days`. Capital uses a CRF annuity, charged once per season, plus
  annual maintenance. Passive options are costed the same way.

## 9. Numerics

* SciPy `solve_ivp` (BDF by default; Radau, LSODA, and RK45 are selectable) with a Jacobian
  sparsity pattern.
* Integration restarts at every schedule discontinuity: weather and occupancy rows, schedule
  edges, time-of-use edges, hourly breakpoints. Devices and occupancy are constant within a
  segment.
* Warm-up: `warmup_days` cycles the first 24 h of the window, so thermal storage starts near
  its periodic state.
* Residuals reported per run:
  * global energy and water residuals relative to gross flux;
  * `internal_cancellation_rel_max`, an algebraic check that internal terms cancel at sampled
    states;
  * dry-air imbalance;
  * invalid-state counts;
  * the evaporation reconciliation ratio.

  Because BDF preserves linear invariants, the integrated residuals mainly confirm the
  equal-and-opposite bookkeeping. Discretisation error is tested separately by refinement
  (`venuecool refine`, `test_numerical_refinement`).

## 10. Search and uncertainty

* **Search** (`search.py`): enumerates slot options (none / preset / preset×N) × schedule ×
  passive subsets. It prunes by device count and investment before simulating, and classifies
  each candidate as `meets_target`, `physical_shortfall`, `unsupported_evaluation`, or
  `constraint_violation`.
  * The lowest-energy (or lowest-cost) feasible candidate is **re-simulated independently** at
    rtol 1e-8 with a 150 s step.
  * Results stream to CSV as candidates finish.
  * If nothing meets the target, the run says so and lists the closest physical shortfalls.
* **Uncertainty** (`uncertainty.py`): seeded Latin hypercube sampling over every bounded
  quantity. The baseline and every intervention share each draw (common random numbers). It
  reports quantiles, P(relief < 0), and Spearman rank sensitivities. `sweep` gives one-at-a-time
  low/high tornado data.

## 11. Software/physics checks vs validation

The test suite (`tests/`) checks units, configuration, conservative mixing, fan behaviour,
evaporative thermodynamics, shared-device accounting, baseline equivalence, occupancy
aggregation, negative-benefit preservation, matched draws, search reporting, and numerical
refinement. These are **verification** checks. **No validation against measurements has been
done**: all example inputs are synthetic.

To validate, you would need:

* measured section air temperature and humidity (shaded, aspirated sensors) at several points;
* globe or MRT measurements, sunlit and shaded;
* air speeds in covered zones;
* tracer-gas exchange rates (which can replace the parametric airflows through `TableAirflow`);
* device airflow, power, and water metering;
* occupancy counts.

## 12. Limitations and extension points

* Each section is well mixed. There is no stratification and no resolved jets; the plume
  fraction and covered speed stand in for them. CFD or measurements can supply these per device
  and group without code changes (`plume_fraction`, `covered_air_speed`, `TableAirflow`).
* Wind direction, stack-driven exchange, and train- or vehicle-induced flows are all folded into
  \(Q_{min}+C_wAU\).
* Clear-sky longwave, no cloud model, and no shortwave transmission through translucent roofs.
* SET is a steady-state index for an average adult. It does not represent dynamic
  physiology, acclimatisation, age, or health status, and **is not a heat-safety guarantee**.
  Later extensions (e.g. JOS-3 or two-node transient models) can plug in where `comfort._set`
  is called.
* Representative design day(s) scaled to a season. Running multi-day measured weather is
  supported (`end_h` > 24) and avoids that scaling.
* Automatic CAD ingestion is not implemented. Sections take simple 2D polygons, and the
  `footprint` and `surfaces` fields are the interface for a future importer.
