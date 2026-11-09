# VenueCool

VenueCool is a section-resolved, transient tool for assessing cooling options in open-air and
semi-open venues, built around the HiMCM 2026 Problem B setting. A venue is modelled as
connected **sections**, such as a shelter, a curb, a platform under a canopy, or a waiting room.
Each section has its own geometry, surfaces, sun exposure, occupants and devices, and exchanges
air with the outdoors and its neighbours.

For each candidate intervention, VenueCool simulates the venue and a *matched* baseline under
identical weather and occupancy. It then reports:

* discomfort burden, as SET above a target, integrated over person-hours;
* relief, which may be negative;
* attainment and poorly served sections;
* electricity (kWh), water and season cost;
* relief per kWh and per USD;
* which person-hours could and could not be assessed.

> **Status of results.** Every example input in this repository is **synthetic demonstration
> data** (labelled `SYNTHETIC_DEMONSTRATION` in the files). The outputs demonstrate the method;
> they are not predictions for a real site. VenueCool reports thermal-comfort indices, not heat
> safety, and it never invents satisfaction percentages.

Equations, assumptions and limitations are in [`docs/MODEL.md`](docs/MODEL.md).

## Install

```bash
cd COMAP_Simulation
python3.13 -m venv .venv            # Python >= 3.10 should work; developed on 3.13
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[dashboard,dev]"      # optional: adds the `venuecool` command
export NUMBA_CACHE_DIR=$PWD/.numba_cache          # pythermalcomfort uses numba; see note
```

`venuecool/__init__.py` sets `NUMBA_CACHE_DIR` itself if it is unset. Exporting it in the
shell is still recommended for sandboxed or read-only environments. All commands below use
`.venv/bin/python -m venuecool`, which works without the editable install, from the repository
root.

## Exact commands

```bash
# software/physics checks (36 tests, ~30 s)
.venv/bin/python -m pytest -q

# regenerate the synthetic weather + occupancy inputs (already committed)
.venv/bin/python examples/make_synthetic_inputs.py

# validate a configuration and print the provenance of every quantity
.venv/bin/python -m venuecool validate examples/bus_stop.yaml

# single-zone Problem B slice: one shelter, baseline + fan + two evaporative coolers,
# CSV exports + Tair / RH / SET / cumulative-kWh graphs
.venv/bin/python -m venuecool report examples/bus_shelter_single.yaml
# assumptions: docs/ASSUMPTIONS_SINGLE_ZONE.md
# airflow / solar / coverage bounds and SET targets 28, 30, 32 C
.venv/bin/python -m venuecool robust examples/bus_shelter_single.yaml
# what changed: docs/IMPROVEMENTS.md

# baseline + every scenario in the config, with exports
.venv/bin/python -m venuecool compare examples/bus_stop.yaml --weather examples/data/weather_hot_dry.csv

# all three examples under both synthetic climates -> outputs/examples_overview.csv  (~70 s)
.venv/bin/python -m venuecool examples

# bounded configuration search (parallel); --limit N for a quick trial
.venv/bin/python -m venuecool search examples/bus_stop.yaml --weather examples/data/weather_hot_dry.csv --workers 11
.venv/bin/python -m venuecool search examples/rail_station.yaml --workers 11
.venv/bin/python -m venuecool search examples/venue3.yaml --workers 11

# seeded Latin-hypercube uncertainty with matched draws, and a one-at-a-time sweep
.venv/bin/python -m venuecool uncertainty examples/bus_stop.yaml --weather examples/data/weather_hot_dry.csv --samples 128 --workers 11
.venv/bin/python -m venuecool sweep examples/bus_stop.yaml --scenario B_evap --workers 11

# checks of device presets against the problem's quoted effects; numerical refinement; runtime
.venv/bin/python -m venuecool calibrate examples/bus_stop.yaml
.venv/bin/python -m venuecool refine examples/bus_stop.yaml --scenario B_evap
.venv/bin/python -m venuecool benchmark examples/venue3.yaml --runs 22 --workers 11

# dashboard
.venv/bin/streamlit run app/dashboard.py
```

Multiprocessing (`--workers > 1`) uses `ProcessPoolExecutor`. Some sandboxes forbid the POSIX
semaphores this needs. In that case use `--workers 1`.

Outputs go to `outputs/<config>/<weather tag>/<command>/` unless `--out` is given. The weather
tag comes from the weather file (`weather_hot_dry.csv` becomes `hot_dry`); when there is no
`--weather`, the configuration's own weather file is used. The dashboard reads saved search and
uncertainty results from the same place.

## Repository layout

| path | contents |
|---|---|
| `venuecool/` | the engine; one module per concern (see table below) |
| `app/dashboard.py` | Streamlit + Plotly dashboard |
| `examples/*.yaml` | single-zone bus shelter, two-section bus stop, rail station, three-section venue |
| `docs/ASSUMPTIONS_SINGLE_ZONE.md` | labelled DEMO-ASSUMPTIONS for the single-zone Problem B slice |
| `examples/common/` | shared physics, materials, tariffs, device presets, section templates |
| `examples/data/` | synthetic weather (hot/dry, warm/humid) and occupancy CSVs |
| `tests/` | software and physics checks with a small two-section test venue |
| `docs/MODEL.md` | equations, assumptions, limitations, extension points |
| `outputs/` | the actual outputs produced by the commands above |

| module | role |
|---|---|
| `quantity.py`, `config.py` | units, provenance and bounds; YAML/JSON schema; includes, templates, overrides |
| `inputs.py`, `solar.py` | weather/occupancy CSV loading and checks; solar geometry |
| `psychro.py` | moist-air properties, wet bulb, validity flags (no clipping) |
| `airflow.py` | pluggable air-exchange providers (parametric, tabulated from measurements or CFD) |
| `devices.py` | fan and direct-evaporative outlet states |
| `model.py` | coupled ODE system, segmented SciPy integration, ledgers, conservation |
| `comfort.py` | MRT (longwave plus SolarCal solar), SET, applicability classes |
| `metrics.py` | burden, matched relief, attainment, costs |
| `runner.py` | `Study` (baseline and scenario runs), exports |
| `search.py`, `uncertainty.py` | bounded parallel search; LHS uncertainty and sensitivity |
| `calibrate.py`, `cli.py` | preset checks; command-line interface |

## Configuration

A venue file is YAML or JSON. It can `include:` shared files; the including file wins, and lists
of `id`-keyed items merge by id. Every physical number is a quantity:

```yaml
saturation_effectiveness: {value: 0.75, unit: "-", source: "assumed typical rigid-media pad (demonstration)", bounds: [0.6, 0.85]}
```

* The unit must match, or be convertible to, the field's canonical unit. A `kWh` where `W` is
  expected is rejected.
* `source` is mandatory.
* `bounds` makes the quantity part of the uncertainty analysis, and the nominal value must lie
  inside them.
* Unknown keys are errors, so typos cannot silently fall back to defaults.

Missing required data (weather columns, NaNs, gaps over 3 h, a run outside the weather period,
a missing occupancy column) raises an error. Nothing is filled in silently.

Key blocks:

* `sections`: each section, possibly from a `template`, keeps its own id. A section has:
  * a footprint polygon, height and volume;
  * `surfaces` (roof, floor, walls) built from `materials`;
  * `sun` exposure (`constant` / `schedule` / `canopy`) and an `outdoor` exchange;
  * `occupancy` (CSV column, met, clo);
  * `exposure_groups` (share of occupants and which devices cover them);
  * `radiation` view factors and `initial` conditions.
* `connections`: openings between sections.
* `device_presets` and `devices`: physical devices with unique ids. A device has `serves`
  (sections with airflow fractions summing to 1), `intake` (outdoor or recirculated), a
  schedule, and optional per-device overrides.
* `passive_options`: named path overrides, e.g.
  `sections[shelter].surfaces[roof].solar_absorptance_exterior`, with capital cost.
* `scenarios`, `search`, `uncertainty`: what to compare, search and sample.

Weather and occupancy files are CSVs with `# key: value` metadata headers (data status, site).

## Demonstration results (synthetic inputs, one design day scaled to a 120-day season)

Comfort target: SET ≤ 30 °C. "Relief" is the reduction in K·person-hours above target
against the matched baseline. Full table: `outputs/examples_overview.csv`.

Bus stop (hot/dry):

| scenario | attainment | relief (K·ph/day) | kWh/day | water m³/day | season cost USD |
|---|---|---|---|---|---|
| baseline | 0.26 | 0 | 0 | 0 | 0 |
| A: fan (0.25 kW) | 0.29 | 19.8 | 4.0 | 0 | 220 |
| B: evaporative (1.2 kW) | 0.52 | 91.2 | 19.2 | 1.28 | 1482 |
| C: large evaporative (2.5 kW) | 0.65 | 137.5 | 40.0 | 2.79 | 2882 |
| B + curb shade sail | 0.62 | 149.0 | 19.2 | 1.28 | 1775 |

Under warm/humid weather, B's relief falls from 91 to 64 K·ph, and its water use falls more
sharply, from 1.28 to 0.38 m³. The fan's relief rises from 20 to 40 K·ph, because in dry heat
above skin temperature extra air speed helps less. The fan slightly *worsens* the unshaded curb
through motor heat in the shared air, and this negative contribution is kept.

`venuecool calibrate` compares the presets with the problem's quoted effects:

* the shelter-mean air drop is 4.6 K for B (claim "up to 5 °C") and 7.9 K for C (claim "up
  to 8 °C");
* the fan's SET reduction against still air is 2.5–3.7 K (claim "feels up to 3 °C"), but only
  0.3–0.9 K in the already breezy simulated shelter.

Search outcomes (`outputs/<example>/hot_dry/search/search_summary.json`); the selected
configuration is re-simulated at tighter tolerance before being reported:

* **Bus stop**: 160 candidates in 23 s. 5 meet the target, 51 are physical shortfalls, and 104
  violate the water constraint. The selected configuration is one evaporative cooler,
  11:00–19:00, plus the curb shade sail: 9.6 kWh/day, about 1,430 USD/season.
* **Rail station**: 1,096 candidates in 194 s, of which 595 meet the target. The selected
  configuration is one evaporative cooler in the waiting room (06:00–22:00) plus shade at the
  open platform end: 19.2 kWh/day.
* **Three-section venue**: 11,254 candidates over 7 slots in 35 min on 11 workers. 887 meet the
  target, 9,807 are physical shortfalls, and 560 violate constraints; none was unsupported.
  * The selected configuration is two evaporative coolers in the concourse (stalls and
    corridor), plus the shared gate cooler (large) running only when hot, plus the plaza shade.
  * It uses 71.1 kWh/day and costs about 9,100 USD/season, with attainment 0.61.
  * The independent re-simulation reproduced the attainment within 0.002.

Uncertainty, hot/dry, with matched LHS draws. Relief is in K·ph per day, given as 5–95% ranges.
No draw in any example gave negative relief.

| example | draws, parameters | scenario | relief, 5–95% |
|---|---|---|---|
| bus stop | 128 draws, 38 parameters | A (fan) | 8–50 |
| | | B (evaporative) | 59–115 |
| | | C (large evaporative) | 93–190 |
| rail station | 96 draws, 50 parameters | shared large cooler | 717–1305 |
| | | outdoor-intake waiting-room cooler | 343–641 |
| | | recirculating waiting-room cooler | 356–640 |
| three-section venue | 64 draws, 53 parameters | concourse evaporative | 2106–3444 |

The two waiting-room variants are statistically indistinguishable in relief. The recirculating
one uses about 35% less water.

The dominant drivers, consistent between Spearman ranks and the one-at-a-time sweep
(`outputs/bus_stop/hot_dry/sweep/`), are:

* the local wind factor (how breezy the section already is);
* the open-side sun fraction;
* clothing insulation;
* the coverage air speed of the device.

Two kinds of correlation are flagged rather than reported as drivers:

* correlations within the noise level (≈ 2/√n);
* correlations with parameters the scenario cannot depend on (unused presets, devices and
  passive options), which are pure sampling noise.

Numerics:

* Halving the tolerance and output step changes relief by under 0.3%.
* Conservation residuals are at round-off level.
* A three-section run takes about 0.8 s, and 11 workers give about 8.6 runs/s.

## Checks vs validation

`pytest` runs **verification** checks:

* unit handling, schema rejection, bounds and include merging;
* psychrometric reference values and the wet-bulb limit;
* conservative mixing between sealed sections;
* a 0.25 kW fan for one hour giving 0.25 kWh;
* a fan adding air speed but no outdoor exchange or cooling;
* the evaporative water ledger matching the moisture added;
* recirculation in a sealed room humidifying the air and weakening the cooling;
* a shared device counted once, and overlapping coverage not double counting beneficiaries;
* an idle device reproducing the baseline exactly;
* negative relief kept, occupancy integrated exactly, ratios formed after summing;
* numerical refinement and periodic warm-up;
* search status reporting, including "no configuration meets target", and matched draws.

These checks confirm the code implements the stated model. **They are not validation against
measurements**: no field data were available. `docs/MODEL.md` §11 lists the measurements
needed for validation.

## Extending

* **Airflows**: provide a `TableAirflow` CSV from measurements or CFD
  (`airflow: {provider: table, file: flows.csv, source: "..."}`, with columns `wind_ms`,
  `out:<section>`, `conn:<connection>` in m³/s). Missing columns or negative values raise errors.
* **Device plumes**: set per-device `plume_fraction` and `covered_air_speed` overrides.
* **Geometry**: sections accept 2D polygons and explicit surfaces, which is where a CAD
  importer would plug in.
* **Comfort models**: add another index in `comfort._set`.
