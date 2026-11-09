"""Validated venue configuration (YAML/JSON) with section templates and provenance.

Pipeline: ``load_raw`` -> ``expand_templates`` -> (optional uncertainty draw / passive overrides via
``set_path``) -> ``validate``. Each section keeps its own identity after template expansion: the
template is deep-copied and merged with the section's own entries.

Override paths use dots with ``[id]`` selectors for lists of objects, e.g.
``sections[shelter].surfaces[roof].solar_absorptance_exterior.value``.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .quantity import Q, Quantity


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- building blocks
class Meta(_M):
    name: str
    description: str = ""
    data_status: str = Field(
        description="e.g. SYNTHETIC_DEMONSTRATION or MEASURED; reported in every output")
    notes: List[str] = []


class Site(_M):
    latitude_deg: float = Field(ge=-90, le=90)
    longitude_deg: float = Field(ge=-180, le=180)
    utc_offset_h: float
    day_of_year: int = Field(ge=1, le=366)


class FileRef(_M):
    file: str
    solar_split: Literal["provided", "erbs"] = "provided"


class AirflowSpec(_M):
    """Air-exchange provider. ``table``: CSV of flows vs wind speed (``wind_ms``, ``out:<section>``,
    ``conn:<connection>`` in m3/s) from tracer-gas measurements or CFD."""
    provider: Literal["parametric", "table"] = "parametric"
    file: Optional[str] = None
    source: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.provider == "table" and not (self.file and self.source and self.source.strip()):
            raise ValueError("airflow provider 'table' needs a file and a source")
        return self


class Simulation(_M):
    start_h: float = 0.0
    end_h: float = 24.0
    warmup_days: int = Field(default=2, ge=0)
    output_step_s: float = Field(default=300.0, gt=0)
    method: Literal["BDF", "Radau", "LSODA", "RK45"] = "BDF"
    rtol: float = 1e-6
    atol_scale: float = 1e-6
    max_step_s: float = 900.0


class Physics(_M):
    convection_a: Q("W/(m2*K)")
    convection_b: Q("W*s/(m3*K)")
    still_air_speed: Q("m/s")
    body_surface_area: Q("m2")
    occupant_sensible_fraction: List[Tuple[float, float]] = Field(
        description="[[air_temp_degC, sensible fraction of total metabolic heat], ...]")
    occupant_sensible_fraction_source: str
    sky_model: Literal["berdahl_martin"] = "berdahl_martin"


class ExtendedLimits(_M):
    tdb: Tuple[float, float] = (5.0, 50.0)
    tr: Tuple[float, float] = (5.0, 75.0)
    v: Tuple[float, float] = (0.0, 4.0)
    rh: Tuple[float, float] = (0.0, 100.0)


class Comfort(_M):
    index: Literal["SET"] = "SET"
    target: Q("degC")
    decision_basis: Literal["strict", "extended"] = "strict"
    extended_limits: ExtendedLimits = ExtendedLimits()
    poorly_served_attainment: float = Field(default=0.8, ge=0, le=1)
    solar_absorptance_body: Q("-")
    occupant_sharp_deg: List[float] = Field(
        default=[0, 30, 60, 90, 120, 150, 180],
        description="sun azimuth relative to body front; results averaged (random orientation)")


class TouPeriod(_M):
    start_h: float
    end_h: float
    price: Q("USD/kWh")


class Tariffs(_M):
    electricity: Q("USD/kWh")
    water: Q("USD/m3")
    time_of_use: List[TouPeriod] = []
    discount_rate: Q("1/year")
    season_days: Q("day")
    horizon_note: str = ("Simulated period is a representative design day; operating quantities are "
                         "scaled by season_days / simulated_days; capital is annualised (CRF) and "
                         "charged once per season (one season per year).")


class Material(_M):
    conductivity: Q("W/(m*K)")
    density: Q("kg/m3")
    specific_heat: Q("J/(kg*K)")


class Surface(_M):
    id: str
    kind: Literal["roof", "floor", "wall"]
    material: str
    thickness: Q("m")
    nodes: Literal[1, 2] = 2
    area_m2: Optional[float] = None
    area_fraction: float = Field(default=1.0, gt=0)
    emissivity: Q("-")
    solar_absorptance_exterior: Optional[Q("-")] = None
    solar_absorptance_interior: Optional[Q("-")] = None
    azimuth_deg: float = 180.0
    exterior_sky_fraction: float = Field(default=1.0, ge=0, le=1)
    interior_sky_fraction: float = Field(default=0.0, ge=0, le=1)
    deep_ground_temperature: Optional[Q("degC")] = None
    ground_conductance: Optional[Q("W/(m2*K)")] = None

    @model_validator(mode="after")
    def _kind_fields(self):
        if self.kind == "floor" and (self.deep_ground_temperature is None or self.ground_conductance is None):
            raise ValueError(f"floor '{self.id}' needs deep_ground_temperature and ground_conductance")
        if self.kind == "floor" and self.solar_absorptance_interior is None:
            raise ValueError(f"floor '{self.id}' needs solar_absorptance_interior")
        if self.kind in ("roof", "wall") and self.solar_absorptance_exterior is None:
            raise ValueError(f"{self.kind} '{self.id}' needs solar_absorptance_exterior")
        return self


class SunExposure(_M):
    mode: Literal["constant", "schedule", "canopy"]
    value: Optional[Q("-")] = None
    points: List[Tuple[float, float]] = []
    canopy_height_m: Optional[float] = None
    canopy_depth_m: Optional[float] = None
    open_side_fraction: Optional[Q("-")] = None
    shade_multiplier: Q("-") = Quantity(value=1.0, unit="-", source="default: no added shading")

    @model_validator(mode="after")
    def _check(self):
        if self.mode == "constant" and self.value is None:
            raise ValueError("sun mode 'constant' needs 'value'")
        if self.mode == "schedule" and len(self.points) < 2:
            raise ValueError("sun mode 'schedule' needs >= 2 [hour, fraction] points")
        if self.mode == "canopy" and None in (self.canopy_height_m, self.canopy_depth_m, self.open_side_fraction):
            raise ValueError("sun mode 'canopy' needs canopy_height_m, canopy_depth_m, open_side_fraction")
        return self


class OutdoorExchange(_M):
    opening_area_m2: float = Field(ge=0)
    wind_coefficient: Q("-")
    min_flow: Q("m3/s")
    local_wind_factor: Q("-")


class ViewSet(_M):
    roof: float = Field(default=0.0, ge=0, le=1)
    walls: float = Field(default=0.0, ge=0, le=1)
    sky: float = Field(default=0.0, ge=0, le=1)
    floor: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def _sum(self):
        if self.roof + self.walls + self.sky + self.floor > 1.0 + 1e-9:
            raise ValueError("view factors sum to more than 1 (remainder is 'surroundings')")
        return self


class Radiation(_M):
    floor_view: ViewSet
    occupant_view: ViewSet
    mode: Literal["computed", "prescribed"] = "computed"
    prescribed_mrt_minus_air: Optional[Q("K")] = None
    prescribed_add_solar: bool = True

    @model_validator(mode="after")
    def _p(self):
        if self.mode == "prescribed" and self.prescribed_mrt_minus_air is None:
            raise ValueError("prescribed radiant mode needs prescribed_mrt_minus_air")
        return self


class Occupancy(_M):
    column: Optional[str] = None
    met: Q("met")
    clo: Q("clo")
    posture: Literal["standing", "sitting"] = "standing"


class ExposureGroup(_M):
    id: str
    occupant_share: float = Field(ge=0, le=1)


class Initial(_M):
    air: Union[Literal["outdoor"], Q("degC")] = "outdoor"
    humidity_ratio: Union[Literal["outdoor"], Q("kg/kg")] = "outdoor"
    surfaces: Union[Literal["outdoor_air"], Q("degC")] = "outdoor_air"


class Section(_M):
    id: str
    name: str
    template: Optional[str] = None
    template_description: str = ""
    footprint: List[Tuple[float, float]] = Field(min_length=3)
    height_m: float = Field(gt=0)
    volume_factor: Q("-")
    surfaces: List[Surface]
    sun: SunExposure
    outdoor: OutdoorExchange
    radiation: Radiation
    occupancy: Occupancy
    exposure_groups: List[ExposureGroup] = [ExposureGroup(id="all", occupant_share=1.0)]
    initial: Initial = Initial()

    @model_validator(mode="after")
    def _check(self):
        ids = [s.id for s in self.surfaces]
        if len(set(ids)) != len(ids):
            raise ValueError(f"section {self.id}: duplicate surface ids")
        if sum(1 for s in self.surfaces if s.kind == "floor") != 1:
            raise ValueError(f"section {self.id}: exactly one floor surface is required")
        if sum(1 for s in self.surfaces if s.kind == "roof") > 1:
            raise ValueError(f"section {self.id}: at most one roof surface")
        share = sum(g.occupant_share for g in self.exposure_groups)
        if abs(share - 1.0) > 1e-6:
            raise ValueError(f"section {self.id}: exposure group shares sum to {share}, not 1")
        return self

    @property
    def area_m2(self) -> float:
        pts = self.footprint
        a = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            a += x1 * y2 - x2 * y1
        return abs(a) / 2.0

    @property
    def centroid(self) -> Tuple[float, float]:
        xs = [p[0] for p in self.footprint]
        ys = [p[1] for p in self.footprint]
        return sum(xs) / len(xs), sum(ys) / len(ys)


class Connection(_M):
    id: str
    a: str
    b: str
    opening_area_m2: float = Field(ge=0)
    wind_coefficient: Q("-")
    min_flow: Q("m3/s")


class DevicePreset(_M):
    kind: Literal["fan", "evaporative"]
    label: str
    claim: str = ""
    electric_power: Q("W")
    airflow: Q("m3/s")
    covered_air_speed: Q("m/s")
    saturation_effectiveness: Optional[Q("-")] = None
    bleed_fraction: Q("-") = Quantity(value=0.0, unit="-", source="default: none")
    plume_fraction: Q("-") = Quantity(value=0.0, unit="-", source="default: fully mixed")
    motor_heat_to_air: Q("-") = Quantity(value=1.0, unit="-", source="all electrical input ends as heat in served air")
    capital_cost: Q("USD")
    lifetime: Q("year")
    maintenance: Q("USD/year")
    default_intake: Literal["outdoor", "recirculate"] = "recirculate"

    @model_validator(mode="after")
    def _physical(self):
        if self.electric_power.value < 0:
            raise ValueError("electric_power must be >= 0")
        if self.airflow.value <= 0:
            raise ValueError("airflow must be > 0")
        if self.covered_air_speed.value < 0:
            raise ValueError("covered_air_speed must be >= 0")
        if not 0.0 <= self.plume_fraction.value <= 1.0:
            raise ValueError("plume_fraction (coverage fraction) must lie in [0, 1]")
        if self.bleed_fraction.value < 0:
            raise ValueError("bleed_fraction must be >= 0")
        if not 0.0 <= self.motor_heat_to_air.value <= 1.0:
            raise ValueError("motor_heat_to_air must lie in [0, 1]")
        if self.kind == "evaporative" and self.saturation_effectiveness is None:
            raise ValueError("evaporative presets need saturation_effectiveness")
        if self.saturation_effectiveness is not None and not 0 <= self.saturation_effectiveness.value <= 1:
            raise ValueError("saturation_effectiveness must lie in [0, 1]")
        return self


class Serve(_M):
    section: str
    airflow_fraction: float = Field(gt=0, le=1)
    groups: Optional[List[str]] = None


class Schedule(_M):
    mode: Literal["always", "hours", "occupied_hot", "never"]
    intervals: List[Tuple[float, float]] = []
    outdoor_threshold: Optional[Q("degC")] = None

    @model_validator(mode="after")
    def _c(self):
        if self.mode == "occupied_hot" and self.outdoor_threshold is None:
            raise ValueError("occupied_hot schedules need outdoor_threshold")
        return self


class Device(_M):
    id: str
    preset: str
    intake: Optional[Literal["outdoor", "recirculate"]] = None
    serves: List[Serve] = Field(min_length=1)
    schedule: str
    overrides: Dict[str, Quantity] = {}

    @model_validator(mode="after")
    def _f(self):
        tot = sum(s.airflow_fraction for s in self.serves)
        if abs(tot - 1.0) > 1e-6:
            raise ValueError(f"device {self.id}: airflow fractions sum to {tot}; must reconcile to 1")
        if len({s.section for s in self.serves}) != len(self.serves):
            raise ValueError(f"device {self.id}: a section is listed twice in 'serves'")
        return self


def apply_device_overrides(preset: DevicePreset, overrides: Dict[str, Quantity], where: str) -> DevicePreset:
    """Merge overrides into a preset, convert each quantity to its canonical unit, and re-validate.

    Overrides are stored as plain Quantities (so a kW value is not converted at parse time).
    Re-validating the merged preset runs the ``Q(...)`` converters and the physical bounds.
    """
    if not overrides:
        return preset
    data = preset.model_dump()
    for k, q in overrides.items():
        if k not in DevicePreset.model_fields:
            raise ValueError(f"{where}: override of unknown preset field {k}")
        data[k] = q.model_dump() if isinstance(q, Quantity) else q
    try:
        return DevicePreset.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"{where}: invalid override: {e}") from e


class PassiveOption(_M):
    id: str
    description: str
    set: Dict[str, Any]
    capital_cost: Q("USD")
    lifetime: Q("year")
    maintenance: Q("USD/year")


class Scenario(_M):
    id: str
    description: str = ""
    devices: List[str] = []
    passive: List[str] = []


class SearchSlot(_M):
    id: str
    serves: List[Serve]
    options: List[str] = Field(description="preset ids or 'none'; 'preset*N' places N units")
    intake: Optional[Literal["outdoor", "recirculate"]] = None


class SearchTarget(_M):
    min_attainment: float = Field(ge=0, le=1)
    min_section_attainment: float = Field(default=0.0, ge=0, le=1)
    max_unassessed_fraction: float = Field(default=0.05, ge=0, le=1)


class SearchConstraints(_M):
    max_devices: int = 6
    max_investment_usd: float = 1e9
    max_water_m3_per_day: float = 1e9


class SearchSpec(_M):
    objective: Literal["energy", "cost"] = "energy"
    target: SearchTarget
    constraints: SearchConstraints = SearchConstraints()
    slots: List[SearchSlot]
    schedules: List[str]
    passive_options: List[str] = []
    max_configurations: int = 20000


class UncertaintySpec(_M):
    samples: int = 64
    seed: int = 12345
    parameters: Union[Literal["all"], List[str]] = "all"
    scenarios: List[str] = []


class VenueConfig(_M):
    meta: Meta
    site: Site
    weather: FileRef
    occupancy: FileRef
    simulation: Simulation = Simulation()
    airflow: AirflowSpec = AirflowSpec()
    physics: Physics
    comfort: Comfort
    tariffs: Tariffs
    materials: Dict[str, Material]
    templates: Dict[str, Any] = {}
    sections: List[Section]
    connections: List[Connection] = []
    device_presets: Dict[str, DevicePreset]
    devices: List[Device] = []
    schedules: Dict[str, Schedule] = {}
    passive_options: List[PassiveOption] = []
    scenarios: List[Scenario] = []
    search: Optional[SearchSpec] = None
    uncertainty: Optional[UncertaintySpec] = None
    base_dir: str = "."

    @model_validator(mode="after")
    def _cross(self):
        sec = {s.id: s for s in self.sections}
        if len(sec) != len(self.sections):
            raise ValueError("duplicate section ids")
        for s in self.sections:
            for surf in s.surfaces:
                if surf.material not in self.materials:
                    raise ValueError(f"section {s.id} surface {surf.id}: unknown material {surf.material}")
        cids = set()
        for c in self.connections:
            if c.a not in sec or c.b not in sec or c.a == c.b:
                raise ValueError(f"connection {c.id} must join two distinct existing sections")
            if c.id in cids:
                raise ValueError(f"duplicate connection id {c.id}")
            cids.add(c.id)
        dids = set()
        for d in self.devices:
            if d.id in dids:
                raise ValueError(f"duplicate device id {d.id}: each physical device has one ledger")
            dids.add(d.id)
            if d.preset not in self.device_presets:
                raise ValueError(f"device {d.id}: unknown preset {d.preset}")
            if d.schedule not in self.schedules:
                raise ValueError(f"device {d.id}: unknown schedule {d.schedule}")
            _check_serves(d.serves, sec, f"device {d.id}")
            apply_device_overrides(self.device_presets[d.preset], d.overrides, f"device {d.id}")
        pids = {p.id for p in self.passive_options}
        for sc in self.scenarios:
            for d in sc.devices:
                if d not in dids:
                    raise ValueError(f"scenario {sc.id}: unknown device {d}")
            for p in sc.passive:
                if p not in pids:
                    raise ValueError(f"scenario {sc.id}: unknown passive option {p}")
        if self.search:
            for sl in self.search.slots:
                _check_serves(sl.serves, sec, f"search slot {sl.id}")
                for o in sl.options:
                    name = o.split("*")[0]
                    if name != "none" and name not in self.device_presets:
                        raise ValueError(f"search slot {sl.id}: unknown preset option {o}")
            for s in self.search.schedules:
                if s not in self.schedules:
                    raise ValueError(f"search: unknown schedule {s}")
            for p in self.search.passive_options:
                if p not in pids:
                    raise ValueError(f"search: unknown passive option {p}")
        return self

    def section(self, sid: str) -> Section:
        return next(s for s in self.sections if s.id == sid)

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else Path(self.base_dir) / p


def _check_serves(serves, sec, where):
    for s in serves:
        if s.section not in sec:
            raise ValueError(f"{where}: unknown section {s.section}")
        gids = {g.id for g in sec[s.section].exposure_groups}
        for g in s.groups or []:
            if g not in gids:
                raise ValueError(f"{where}: section {s.section} has no exposure group {g}")
    if abs(sum(s.airflow_fraction for s in serves) - 1.0) > 1e-6:
        raise ValueError(f"{where}: airflow fractions must sum to 1")


# --------------------------------------------------------------------------- raw dict handling
def _read_mapping(path: Path) -> dict:
    text = path.read_text()
    raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: configuration must be a mapping")
    return raw


def load_raw(path: Union[str, Path]) -> dict:
    """Read YAML/JSON. ``include: [files]`` are merged first (the including file wins)."""
    path = Path(path)
    return _with_includes(_read_mapping(path), path.resolve().parent)


def load_raw_text(text: str, base_dir: Union[str, Path], fmt: str = "yaml") -> dict:
    """Same as :func:`load_raw` for configuration text (e.g. edited in the dashboard)."""
    raw = json.loads(text) if fmt == "json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    return _with_includes(raw, Path(base_dir).resolve())


def _with_includes(raw: dict, base: Path) -> dict:
    merged: dict = {}
    for inc in raw.pop("include", []) or []:
        merged = deep_merge(merged, load_raw(base / inc))
        merged.pop("base_dir", None)
    raw = deep_merge(merged, raw) if merged else raw
    raw.setdefault("base_dir", str(base))
    return raw


def deep_merge(base: Any, over: Any) -> Any:
    """Merge ``over`` into a deep copy of ``base``. Lists of dicts with 'id' are merged by id.

    Quantity dicts (value/unit/source) are atomic: an override replaces value, unit, provenance
    and bounds together, so template bounds never leak onto an overridden value.
    """
    if isinstance(over, dict) and {"value", "unit", "source"} <= set(over):
        return copy.deepcopy(over)
    if isinstance(base, dict) and isinstance(over, dict):
        out = copy.deepcopy(base)
        for k, v in over.items():
            out[k] = deep_merge(out[k], v) if k in out else copy.deepcopy(v)
        return out
    if (isinstance(base, list) and isinstance(over, list) and base and over
            and all(isinstance(x, dict) and "id" in x for x in base + over)):
        out = [copy.deepcopy(x) for x in base]
        idx = {x["id"]: i for i, x in enumerate(out)}
        for x in over:
            if x["id"] in idx:
                out[idx[x["id"]]] = deep_merge(out[idx[x["id"]]], x)
            else:
                out.append(copy.deepcopy(x))
        return out
    return copy.deepcopy(over)


def expand_templates(raw: dict) -> dict:
    raw = copy.deepcopy(raw)
    templates = raw.get("templates", {}) or {}
    out = []
    for s in raw.get("sections", []):
        t = s.get("template")
        if t:
            if t not in templates:
                raise ValueError(f"section {s.get('id')}: unknown template {t}")
            tmpl = copy.deepcopy(templates[t])
            desc = tmpl.pop("description", "")
            merged = deep_merge(tmpl, s)
            merged.setdefault("template_description", desc)
            out.append(merged)
        else:
            out.append(copy.deepcopy(s))
    raw["sections"] = out
    return raw


_TOK = re.compile(r"([^.\[\]]+)(?:\[([^\]]+)\])?")


def _split(path: str):
    toks = []
    for part in path.split("."):
        m = _TOK.fullmatch(part)
        if not m:
            raise ValueError(f"bad override path element '{part}' in '{path}'")
        toks.append((m.group(1), m.group(2)))
    return toks


def _descend(node, key, sel, create=False):
    if key not in node:
        if create:
            node[key] = {}
        else:
            raise KeyError(f"path key '{key}' not found")
    node = node[key]
    if sel is not None:
        if not isinstance(node, list):
            raise KeyError(f"'{key}' is not a list; cannot select [{sel}]")
        hits = [x for x in node if isinstance(x, dict) and str(x.get("id")) == sel]
        if not hits:
            raise KeyError(f"no item with id '{sel}' in '{key}'")
        node = hits[0]
    return node


def get_path(raw: dict, path: str):
    toks = _split(path)
    node = raw
    for key, sel in toks:
        node = _descend(node, key, sel)
    return node


def set_path(raw: dict, path: str, value, create_missing: bool = False) -> None:
    if "::" in path:
        # "<path to a passive option's set mapping>::<override path>[.value]"
        outer, inner = path.split("::", 1)
        mapping = get_path(raw, outer)
        key = inner[:-len(".value")] if inner.endswith(".value") and inner not in mapping else inner
        if key not in mapping:
            raise KeyError(f"passive override '{inner}' not found under '{outer}'")
        if key != inner:
            mapping[key] = {**mapping[key], "value": copy.deepcopy(value)}
        else:
            mapping[key] = copy.deepcopy(value)
        return
    toks = _split(path)
    node = raw
    for key, sel in toks[:-1]:
        node = _descend(node, key, sel, create=create_missing)
    key, sel = toks[-1]
    if sel is not None:
        node = _descend(node, key, None)
        hits = [i for i, x in enumerate(node) if isinstance(x, dict) and str(x.get("id")) == sel]
        if not hits:
            raise KeyError(f"no item with id '{sel}' in '{key}'")
        node[hits[0]] = copy.deepcopy(value)
        return
    if not isinstance(node, dict):
        raise KeyError(f"override target parent of '{path}' is not a mapping")
    # creating the final key is allowed: the schema forbids unknown fields, so typos still fail
    node[key] = copy.deepcopy(value)


def find_uncertain(raw: dict, prefix: str = "") -> List[Tuple[str, dict]]:
    """List (path_to_value, quantity_dict) for every quantity that declares bounds."""
    found = []
    if isinstance(raw, dict):
        if {"value", "unit", "source"} <= set(raw) and raw.get("bounds") is not None:
            return [(prefix + ".value" if prefix else "value", raw)]
        for k, v in raw.items():
            if k == "templates":
                continue
            if k == "set" and prefix.startswith("passive_options[") and isinstance(v, dict):
                for inner, q in v.items():
                    if isinstance(q, dict) and {"value", "unit", "source"} <= set(q) and q.get("bounds") is not None:
                        found.append((f"{prefix}.set::{inner}.value", q))
                continue
            found += find_uncertain(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(raw, list):
        for i, x in enumerate(raw):
            sel = x.get("id") if isinstance(x, dict) else None
            if sel is None:
                continue
            found += find_uncertain(x, f"{prefix}[{sel}]")
    return found


def apply_passive(raw: dict, option: dict) -> None:
    for path, value in option["set"].items():
        set_path(raw, path, value)


def build(raw: dict, passive: Optional[List[str]] = None) -> VenueConfig:
    """Expand templates, apply passive options (by id), validate."""
    r = expand_templates(raw)
    opts = {p["id"]: p for p in r.get("passive_options", [])}
    for pid in passive or []:
        if pid not in opts:
            raise ValueError(f"unknown passive option {pid}")
        apply_passive(r, opts[pid])
    return VenueConfig.model_validate(r)


def load_config(path, passive=None) -> VenueConfig:
    return build(load_raw(path), passive)
