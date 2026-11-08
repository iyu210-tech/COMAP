"""VenueCool local dashboard (Streamlit + Plotly).

    streamlit run app/dashboard.py

The dashboard only calls the engine in ``venuecool``; no physics lives here.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import venuecool  # noqa: E402,F401  (sets NUMBA_CACHE_DIR)
from venuecool import metrics  # noqa: E402
from venuecool.config import expand_templates, load_raw_text  # noqa: E402
from venuecool.runner import Study, comparison_rows, provenance_table, timeseries_frame, _clean  # noqa: E402

EX = ROOT / "examples"
OUT = ROOT / "outputs"
st.set_page_config(page_title="VenueCool", layout="wide")


# ------------------------------------------------------------------ cached engine calls
@st.cache_resource(show_spinner="Simulating matched baseline and scenarios...")
def run_study(yaml_text: str, base_dir: str, weather: str | None, scenarios: tuple):
    raw = load_raw_text(yaml_text, base_dir)
    study = Study(raw, weather_file=weather)
    outs, comps = study.compare_all(list(scenarios))
    return study, outs, comps


def section_polygons(cfg):
    return {s.id: (s.name, np.array(s.footprint + [s.footprint[0]], float)) for s in cfg.sections}


# ------------------------------------------------------------------ sidebar: configuration
st.sidebar.title("VenueCool")
examples = sorted(p.name for p in EX.glob("*.yaml"))
choice = st.sidebar.selectbox("Example configuration", examples, index=examples.index("bus_stop.yaml") if "bus_stop.yaml" in examples else 0)
uploaded = st.sidebar.file_uploader("...or upload a YAML configuration", type=["yaml", "yml"])
base_dir = str(EX)
default_text = uploaded.getvalue().decode() if uploaded else (EX / choice).read_text()
wopts = {"configuration default": None,
         "synthetic hot/dry": str(EX / "data" / "weather_hot_dry.csv"),
         "synthetic warm/humid": str(EX / "data" / "weather_warm_humid.csv")}
wsel = st.sidebar.radio("Weather", list(wopts))
wup = st.sidebar.file_uploader("...or upload weather CSV", type=["csv"])
weather = wopts[wsel]
if wup is not None:
    p = OUT / "uploads"
    p.mkdir(parents=True, exist_ok=True)
    (p / wup.name).write_bytes(wup.getvalue())
    weather = str(p / wup.name)

with st.sidebar.expander("Edit configuration (YAML)", expanded=False):
    key = f"yaml::{choice}::{uploaded.name if uploaded else ''}"
    text = st.text_area("YAML", value=default_text, height=400, key=key)
try:
    raw_preview = expand_templates(load_raw_text(text, base_dir))
    scen_ids = [s["id"] for s in raw_preview.get("scenarios", []) if s["id"] != "baseline"]
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Configuration error: {e}")
    st.stop()
sel = st.sidebar.multiselect("Interventions (matched against baseline)", scen_ids, default=scen_ids)
go_btn = st.sidebar.button("Run / refresh", type="primary")

if "ran" not in st.session_state:
    st.session_state.ran = False
if go_btn:
    st.session_state.ran = True
if not st.session_state.ran:
    st.title("VenueCool - cooling assessment for open-air and semi-open venues")
    st.info("Choose a configuration and weather in the sidebar, then press **Run / refresh**.")
    st.stop()

try:
    study, outs, comps = run_study(text, base_dir, weather, tuple(sel))
except Exception as e:  # noqa: BLE001
    st.error(f"Run failed: {type(e).__name__}: {e}")
    st.stop()

cfg = study.cfg0
st.title(cfg.meta.name)
status = cfg.meta.data_status
wstat = study.weather.meta.get("data_status", "UNSPECIFIED")
(st.warning if "SYNTHETIC" in (status + wstat) else st.success)(
    f"Configuration data status: **{status}** | weather: **{wstat}** ({Path(cfg.weather.file).name}) | "
    f"solar site from {study.site_source}. Comfort index: SET, target {cfg.comfort.target.value:.1f} C, "
    f"decision basis: {cfg.comfort.decision_basis}.")
st.caption("Results are modeled person-equivalent-degree-hours for comparing configurations; they are not measured "
           "satisfaction and not a medical heat-safety guarantee. Person-hours are not unique visitors.")

ok = {k: o for k, o in outs.items() if o.ok}
for k, o in outs.items():
    if not o.ok:
        st.error(f"{k}: solver failed - {o.sim.message}")
comp_df = comparison_rows(outs, comps)

tabs = st.tabs(["Section map", "Local conditions", "Venue comparison", "Target & validity", "Sensitivity", "Search",
                "Configuration & provenance", "Exports"])

# ------------------------------------------------------------------ map
with tabs[0]:
    c1, c2, c3 = st.columns([2, 2, 3])
    scen = c1.selectbox("Scenario", list(ok), key="map_s")
    var = c2.selectbox("Colour by", ["mean SET (occupied populations)", "air temperature", "relative humidity",
                                     "sun fraction", "floor surface temperature"])
    o = ok[scen]
    t_h = o.sim.t / 3600
    k = c3.slider("Time (h)", float(t_h[0]), float(t_h[-2]), float(min(15.0, t_h[-2])), step=float(t_h[1] - t_h[0]))
    ki = int(np.argmin(np.abs(t_h - k)))
    vals = {}
    for i, sid in enumerate(o.sim.section_ids):
        if var == "air temperature":
            vals[sid] = o.sim.T_air[ki, i]
        elif var == "relative humidity":
            vals[sid] = o.sim.RH[ki, i] * 100
        elif var == "sun fraction":
            vals[sid] = o.sim.drivers["fsun"][ki, i]
        elif var == "floor surface temperature":
            vals[sid] = o.sim.T_nodes[ki, o.model.floor_node[i]]
        else:
            d = o.comfort_df
            d = d[(d.section == sid) & (d.k == min(ki, d.k.max()))]
            w = d.people.to_numpy()
            vals[sid] = float(np.nansum(d.set_extended * w) / w.sum()) if w.sum() > 0 else float(d.set_extended.mean())
    vmin, vmax = np.nanmin(list(vals.values())), np.nanmax(list(vals.values()))
    fig = go.Figure()
    cs = px.colors.sequential.YlOrRd
    for sid, (name, poly) in section_polygons(cfg).items():
        v = vals[sid]
        frac = 0.5 if vmax == vmin else (v - vmin) / (vmax - vmin)
        col = px.colors.sample_colorscale(cs, [frac])[0]
        fig.add_trace(go.Scatter(x=poly[:, 0], y=poly[:, 1], fill="toself", fillcolor=col, line=dict(color="black"),
                                 name=sid, hovertext=f"{name}: {v:.2f}", hoverinfo="text", mode="lines"))
        cx, cy = poly[:-1, 0].mean(), poly[:-1, 1].mean()
        fig.add_annotation(x=cx, y=cy, text=f"<b>{name}</b><br>{v:.1f}", showarrow=False,
                           font=dict(color="black"), bgcolor="rgba(255,255,255,0.8)")
    for dev in o.model.devs:
        for si, frac_, groups in dev.serves:
            poly = section_polygons(cfg)[cfg.sections[si].id][1]
            fig.add_trace(go.Scatter(x=[poly[:-1, 0].mean()], y=[poly[:-1, 1].min() + 0.15 * np.ptp(poly[:, 1])],
                                     mode="markers+text", marker=dict(symbol="star", size=14, color="royalblue"),
                                     text=[dev.id], textposition="top center", showlegend=False,
                                     hovertext=f"{dev.id} ({dev.preset}) -> {', '.join(groups)} [{frac_:.0%} airflow]"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(colorscale=cs, cmin=vmin, cmax=vmax, color=[vmin], showscale=True,
                                         colorbar=dict(title=var.split(" (")[0]))))
    fig.update_layout(height=520, yaxis=dict(scaleanchor="x", title="y (m)"), xaxis=dict(title="x (m)"),
                      title=f"{var} at {t_h[ki]:.2f} h - {scen}", showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Values are shown at the modeled section resolution (one value per section). Stars mark devices "
               "(placed at their served sections).")

# ------------------------------------------------------------------ local conditions
with tabs[1]:
    c1, c2 = st.columns(2)
    sec = c1.selectbox("Section", [s.id for s in cfg.sections])
    scs = c2.multiselect("Scenarios", list(ok), default=list(ok)[:3])
    ts = pd.concat([timeseries_frame(ok[s]) for s in scs]) if scs else pd.DataFrame()
    if len(ts):
        t = ts[ts.section == sec]
        fig = px.line(t, x="t_h", y="T_air_C", color="scenario", title=f"Air temperature - {sec}")
        base_t = t[t.scenario == scs[0]]
        fig.add_scatter(x=base_t.t_h, y=base_t.T_out_C, name="outdoor air", line=dict(dash="dot", color="grey"))
        st.plotly_chart(fig, width="stretch")
        st.plotly_chart(px.line(t, x="t_h", y="RH_pct", color="scenario", title=f"Relative humidity - {sec}"),
                        width="stretch")
        cf = pd.concat([ok[s].comfort_df.assign(scenario=s) for s in scs])
        cf = cf[cf.section == sec]
        cf["population"] = cf.group + "/" + cf.sun_state
        fig = px.line(cf, x="t_h", y="set_extended", color="scenario", line_dash="population",
                      title=f"SET by exposure group and sun state - {sec}")
        fig.add_hline(y=cfg.comfort.target.value, line_dash="dash", annotation_text="target")
        st.plotly_chart(fig, width="stretch")
        st.plotly_chart(px.line(cf, x="t_h", y="mrt", color="scenario", line_dash="population",
                                title=f"Mean radiant temperature - {sec}"), width="stretch")
        st.plotly_chart(px.area(cf[cf.scenario == scs[0]], x="t_h", y="people", color="population",
                                title=f"People present (mean) - {sec}"), width="stretch")

# ------------------------------------------------------------------ venue comparison
with tabs[2]:
    st.dataframe(comp_df, width="stretch")
    good = comp_df[comp_df.status == "ok"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(good, x="scenario", y="relief_Kph", title="Useful relief vs baseline (K person-h, matched)"),
                    width="stretch")
    c2.plotly_chart(px.bar(good, x="scenario", y="attainment", title="Fraction of assessed person-hours meeting target"),
                    width="stretch")
    c1.plotly_chart(px.bar(good, x="scenario", y=["electricity_kWh", "water_m3"], barmode="group",
                           title="Electricity (kWh) and water (m3) per simulated period"), width="stretch")
    c2.plotly_chart(px.scatter(good, x="season_total_cost_usd", y="relief_Kph", text="scenario",
                               title="Relief vs season total cost (USD)"), width="stretch")
    rows = []
    for sid, c in comps.items():
        for s, v in c["sections"].items():
            rows.append(dict(scenario=sid, section=s, **v))
    if rows:
        st.subheader("Relief by section")
        st.dataframe(pd.DataFrame(rows), width="stretch")
    bd = []
    for sid, o in ok.items():
        for s, d in o.sim.breakdown_J.items():
            bd.append(dict(scenario=sid, section=s, **{k: v / 3.6e6 for k, v in d.items()}))
    st.subheader("Enthalpy flows into section air (kWh, reporting period)")
    st.caption("Negative outdoor_exchange = cooling (or heat) carried away to the surroundings - the 'open freezer' loss.")
    st.dataframe(pd.DataFrame(bd).round(2), width="stretch")

# ------------------------------------------------------------------ validity
with tabs[3]:
    rows = []
    for sid, o in ok.items():
        for s, v in o.summary["sections"].items():
            rows.append(dict(scenario=sid, section=s, **v))
    sdf = pd.DataFrame(rows)
    st.dataframe(sdf, width="stretch")
    fig = px.bar(sdf, x="scenario", y=["supported_person_hours", "extrapolated_person_hours", "unassessed_person_hours"],
                 facet_col="section", title="Model-validity classes of occupied person-hours")
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"**Poorly served (attainment < {cfg.comfort.poorly_served_attainment:.0%})**")
    st.write({sid: o.summary["poorly_served"] for sid, o in ok.items()})
    cons = pd.DataFrame([dict(scenario=sid, **o.sim.conservation, **{f"invalid:{k}": v for k, v in o.sim.invalid.items()})
                         for sid, o in ok.items()])
    st.subheader("Conservation residuals and invalid states")
    st.dataframe(cons, width="stretch")

# ------------------------------------------------------------------ sensitivity
with tabs[4]:
    wtag = Path(weather or cfg.weather.file).stem.replace("weather_", "")
    udir = OUT / Path(choice).stem / wtag / "uncertainty"
    if (udir / "uncertainty_summary.json").exists():
        us = json.loads((udir / "uncertainty_summary.json").read_text())
        dr = pd.read_csv(udir / "uncertainty_draws.csv")
        st.caption(f"Loaded saved batch: {us['samples']} LHS draws, seed {us['seed']}, {us['workers']} workers, "
                   f"{us['wall_s']:.0f} s (matched baseline per draw).")
        rel = [c for c in dr.columns if c.endswith(":relief_Kph")]
        st.plotly_chart(px.box(dr[rel].melt(), x="variable", y="value", points="all",
                               title="Relief distribution across uncertainty draws (K person-h)"), width="stretch")
        s_sel = st.selectbox("Scenario for drivers", list(us["sensitivity"]))
        sens = pd.DataFrame(us["sensitivity"][s_sel]).head(15)
        if len(sens):
            noise = us.get("spearman_noise_level")
            fig = px.bar(sens, x="spearman_rho", y="parameter", orientation="h",
                         title=f"Spearman rank correlation with relief - {s_sel}")
            if noise:
                fig.add_vrect(x0=-noise, x1=noise, fillcolor="grey", opacity=0.2, line_width=0)
                st.caption(f"Grey band: |rho| <= {noise:.2f} is within sampling noise for {us.get('n_ok')} draws.")
            st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(us["outputs"]).T, width="stretch")
    else:
        st.info(f"No saved uncertainty batch in {udir}. Run e.g. `venuecool uncertainty examples/{choice} "
                f"--samples 64 --workers 11`.")
    from venuecool.uncertainty import uncertain_parameters
    up = uncertain_parameters(study)
    st.write(f"{len(up)} parameters with declared bounds:")
    st.dataframe(pd.DataFrame([dict(path=p, nominal=q['value'], unit=q['unit'], low=q['bounds'][0], high=q['bounds'][1],
                                    source=q['source']) for p, q in up]), width="stretch")

# ------------------------------------------------------------------ search
with tabs[5]:
    sdir = OUT / Path(choice).stem / wtag / "search"
    if (sdir / "search_results.csv").exists():
        sr = pd.read_csv(sdir / "search_results.csv")
        ss = json.loads((sdir / "search_summary.json").read_text()) if (sdir / "search_summary.json").exists() else {}
        st.caption(f"{len(sr)} configurations evaluated; status counts: {ss.get('status_counts')}")
        fig = px.scatter(sr, x="electricity_kWh", y="attainment", color="status", hover_name="label",
                         title="Tested configurations: energy vs target attainment")
        if ss.get("target"):
            fig.add_hline(y=ss["target"]["min_attainment"], line_dash="dash", annotation_text="target")
        st.plotly_chart(fig, width="stretch")
        if ss.get("selected"):
            st.success(f"Selected: **{ss['selected']['label']}** - verification: {ss.get('verification')}")
        elif ss:
            st.error(ss.get("message"))
        st.dataframe(sr.sort_values(["status", "electricity_kWh"]), width="stretch")
    else:
        st.info(f"No saved search in {sdir}. Run `venuecool search examples/{choice} --workers 11`.")

# ------------------------------------------------------------------ provenance
with tabs[6]:
    st.dataframe(provenance_table(study.raw), width="stretch")
    st.json({"weather_metadata": study.weather.meta, "occupancy_metadata": study.occupancy.meta,
             "notes": cfg.meta.notes})

# ------------------------------------------------------------------ exports
with tabs[7]:
    st.download_button("comparison.csv", comp_df.to_csv(index=False), "comparison.csv")
    ts_all = pd.concat([timeseries_frame(o) for o in ok.values()])
    st.download_button("timeseries.csv", ts_all.to_csv(index=False), "timeseries.csv")
    cf_all = pd.concat([o.comfort_df.assign(scenario=s) for s, o in ok.items()])
    st.download_button("comfort_populations.csv", cf_all.to_csv(index=False), "comfort_populations.csv")
    summary = {sid: dict(metrics=o.summary, costs=o.costs, ledgers=o.sim.ledgers, conservation=o.sim.conservation,
                         comparison=comps.get(sid)) for sid, o in ok.items()}
    st.download_button("summary.json", json.dumps(_clean(summary), indent=2), "summary.json")
    st.download_button("config_used.yaml", text, "config_used.yaml")
