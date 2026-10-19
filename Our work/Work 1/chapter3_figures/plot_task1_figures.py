from pathlib import Path
import hashlib
import json
import warnings

warnings.filterwarnings("ignore", message="Pandas requires version")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TEAL, BLUE, OCHRE = "#277F8A", "#537FAF", "#C58B3C"
INK, MUTED, GRID = "#243B49", "#6F7F89", "#E3E9EC"
PALE, IDLE, SHARED, RED = "#CAD3D7", "#7C9FA6", "#BDCDD2", "#BD604A"
COLORS = {0.5: BLUE, 1.0: TEAL, 2.0: OCHRE}
MARKERS = {1: "o", 2: "s", 4: "^"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#9BAAB1", "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlepad": 17, "axes.labelsize": 10,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def style(ax, grid="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid)
    ax.tick_params(length=4, pad=5)


def title(ax, letter, text):
    ax.set_title(f"({letter})  {text}", loc="left")


def export(fig, stem):

    for suffix in ("pdf", "svg", "png"):
        fig.savefig(ROOT / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)


def load_and_validate():
    q = pd.read_csv(DATA / "quality_reported_aggregates.csv")
    require(q.variant.tolist() == ["V", "VC2", "VC11"], "Unexpected quality variants")
    require(q.additional_calls.tolist() == [0, 1, 10], "Unexpected call counts")
    require(q.low_score_roc_auc.isna().tolist() == [False, False, True],
            "VC11 AUC must stay missing")
    c = pd.read_csv(DATA / "cost_760_reported_27_cases.csv")
    require(len(c) == 27 and not c.duplicated(["region", "arrival_multiplier", "replicas"]).any(),
            "Expected 27 distinct cost cases")
    require(set(c.n_requests) == {5000}, "Cost table must contain 5000-request cases")
    require(np.allclose(c.cost_total_eur, c.cost_electricity_eur + c.cost_hardware_eur
                        + c.cost_maintenance_eur, atol=1.6e-6, rtol=0), "Cost identity failed")
    require((c.cost_maintenance_eur == 0).all(), "Price response assumes zero maintenance")
    s = pd.read_csv(DATA / "simulation_summary.csv")
    r = pd.read_csv(DATA / "simulation_runs.csv")
    manifest = json.loads((DATA / "run_manifest.json").read_text(encoding="utf-8"))
    profile = manifest["active_profile"]
    for key, expected in {"shared_w": 1500, "gpu_idle_w": 175, "gpu_busy_w": 760,
                          "gpu_count_per_server": 8, "pue": 1.2}.items():
        require(profile[key] == expected, f"Wrong power boundary: {key}")
    require(manifest["config"]["network_control_s"] == .05, "Unexpected fixed overhead")
    require(manifest["config"]["repetitions"] == 5, "Expected five repetitions")
    require(set(r.requests) == {5000}, "Use 5000 replayed requests per workload")
    for column, expected in {"shared_w": 1500, "gpu_idle_w": 175, "gpu_busy_w": 760}.items():
        require(set(r[column]) == {expected}, f"Mixed versions in runs: {column}")
    keys = ["workload", "rate_multiplier", "replicas", "repeat"]
    require(not r.duplicated(keys + ["carbon_scenario"]).any(), "Duplicate carbon bookkeeping")

    for column in ("facility_kwh", "horizon_including_drain_s", "p95_latency_s", "late_fraction"):
        require((r.groupby(keys)[column].max() - r.groupby(keys)[column].min()).max() < 1e-9,
                f"Carbon scenarios unexpectedly change trajectory: {column}")
    eea = r[r.carbon_scenario == "EEA_EU27_2024"].copy()
    summary = s[s.carbon_scenario == "EEA_EU27_2024"].copy()
    require(len(eea) == 90 and len(summary) == 18, "Expected 18 configurations x 5 runs")
    require((summary.repetitions == 5).all(), "Expected five-repetition means")
    group = ["workload", "rate_multiplier", "replicas"]
    require((eea.groupby(group).size() == 5).all(), "Wrong per-configuration repetition count")
    for run_col, summary_col in [("facility_kwh", "facility_kwh_mean"),
                                  ("p95_latency_s", "p95_latency_s_mean"),
                                  ("late_fraction", "late_fraction_mean")]:
        means = eea.groupby(group)[run_col].mean().sort_index()
        supplied = summary.set_index(group)[summary_col].sort_index()
        require(np.allclose(means, supplied, rtol=1e-11, atol=1e-10),
                f"Runs/summary mismatch: {run_col}")
    summary["facility_kwh_per_1000_requests"] = summary.facility_kwh_mean / 5
    summary["service_feasible"] = ((summary.p95_latency_s_mean <= 10)
                                    & (summary.late_fraction_mean <= .05))
    summary.to_csv(DATA / "environment_18_configuration_means.csv", index=False)


    eea["shared_kwh"] = 1.2 * 1500 * eea.horizon_including_drain_s / 3.6e6
    eea["eight_gpu_idle_kwh"] = 1.2 * 8 * 175 * eea.horizon_including_drain_s / 3.6e6
    eea["busy_increment_kwh"] = eea.facility_kwh - eea.shared_kwh - eea.eight_gpu_idle_kwh
    service_work_s = (eea.mean_latency_s - eea.mean_queue_wait_s - .05) * eea.requests
    independent_busy = 1.2 * (760 - 175) * service_work_s / 3.6e6
    identity_error = float(np.max(np.abs(eea.busy_increment_kwh - independent_busy)))
    require(identity_error < 1e-9, "Independent busy-work energy identity failed")
    require((eea.busy_increment_kwh >= 0).all(), "Negative busy increment")
    energy = eea[(eea.workload == "azure_conversation") & (eea.rate_multiplier == 1)].copy()
    cols = ["shared_kwh", "eight_gpu_idle_kwh", "busy_increment_kwh", "facility_kwh"]
    energy = energy.groupby("replicas")[cols].mean() / 5
    energy.to_csv(DATA / "conversation_baseline_energy_components_per_1000.csv")

    emissions = s[(s.workload == "azure_conversation") & (s.rate_multiplier == 1)
                  & (s.replicas == 2) & s.carbon_scenario.str.startswith("Ember_")].copy()
    require(len(emissions) == 3 and (emissions.carbon_unit == "gCO2e/kWh").all(),
            "Regional panel needs three Ember CO2e scenarios")
    require(np.allclose(emissions.facility_kwh_mean, energy.loc[2, "facility_kwh"] * 5),
            "Regional panel must use the same energy")
    emissions.to_csv(DATA / "conversation_two_replica_ember_emissions.csv", index=False)
    c["whole_server_cost_eur"] = c.cost_electricity_eur + c.cost_hardware_eur * 8 / c.replicas
    c["acquisition_plus25_total_change_pct"] = 25 * c.cost_hardware_eur / c.cost_total_eur
    c["electricity_plus50_total_change_pct"] = 50 * c.cost_electricity_eur / c.cost_total_eur
    c.to_csv(DATA / "cost_derived_accounting_responses.csv", index=False)
    report = {
        "status": "passed", "cost_cases": len(c), "environment_configurations": len(summary),
        "independent_environment_runs": len(eea), "carbon_bookkeeping_rows": len(r),
        "repetitions_per_configuration": 5, "requests_per_run": 5000,
        "whole_server_baseline_w": 2900, "busy_increment_identity_max_error_kwh": identity_error,
        "quality_macro_mae_reduction_pct": float(100 * (1 - q.macro_mae.iloc[1] / q.macro_mae.iloc[0])),
        "acquisition_plus25_range_pct": [float(c.acquisition_plus25_total_change_pct.min()),
                                         float(c.acquisition_plus25_total_change_pct.max())],
        "electricity_plus50_range_pct": [float(c.electricity_plus50_total_change_pct.min()),
                                         float(c.electricity_plus50_total_change_pct.max())],
        "minimum_feasible_replicas": summary[summary.service_feasible].groupby(
            ["workload", "rate_multiplier"]).replicas.min().reset_index().to_dict("records"),
        "conversation_two_to_four_p95_reduction_pct": float(100 * (1 - summary[
            (summary.workload == "azure_conversation") & (summary.rate_multiplier == 1)
            & (summary.replicas == 4)].p95_latency_s_mean.iloc[0] / summary[
            (summary.workload == "azure_conversation") & (summary.rate_multiplier == 1)
            & (summary.replicas == 2)].p95_latency_s_mean.iloc[0])),
        "source_sha256": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in
            ["quality_reported_aggregates.csv", "cost_760_reported_27_cases.csv",
             "simulation_summary.csv", "simulation_runs.csv", "run_manifest.json"]},
        "interpretation": "Existing aggregates and fixed-trajectory accounting; no retraining or simulation rerun.",
    }
    return q, c, summary, energy, emissions, report


def quality_figure(q):
    fig = plt.figure(figsize=(10.6, 5.5))
    gs = fig.add_gridspec(2, 2, left=.09, right=.965, bottom=.19, top=.88,
                          width_ratios=[1.22, 1], hspace=1.05, wspace=.56)
    ax = fig.add_subplot(gs[:, 0])
    title(ax, "a", "Evidence has a generation cost")
    x = np.log1p(q.additional_calls)
    ax.plot(x, q.macro_mae, color="#B8C6CC", lw=2, zorder=2)
    colors = ["#889CA6", TEAL, OCHRE]
    for i, row in q.iterrows():
        ax.scatter(x.iloc[i], row.macro_mae, s=130, color=colors[i], edgecolors="white", linewidths=1.6, zorder=3)
        offset, align = ((10, 10), "left") if i < 2 else ((-10, 10), "right")
        ax.annotate(f"{row.variant}\n{row.macro_mae:.4f}", (x.iloc[i], row.macro_mae),
                    xytext=offset, textcoords="offset points", color=colors[i], ha=align, fontsize=11)
    ax.annotate("17.5% lower macro MAE", (x.iloc[1], q.macro_mae.iloc[1]),
                xytext=(.06, .28), textcoords="axes fraction", fontsize=10, color=TEAL,
                bbox={"boxstyle": "round,pad=.3", "fc": "#EFF6F6", "ec": "none"},
                arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.1})
    ax.set(xlim=(-.16, 2.67), ylim=(.175, .385), xticks=x, xticklabels=["0", "1", "10"],
           ylabel="Macro MAE (lower is better)",
           xlabel="Additional calls per assessed response\n(log(1 + calls) spacing)")
    ax.text(.04, .045, "60 held-out benchmarks", transform=ax.transAxes, color=MUTED, fontsize=9)
    style(ax)
    for panel, field, rows, limits, ticks, label, heading in [
        ("b", "low_score_roc_auc", q.iloc[:2], (.765, .90), [.775, .80, .825, .85, .875],
         "ROC-AUC (higher is better)", "Detecting low-score responses"),
        ("c", "within_request_spearman", q, (.53, .633), [.54, .56, .58, .60, .62],
         "Spearman correlation (higher is better)", "Within-request model ranking")]:
        a = fig.add_subplot(gs[0 if panel == "b" else 1, 1])
        title(a, panel, heading)
        for i, row in rows.iterrows():
            y, val = i, row[field]
            a.hlines(y, limits[0], val, color=GRID, lw=7)
            a.scatter(val, y, s=95, c=colors[i], edgecolors="white", linewidths=1.2, zorder=3)
            a.annotate(f"{val:.4f}", (val, y), xytext=(6, 0), textcoords="offset points",
                       va="center", fontsize=10)
        a.set(xlim=limits, ylim=(len(rows)-.5, -.5), yticks=range(len(rows)),
              yticklabels=rows.variant, xticks=ticks, xlabel=label)
        a.spines["left"].set_visible(False)
        a.tick_params(axis="y", length=0)
        style(a, "x")
    export(fig, "fig_3_2_evidence_value")


def cost_figure(c):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.1))
    fig.subplots_adjust(left=.09, right=.96, bottom=.105, top=.91, hspace=.92, wspace=.58)
    a, b, d, e = axes.flat
    title(a, "a", "Capacity dominates baseline cost")
    base = c[c.arrival_multiplier == 1]
    for j, rep in enumerate([1, 2, 4]):
        part = base[base.replicas == rep].set_index("region").loc[["FR", "DE-LU", "PL"]]
        x = j * 4 + np.arange(3)
        a.bar(x, part.cost_hardware_eur, width=.7, color=PALE)
        a.bar(x, part.cost_electricity_eur, width=.7, bottom=part.cost_hardware_eur,
              color=[TEAL, BLUE, OCHRE])
        a.text(j*4+1, part.cost_total_eur.max()+.23,
               f"{part.cost_total_eur.min():.2f} - {part.cost_total_eur.max():.2f}", ha="center", fontsize=9)
        a.text(j*4+1, -.23, f"{rep} replica" + ("s" if rep > 1 else ""),
               transform=a.get_xaxis_transform(), ha="center", fontsize=10, fontweight="bold")
    a.set(ylim=(0, 12.5), ylabel="Batch cost (EUR)", xticks=[0,1,2,4,5,6,8,9,10],
          xticklabels=["FR", "DE-LU", "PL"]*3)
    a.tick_params(axis="x", labelsize=8)
    a.text(.02, .96, "Grey: allocated depreciation\nColoured caps: electricity", transform=a.transAxes,
           va="top", color=MUTED, fontsize=8.5)
    style(a)

    title(b, "b", "Depreciation allocation")
    fr = base[base.region == "FR"].set_index("replicas").loc[[1, 2, 4]]
    for y, (_, row) in enumerate(fr.iterrows()):
        b.hlines(y, row.cost_total_eur, row.whole_server_cost_eur, color="#D7DFE2", lw=4)
        b.scatter(row.cost_total_eur, y, s=95, c=TEAL, zorder=3)
        b.scatter(row.whole_server_cost_eur, y, s=85, marker="D", c=RED, zorder=3)
        b.annotate(f"{row.cost_total_eur:.2f}", (row.cost_total_eur, y), xytext=(-4, 10),
                   textcoords="offset points", ha="center", color=TEAL, fontsize=9)
        b.annotate(f"{row.whole_server_cost_eur:.2f}", (row.whole_server_cost_eur, y), xytext=(10, 0),
                   textcoords="offset points", va="center", color=RED, fontsize=9)
    b.set(xlim=(0, 25.8), ylim=(2.6, -.65), yticks=[0, 1, 2],
          yticklabels=["1 replica", "2 replicas", "4 replicas"], xlabel="Batch cost in France (EUR)")
    b.text(.02, 1.01, "Allocated share", transform=b.transAxes, color=TEAL, fontsize=9)
    b.text(.68, 1.01, "Whole server", transform=b.transAxes, color=RED, fontsize=9)
    style(b, "x")

    title(d, "c", "Price response: France, 1 replica")
    row = fr.loc[1]
    mx, my = np.linspace(.75, 1.25, 41), np.linspace(.5, 1.5, 41)
    xx, yy = np.meshgrid(mx, my)
    zz = 100 * ((xx*row.cost_hardware_eur + yy*row.cost_electricity_eur) / row.cost_total_eur - 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("price", [TEAL, "#F4F2E9", "#CC8E72"])

    im = d.pcolormesh(xx, yy, zz, cmap=cmap, shading="gouraud", vmin=-30, vmax=30,
                     edgecolors="none", linewidth=0, antialiased=False, rasterized=False)
    contours = d.contour(xx, yy, zz, levels=[-20,-10,0,10,20], colors=INK, linewidths=.65, alpha=.7)

    label_positions = [(.79,.61), (.84,.93), (.96,1.2), (1.12,.97), (1.19,1.35)]
    d.clabel(contours, fmt=lambda value: f"{value:+.0f}%", fontsize=8, manual=label_positions)
    d.scatter(1, 1, s=145, marker="*", edgecolors=INK, facecolors="white", linewidths=1.2, zorder=5)
    d.set(xlabel="Acquisition-price multiplier", ylabel="Electricity-price multiplier",
          xticks=[.75,1,1.25], yticks=[.5,1,1.5], xlim=(.75,1.25), ylim=(.5,1.5))

    pos = d.get_position()
    ca = fig.add_axes([pos.x1+.012, pos.y0, .011, pos.height])


    cy = np.linspace(-30, 30, 41)
    cx, cy_grid = np.meshgrid([0, 1], cy)
    ca.pcolormesh(cx, cy_grid, cy_grid, cmap=cmap, shading="gouraud", vmin=-30, vmax=30)
    ca.set(xticks=[], yticks=[-30,-15,0,15,30], ylim=(-30,30))
    ca.yaxis.tick_right()
    ca.yaxis.set_label_position("right")
    ca.tick_params(labelsize=8)
    ca.set_ylabel("Cost change (%)", fontsize=9)
    flat = pd.DataFrame({"acquisition_multiplier": xx.ravel(), "electricity_multiplier": yy.ravel(),
                         "cost_change_pct": zz.ravel()})
    flat.to_csv(DATA / "cost_fr_one_replica_analytical_surface.csv", index=False)

    title(e, "d", "Economic sensitivity: 27 cases")

    jitter = .075 * np.sin(np.arange(len(c)) * 2.399963229728653)
    for y, col, color in [(1, "acquisition_plus25_total_change_pct", TEAL),
                          (0, "electricity_plus50_total_change_pct", OCHRE)]:
        vals = c[col]
        e.scatter(vals, y+jitter, color=color, s=25, alpha=.76, edgecolors="white", linewidths=.35)
        e.hlines(y, vals.min(), vals.max(), color=color, lw=3, zorder=4)
        e.text((vals.min()+vals.max())/2, y+.19, f"{vals.min():.2f}% - {vals.max():.2f}%",
               color=color, ha="center", fontsize=10)
    e.set(xlim=(0, 27.5), ylim=(-.45,1.55), yticks=[0,1],
          yticklabels=["Electricity\nprice +50%", "Acquisition\nprice +25%"],
          xlabel="Increase in total cost (%)")
    e.text(.98, .025, "Enumerated ranges; unequal price shocks", transform=e.transAxes,
           ha="right", color=MUTED, fontsize=7.5)
    style(e, "x")
    export(fig, "fig_3_3_cost_structure_boundary")


def environment_figure(s, energy, emissions):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0))
    fig.subplots_adjust(left=.09, right=.965, bottom=.105, top=.81, hspace=.82, wspace=.55)
    a, b, c, d = axes.flat
    legend = [Line2D([],[],color=COLORS[k],lw=2,label=f"Arrival rate x{k:g}") for k in [.5,1,2]]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.52,.995), ncol=3,
               frameon=False, fontsize=10, columnspacing=2.2)
    marker_legend = [Line2D([],[],linestyle="none",marker=MARKERS[k],color=MUTED,
                          markersize=7,label=f"{k} replica"+("s" if k>1 else "")) for k in [1,2,4]]
    marker_legend.append(Line2D([],[],linestyle="none",marker="o",markerfacecolor="white",
                               markeredgecolor=MUTED,markersize=7,label="Hollow: service requirement fails"))
    fig.legend(handles=marker_legend, loc="upper center", bbox_to_anchor=(.52,.952), ncol=4,
               frameon=False, fontsize=9, columnspacing=1.8)
    for ax, workload, label, letter in [(a,"azure_code","Code generation","a"),
                                        (b,"azure_conversation","Conversation","b")]:
        title(ax, letter, label)
        part = s[s.workload == workload]
        for rate in [.5,1,2]:
            rows = part[part.rate_multiplier == rate].set_index("replicas").loc[[1,2,4]]
            color = COLORS[rate]
            ax.plot(rows.p95_latency_s_mean, rows.facility_kwh_per_1000_requests, color=color, lw=1.4, alpha=.8)
            for rep, row in rows.iterrows():
                px, py = row.p95_latency_s_mean, row.facility_kwh_per_1000_requests
                ax.scatter(px, py, marker=MARKERS[rep], s=95, edgecolors=color, linewidths=1.5,
                           facecolors=color if row.service_feasible else "white", zorder=4)

                dx, dy = {1:(8,7), 2:(10,-15), 4:(-6,10)}[rep]
                ax.annotate(str(rep), (px,py), xytext=(dx,dy), textcoords="offset points", color=color, fontsize=8.5)
        ax.set_xscale("log")
        ax.set(xlim=(1.8,1600), xlabel="P95 response time (s; log scale)",
               ylabel="Facility energy (kWh / 1,000 requests)")
        ax.xaxis.set_major_locator(FixedLocator([2,10,100,1000]))
        ax.xaxis.set_major_formatter(FixedFormatter(["2","10","100","1000"]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim((.45,3.25) if workload == "azure_code" else (.7,5.25))
        ax.axvline(10, color=RED, lw=1.1, ls=(0,(4,3)))
        ax.text(10.7, .68, "10 s", color=RED, transform=ax.get_xaxis_transform(), fontsize=8.5)
        style(ax)

    title(c, "c", "Energy decomposition: baseline load")
    x, bottom = np.arange(3), np.zeros(3)
    for col, color in [("shared_kwh", SHARED), ("eight_gpu_idle_kwh", IDLE), ("busy_increment_kwh", OCHRE)]:
        vals = energy.loc[[1,2,4], col].to_numpy()
        c.bar(x, vals, bottom=bottom, width=.56, color=color)
        bottom += vals
    for i, total in enumerate(bottom):
        c.text(i, total+.06, f"{total:.4f}", ha="center", fontsize=10)
    c.set(xticks=x, xticklabels=["1","2","4"], xlabel="Service replicas",
           ylabel="Facility energy (kWh / 1,000 requests)", ylim=(0,3.55))
    c.legend(handles=[Patch(color=SHARED,label="Shared host"), Patch(color=IDLE,label="8-GPU idle baseline"),
                       Patch(color=OCHRE,label="Busy increment")], loc="upper left", frameon=False,
             fontsize=8, ncol=2, handlelength=1.1, handletextpad=.5, columnspacing=1)
    style(c)

    title(d, "d", "Same energy, different grid factors")
    for y, (country, color) in enumerate([("France",TEAL),("Germany",BLUE),("Poland",OCHRE)]):
        row = emissions[emissions.carbon_scenario == f"Ember_{country}_2024"].iloc[0]
        val = row.emissions_g_per_1000_requests_mean
        d.hlines(y,0,val,color=color,lw=4,alpha=.5)
        d.scatter(val,y,s=110,color=color,zorder=3)
        d.annotate(f"{val:,.1f}",(val,y),xytext=(9,0),textcoords="offset points",va="center",color=color,fontsize=10)
    d.set(xlim=(0,1850), ylim=(2.55,-.6), yticks=[0,1,2], yticklabels=["France","Germany","Poland"],
           xticks=[0,500,1000,1500], xlabel="Emissions (g CO$_2$e / 1,000 requests)")
    d.text(.98,.04,"Conversation; 2 replicas; baseline load\nEmber 2024 annual factors", transform=d.transAxes,
           ha="right",color=MUTED,fontsize=8)
    style(d,"x")
    export(fig,"fig_3_4_latency_energy_carbon")


def main():
    q, c, s, energy, emissions, report = load_and_validate()
    quality_figure(q)
    cost_figure(c)
    environment_figure(s, energy, emissions)
    report["exports"] = [f"{stem}.{suffix}" for stem in ["fig_3_2_evidence_value",
        "fig_3_3_cost_structure_boundary", "fig_3_4_latency_energy_carbon"] for suffix in ["pdf","svg","png"]]
    (ROOT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Validated inputs and energy identities; exported 3 PDF, 3 SVG, 3 PNG figures at 300 dpi.")
    print("All figures use existing data. No model, queue simulation, or power scan was rerun.")


if __name__ == "__main__":
    main()
