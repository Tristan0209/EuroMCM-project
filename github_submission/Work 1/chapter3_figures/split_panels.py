from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
import pandas as pd

import plot_task1_figures as source

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "separate_panels"
OUT.mkdir(exist_ok=True)


def export_panels(fig, stem):
    axes = list(fig.axes)
    count = 3 if "3_2" in stem else 4
    for original_legend in fig.legends:
        original_legend.set_visible(False)
    for i in range(count):
        for ax in axes:
            ax.set_visible(False)
        ax = axes[i]
        ax.set_visible(True)
        selected = [ax]
        if "3_3" in stem and i == 2:
            axes[4].set_visible(True)
            selected.append(axes[4])
        legend = None
        if "3_4" in stem and i in (0, 1):

            handles = [Line2D([], [], color=source.COLORS[k], lw=2,
                              label=f"Arrival rate x{k:g}") for k in [.5, 1, 2]]
            handles += [Line2D([], [], linestyle="none", marker=source.MARKERS[k],
                               color=source.MUTED, markersize=6,
                               label=f"{k} replica" + ("s" if k > 1 else "")) for k in [1, 2, 4]]
            handles += [Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
                               markeredgecolor=source.MUTED, markersize=6,
                               label="Hollow: service requirement fails")]
            legend = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.23),
                               ncol=3, fontsize=7.5, frameon=False, borderaxespad=0,
                               columnspacing=1, handlelength=1.4, handletextpad=.4)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = Bbox.union([a.get_tightbbox(renderer) for a in selected])
        bounds = bounds.transformed(fig.dpi_scale_trans.inverted()).padded(.12)
        fig.savefig(OUT / f"{stem}_{chr(97+i)}.pdf", bbox_inches=bounds, facecolor="white")
        if legend is not None:
            legend.remove()
    plt.close(fig)


def main():
    source.export = export_panels
    d = ROOT / "data"
    q = pd.read_csv(d / "quality_reported_aggregates.csv")
    c = pd.read_csv(d / "cost_derived_accounting_responses.csv")
    s = pd.read_csv(d / "environment_18_configuration_means.csv")
    energy = pd.read_csv(d / "conversation_baseline_energy_components_per_1000.csv", index_col="replicas")
    emissions = pd.read_csv(d / "conversation_two_replica_ember_emissions.csv")
    source.quality_figure(q)
    source.cost_figure(c)
    source.environment_figure(s, energy, emissions)
    paths = sorted(OUT.glob("*.pdf"))
    if len(paths) != 11:
        raise RuntimeError(f"Expected 11 panels; found {len(paths)}")
    with zipfile.ZipFile(ROOT / "separate_panels.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.name)
    print("Exported 11 individual panel PDFs and separate_panels.zip")


if __name__ == "__main__":
    main()
