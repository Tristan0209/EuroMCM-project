from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

METRICS = {
    'V': {'MAE': .3545, 'RMSE': .4198, 'Spearman': .4943, 'Low-quality AUC': .7935},
    'VC₂': {'MAE': .2925, 'RMSE': .3789, 'Spearman': .6063, 'Low-quality AUC': .8604},
    'VC₁₁': {'MAE': .2023, 'RMSE': .3101, 'Spearman': .6970, 'Low-quality AUC': .9146},
}

COLORS = {
    'MAE': '#EF8979',
    'RMSE': '#71B8ED',
    'Spearman': '#84D8D7',
    'Low-quality AUC': '#B8AEEA',
}
OUTPUT_DIR = Path(__file__).resolve().parent


def draw_panel(ax, names, metrics, ylim, ylabel, subtitle):
    x = np.arange(len(names), dtype=float)
    width = .31
    offsets = (-.17, .17)
    for metric, offset in zip(metrics, offsets):
        vals = np.array([METRICS[n][metric] for n in names])
        bars = ax.bar(x + offset, vals, width=width, color=COLORS[metric],
                      linewidth=0, zorder=3)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width()/2, v + (ylim[1]-ylim[0])*.018,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9.2,
                    color='#344054', fontweight='medium')
    ax.set_xticks(x, names)
    ax.tick_params(axis='x', length=0, pad=10, labelsize=12)
    ax.tick_params(axis='y', length=0, labelsize=9, colors='#667085')
    ax.set_ylim(*ylim)
    ax.set_xlim(-.56, len(names)-.44)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=9, color='#475467')

    ax.text(.5, -.19, subtitle, transform=ax.transAxes, ha='center', va='top',
            fontsize=11.5, fontweight='semibold', color='#202939')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#E9EDF2', linewidth=.9)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#D9DFE7')


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'pdf.fonttype':42, 'ps.fonttype':42,
                         'savefig.facecolor':'white'})
    names = list(METRICS)
    fig, axs = plt.subplots(1, 2, figsize=(11.9, 4.9), dpi=150)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.82, wspace=.24)

    legend = [Patch(facecolor=COLORS[m], label=m) for m in COLORS]
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.075,.975),
               ncol=4, frameon=False, columnspacing=2.2, handlelength=1.4,
               handleheight=.95, handletextpad=.55, fontsize=10.5)
    draw_panel(axs[0], names, ('MAE','RMSE'), (0,.49), 'Error  ↓  lower is better',
               '(a)  Prediction error')
    draw_panel(axs[1], names, ('Spearman','Low-quality AUC'), (0,1.05),
               'Score  ↑  higher is better', '(b)  Ranking and detection')
    fig.text(.075, .07, 'V: validity only      VC₂: validity + 2-model consensus      VC₁₁: validity + 11-model consensus',
             fontsize=9.2, color='#667085', ha='left')
    for ext in ('png','pdf'):
        path = OUTPUT_DIR / f'tacva_comparison_refined.{ext}'
        fig.savefig(path, dpi=350, bbox_inches='tight', pad_inches=.2)
        print('Saved:', path)
    plt.close(fig)

if __name__ == '__main__':
    main()
