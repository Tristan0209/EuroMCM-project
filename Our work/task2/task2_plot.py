def _plot_architecture_and_risk():
    from pathlib import Path
    import csv, zipfile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import numpy as np

    OUT = Path(__file__).resolve().parent / "outputs" / "task2_figures"
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
         'axes.labelcolor': '#23384a', 'text.color': '#23384a', 'axes.edgecolor': '#b5c0ca', 'xtick.color': '#506273',
         'ytick.color': '#506273'})
    NAVY = '#23384a';
    BLUE = '#3269a8';
    TEAL = '#258778';
    AMBER = '#be8c36';
    RUST = '#ba604e';
    GREY = '#8393a1'

    def save(fig, name):
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.08)
        plt.close(fig)

    def card(ax, x, y, w, h, title, detail='', color=BLUE, fill='#f0f5fb', size=10):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5,rounding_size=2', facecolor=fill, edgecolor=color,
                           linewidth=.9, zorder=3))
        ax.text(x + w / 2, y + h * .66 if detail else y + h / 2, title, ha='center', va='center', fontsize=size,
                fontweight='bold', color=color, zorder=4)
        if detail: ax.text(x + w / 2, y + h * .28, detail, ha='center', va='center', fontsize=8.5, color=NAVY, zorder=4)

    def arrow(ax, start, end, color=GREY, style='-', rad=0):
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10, linewidth=1, color=color, linestyle=style,
                            connectionstyle=f'arc3,rad={rad}', zorder=2))

    def label(ax, x, y, s, color=NAVY, ha='center', size=8.5):
        ax.text(x, y, s, ha=ha, va='center', fontsize=size, color=color,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .7}, zorder=5)

    fig, ax = plt.subplots(figsize=(10.4, 6.3));
    fig.subplots_adjust(0, 0, 1, 1);
    ax.set_xlim(0, 108);
    ax.set_ylim(0, 70);
    ax.axis('off')

    for xy, w, h, fc in [((1, 48), 106, 21, '#f6f8fb'), ((1, 14), 106, 32, '#fbfcfd')]:
        ax.add_patch(
            FancyBboxPatch(xy, w, h, boxstyle='round,pad=.5,rounding_size=1.5', facecolor=fc, edgecolor='#d7e0e8',
                           lw=.8))
    ax.text(3, 66, 'A  OFFLINE POLICY CALIBRATION', fontsize=11.3, fontweight='bold', color=NAVY)
    ax.text(3, 43, 'B  REQUEST-TIME EXECUTION', fontsize=11.3, fontweight='bold', color=NAVY)
    items = [(4, 52, 22, 10, 'Grouped partitions', 'Train / select / confirm', GREY, '#ffffff'),
             (30, 52, 22, 10, 'Fit predictors', 'Quality + calling cost', BLUE, '#edf3fb'),
             (56, 52, 22, 10, 'Search policies', 'Thresholds + answer checks', BLUE, '#edf3fb'),
             (82, 52, 22, 10, 'Calibrate task risk', 'Choose fallback tasks', TEAL, '#eaf5f1')]
    for a in items: card(ax, *a)
    for x in [26, 52, 78]: arrow(ax, (x + 1, 57), (x + 3, 57))

    arrow(ax, (41, 52), (34, 36), BLUE, '--', -.06);
    label(ax, 35, 48, 'Frozen predictors', BLUE)
    arrow(ax, (93, 52), (72, 36), TEAL, '--', -.08);
    label(ax, 83, 48, 'Threshold + fallback map', TEAL)
    card(ax, 4, 26, 17, 10, 'New request', 'Observable features', GREY, '#ffffff')
    card(ax, 25, 26, 20, 10, 'Predict both models', r'$\widehat q_S,\widehat q_L,\widehat C_S,\widehat C_L$', BLUE,
         '#edf3fb', 9.3)
    card(ax, 49, 26, 15, 10, 'Routing score', r'$H_i=\Delta\widehat q_i-0.5\Delta\widehat C_i/c_0$', BLUE, '#edf3fb',
         9.7)
    card(ax, 68, 26, 17, 10, 'Task fallback', 'Registered General?', TEAL, '#eaf5f1', 9.7)
    for x0, x1 in [(21, 25), (45, 49), (64, 68)]: arrow(ax, (x0 + 1, 31), (x1 - 1, 31))
    card(ax, 90, 31, 13, 8, 'Call L', '', BLUE, '#edf3fb', 10)
    card(ax, 90, 17, 13, 8, 'Call S', '', TEAL, '#eaf5f1', 10)
    arrow(ax, (85, 34), (89, 35), BLUE);
    label(ax, 88, 40, 'Code / Math / unknown', BLUE, size=8)

    arrow(ax, (77, 26), (77, 20), TEAL);
    label(ax, 72, 23, 'General', TEAL, size=8)
    ax.text(77, 17.3, r'$H_i\geq\tau\ ?$', ha='center', va='center', fontsize=10, color=TEAL, zorder=4)
    arrow(ax, (82, 19), (90, 32), BLUE);
    label(ax, 86, 25, 'Yes', BLUE, size=8)
    arrow(ax, (82, 17.5), (89, 20.5), TEAL);
    label(ax, 87, 16, 'No', TEAL, size=8)
    ax.text(5, 17, r'Final settings: $\tau=-0.6170992537$; one generation call per request.', fontsize=9, color=NAVY)

    ax.add_patch(
        FancyBboxPatch((1, 1), 106, 10.5, boxstyle='round,pad=.5,rounding_size=1.5', fc='white', ec='#b7bfc8', lw=.9,
                       ls=(0, (4, 3))))
    ax.text(3, 8.6, 'EVALUATED OPTIONS  |  inactive in the selected policy', fontsize=9, fontweight='bold', color=GREY)
    ax.text(3, 4.3, r'Cascade: $S\rightarrow$ check $\rightarrow L$; both calls charged.', fontsize=8.8, color=NAVY)
    ax.text(60, 4.3, 'Feedback: paid audit after delivery; next-batch bias update.', fontsize=8.5, color=NAVY)
    save(fig, 'fig4_routing_architecture_en')


    tasks = ['General', 'Math', 'Code'];
    n = [2746, 853, 43];
    harm = [29, 0, 3]
    uq = np.array([.025963, .021900, .668890]);
    uh = np.array([.019307, .008862, .286013]);
    rat = np.column_stack([uq / .05, uh / .02])
    fig = plt.figure(figsize=(10.2, 4.5));
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=.42, left=.10, right=.97, bottom=.28, top=.79)
    a = fig.add_subplot(gs[0, 0]);
    b = fig.add_subplot(gs[0, 1])
    fig.text(.10, .96, 'a  Task-level checks', fontsize=11, fontweight='bold', color=NAVY)
    fig.text(.56, .96, 'b  Joint score-loss check', fontsize=11, fontweight='bold', color=NAVY)
    cmap = ListedColormap(['#eaf5f1', '#b9dfd1', '#f3d6bc', '#d89b89']);
    norm = BoundaryNorm([0, .5, 1, 5, 15], 4)
    a.imshow(rat, cmap=cmap, norm=norm, aspect='auto');
    a.set_xticks([0, 1], ['Score loss\nbudget = 0.05', 'Harmful events\nbudget = 2%'], fontsize=9);
    a.xaxis.tick_top();
    a.tick_params(axis='both', length=0, pad=8)
    a.set_yticks(range(3), [f'{t}\nn={ng:,}; h={h}' for t, ng, h in zip(tasks, n, harm)], fontsize=9)
    for i in range(3):
        for j in range(2):
            actual = f'{uq[i]:.6f}' if j == 0 else f'{100 * uh[i]:.4f}%'
            a.text(j, i - .10, actual, ha='center', va='center', fontweight='bold', fontsize=11, color=NAVY)
            a.text(j, i + .20, f'{rat[i, j]:.2f}' + r'$\times$ budget', ha='center', va='center', fontsize=8.5,
                   color=NAVY)
    a.set_xticks(np.arange(-.5, 2, .999999), minor=True);
    a.set_yticks(np.arange(-.5, 3, 1), minor=True);
    a.grid(which='minor', color='white', linewidth=3);
    a.tick_params(which='minor', length=0)
    for sp in a.spines.values(): sp.set_visible(False)


    y = np.array([2, 1, 0]);
    v = [0, .019606, .0247];
    colors = [GREY, TEAL, RUST]
    b.axvspan(0, .02, color='#eaf5f1', zorder=0);
    b.axvline(.02, color=RUST, linestyle='--', lw=1.1)
    for yy, val, c in zip(y, v, colors):
        b.hlines(yy, 0, val, color=c, lw=11, alpha=.8, zorder=2);
        b.scatter(val, yy, color=c, s=55, zorder=3, edgecolors='white', linewidth=.8)
        b.text(val + .0007, yy, f'{val:.6f}' if val == .019606 else ('0' if val == 0 else r'$\approx0.0247$'),
               va='center', fontsize=9.3, color=c, fontweight='bold')
    b.set_yticks(y, ['All L', 'General', 'General + Math'], fontsize=9.5);
    b.set_xlim(-.0005, .0325);
    b.set_ylim(-.6, 2.65);
    b.set_xticks([0, .01, .02, .03]);
    b.set_xlabel('Aggregate positive score-loss upper bound', fontsize=9.3, labelpad=9);
    b.grid(axis='x', color='#e3e8ed', lw=.7);
    b.set_axisbelow(True)
    for sp in ['top', 'right', 'left']: b.spines[sp].set_visible(False)
    b.tick_params(axis='y', length=0, pad=9);
    b.tick_params(axis='x', length=3)
    b.text(.02, 2.48, 'Budget = 0.02', fontsize=8.7, color=RUST, ha='center')

    fig.text(.10, .11, 'Below budget', fontsize=8.5, color=TEAL,
             bbox={'boxstyle': 'round,pad=.25', 'fc': '#eaf5f1', 'ec': 'none'})
    fig.text(.265, .11, 'Above budget', fontsize=8.5, color=RUST,
             bbox={'boxstyle': 'round,pad=.25', 'fc': '#f3d6bc', 'ec': 'none'})
    fig.text(.56, .08, 'Selected mix: harmful-event bound = 1.458% < 2%.', fontsize=8.6, color=TEAL)
    fig.text(.025, .015,
             'n: prompt groups; h: harmful groups. Only reported joint cases are shown; General + Math is approximate. Bounds are not error bars.',
             fontsize=8, color='#667785')
    save(fig, 'fig4_risk_fallback_en')

    with (OUT / 'risk_values.csv').open('w', newline='') as f:
        w = csv.writer(f);
        w.writerow(['task', 'prompt_groups', 'harmful_groups', 'score_loss_upper_bound', 'harmful_event_upper_bound',
                    'score_budget', 'event_budget'])
        for row in zip(tasks, n, harm, uq, uh, [.05] * 3, [.02] * 3): w.writerow(row)
    with (OUT / 'joint_score_bounds.csv').open('w', newline='') as f:
        w = csv.writer(f);
        w.writerow(['routing_enabled', 'aggregate_positive_score_loss_upper_bound', 'approximate'])
        w.writerows([['none', 0, False], ['General', .019606, False], ['General+Math', .0247, True]])
    (OUT / 'README.txt').write_text(
        'Chapter 4 visual design examples\n\n1. fig4_routing_architecture_en: schematic of fitted policy and request-time execution. Dashed options were evaluated but are disabled in the selected policy. No request scores are used before generation.\n2. fig4_risk_fallback_en: plotted from supplied chapter aggregates; not a rerun. Panel a colors bounds relative to task budgets; panel b shows only the three reported score-bound cases. General+Math is approximate. All-L relative loss is zero by identical action. The full eight-combination result is not available.\n\nPDF and SVG are vector formats; PNG is 300 dpi. plot_chapter4_design.py reproduces both figures. CSVs contain all plotted numerical data.\n\nFor a final eight-combination constraint-plane chart, export each combination name, task-fallback mask, aggregate Uq/Uh, task-budget feasibility, selection cost, and task-mixture weights. For a request-level decision map and paired-score histogram, export request/group id, partition, task, qhat_S/L, Chat_S/L, H, actual initial/final model, q_S/L, and actual call charges.\n')
    print('Created figure 1 and 2.')


def _plot_decision_and_results():
    from pathlib import Path
    import csv, zipfile
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    OUT = Path(__file__).resolve().parent / "outputs" / "task2_figures"
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#aab8c4',
                         'axes.labelcolor': '#253c4f', 'text.color': '#253c4f', 'xtick.color': '#556b7b',
                         'ytick.color': '#556b7b', 'axes.titlesize': 11, 'axes.titleweight': 'bold'})
    BLUE = '#3269a8';
    TEAL = '#258778';
    NAVY = '#253c4f';
    GREY = '#9daab5';
    AMBER = '#bd8b36';
    RUST = '#bc6553'
    names = ['All L', 'All S', 'Fixed cascade', 'Random routing', 'Pre + fallback']
    cost = np.array([3.22199, .13313, 1.36053, 3.04476, 2.98872]);
    score = np.array([.780641, .547648, .743422, .767451, .776142]);
    pass_pct = np.array([84.199, 56.864, 76.442, 82.656, 83.817]);
    colors = [NAVY, GREY, AMBER, TEAL, BLUE];
    markers = ['s', 'o', '^', 'D', '*']
    fig = plt.figure(figsize=(12.4, 5.1));
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.07], left=.055, right=.99, bottom=.29, top=.84, wspace=.42)
    a, b, c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    x = np.linspace(0, 3, 401);
    tau = -.6170992537;
    boundary = .5 * x + tau
    a.fill_between(x, boundary, 1, color='#e5edf8');
    a.fill_between(x, -1, boundary, color='#e3f2ec');
    a.plot(x, boundary, color=BLUE, lw=1.6)
    a.axhline(0, color='white', lw=.9);
    a.set_xlim(0, 3);
    a.set_ylim(-1, 1);
    a.set_xticks([0, 1, 2, 3]);
    a.set_yticks([-1, -.5, 0, .5, 1]);
    a.set_xlabel('Normalized predicted cost increment\n' + r'$(\widehat C_L-\widehat C_S)/c_0$', labelpad=7,
                 fontsize=9.5);
    a.set_ylabel('Predicted score gain\n' + r'$\widehat q_L-\widehat q_S$', fontsize=9.5, labelpad=7)
    a.text(.32, .72, 'Choose L', color=BLUE, fontweight='bold', fontsize=11);
    a.text(1.25, -.70, 'Choose S', color=TEAL, fontweight='bold', fontsize=11)
    a.text(.13, -.93, r'$\tau=-0.6171$', fontsize=8.6, color=NAVY)
    a.text(.11, .965, r'$\Delta\widehat q=0.5x+\tau$', transform=a.transAxes, va='top', fontsize=9, color=BLUE)
    a.set_title('a  Routing boundary', loc='left', pad=31);
    a.text(0, 1.045, 'Schematic; registered General only', transform=a.transAxes, fontsize=8.6, color='#677b89')

    for xx, yy, col, m in zip(cost, score, colors, markers): b.scatter(xx, yy, c=col, marker=m,
                                                                       s=130 if m == '*' else 55, zorder=4,
                                                                       edgecolors='white', linewidth=.6)
    b.set_xlim(-.05, 3.48);
    b.set_ylim(.525, .805);
    b.set_xticks([0, 1, 2, 3]);
    b.set_yticks([.55, .65, .75, .80]);
    b.grid(color='#e7ecf0', lw=.7);
    b.set_axisbelow(True);
    b.set_xlabel('Historical API cost\n(USD / 1,000 requests)', labelpad=7, fontsize=9.5);
    b.set_ylabel('Mean Score', fontsize=9.5, labelpad=7)
    b.set_title('b  Quality and calling cost', loc='left', pad=31);
    b.text(0, 1.045, 'Reported old-test aggregates', transform=b.transAxes, fontsize=8.6, color='#677b89')
    b.annotate('All S', (cost[1], score[1]), xytext=(7, 8), textcoords='offset points', fontsize=8.8, color=GREY)
    b.annotate('Fixed cascade', (cost[2], score[2]), xytext=(-30, -17), textcoords='offset points', fontsize=8.8,
               color=AMBER)
    inset = b.inset_axes([.33, .12, .63, .44]);
    inset.set_facecolor('#f7f9fb')
    for k in [0, 3, 4]: inset.scatter(cost[k], score[k], c=colors[k], marker=markers[k], s=85 if k == 4 else 35,
                                      zorder=4, edgecolors='white', linewidth=.4)
    inset.set_xlim(2.94, 3.27);
    inset.set_ylim(.764, .784);
    inset.set_xticks([3.0, 3.2]);
    inset.set_yticks([.765, .780]);
    inset.tick_params(labelsize=7.5, length=2);
    inset.grid(color='#e5ebef', lw=.5);
    inset.set_title('High-score detail', loc='left', fontsize=8, pad=3)

    quality_loss = np.array([.232993, .037219, .013190, .004499]);
    pass_loss = np.array([.27335, .07757, .01543, .003817]);
    qratio = quality_loss / .02;
    pratio = pass_loss / .02
    c.axvspan(.1, 1, color='#eaf5f1', zorder=0);
    c.axvline(1, color=RUST, lw=1.1, ls='--')
    y = np.arange(4)[::-1]
    for yy, qr, pr in zip(y, qratio, pratio):
        c.scatter(qr, yy + .12, s=40, c=BLUE, marker='o', zorder=4, edgecolor='white', lw=.5)
        c.scatter(pr, yy - .12, s=48, c=AMBER, marker='^', zorder=4, edgecolor='white', lw=.5)

        c.text(qr * 1.12, yy + .12, f'{qr:.2f}' + r'$\times$', fontsize=7.8, va='center', color=BLUE)
        c.text(pr * 1.12, yy - .12, f'{pr:.2f}' + r'$\times$', fontsize=7.8, va='center', color=AMBER)
    c.set_xscale('log');
    c.set_xlim(.1, 33);
    c.set_ylim(-.65, 3.65);
    c.set_xticks([.1, 1, 10]);
    c.set_xticklabels(['0.1', '1', '10']);
    c.set_yticks(y, ['All S', 'Cascade', 'Random', 'Pre'], fontsize=9);
    c.tick_params(axis='y', length=0, pad=6);
    c.minorticks_off();
    c.grid(axis='x', color='#e5ebef', lw=.6);
    c.set_axisbelow(True);
    c.spines['left'].set_visible(False)
    c.set_xlabel('Observed loss / overall budget\n(log scale; each budget = 0.02)', labelpad=7, fontsize=9.5);
    c.set_title('c  Observed budget use', loc='left', pad=31);
    c.text(0, 1.045, 'Relative to all L; descriptive only', transform=c.transAxes, fontsize=8.6, color='#677b89');
    c.text(1, 3.43, 'Budget', ha='center', fontsize=8, color=RUST)
    c.legend(handles=[Line2D([0], [0], marker='o', ls='', color=BLUE, label='Mean Score loss', markersize=5),
                      Line2D([0], [0], marker='^', ls='', color=AMBER, label='Pass-rate loss', markersize=5)],
             loc='lower left', bbox_to_anchor=(-.02, -.02), frameon=False, fontsize=7.8, handletextpad=.3, borderpad=.1,
             labelspacing=.3)
    fig.legend(handles=[
        Line2D([0], [0], marker=markers[k], ls='', color=colors[k], label=names[k], markersize=8 if k == 4 else 6) for k
        in [0, 3, 4]], loc='lower center', bbox_to_anchor=(.50, .08), ncol=3, frameon=False, fontsize=9,
               handletextpad=.3, columnspacing=1.5)
    fig.text(.055, .015,
             'Panel a shows the implemented rule without synthetic requests. Panels b–c use supplied results; budget use is not a calibrated risk bound.',
             fontsize=8.2, color='#657b8b')
    for ext in ['pdf', 'svg', 'png']: fig.savefig(OUT / f'fig4_decision_results_example_en.{ext}', dpi=300,
                                                  bbox_inches='tight', pad_inches=.1, facecolor='white')
    plt.close(fig)
    with (OUT / 'reported_policy_results.csv').open('w', newline='') as f:
        w = csv.writer(f);
        w.writerow(['policy', 'cost_usd_per_1000', 'mean_score', 'pass_rate_percent'])
        w.writerows(zip(names, cost, score, pass_pct))
    with (OUT / 'observed_budget_use.csv').open('w', newline='') as f:
        w = csv.writer(f);
        w.writerow(['policy', 'mean_score_loss', 'pass_rate_loss_fraction', 'score_budget', 'pass_budget',
                    'score_budget_ratio', 'pass_budget_ratio'])
        w.writerows(zip(names[1:], quality_loss, pass_loss, [.02] * 4, [.02] * 4, qratio, pratio))
    (OUT / 'README.txt').write_text(
        'Third-figure design example, not a new experiment.\n\na: exact mathematical boundary from the selected H rule for General. It contains no simulated or observed request points. Code, Math, and unregistered sources bypass this threshold and use L.\nb: supplied aggregate results for 7,335 previously examined test requests.\nc: observed overall score/pass-rate losses divided by each research budget, 0.02. This is a descriptive display, not a new optimization score, risk calibration, or production guarantee. Non-Pre pass losses are computed from rounded reported rates; Pre uses the reported 0.3817 percentage-point loss. The request-level paired-score distribution cannot be reconstructed from these means.\n\nFor the final requested 431-request paired-score histogram, provide actual request-level initial/final model choices and q_S/q_L. For a real decision cloud, also provide qhat_S/L, Chat_S/L, H, partition, task, and prompt-group ID.\n\nPDF/SVG are vector; PNG is 300 dpi. CSV and plotting code are included.\n')
    print('Created figure 3.')


if __name__ == "__main__":
    _plot_architecture_and_risk()
    _plot_decision_and_results()
    from pathlib import Path

    out = Path(__file__).resolve().parent / "outputs" / "task2_figures"
    print("Done. All figures saved in:", out)
