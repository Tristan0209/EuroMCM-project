from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics as st
import sys
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from matplotlib.transforms import blended_transform_factory
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
SCENARIOS = ['Baseline', 'Difficult', 'Prediction Error', 'Demand Growth']
LABELS = ['Baseline', 'Math/Code 40%', 'Prediction error', 'Demand +50%']
POLICIES = ['Greedy', 'Spatial_MILP']
TASKS = ['General', 'Math', 'Code']
BLUE, ORANGE, INK = '#2878B5', '#E6862A', '#414B56'
COLORS = dict(zip(POLICIES, [BLUE, ORANGE]))
NAMES = dict(zip(POLICIES, ['Greedy', 'Spatial MILP']))
STEMS = ['Fig1_Scenario_Dashboard', 'Fig2_Carbon_Accounting', 'Fig3_Group_Tradeoffs']
TRACES, LAYOUT = [], []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def pick(rows, **filters):
    result = sorted([r for r in rows if all(r[k] == str(v) for k, v in filters.items())], key=lambda r: int(r['seed']))
    assert len(result) == 3, (filters, len(result))
    assert len({r['seed'] for r in result}) == 3
    return result


def series(rows, field, scale=1, **filters):
    return [float(r[field]) * scale for r in pick(rows, **filters)]


def trace(figure, panel, category, policy, metric, values, unit, comparison='raw seed-level metric'):
    assert len(values) == 3 and all(math.isfinite(v) for v in values)
    record = dict(figure=figure, panel=panel, category=category, policy=policy, metric=metric,
                  values=values, mean=st.mean(values), minimum=min(values), maximum=max(values),
                  unit=unit, comparison=comparison)
    TRACES.append(record)
    return record['mean']


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9.5,
        'axes.titlesize': 10.5, 'axes.titleweight': 'semibold', 'axes.titlepad': 10,
        'axes.labelsize': 9.5, 'xtick.labelsize': 8.7, 'ytick.labelsize': 9,
        'axes.edgecolor': '#7D858E', 'axes.linewidth': .65,
        'axes.spines.top': False, 'axes.spines.right': False,
        'text.color': '#25313C', 'axes.labelcolor': '#25313C',
        'xtick.color': '#4B5661', 'ytick.color': '#4B5661',
        'grid.color': '#D8DDE3', 'grid.linewidth': .55, 'grid.alpha': .65,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 400, 'axes.unicode_minus': False,
    })


def base(ax, title, categories):
    ax.set_title(title, loc='left')
    ax.set_yticks(range(len(categories)), categories)
    ax.set_ylim(len(categories) - .65, -.65)
    ax.tick_params(axis='y', length=0, pad=6)
    ax.grid(axis='x', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)


def legend(fig, extra=False, y=.985):
    handles = [Line2D([], [], marker='o', linestyle='none', color=COLORS[p], markersize=5.6, label=NAMES[p]) for p in POLICIES]
    if extra:
        handles.insert(0, Line2D([], [], marker='D', linestyle='none', color=INK, markersize=4.5, label='Shared model choices'))
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               bbox_to_anchor=(.52, y), fontsize=9, handletextpad=.45, columnspacing=1.8)


def pair(ax, y, greedy, spatial, labels=False, precision=3):
    ax.plot([greedy, spatial], [y, y], color='#CDD4DC', lw=1.4, solid_capstyle='round', zorder=1)
    for p, x in zip(POLICIES, [greedy, spatial]):
        ax.plot(x, y, 'o', color=COLORS[p], markersize=5.9, markeredgecolor='white', markeredgewidth=.6, zorder=3)
        if labels:
            offset = 8 if p == 'Greedy' else -10
            ax.annotate(f'{x:.{precision}f}', (x, y), xytext=(0, offset), textcoords='offset points',
                        color=COLORS[p], ha='center', va='center', fontsize=8.5, annotation_clip=False)


def finish(fig, out, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text().strip():
            continue
        bbox = obj.get_window_extent(renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > width+1 or bbox.y1 > height+1:
            outside.append(obj.get_text())
    assert not outside, ('Text beyond figure canvas', stem, outside)
    sizes = {}
    for ext in ['png', 'svg', 'pdf']:
        directory = out / ('output/pdf' if ext == 'pdf' else ext)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f'{stem}.{ext}'

        fig.savefig(target, dpi=400, transparent=False)
        sizes[ext] = {'path': str(target.relative_to(out)), 'sha256': sha(target)}
    LAYOUT.append({'figure': stem, 'size_inches': list(fig.get_size_inches()), 'dpi': 400,
                   'canvas_text_check': 'PASS', 'files': sizes})
    plt.close(fig)


def figure1(results, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.85))
    fig.subplots_adjust(left=.195, right=.98, bottom=.19, top=.86, wspace=.68, hspace=.77)
    legend(fig, extra=True, y=1)
    for ax, field, scale, title, unit, precision in [
        (axes[0, 0], 'mean_historical_quality', 1, '(a) Historical Score', 'Score', 6),
        (axes[0, 1], 'answer_pass_rate', 100, '(b) Threshold-pass rate', '% of all inputs', 3),
    ]:
        panel = 'a' if scale == 1 else 'b'
        base(ax, title, LABELS)
        baseline = series(results, field, scale, scenario='Baseline', policy='Greedy')
        for y, scenario in enumerate(SCENARIOS):
            values = series(results, field, scale, scenario=scenario, policy='Greedy')
            other = series(results, field, scale, scenario=scenario, policy='Spatial_MILP')
            assert all(abs(a-b)<1e-12 for a,b in zip(values, other)), 'Quality must be shared across policies'
            actual = trace(1, panel, scenario, 'Shared routing', field, values, unit)
            delta = trace(1, panel, scenario, 'Shared routing', field+'_delta',
                          [a-b for a,b in zip(values, baseline)], 'Score' if scale==1 else 'percentage points',
                          'same-seed scenario minus Baseline; shared model choices')
            ax.plot([0, delta], [y, y], color=INK, lw=1.6, zorder=2)
            ax.plot(delta, y, 'D', color=INK, ms=4.8, zorder=3)
            label = f'{actual:.{precision}f}' + ('%' if scale==100 else '')
            left = delta < -.004 if scale==1 else delta < -.1
            ax.annotate(label, (delta,y), xytext=(-6 if left else 6,0), textcoords='offset points',
                        ha='right' if left else 'left', va='center', fontsize=8.3)
        ax.axvline(0, color='#A5ADB6', lw=.9, zorder=1)
        if scale == 1:
            ax.set_xlim(-.07, .035)
            ax.set_xticks([-.06, -.03, 0, .03])
            ax.set_xlabel('Change from baseline\n(Score units)')
        else:
            ax.set_xlim(-.35, 1.8)
            ax.set_xticks([0, .5, 1, 1.5])
            ax.set_xlabel('Change from baseline\n(percentage points)')
    for ax, field, scale, title, panel in [
        (axes[1,0], 'p95_response_s', 1, '(c) Completed-request P95', 'c'),
        (axes[1,1], 'service_violation_rate', 100, '(d) Service violation', 'd')]:
        base(ax, title, LABELS)
        for y, scenario in enumerate(SCENARIOS):
            means = [trace(1,panel,scenario,p,field,series(results,field,scale,scenario=scenario,policy=p),
                           's; completed requests' if scale==1 else '% of all inputs') for p in POLICIES]
            pair(ax, y, *means)
        if scale == 1:
            ax.set_xlim(0,14)
            ax.set_xticks([0,4,8,12])
            ax.set_xlabel('Response time (s)')
        else:
            values = [float(r[field])*scale for r in results]
            assert min(values)>0, 'Do not replace zeros or add a pseudocount for the log scale'
            ax.set_xscale('log')
            ax.set_xlim(.01,2.5)
            ax.xaxis.set_major_locator(FixedLocator([.01,.1,1]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: f'{x:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel('Violation (% of all inputs)\nLog scale')
    fig.text(.5,.022,'Means of 3 paired seeds; no confidence intervals. Quality panels share model choices.',
             ha='center',fontsize=8.5,color='#5E6873')
    finish(fig,out,STEMS[0])


def figure2(results, out):
    fig, axes = plt.subplots(1,3,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[1,1,.95]})
    fig.subplots_adjust(left=.075,right=.89,bottom=.29,top=.73,wspace=.95)
    legend(fig, y=1.01)
    scenarios = ['Baseline','Demand Growth']
    for ax,field,scale,panel,title,unit in [
        (axes[0],'co2e_g',.001,'a','(a) Total facility\ncarbon','kgCO2e'),
        (axes[1],'incremental_co2e_g',1,'b','(b) Operational\nincrement','gCO2e')]:
        ax.set_title(title,loc='left')
        ax.grid(axis='y',zorder=0)
        ax.set_axisbelow(True)
        for s,scenario in enumerate(scenarios):
            for p,policy in enumerate(POLICIES):
                value=trace(2,panel,scenario,policy,field,series(results,field,scale,scenario=scenario,policy=policy),unit)
                x=s+(p-.5)*.32
                ax.bar(x,value,width=.29,color=COLORS[policy],zorder=2)
                if scale<1:

                    ax.annotate(f'{value:.3f}',(s,value),xytext=(0,5 if p==0 else 18),
                                textcoords='offset points',ha='center',va='bottom',fontsize=8.5,color=COLORS[policy])
                else:
                    ax.annotate(f'{value:.1f}',(x,value),xytext=(0,4),
                                textcoords='offset points',ha='center',va='bottom',fontsize=8.5)
        ax.set_xticks([0,1],['Baseline','Demand\n+50%'])
        ax.set_xlim(-.5,1.5)
        ax.set_ylabel(unit,labelpad=4)
        ax.set_ylim(0,57 if scale<1 else 290)
        ax.set_yticks([0,20,40] if scale<1 else [0,100,200])
    ax=axes[2]
    base(ax,'(c) Signed carbon\ncontrasts',['Baseline\nmatched demand','Growth\nmatched demand','Growth vs\noriginal Greedy'])
    base_g=series(results,'co2e_g',scenario='Baseline',policy='Greedy')
    base_s=series(results,'co2e_g',scenario='Baseline',policy='Spatial_MILP')
    growth_g=series(results,'co2e_g',scenario='Demand Growth',policy='Greedy')
    growth_s=series(results,'co2e_g',scenario='Demand Growth',policy='Spatial_MILP')
    for y,(category,a,b) in enumerate([('Baseline matched',base_s,base_g),('Growth matched',growth_s,growth_g),('Growth vs original',growth_s,base_g)]):
        value=trace(2,'c',category,'Spatial - Greedy','co2e_g_difference',[x-z for x,z in zip(a,b)],'gCO2e',
                    'same-demand Spatial minus Greedy' if y<2 else 'Growth Spatial minus Baseline Greedy')
        ax.plot([0,value],[y,y],color=INK,lw=1.7,zorder=2)
        ax.plot(value,y,'o',color=INK,ms=5.5,zorder=3)
        transform=blended_transform_factory(ax.transAxes,ax.transData)
        ax.text(1.07,y,f'{value:+.3f}',transform=transform,ha='left',va='center',fontsize=8.8)
    ax.axvline(0,color='#A5ADB6',lw=.9,zorder=1)
    ax.set_xlim(-120,20)
    ax.set_xticks([-100,-50,0])
    ax.set_xlabel('gCO2e; Spatial - Greedy',fontsize=8.7)
    ax.tick_params(axis='y',labelsize=8.5)
    fig.text(.5,.025,'Fixed 12-hour idle retained in (a). Contrasts are 3-seed means; growth is externally imposed.',
             ha='center',fontsize=8.5,color='#5E6873')
    finish(fig,out,STEMS[1])


def figure3(groups, out):
    fig, axes = plt.subplots(1,3,figsize=(7.2,2.65))
    fig.subplots_adjust(left=.105,right=.985,bottom=.25,top=.75,wspace=.55)
    legend(fig,y=1.02)
    specs = [('violation_rate',100,'(a) Violation\nrate','% of all inputs',4,(0,2.05),[0,1,2]),
             ('completed_p95_s',1,'(b) Completed\nP95','Response time (s)',3,(0,21),[0,10,20]),
             ('valid_pass_rate',100,'(c) Valid delivered\npass','% of all inputs',3,(60,100),[60,80,100])]
    for panel,(ax,(field,scale,title,unit,precision,limits,ticks)) in enumerate(zip(axes,specs)):
        base(ax,title,TASKS)
        ax.set_ylim(2.7,-.65)
        if panel:
            ax.tick_params(axis='y',labelleft=False)
        for y,task in enumerate(TASKS):
            means=[trace(3,chr(97+panel),task,p,field,series(groups,field,scale,scenario='Demand Growth',policy=p,task_type=task),
                         's; completed requests' if scale==1 else '% of task-category inputs') for p in POLICIES]
            pair(ax,y,*means,labels=True,precision=precision)
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.set_xlabel(unit)
    fig.text(.5,.02,'3-seed means | Rates: all category inputs | P95: completions only | Labels: exact rounded means',
             ha='center',fontsize=8.5,color='#5E6873')
    finish(fig,out,STEMS[2])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default_source=HERE/'source_tables' if (HERE/'source_tables/Task4_Simple_Results.csv').is_file() else HERE.parent
    parser.add_argument('--source-dir',type=Path,default=default_source)
    parser.add_argument('--output-dir',type=Path,default=HERE)
    args=parser.parse_args()
    source,out=args.source_dir.resolve(),args.output_dir.resolve()
    assert out!=source and out!=source/'figures','Never overwrite frozen source directories'
    out.mkdir(parents=True,exist_ok=True)
    paths=[source/'Task4_Simple_Results.csv',source/'final_review/group_service.csv']
    source_hashes={str(p):sha(p) for p in paths}
    results,groups=map(load,paths)
    assert len(results)==24 and len(groups)==72
    assert {r['scenario'] for r in results}==set(SCENARIOS)
    assert {r['policy'] for r in results}==set(POLICIES)
    seeds=sorted({int(r['seed']) for r in results})
    assert len(seeds)==3
    for scenario in SCENARIOS:
        g,s=[pick(results,scenario=scenario,policy=p) for p in POLICIES]
        assert [r['world_id'] for r in g]==[r['world_id'] for r in s]
        assert all(float(r['idle_co2e_g'])==float(g[0]['idle_co2e_g']) for r in g+s)
    for r in results:
        assert abs(float(r['co2e_g'])-float(r['idle_co2e_g'])-float(r['incremental_co2e_g']))<1e-8
    code=pick(groups,scenario='Baseline',policy='Greedy',task_type='Code')
    assert {int(r['n_input']) for r in code}=={86} and {int(r['unique_prompt_groups']) for r in code}=={85}
    style()
    figure1(results,out)
    figure2(results,out)
    figure3(groups,out)
    audit=out/'audit'
    audit.mkdir(exist_ok=True)
    manifest=dict(status='PASS',seeds=seeds,original_executions=24,new_experiments=0,new_inference=0,new_training=0,
        source_hashes=source_hashes,policies=COLORS,shared_quality_color=INK,traces=TRACES,layout=LAYOUT,
        summaries='Arithmetic mean of seed-level quantities, no pooling. No CI or range bars shown.',
        log_axis='Figure1(d) only; no zero, epsilon replacement, pseudocount or omitted nonpositive observation.',
        scope='Display-only; never import or rerun Task2/Task3/Task4 experiment code.')
    (audit/'plot_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'plot_data.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields=['figure','panel','category','policy','metric','seed','value','unit','comparison']
        writer=csv.DictWriter(handle,fields)
        writer.writeheader()
        for t in TRACES:
            for seed,value in zip(seeds,t['values']):
                writer.writerow({**{k:t[k] for k in fields if k not in ['seed','value']},'seed':seed,'value':repr(value)})

    for path in paths:
        copy=out/'source_tables'/path.relative_to(source)
        copy.parent.mkdir(parents=True,exist_ok=True)
        if path.resolve()!=copy.resolve():
            shutil.copyfile(path,copy)
        assert sha(copy)==source_hashes[str(path)]
    assert {str(p):sha(p) for p in paths}==source_hashes
    print('PASS: 3 figures x PNG(400dpi)/editable SVG/vector PDF; saved observations only; no new experiment')


if __name__=='__main__':
    main()
