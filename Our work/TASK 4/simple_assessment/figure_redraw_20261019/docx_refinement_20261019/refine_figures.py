from pathlib import Path
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OLD=HERE.parent
basis=HERE/'plotting_basis.py' if (HERE/'plotting_basis.py').is_file() else OLD/'redraw_task4_figures.py'
spec=importlib.util.spec_from_file_location('task4_original_plot',basis)
plot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
captured=[]
plot.finish=lambda fig,out,stem:captured.append((stem,fig))
plot.style()
source=HERE/'source_tables' if (HERE/'source_tables').is_dir() else OLD/'source_tables'
results=plot.load(source/'Task4_Simple_Results.csv')
groups=plot.load(source/'final_review/group_service.csv')
plot.figure1(results,HERE)
plot.figure2(results,HERE)
plot.figure3(groups,HERE)
reference=HERE/'reference_plot_manifest.json' if (HERE/'reference_plot_manifest.json').is_file() else OLD/'audit/plot_manifest.json'
prior=json.loads(reference.read_text(encoding='utf-8'))
assert plot.TRACES==prior['traces'],'Original numerical observations and comparisons must be identical'
exports=[]


def save(fig,stem,crop=False):
    fig.canvas.draw()
    bbox=fig.get_tightbbox(fig.canvas.get_renderer()).padded(.09) if crop else None
    files={}
    for ext in ['png','svg','pdf']:
        folder=HERE/ext
        folder.mkdir(exist_ok=True)
        path=folder/f'{stem}.{ext}'
        fig.savefig(path,dpi=400,facecolor='white',transparent=False,bbox_inches=bbox)
        files[ext]={'path':str(path.relative_to(HERE)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    exports.append({'figure':stem,'files':files})


for stem,fig in captured:

    assert len(fig.texts)==1
    fig.texts[0].remove()
    if stem=='Fig1_Scenario_Dashboard':
        fig.legends[0].remove()
        plot.legend(fig,extra=False,y=1)
        fig.axes[1].set_title('(b) Historical\nthreshold-pass rate',loc='left')
        for ax,panel,precision in [(fig.axes[0],'a',4),(fig.axes[1],'b',3)]:
            annotations=list(ax.texts)
            traces=[t for t in plot.TRACES if t['figure']==1 and t['panel']==panel and t['metric'].endswith('_delta')]
            assert len(annotations)==len(traces)==4
            for text,trace in zip(annotations,traces):
                text.set_text(f"{trace['mean']:+.{precision}f}")
                text.set_fontsize(9)
        save(fig,stem)
    elif stem=='Fig2_Carbon_Accounting':
        a,b,c=fig.axes
        a.set_position([.10,.29,.25,.44])
        b.set_position([.52,.29,.25,.44])
        b.set_title('(b) Operational\ncarbon increment',loc='left')
        c.set_visible(False)
        fig.legends[0].set_bbox_to_anchor(((.10+.77)/2,1.01),transform=fig.transFigure)
        save(fig,'Fig2ab_Facility_and_Operational',crop=True)
        a.set_visible(False)
        b.set_visible(False)
        c.set_visible(True)
        fig.legends[0].set_visible(False)
        c.set_position([.48,.29,.30,.44])
        c.set_title('(c) Key carbon differences',loc='left')
        c.set_xlabel('Difference in total carbon\n(gCO2e)',fontsize=9.5)
        c.set_yticks([0,1,2],[
            'Baseline matched demand\nSpatial - Greedy',
            'Growth matched demand\nSpatial - Greedy',
            'Cross-demand comparison\nGrowth Spatial - Baseline Greedy'])
        c.tick_params(axis='y',labelsize=9)
        save(fig,'Fig2c_Key_Carbon_Differences',crop=True)
    else:
        titles=['(a) Service\nviolation rate','(b) Completed-request\nP95','(c) Valid delivered\npass rate']
        for j,(ax,title) in enumerate(zip(fig.axes,titles)):
            ax.set_title(title,loc='left')
            ax.set_xlabel('Response time (s)' if j==1 else 'Rate (%)')

            annotations=list(ax.texts)
            values=[t['mean'] for t in plot.TRACES if t['figure']==3 and t['panel']==chr(97+j)]
            assert len(values)==len(annotations)==6
            for text,value in zip(annotations,values):
                if j==2:
                    text.remove()
                else:

                    text.set_text(f'{value:.{3 if j==0 else 2}f}')
                    text.set_fontsize(8.8)
        save(fig,stem)
    plot.plt.close(fig)
(HERE/'verification.json').write_text(json.dumps({'status':'PASS','data_exactly_equal_to_original':True,
    'checked_traces':len(plot.TRACES),'new_experiments':0,'bottom_captions':False,
    'figure2_split_preserved':True,'exports':exports},indent=2),encoding='utf-8')
print('PASS: original 61 numerical traces unchanged; DOCX display refinements applied')
