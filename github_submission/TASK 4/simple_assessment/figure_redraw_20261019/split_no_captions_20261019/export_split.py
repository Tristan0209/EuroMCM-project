from pathlib import Path
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OLD=HERE.parent
spec=importlib.util.spec_from_file_location('original_task4_plot',OLD/'redraw_task4_figures.py')
plot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
captured=[]
plot.finish=lambda fig,out,stem: captured.append((stem,fig))
plot.style()
results=plot.load(OLD/'source_tables/Task4_Simple_Results.csv')
groups=plot.load(OLD/'source_tables/final_review/group_service.csv')
plot.figure1(results,HERE)
plot.figure2(results,HERE)
plot.figure3(groups,HERE)
prior=json.loads((OLD/'audit/plot_manifest.json').read_text(encoding='utf-8'))
assert plot.TRACES==prior['traces'],'All original plotted data must match exactly'
records=[]


def save(fig,stem,crop=False):
    fig.canvas.draw()
    bbox=fig.get_tightbbox(fig.canvas.get_renderer()).padded(.08) if crop else None
    for ext in ['png','svg','pdf']:
        target=HERE/ext/f'{stem}.{ext}'
        target.parent.mkdir(exist_ok=True)
        fig.savefig(target,dpi=400,facecolor='white',transparent=False,bbox_inches=bbox)
    records.append(stem)


for stem,fig in captured:

    assert len(fig.texts)==1
    fig.texts[0].remove()
    if stem!='Fig2_Carbon_Accounting':
        save(fig,stem)
    else:
        a,b,c=fig.axes
        c.set_visible(False)

        center=(a.get_position().x0+b.get_position().x1)/2
        fig.legends[0].set_bbox_to_anchor((center,1.01),transform=fig.transFigure)
        save(fig,'Fig2ab_Facility_and_Operational',crop=True)
        a.set_visible(False)
        b.set_visible(False)
        c.set_visible(True)
        fig.legends[0].set_visible(False)
        save(fig,'Fig2c_Signed_Carbon_Contrasts',crop=True)
    plot.plt.close(fig)
(HERE/'verification.json').write_text(json.dumps({'status':'PASS','data_exactly_equal_to_original':True,
    'only_changes':['remove bottom figure-wide text','split Figure2 into a+b and c; center a+b legend'],
    'outputs':records,'new_experiments':0},indent=2),encoding='utf-8')
print('PASS: four figures; all original plot traces exactly unchanged')
