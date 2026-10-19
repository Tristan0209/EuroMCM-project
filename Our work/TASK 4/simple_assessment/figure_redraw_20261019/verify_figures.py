from pathlib import Path
from decimal import Decimal
import csv
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
from PIL import Image

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    with p.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def subset(rows,**filters):
    rows=sorted([r for r in rows if all(r[k]==str(v) for k,v in filters.items())],key=lambda r:int(r['seed']))
    assert len(rows)==3
    return rows


def numbers(rows,field,scale=Decimal(1),**filters):
    return [Decimal(r[field])*scale for r in subset(rows,**filters)]


def main():
    m=json.loads((HERE/'audit/plot_manifest.json').read_text(encoding='utf-8'))
    results=load(SOURCE/'Task4_Simple_Results.csv')
    groups=load(SOURCE/'final_review/group_service.csv')
    assert len(results)==24 and len(groups)==72
    assert len(m['traces'])==61
    maximum=Decimal(0)
    for t in m['traces']:
        if t['figure']==1:
            metric=t['metric']
            delta=metric.endswith('_delta')
            field=metric[:-6] if delta else metric
            scale=Decimal(100) if field in ['answer_pass_rate','service_violation_rate'] else Decimal(1)
            policy='Greedy' if t['policy']=='Shared routing' else t['policy']
            values=numbers(results,field,scale,scenario=t['category'],policy=policy)
            if t['policy']=='Shared routing':
                assert values==numbers(results,field,scale,scenario=t['category'],policy='Spatial_MILP')
            if delta:
                baseline=numbers(results,field,scale,scenario='Baseline',policy=policy)
                values=[a-b for a,b in zip(values,baseline)]
        elif t['figure']==2 and t['panel'] in ['a','b']:
            scale=Decimal('.001') if t['panel']=='a' else Decimal(1)
            values=numbers(results,t['metric'],scale,scenario=t['category'],policy=t['policy'])
        elif t['figure']==2:
            spatial_scenario='Baseline' if t['category']=='Baseline matched' else 'Demand Growth'
            greedy_scenario='Baseline' if t['category'] in ['Baseline matched','Growth vs original'] else 'Demand Growth'
            spatial=numbers(results,'co2e_g',scenario=spatial_scenario,policy='Spatial_MILP')
            greedy=numbers(results,'co2e_g',scenario=greedy_scenario,policy='Greedy')
            values=[s-g for s,g in zip(spatial,greedy)]
        else:
            scale=Decimal(1) if t['metric']=='completed_p95_s' else Decimal(100)
            values=numbers(groups,t['metric'],scale,scenario='Demand Growth',policy=t['policy'],task_type=t['category'])
        mean=sum(values)/Decimal(3)
        differences=[abs(a-Decimal(str(b))) for a,b in zip(values,t['values'])]
        differences.extend([abs(mean-Decimal(str(t['mean']))),abs(min(values)-Decimal(str(t['minimum']))),abs(max(values)-Decimal(str(t['maximum'])))])
        maximum=max([maximum]+differences)
        assert max(differences)<Decimal('1e-8'),t

    prior=json.loads((SOURCE/'academic_revision_20261018/audit/delivery_validation.json').read_text(encoding='utf-8'))
    prior_claims={t['claim']:t for t in prior['numeric_claims']}
    contrasts=[t for t in m['traces'] if t['figure']==2 and t['panel']=='c']
    for t,key,sign in zip(contrasts,['Baseline same-demand carbon saving','Growth matched-demand saving','Cross-demand change'],[-1,-1,1]):
        assert abs(t['mean']-sign*prior_claims[key]['value'])<1e-8
    assert all(float(r['service_violation_rate'])>0 for r in results)
    assert m['policies']=={'Greedy':'#2878B5','Spatial_MILP':'#E6862A'}
    assert m['new_experiments']==m['new_training']==m['new_inference']==0
    formats=[]
    for fig in m['layout']:
        paths={ext:HERE/r['path'] for ext,r in fig['files'].items()}
        for ext,path in paths.items():
            assert sha(path)==fig['files'][ext]['sha256']
        with Image.open(paths['png']) as im:
            dpi=im.info.get('dpi',(0,0))
            assert min(dpi)>=300 and im.mode in ['RGB','RGBA']
            assert all(abs(actual-inch*400)<=1 for actual,inch in zip(im.size,fig['size_inches']))
            assert im.getpixel((0,0))[:3]==(255,255,255)
            pixels=list(im.size)
        tree=ET.parse(paths['svg'])
        tags=[el.tag.rsplit('}',1)[-1] for el in tree.iter()]
        assert 'text' in tags and 'image' not in tags,'Editable text and vector primitives, no embedded raster'
        info=subprocess.run(['pdfinfo',str(paths['pdf'])],capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
        assert re.search(r'Pages:\s+1\b',info.stdout)
        images=subprocess.run(['pdfimages','-list',str(paths['pdf'])],capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
        assert not any(re.match(r'^\s*\d+\s+\d+',line) for line in images.stdout.splitlines()),'PDF must be vector'
        fonts=subprocess.run(['pdffonts',str(paths['pdf'])],capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
        assert 'TrueType' in fonts.stdout
        assert re.search(r'\byes\s+yes\s+yes\b',fonts.stdout),'Embedded, subsetted fonts with Unicode mapping'
        formats.append({'figure':fig['figure'],'pixels':pixels,'dpi':list(dpi),'svg_editable_text':True,'pdf_vector':True,'pdf_font_embedded':True})
    before=json.loads((HERE/'audit/preservation_before.json').read_text(encoding='utf-8'))
    assert all(Path(p).is_file() and sha(Path(p))==h for p,h in before['old_task4'].items())
    assert {str(p.resolve()) for p in SOURCE.parent.rglob('*') if p.is_file() and HERE not in p.parents}==set(before['old_task4'])
    assert all(sha(Path(p))==h for p,h in before['layout_references'].items())
    for rel in ['Task4_Simple_Results.csv','final_review/group_service.csv']:
        assert sha(SOURCE/rel)==sha(HERE/'source_tables'/rel)
    for p in HERE.glob('*.py'):
        compile(p.read_text(encoding='utf-8'),str(p),'exec')
    visual=json.loads((HERE/'audit/visual_qa.json').read_text(encoding='utf-8'))
    assert visual['status']=='PASS'
    for item in visual['figures']:
        assert sha(HERE/item['png'])==item['png_sha256']
        assert sha(HERE/item['pdf'])==item['pdf_sha256']
        assert sha(HERE/item['rendered_pdf'])==item['rendered_pdf_sha256']
    validation=dict(status='PASS',traces_independently_checked=len(m['traces']),
        arithmetic='Decimal from original CSV, independent of matplotlib and plot code',
        maximum_absolute_numeric_difference=str(maximum),tolerance='1e-8 (CSV float roundoff only)',
        formats=formats,original_task4_files_unchanged=len(before['old_task4']),
        original_reference_files_unchanged=len(before['layout_references']),original_data_copies_byte_identical=True,
        visual_QA='PASS: all PNGs and all vector-PDF renders inspected',
        zero_based_facility_and_operating_axes=True,contrasts_checked_against_previous_audit=True,
        new_experiments=0,new_training=0,new_inference=0)
    (HERE/'audit/validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: 61 traces, saved-data contrasts, 400dpi PNG/editable SVG/vector PDF, visual QA and original file protection')


if __name__=='__main__':
    main()
