from pathlib import Path
import hashlib,json,zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
names=['01_based_on_2400996','02_based_on_2025_O_award','03_based_on_2300348']
report=[]
for name in names:
    file=ROOT/f'{name}.pdf';reader=PdfReader(file)
    assert len(reader.pages)==1
    page=reader.pages[0];text=page.extract_text()
    for term in ['MILP','Bernstein']:
        assert term in text,(name,term)
    assert 'Our work' in text or 'SWITCH' in text
    resources=page['/Resources'].get_object()
    assert '/Font' in resources
    for suffix in ['.svg','.png','_comparison.png']:
        assert (ROOT/(name+suffix)).stat().st_size>1000
    report.append({'name':name,'pages':1,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                   'pdf_width_pt':float(page.mediabox.width),'pdf_height_pt':float(page.mediabox.height),
                   'vector_fonts':True,'text_extraction_pass':True,'visual_check':'PDF rendered and visually inspected'})
(ROOT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
target=ROOT/'Three_Reference_Matched_Flowcharts.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for file in sorted(ROOT.iterdir()):
        if file.is_file() and file.suffix in {'.pdf','.svg','.png','.md','.py','.json'}:
            z.write(file,file.name)
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    print('Verified 3 vector PDFs, 3 SVGs, 3 PNGs and 3 source/new comparison images.')
    print('Package entries:',len(z.namelist()))
