from pathlib import Path
import io, json, zipfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader, PdfWriter, Transformation

ROOT=Path(__file__).resolve().parent
FONTS=Path('C:/Windows/Fonts')
for name,file in [('Times','times.ttf'),('Times-Bold','timesbd.ttf'),('Times-Italic','timesi.ttf'),
                  ('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONTS/file)))
pdfmetrics.registerFontFamily('Times', normal='Times',bold='Times-Bold',italic='Times-Italic',boldItalic='Times-Bold')
W,H=A4
INK='#263947'
styles={
 'body':ParagraphStyle('body',fontName='Times',fontSize=11,leading=14.5,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Arial',fontSize=8.5,leading=12,textColor='#677782'),
 'caption':ParagraphStyle('caption',fontName='Times-Italic',fontSize=9.5,leading=12,textColor=INK),
 'note':ParagraphStyle('note',fontName='Times',fontSize=10.5,leading=14,textColor=INK)
}

def p(c,s,x,y,width,style='body'):
    q=Paragraph(s,styles[style]);_,height=q.wrap(width,H)
    q.drawOn(c,x,y-height)
    return y-height-styles[style].spaceAfter

def heading(c,label,subtitle,page):
    c.setFillColor(INK);c.setFont('Arial-Bold',17);c.drawString(42,H-52,label)
    p(c,subtitle,42,H-67,W-84,'small')
    c.setStrokeColorRGB(.82,.86,.89);c.line(42,H-97,W-42,H-97)
    c.setFont('Arial',8);c.setFillColorRGB(.4,.46,.5)
    c.drawString(42,24,'EUROMCM  /  OUR WORK')
    c.drawRightString(W-42,24,f'{page} / 4')

data=json.loads((ROOT/'content.json').read_text(encoding='utf-8'))
buf=io.BytesIO();c=canvas.Canvas(buf,pagesize=A4)
c.setTitle('Our Work: manuscript text and three technical roadmaps')
c.setAuthor('EuroMCM project')
heading(c,'1.3  Our Work','Manuscript-ready text, followed by three alternative flowchart layouts.',1)
y=H-121
for paragraph in data['paragraphs']:
    y=p(c,paragraph,49,y,W-98)
y-=6
y=p(c,'Suggested figure caption',49,y,W-98,'small')-5
y=p(c,'Overview of our framework: performance evaluation, protected model routing, carbon-aware scheduling, and robustness and sustainability assessment.',49,y,W-98,'caption')-24
c.setFillColor(INK);c.setFont('Arial-Bold',11);c.drawString(49,y,'Compact alternative')
y-=16
y=p(c,data['short'],49,y,W-98,'note')
if y<55:raise ValueError(f'Text page exceeds available space: {y}')
c.showPage()

variants=[
 ('A_dependency_roadmap','A  |  Task dependency roadmap',
  'A sequential task map, inspired by the dependency structure of the 2024 reference.',
  'Reading order: Task 1 to Task 2, then down to Task 3 and left to Task 4. The arrows distinguish model selection, execution scheduling and final assessment.',
  'Use this version when the paper can allocate more vertical space to explain how the four tasks depend on one another.'),
 ('B_four_column_framework','B  |  Four-column framework',
  'A compact overview, inspired by the parallel columns of the 2025 reference.',
  'Four colored columns separate performance evidence, routing, scheduling and validation. The bottom band collects the resulting decision support for SWITCH.',
  'Recommended for the main manuscript: it offers the clearest balance between coverage, compactness and consistency with a conventional Our Work section.'),
 ('C_core_and_validation','C  |  Core and validation',
  'A modular architecture, inspired by the grouped panels of the 2023 reference.',
  'The center highlights the selected routing and scheduling core. Evidence sits to the left, and stress testing and environmental assessment sit to the right. The dashed loop is observed scheduling-state feedback.',
  'Use this version when the introduction should emphasize the technical contribution and distinguish the execution core from its independent assessment.')
]
placements=[]
for idx,(name,title,subtitle,explanation,recommendation) in enumerate(variants,2):
    heading(c,title,subtitle,idx)
    src=PdfReader(ROOT/'figures'/f'{name}.pdf').pages[0]
    dw=W-70
    scale=dw/float(src.mediabox.width)
    dh=float(src.mediabox.height)*scale
    top=H-116
    bottom=top-dh
    placements.append((idx-1,name,scale,35,bottom))
    y=bottom-22
    y=p(c,'Figure X. '+ 'Overview of our work.',42,y,W-84,'caption')-10
    y=p(c,explanation,42,y,W-84,'note')-7
    y=p(c,recommendation,42,y,W-84,'note')-10
    y=p(c,'All figures use the same evidence boundary: historical replay and parameterized simulation. The selected router does not activate post-response escalation; temporal policies are comparisons rather than an assumed source of savings.',42,y,W-84,'small')
    if y<46:raise ValueError(f'Figure page exceeds available space: {name}, {y}')
    c.showPage()
c.save();buf.seek(0)
base=PdfReader(buf);writer=PdfWriter()
for page in base.pages:writer.add_page(page)
for index,name,scale,x,y in placements:
    src=PdfReader(ROOT/'figures'/f'{name}.pdf').pages[0]
    writer.pages[index].merge_transformed_page(src,Transformation().scale(scale).translate(x,y),over=False)
writer.add_metadata({'/Title':'Our Work: text and three alternative flowcharts','/Author':'EuroMCM project'})
with (ROOT/'Our_Work_Review.pdf').open('wb') as out:writer.write(out)
check=PdfReader(ROOT/'Our_Work_Review.pdf')
assert len(check.pages)==4
assert all('Our Work' in check.pages[0].extract_text() for _ in [0])
for j,(name,*_) in enumerate(variants,1):
    text=check.pages[j].extract_text()
    for term in ['Pre-LightGBM','MILP','SWITCH']:
        assert term in text,(name,term)

with zipfile.ZipFile(ROOT/'Our_Work_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.pdf','.png','.svg','.md','.tex','.py','.json'}:
            z.write(path,path.relative_to(ROOT))
print('Four-page review PDF verified; deliverable ZIP ready.')
