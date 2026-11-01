"""Build the experiment record from saved run artifacts; no ML runs."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'runs/supervisor_plot_dependencies'))
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
import pdfplumber

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=8,textColor=colors.HexColor('#243449')))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=5))
styles['Title'].textColor=colors.HexColor('#163A75')
styles['Title'].fontSize=22
styles['Heading2'].textColor=colors.HexColor('#163A75')
styles['Heading2'].fontSize=12
story=[]
def p(t,style='BodyCustom'):story.append(Paragraph(t,styles[style]))
def h(t):p(t,'Heading2')
def table(rows,widths):
    data=[[Paragraph(str(v),styles['SmallCustom']) for v in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF0F6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#B8C8D8')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F7F9FB')]),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.extend([t,Spacer(1,9)])

p('Hardware and training setup','Title')
p('Standardised comparison | 1 November 2026')
h('Hardware')
p('<b>CPU:</b> AMD Ryzen 7 9800X3D (4 PyTorch threads used).<br/><b>GPU:</b> AMD Radeon RX 9070 XT; all models trained on GPU.<br/><b>Software:</b> Windows, PyTorch 2.12.0+rocm7.14.0; NumPy 2.5.4; h5py 3.16.0.<br/><b>Precision:</b> FP32. <b>Data-loader workers:</b> 4.')
h('Data and training budget')
table([['Setting','Shared configuration'],['Input / SNR range','50 x 100 binary grids; -20 to +5 dB in 1 dB steps'],['Dataset','104,000 images: 4,000 per SNR'],['Identical data split','62,400 training / 20,800 validation / 20,800 test'],['Training','8 epochs; batch size 64; Adam; learning rate 0.001; seed 42'],['Model selection','Minimum validation loss; no early stopping or learning-rate schedule']],[125,386])
p('The data generator was unchanged. No model receives SNR as an input. U-Net, EvidenceAxialNet and FastUNet use BCE + Dice loss; FCNN uses MSE on bounding-box coordinates.')
h('Model configurations')
table([['Model','Training configuration'],['Original U-Net','Baseline U-Net implementation; native BatchNorm backend'],['EvidenceAxialNet','Base width 16; evidence features enabled; 3 refinement steps'],['FastUNet','Base width 8; multi-branch training blocks; native BatchNorm backend'],['FCNN recreation','5,000 inputs; four 256-unit hidden layers; 80 outputs (20 boxes); dropout 0.4']],[145,366])
p('Use the saved model-source snapshots, configurations, dataset manifests and split indices from the original runs to reproduce these settings. The same seed alone does not guarantee identical results on different hardware.','SmallCustom')

out=ROOT/'output/pdf/Experiment Setup and Metrics.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
(ROOT/'tmp/pdfs').mkdir(parents=True, exist_ok=True)
def footer(c,d):
    c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#526173'))
    c.drawString(42,23,'Standardised local experiment | Seed 42')
    c.drawRightString(A4[0]-42,23,str(d.page))
SimpleDocTemplate(str(out),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=40,title='Hardware and Training Setup',author='Isaac').build(story,onFirstPage=footer,onLaterPages=footer)
with pdfplumber.open(out) as pdf:
    print('Pages:',len(pdf.pages))
    for i,page in enumerate(pdf.pages):page.to_image(resolution=110).save(ROOT/f'tmp/pdfs/experiment_record_{i+1}.png')
print(out)
