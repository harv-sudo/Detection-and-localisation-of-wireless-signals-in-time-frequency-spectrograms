"""Create two comparison PDFs from saved matched evaluations and benchmarks."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runs/supervisor_plot_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pdfplumber

OUT = ROOT / 'output/pdf'
TMP = ROOT / 'tmp/pdfs'
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
def read(path):
    source = ROOT / path
    if not source.exists() and path.startswith('runs/'):
        source = ROOT / 'review_artifacts/standardised_results' / path[5:]
    return json.loads(source.read_text())

specs = [
    ('unet', 'Original U-Net', '#163A75', 'o', '--', 'unet_bn_native_seed42/evaluation.json'),
    ('evidence_axial', 'EvidenceAxialNet', '#D55E00', 's', ':', 'evidence_axial_seed42/evaluation.json'),
    ('fast_unet_b8_deploy', 'FastUNet', '#008577', '^', '-', 'fast_unet_b8_seed42/deployment_evaluation.json'),
    ('fcnn', 'FCNN', '#56236D', 'D', '-.', 'fcnn_matched_seed42/evaluation.json'),
]
evaluations = [read('runs/' + s[5]) for s in specs]
for d in evaluations:
    assert d['overall']['samples'] == 20800
    assert d['test_indices'] == evaluations[0]['test_indices']
snrs = np.arange(-20, 6)
series = [[d['per_snr'][str(s)]['macro']['f1'] for s in snrs] for d in evaluations]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})

def page(title):
    fig = plt.figure(figsize=(13.62, 8.82), facecolor='white')
    fig.text(.045, .95, title, fontsize=23, weight='bold', color='#172B42')
    return fig

def save(fig, name):
    path = OUT / (name + '.pdf')
    fig.savefig(path, dpi=400)
    plt.close(fig)
    with pdfplumber.open(path) as pdf:
        assert len(pdf.pages) == 1
        pdf.pages[0].to_image(resolution=110).save(TMP / (name + '.png'))
    print(path)

fig = page('Detection performance')
reference = read('review_artifacts/reference_figure16a_digitized.json')
b = reference['plot_bounds_pdf_points']
left, right, top, bottom = [b[k] for k in ('left', 'right', 'top', 'bottom')]
crop = (64, 56, 273, 221)
with pdfplumber.open(ROOT / 'review_artifacts/reference_signal_area_2023.pdf') as p:
    original = p.pages[11].crop(crop).to_image(resolution=500).original
ax = fig.add_axes([.025, .14, .64, .72])
ax.imshow(original, extent=(64, 273, 221, 56), aspect='equal')
ax.set(xlim=(64, 273), ylim=(221, 56))
ax.axis('off')
xp = left + (snrs + 20) * (right - left) / 25
for i, (s, vals) in enumerate(zip(specs, series)):
    ax.plot(xp, bottom - np.array(vals) * (bottom-top), color=s[2], ls=s[4],
        lw=2, marker=s[3], markevery=(i % 3, 3), ms=5, mfc='white', mew=1.2)
fig.text(.71, .82, 'Our models', fontsize=12, weight='bold', color='#172B42')
handles = [Line2D([], [], color=s[2], ls=s[4], marker=s[3], mfc='white', lw=2,
    label=s[1]) for s in specs]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.697,.804),
    frameon=False, fontsize=12, labelspacing=.9, handlelength=3)
fig.text(.71,.565,'Overall macro F1', fontsize=11, weight='bold',color='#172B42')
for i,(s,d) in enumerate(zip(specs,evaluations)):
    fig.text(.71,.535-i*.029,s[1],fontsize=10,color=s[2])
    fig.text(.96,.535-i*.029,f"{d['overall']['macro']['f1']:.4f}",ha='right',fontsize=10)
zoom = fig.add_axes([.735,.17,.225,.215])
for i,(s,vals) in enumerate(zip(specs[:3],series[:3])):
    zoom.plot(snrs[:9], vals[:9], color=s[2],ls=s[4],lw=1.7,marker=s[3],
        markevery=(i,3),ms=4,mfc='white')
zoom.set(xlim=(-20.3,-11.7),ylim=(.765,.95),xlabel='SNR (dB)',ylabel='F1 score',
    title='Low-SNR detail')
zoom.title.set_fontsize(9)
zoom.set_xticks([-20,-18,-16,-14,-12])
zoom.set_yticks([.78,.82,.86,.90,.94])
zoom.tick_params(labelsize=9)
zoom.grid(alpha=.18)
if '--speed-only' not in sys.argv:
    save(fig,'Performance')
else:
    plt.close(fig)

files = ['cpu_b1_eager','cpu_b1_channels_last','cpu_b1_prepared','cuda_b1_eager','cuda_b64_eager']
bench = [read('runs/fcnn_matched_seed42/benchmark_'+f+'.json') for f in files]
records = [{r['model']:r for r in b['records']} for b in bench]
times = np.array([[r[s[0]]['median_batch_ms'] for s in specs] for r in records])
for b in bench:
    assert b['trained_weights'] and b['rounds']==3 and b['dtype']=='float32'
for s in specs:
    assert len({r[s[0]]['checkpoint_sha256'] for r in records})==1
fig = page('Inference speed')
for j,(title,vals) in enumerate(zip(['CPU | batch 1\nBest measured mode','GPU | batch 1','GPU | batch 64'],
    [times[:3].min(axis=0),times[3],times[4]])):
    ax=fig.add_axes([.18+j*.275,.49,.23,.30])
    ax.barh(range(4),vals,color=[s[2] for s in specs],height=.52)
    ax.set_yticks(range(4),[s[1] for s in specs] if j==0 else ['']*4)
    ax.invert_yaxis()
    ax.set(xlim=(0,max(vals)*1.3),xlabel='Median forward time (ms/batch)',title=title)
    ax.grid(axis='x',alpha=.16)
    ax.set_axisbelow(True)
    for i,v in enumerate(vals):ax.text(v+max(vals)*.025,i,f'{v:.3f}',va='center',fontsize=11)
fig.text(.045,.365,'Published results | Figure 16b',fontsize=14,weight='bold',color='#172B42')
fig.text(.045,.328,'Approximate graph readings; different setup, not a direct speed comparison.',fontsize=11,color='#526173')
tax=fig.add_axes([.045,.065,.91,.235]);tax.axis('off')
paper_times = [
    ['Energy detection (ED)', '0.2-0.4'],
    ['CT-SA', '2-3'],
    ['SSA', '7-11'],
    ['DL-ANN Option 1', '10'],
    ['DL-ANN Option 2', '35'],
    ['DL-ANN Option 3', '300'],
    ['DL-ANN Option 4', '300'],
]
table=tax.table(cellText=paper_times,
    colLabels=['Paper method','Reported computation time (ms)'],cellLoc='center',bbox=[0,0,1,1])
table.auto_set_font_size(False);table.set_fontsize(10)
for (r,c),cell in table.get_celld().items():
    cell.set_edgecolor('#DFE5EA');cell.set_linewidth(.5)
    if r==0:cell.set_facecolor('#EDF2F6');cell.set_text_props(weight='bold',color='#172B42')
    elif r%2==0:cell.set_facecolor('#F7F9FB')
save(fig,'Inference Speed')
