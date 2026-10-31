"""Reproducible architecture and training-only CIFAR-10 sample figures."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parents[1]/'work'/'matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from torchvision.datasets import CIFAR10

out=ROOT/'figures'
out.mkdir(exist_ok=True)
def save(fig,name):
    fig.savefig(out/f'{name}.png',dpi=220,bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(6.5,5.8))
ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
centers=np.linspace(.855,.12,6)
width=.42
height=.112
for x,regularized,title in [(.25,False,'Baseline'),(.75,True,'BN + Dropout')]:
    ax.text(x,.96,title,fontsize=11,fontweight='bold',va='center',ha='center')
    labels=['RGB input\n3 x 32 x 32']
    for cin,cout,size in [(3,32,16),(32,64,8),(64,128,4)]:
        labels.append(f'Conv 3x3: {cin} -> {cout}\n'+('BN / ' if regularized else '')+f'ReLU / MaxPool\n{cout} x {size} x {size}')
    labels += ['Global average pool\nFlatten\n128 features',
               ('Dropout p=0.3\n' if regularized else '')+'Linear 128 -> 10\n10 logits']
    for j,(y,label) in enumerate(zip(centers,labels)):
        ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,
            boxstyle='round,pad=0.005',linewidth=1,edgecolor='#444444',facecolor='#f4f4f4'))
        ax.text(x,y,label,ha='center',va='center',fontsize=9.3,linespacing=1.15)
        if j<5:
            ax.annotate('',xy=(x,centers[j+1]+height/2+.007),xytext=(x,y-height/2-.007),
                arrowprops=dict(arrowstyle='->',lw=1,color='#333333'))
    ax.text(x,.015,'94,986 parameters' if regularized else '94,538 parameters',fontsize=9,ha='center')
fig.suptitle('Small CNN architectures',fontsize=12,y=.99)
fig.tight_layout()
save(fig,'network_architecture')

data=CIFAR10(str(ROOT.parents[1]/'work'/'cifar-data'),train=True,download=False)
split=json.loads((ROOT/'runs'/'baseline_42'/'split.json').read_text(encoding='utf-8'))
train_ids=np.array(split['train']); val_ids=np.array(split['validation'])
assert len(train_ids)==45000 and len(val_ids)==5000
assert len(set(train_ids.tolist()) & set(val_ids.tolist()))==0
assert len(set(train_ids.tolist()) | set(val_ids.tolist()))==50000
labels=np.array(data.targets)
rng=np.random.default_rng(2026)
fig,axes=plt.subplots(4,5,figsize=(6.5,5.3))
selected=[]
for cls in range(10):
    ids=rng.choice(train_ids[labels[train_ids]==cls],size=2,replace=False)
    for row,idx in enumerate(ids):
        gridrow=(cls//5)*2+row
        ax=axes[gridrow,cls%5]; ax.imshow(data.data[idx],interpolation='nearest'); ax.axis('off')
        if row==0: ax.set_title(data.classes[cls],fontsize=9)
        selected.append(dict(class_id=cls,train_index=int(idx),row=gridrow,column=cls%5))
fig.suptitle('CIFAR-10 training samples',fontsize=12)
fig.tight_layout()
save(fig,'cifar10_training_samples')
(out/'sample_indices.json').write_text(json.dumps(dict(selection_seed=2026,source='official training set, project training split only',selected=selected),indent=2),encoding='utf-8')
counts=[dict(class_name=data.classes[c],training=int((labels[train_ids]==c).sum()),validation=int((labels[val_ids]==c).sum())) for c in range(10)]
(out/'split_class_counts.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
print(out)
