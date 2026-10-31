"""Descriptive test examples for preselected baseline/combined seed 42 after final testing."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parents[1]/'work'/'matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from train import model

def predict(variant,loader):
    net=model(variant=='combined')
    net.load_state_dict(torch.load(ROOT/'runs'/f'{variant}_42'/'best.pt',map_location='cpu',weights_only=True))
    net.eval()
    predictions=[]
    with torch.no_grad():
        for x,_ in loader:
            predictions.extend(net(x).argmax(1).tolist())
    return np.array(predictions)

def main():
    status=json.loads((ROOT/'batch_status.json').read_text(encoding='utf-8'))
    if status['phase']!='complete':
        raise RuntimeError('Wait until all formal training and testing are complete')
    torch.set_num_threads(2)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    data=datasets.CIFAR10(str(ROOT.parents[1]/'work/cifar-data'),train=False,download=False,transform=transform)
    loader=DataLoader(data,batch_size=64,shuffle=False)
    baseline=predict('baseline',loader)
    combined=predict('combined',loader)
    truth=np.array(data.targets)
    for name,pred in [('baseline',baseline),('combined',combined)]:
        stored=json.loads((ROOT/'runs'/f'{name}_42'/'test.json').read_text())
        if not np.isclose((pred==truth).mean(),stored['accuracy'],rtol=0,atol=1e-12):
            raise RuntimeError('Re-evaluated weights disagree with stored test result')
    gains=np.flatnonzero((baseline!=truth)&(combined==truth))
    regressions=np.flatnonzero((baseline==truth)&(combined!=truth))
    selected=list(map(int,gains[:4]))+list(map(int,regressions[:4]))
    out=ROOT/'analysis'
    out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,4,figsize=(6.5,5.3))
    for ax in axes.flat: ax.axis('off')
    for ax,idx in zip(axes.flat,selected):
        ax.imshow(data.data[idx],interpolation='nearest')
        kind='gain' if combined[idx]==truth[idx] else 'regression'
        ax.set_title(f"#{idx} ({kind})\nTrue: {data.classes[truth[idx]]}\nBase: {data.classes[baseline[idx]]}\nComb: {data.classes[combined[idx]]}",fontsize=8.5)
    fig.suptitle('Illustrative test cases (seed 42)',fontsize=11)
    fig.tight_layout()
    fig.savefig(out/'error_examples.png',dpi=200,bbox_inches='tight')
    fig.savefig(out/'error_examples.pdf',bbox_inches='tight')
    plt.close(fig)
    (out/'error_examples.json').write_text(json.dumps(dict(seed=42,selection='ascending test index; first 4 per category',
        total_gains=len(gains),total_regressions=len(regressions),selected=selected,
        baseline_predictions=baseline.tolist(),combined_predictions=combined.tolist()),indent=2),encoding='utf-8')
    print('Illustrative cases:',len(selected),'total gains:',len(gains),'total regressions:',len(regressions))

if __name__=='__main__':
    main()
