"""Evaluate supplied best state dictionaries without optimizer checkpoints or retraining."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from train import model,evaluate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs-dir',type=Path,required=True,help='Directory containing variant_seed/best.pt and config.json')
    parser.add_argument('--variant',choices=['baseline','augment','regularized','combined'],required=True)
    parser.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[2]
    folder=args.runs_dir/f'{args.variant}_{args.seed}'
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if config['variant']!=args.variant or config['seed']!=args.seed or config.get('smoke') or config.get('pilot'):
        raise ValueError('Weights/config mismatch')
    net=model(args.variant in ['regularized','combined'])
    net.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True))
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    test=datasets.CIFAR10(str(root/'work/cifar-data'),train=False,download=True,transform=transform)
    loss,accuracy,matrix=evaluate(net,DataLoader(test,batch_size=64),torch.device('cpu'))
    print(json.dumps(dict(variant=args.variant,seed=args.seed,loss=loss,accuracy=accuracy,confusion=matrix),indent=2))

if __name__=='__main__':
    main()
