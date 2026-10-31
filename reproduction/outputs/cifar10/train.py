"""CIFAR-10 controlled experiments. Run from this directory."""
import argparse
import csv
import json
import random
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def model(regularized):
    layers = []
    channels = 3
    for width in (32, 64, 128):
        layers.append(nn.Conv2d(channels, width, 3, padding=1))
        if regularized:
            layers.append(nn.BatchNorm2d(width))
        layers.extend([nn.ReLU(), nn.MaxPool2d(2)])
        channels = width
    layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten()])
    if regularized:
        layers.append(nn.Dropout(0.3))
    layers.append(nn.Linear(128, 10))
    return nn.Sequential(*layers)


def evaluate(net, loader, device):
    net.eval()
    loss_sum, correct, count = 0., 0, 0
    matrix = torch.zeros(10, 10, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            count += len(y)
            matrix += torch.bincount((y * 10 + pred).cpu(), minlength=100).reshape(10, 10)
    return loss_sum / count, correct / count, matrix.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['baseline', 'augment', 'regularized', 'combined'], default='baseline')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--smoke', action='store_true', help='Small pipeline check; not a paper result')
    p.add_argument('--pilot', action='store_true', help='4096/512 timing pilot; never evaluates test data')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate', action='store_true', help='Final test evaluation only after budget is frozen')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1:
        p.error('epochs and batch-size must be positive')
    if a.smoke and a.pilot:
        p.error('Choose smoke or pilot, not both')
    if a.pilot and a.evaluate:
        p.error('Pilot runs must not evaluate the test set')
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(__file__).resolve().parent
    data_root = root.parents[1] / 'work' / 'cifar-data'
    out = root / 'runs' / (a.variant + '_' + str(a.seed) + ('_smoke' if a.smoke else '_pilot' if a.pilot else ''))
    out.mkdir(parents=True, exist_ok=a.resume)
    norm = [transforms.ToTensor(), transforms.Normalize((.5,.5,.5), (.5,.5,.5))]
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train_ds = datasets.CIFAR10(str(data_root), train=True, download=True,
        transform=transforms.Compose((aug if a.variant in ['augment','combined'] else []) + norm))
    val_ds = datasets.CIFAR10(str(data_root), train=True, download=True, transform=transforms.Compose(norm))
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(2026)).tolist()
    train_ids, val_ids = indices[:45000], indices[45000:]
    if a.smoke:
        train_ids, val_ids = train_ids[:256], val_ids[:128]
    if a.pilot:
        train_ids, val_ids = train_ids[:4096], val_ids[:512]
    order_rng = torch.Generator()
    train = DataLoader(Subset(train_ds, train_ids), batch_size=a.batch_size, shuffle=True, num_workers=0, generator=order_rng)
    val = DataLoader(Subset(val_ds, val_ids), batch_size=a.batch_size, num_workers=0)
    net = model(a.variant in ['regularized','combined']).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    best = -1
    start_epoch, history = 1, []
    metadata = dict(vars(a), device=str(device), torch=torch.__version__,
        python=platform.python_version(), platform=platform.platform(),
        gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        parameters=sum(t.numel() for t in net.parameters()), split_seed=2026,
        normalization='fixed mean=std=0.5', protocol_version=2)
    config_file = out / 'config.json'
    if a.resume:
        old = json.loads(config_file.read_text(encoding='utf-8'))
        for key in ['variant','seed','batch_size','smoke','pilot','threads','torch','device','protocol_version']:
            if old[key] != metadata[key]:
                raise ValueError('Resume configuration mismatch: ' + key)
        checkpoint = torch.load(out / 'last.pt', map_location=device, weights_only=False)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best, history = checkpoint['best'], checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        if a.epochs < checkpoint['epoch']:
            raise ValueError('Cannot reduce completed epoch count')
        if not a.smoke and (out / 'test.json').exists() and a.epochs >= start_epoch:
            raise ValueError('Test already evaluated; do not extend training after testing')
    config_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out / 'split.json').write_text(json.dumps({'train':train_ids,'validation':val_ids}), encoding='utf-8')
    with (out / 'history.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','train_accuracy','val_loss','val_accuracy','seconds'])
        writer.writerows(history)
        for epoch in range(start_epoch, a.epochs + 1):
            # Explicit epoch seeds reproduce epoch boundaries on resume and match sample orders.
            order_rng.manual_seed(a.seed * 100000 + epoch)
            torch.manual_seed(a.seed * 100000 + epoch + 10000)
            random.seed(a.seed * 100000 + epoch + 10000)
            np.random.seed(a.seed * 100000 + epoch + 10000)
            start = time.perf_counter()
            net.train()
            total, correct, count = 0., 0, 0
            for x, y in train:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = net(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                count += len(y)
            vl, va, _ = evaluate(net, val, device)
            row = [epoch, total/count, correct/count, vl, va, time.perf_counter()-start]
            writer.writerow(row)
            history.append(row)
            f.flush()
            print(row, flush=True)
            if va > best:
                best = va
                torch.save(net.state_dict(), out / 'best.pt')
            state = dict(epoch=epoch, model=net.state_dict(), optimizer=optimizer.state_dict(),
                         best=best, history=history, epoch_seed_scheme='seed*100000+epoch')
            torch.save(state, out / 'last.tmp')
            (out / 'last.tmp').replace(out / 'last.pt')
    if not (a.evaluate or a.smoke):
        print('Training complete; test set not evaluated. Use --resume --evaluate for final evaluation.')
        return
    net.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True))
    test_ds = datasets.CIFAR10(str(data_root), train=False, download=True, transform=transforms.Compose(norm))
    if a.smoke:
        test_ds = Subset(test_ds, range(128))
    loss, accuracy, confusion = evaluate(net, DataLoader(test_ds, batch_size=a.batch_size), device)
    clean_train = DataLoader(Subset(val_ds, train_ids), batch_size=a.batch_size)
    clean_loss, clean_accuracy, _ = evaluate(net, clean_train, device)
    (out / 'test.json').write_text(json.dumps(dict(loss=loss, accuracy=accuracy, confusion=confusion,
        clean_train_loss=clean_loss, clean_train_accuracy=clean_accuracy, smoke=a.smoke), indent=2))
    print('Test accuracy:', accuracy, 'SMOKE ONLY' if a.smoke else '', flush=True)


if __name__ == '__main__':
    main()
