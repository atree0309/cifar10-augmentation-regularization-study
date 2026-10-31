"""Finite 12-run experiment batch. Re-running safely resumes epoch checkpoints."""
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from filelock import FileLock
import torch

root = Path(__file__).resolve().parent
batch_lock = FileLock(str(root/'batch.lock'))
batch_lock.acquire(timeout=0)
variants = ['baseline','augment','regularized','combined']
seeds = [42,43,44]
epochs = 30
protocol_path = root/'frozen_protocol.json'
source_hash = hashlib.sha256((root/'train.py').read_bytes()).hexdigest()
protocol = dict(epochs=epochs,seeds=seeds,variants=variants,train_sha256=source_hash,
    batch_size=64,threads=2,device='cpu',split_seed=2026,
    rationale='30 epochs retained before formal testing; four timing pilots estimate 9.07 hours plus overhead.')
if protocol_path.exists():
    if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('Frozen protocol differs; inspect before continuing')
else:
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')

def status(phase,variant=None,seed=None):
    (root/'batch_status.json').write_text(json.dumps(dict(phase=phase,variant=variant,
        seed=seed,updated_unix=time.time(),pid=os.getpid()),indent=2),encoding='utf-8')

def execute(variant,seed,evaluate=False):
    out=root/'runs'/f'{variant}_{seed}'
    args=[sys.executable,'-u',str(root/'train.py'),'--variant',variant,'--seed',str(seed),'--epochs',str(epochs)]
    checkpoint=out/'last.pt'
    if checkpoint.exists():
        if not evaluate:
            config=json.loads((out/'config.json').read_text(encoding='utf-8'))
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['epoch']==epochs and len(saved['history'])==epochs:
                for key,value in dict(variant=variant,seed=seed,epochs=epochs,batch_size=64,threads=2,device='cpu').items():
                    if config[key]!=value:
                        raise RuntimeError('Completed-run configuration mismatch: '+str(out))
                if not (out/'best.pt').exists():
                    raise RuntimeError('Completed run is missing best weights: '+str(out))
                print('Already trained; preserving',variant,seed,flush=True)
                return
        args.append('--resume')
    elif out.exists():
        if any(out.iterdir()):
            raise RuntimeError(f'Run exists without checkpoint: {out}. Inspect before retrying.')
        out.rmdir()
    if evaluate:
        if not checkpoint.exists():
            raise RuntimeError('Missing training checkpoint')
        args.append('--evaluate')
    phase='evaluating' if evaluate else 'training'
    status(phase,variant,seed)
    print(phase,variant,seed,flush=True)
    subprocess.run(args,check=True)

try:
    if (root/'batch_error.txt').exists():
        archive=root.parents[1]/'work'/'training-errors'
        archive.mkdir(parents=True,exist_ok=True)
        (root/'batch_error.txt').replace(archive/f'batch_error_{time.time_ns()}.txt')
    # Train every configuration before exposing formal test scores.
    for variant in variants:
        for seed in seeds:
            execute(variant,seed)
    for variant in variants:
        for seed in seeds:
            out=root/'runs'/f'{variant}_{seed}'
            if not (out/'test.json').exists():
                execute(variant,seed,True)
    rows=[]
    for variant in variants:
        scores=[]
        for seed in seeds:
            out=root/'runs'/f'{variant}_{seed}'
            config=json.loads((out/'config.json').read_text(encoding='utf-8'))
            result=json.loads((out/'test.json').read_text(encoding='utf-8'))
            assert not result['smoke']
            with (out/'history.csv').open() as f:
                history=list(csv.DictReader(f))
            if len(history)!=epochs:
                raise RuntimeError('Incomplete run')
            scores.append(result['accuracy'])
            rows.append(dict(variant=variant,seed=seed,test_accuracy=result['accuracy'],
                clean_train_accuracy=result['clean_train_accuracy'],parameters=config['parameters'],
                training_seconds=sum(float(r['seconds']) for r in history)))
        print(variant,statistics.mean(scores),statistics.stdev(scores),flush=True)
    with (root/'formal_results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status('complete')
except Exception as exc:
    status('failed')
    (root/'batch_error.txt').write_text(repr(exc),encoding='utf-8')
    raise
finally:
    batch_lock.release()
