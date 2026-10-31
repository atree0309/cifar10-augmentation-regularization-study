"""Run genuine CIFAR-10 pipeline checks and timing pilots; not paper results."""
import csv
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
variants = ['baseline','augment','regularized','combined']
records = []
for variant in variants:
    for flag, suffix in [('--smoke','smoke'),('--pilot','pilot')]:
        out = root / 'runs' / f'{variant}_42_{suffix}'
        done = out / ('test.json' if suffix == 'smoke' else 'last.pt')
        if not done.exists():
            command = [sys.executable, str(root/'train.py'), '--variant', variant, flag, '--epochs','1']
            if (out/'last.pt').exists():
                command.append('--resume')
            elif out.exists():
                if any(out.iterdir()):
                    raise RuntimeError(f'Incomplete nonempty run, inspect manually: {out}')
                out.rmdir()
            print('Running',variant,suffix,flush=True)
            subprocess.run(command,check=True)
    out = root/'runs'/f'{variant}_42_pilot'
    config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    with (out/'history.csv').open() as f:
        history=list(csv.DictReader(f))
    seconds=float(history[0]['seconds'])
    records.append(dict(variant=variant, parameters=config['parameters'], pilot_epoch_seconds=seconds,
        full_epoch_seconds_estimate=seconds*45000/4096))
summary=dict(status='timing_only_not_paper_results', records=records,
    estimate_30_epochs_3_seeds_hours=sum(r['full_epoch_seconds_estimate'] for r in records)*30*3/3600,
    estimate_10_epochs_3_seeds_hours=sum(r['full_epoch_seconds_estimate'] for r in records)*10*3/3600,
    caveat='Linear extrapolation; excludes startup and final evaluation, validation fraction differs.')
(root/'pilot_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
