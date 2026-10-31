"""Validate and analyze 12 completed runs, or draw explicitly provisional progress.

This script never trains a model or evaluates any images. Formal results are read
only after their configurations, histories, checkpoints and test counts pass.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parents[1] / 'work' / 'matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ['baseline', 'augment', 'regularized', 'combined']
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
HISTORY_FIELDS = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'seconds']
ELAPSED_NOTE = ('Recorded training seconds are elapsed durations and may include sleep, '
                'system load or interruption effects; they are not isolated compute timings.')
EFFECTS = [
    ('augment_minus_baseline', {'augment': 1, 'baseline': -1}),
    ('regularized_minus_baseline', {'regularized': 1, 'baseline': -1}),
    ('combined_minus_baseline', {'combined': 1, 'baseline': -1}),
    ('combined_minus_augment', {'combined': 1, 'augment': -1}),
    ('combined_minus_regularized', {'combined': 1, 'regularized': -1}),
    ('interaction', {'combined': 1, 'augment': -1, 'regularized': -1, 'baseline': 1}),
]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing to write an empty results table: ' + str(path))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run(variant, seed):
    folder = ROOT / 'runs' / f'{variant}_{seed}'
    config_path = folder / 'config.json'
    if not config_path.exists():
        return None
    config = read_json(config_path)
    if config.get('smoke') or config.get('pilot'):
        raise ValueError('Preliminary run in formal directory: ' + str(folder))
    history, history_error, fieldnames = [], None, None
    path = folder / 'history.csv'
    if path.exists():
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            for row in reader:
                try:
                    if set(row) != set(HISTORY_FIELDS):
                        raise ValueError('Unexpected or missing history columns')
                    history.append({key: float(row[key]) for key in HISTORY_FIELDS})
                except (KeyError, TypeError, ValueError) as exc:
                    # A live writer can expose an incomplete last line in progress mode.
                    history_error = repr(exc)
                    break
    result_path = folder / 'test.json'
    result = read_json(result_path) if result_path.exists() else None
    if result is not None and result.get('smoke'):
        raise ValueError('Smoke score cannot be a formal result')
    split_path = folder / 'split.json'
    return dict(variant=variant, seed=seed, folder=folder, config=config,
                history=history, fieldnames=fieldnames, history_error=history_error,
                result=result, split=read_json(split_path) if split_path.exists() else None)


def valid_number(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name + ' is outside its valid range')


def validate_test_result(result, label):
    if result.get('smoke') is not False:
        raise ValueError(label + ': missing explicit formal-result marker')
    matrix = result.get('confusion')
    if not isinstance(matrix, list) or len(matrix) != 10:
        raise ValueError(label + ': confusion matrix must be 10 x 10')
    for row in matrix:
        if not isinstance(row, list) or len(row) != 10:
            raise ValueError(label + ': confusion matrix must be 10 x 10')
        if any(type(value) is not int or value < 0 for value in row):
            raise ValueError(label + ': confusion counts must be nonnegative integers')
        if sum(row) != 1000:
            raise ValueError(label + ': every CIFAR-10 test class must contain 1000 examples')
    if sum(map(sum, matrix)) != 10000:
        raise ValueError(label + ': confusion total must be 10000')
    for metric in ['accuracy', 'clean_train_accuracy']:
        valid_number(result.get(metric), label + ':' + metric, maximum=1)
    for metric in ['loss', 'clean_train_loss']:
        valid_number(result.get(metric), label + ':' + metric)
    reconstructed = sum(matrix[i][i] for i in range(10)) / 10000
    if not math.isclose(result['accuracy'], reconstructed, rel_tol=0, abs_tol=1e-12):
        raise ValueError(label + ': accuracy does not match confusion-matrix trace')


def validate_complete(runs, protocol):
    seeds, epochs = protocol['seeds'], protocol['epochs']
    if protocol['variants'] != VARIANTS or len(seeds) != 3 or len(set(seeds)) != 3 or epochs != 30:
        raise ValueError('This analysis requires the frozen 4 x 3 x 30 protocol')
    incomplete = [f'{v}_{s}' for (v, s), run in zip(
        [(v, s) for v in VARIANTS for s in seeds], runs)
        if run is None or len(run['history']) != epochs or run['result'] is None]
    if len(runs) != 12 or incomplete:
        raise RuntimeError('Final analysis requires all 12 completed runs and test evaluations; '
                           'use --progress. Incomplete: ' + ', '.join(incomplete))
    if sha256(ROOT / 'train.py') != protocol['train_sha256']:
        raise ValueError('train.py no longer matches the frozen source hash')
    import torch
    from train import model
    ids = torch.randperm(50000, generator=torch.Generator().manual_seed(protocol['split_seed'])).tolist()
    expected_split = dict(train=ids[:45000], validation=ids[45000:])
    reference_config = runs[0]['config']
    for run in runs:
        label = f"{run['variant']}_{run['seed']}"
        config, history, folder = run['config'], run['history'], run['folder']
        expected_config = {key: protocol[key] for key in ['epochs', 'batch_size', 'threads', 'device', 'split_seed']}
        expected_config.update(variant=run['variant'], seed=run['seed'], smoke=False, pilot=False,
                               protocol_version=2, normalization='fixed mean=std=0.5')
        for key, value in expected_config.items():
            if key not in config or config[key] != value:
                raise ValueError(label + ': configuration mismatch for ' + key)
        for key in ['torch', 'python', 'platform']:
            if config.get(key) != reference_config.get(key):
                raise ValueError(label + ': environment differs for ' + key)
        if run['split'] != expected_split:
            raise ValueError(label + ': split differs from the frozen split seed/order')
        if run['fieldnames'] != HISTORY_FIELDS or run['history_error'] is not None:
            raise ValueError(label + ': malformed history CSV')
        if [row['epoch'] for row in history] != list(range(1, epochs + 1)):
            raise ValueError(label + ': history must contain each epoch 1..30 exactly once')
        for row in history:
            for metric in ['train_loss', 'val_loss', 'seconds']:
                valid_number(row[metric], label + ':' + metric)
            if row['seconds'] <= 0:
                raise ValueError(label + ': epoch duration must be positive')
            for metric in ['train_accuracy', 'val_accuracy']:
                valid_number(row[metric], label + ':' + metric, maximum=1)
        validate_test_result(run['result'], label)
        if not (folder / 'last.pt').exists() or not (folder / 'best.pt').exists():
            raise ValueError(label + ': missing checkpoint or selected weights')
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        if state['epoch'] != epochs or len(state['history']) != epochs:
            raise ValueError(label + ': checkpoint is incomplete')
        expected_history = [[row[key] for key in HISTORY_FIELDS] for row in history]
        if state['history'] != expected_history:
            raise ValueError(label + ': checkpoint history and CSV differ')
        if state['best'] != max(row['val_accuracy'] for row in history):
            raise ValueError(label + ': selected validation score differs from history')
        net = model(run['variant'] in ['regularized', 'combined'])
        best_state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        net.load_state_dict(best_state, strict=True)
        if sum(p.numel() for p in net.parameters()) != config['parameters']:
            raise ValueError(label + ': parameter count mismatch')
        if any(not torch.isfinite(value).all().item() for value in best_state.values()
               if value.is_floating_point()):
            raise ValueError(label + ': nonfinite selected weights')


def save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def draw_progress(runs, epochs, out):
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    lines = ['# 正式实验进度快照', '', '此文件只展示训练进度，不是最终实验结论。', '',
             '|配置|种子|已完成轮数|正式测试|', '|---|---:|---:|---|']
    for ax, variant in zip(axes.flat, VARIANTS):
        ax.set_title(variant)
        plotted = False
        for run in runs:
            if run is None or run['variant'] != variant:
                continue
            history = run['history']
            lines.append(f"|{variant}|{run['seed']}|{len(history)}/{epochs}|{'已完成' if run['result'] else '未完成'}|")
            if history:
                ax.plot([r['epoch'] for r in history], [r['val_accuracy'] * 100 for r in history],
                        label=f"seed {run['seed']}")
                plotted = True
        if plotted:
            ax.legend()
        else:
            ax.text(.5, .5, 'Pending', transform=ax.transAxes, ha='center')
        ax.set(xlim=(1, epochs), ylim=(0, 100), xlabel='Epoch', ylabel='Validation accuracy (%)')
        ax.grid(alpha=.2)
    fig.suptitle('Training progress only - incomplete experiments')
    fig.tight_layout()
    save(fig, out / 'validation_progress')
    (out / '进度快照.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def final_analysis(runs, protocol, out):
    validate_complete(runs, protocol)  # No final artifact is written before this passes.
    out.mkdir(parents=True, exist_ok=True)
    rows, per_seed, recalls, recall_seeds = [], [], [], []
    for variant in VARIANTS:
        subset = [run for run in runs if run['variant'] == variant]
        for run in subset:
            result, history = run['result'], run['history']
            seconds = [row['seconds'] for row in history]
            best = max(history, key=lambda row: row['val_accuracy'])  # First tie wins, as in training.
            per_seed.append(dict(variant=variant, seed=run['seed'], best_epoch=int(best['epoch']),
                final_epoch=protocol['epochs'], test_accuracy_pct=result['accuracy'] * 100,
                best_validation_accuracy_pct=best['val_accuracy'] * 100, test_loss=result['loss'],
                clean_train_accuracy_pct=result['clean_train_accuracy'] * 100,
                gap_percentage_points=(result['clean_train_accuracy'] - result['accuracy']) * 100,
                parameters=run['config']['parameters'], training_seconds=sum(seconds),
                median_epoch_seconds=statistics.median(seconds), max_epoch_seconds=max(seconds)))
        scores = [run['result']['accuracy'] * 100 for run in subset]
        gaps = [(run['result']['clean_train_accuracy'] - run['result']['accuracy']) * 100 for run in subset]
        durations = [[row['seconds'] for row in run['history']] for run in subset]
        rows.append(dict(variant=variant, n=len(subset), mean_test_accuracy_pct=statistics.mean(scores),
            sample_std_pct=statistics.stdev(scores), mean_gap_percentage_points=statistics.mean(gaps),
            parameters=subset[0]['config']['parameters'],
            mean_training_seconds=statistics.mean(sum(s) for s in durations),
            mean_median_epoch_seconds=statistics.mean(statistics.median(s) for s in durations),
            max_recorded_epoch_seconds=max(max(s) for s in durations)))
        matrices = np.array([run['result']['confusion'] for run in subset], dtype=float)
        normalized_seeds = matrices / matrices.sum(axis=2, keepdims=True)
        normalized = normalized_seeds.mean(axis=0)
        for class_id, class_name in enumerate(CLASSES):
            values = normalized_seeds[:, class_id, class_id] * 100
            recalls.append(dict(variant=variant, class_id=class_id, class_name=class_name, n=len(subset),
                mean_recall_pct=float(values.mean()), sample_std_pct=float(values.std(ddof=1))))
            for run, value in zip(subset, values):
                recall_seeds.append(dict(variant=variant, seed=run['seed'], class_id=class_id,
                    class_name=class_name, recall_pct=float(value)))
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        ax.set_xticks(range(10), CLASSES, rotation=45, ha='right')
        ax.set_yticks(range(10), CLASSES)
        ax.set(xlabel='Predicted class', ylabel='True class',
               title=variant + ' - mean row-normalized confusion matrix (3 seeds)')
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f'{normalized[i, j] * 100:.1f}', ha='center', va='center', fontsize=10.5,
                        color='white' if normalized[i, j] > .5 else 'black')
        fig.colorbar(im, ax=ax, label='Fraction (cell labels are percentages)')
        save(fig, out / f'confusion_{variant}')
        matrix_rows = [dict(true_class=name, **{pred: float(normalized[i, j])
                       for j, pred in enumerate(CLASSES)}) for i, name in enumerate(CLASSES)]
        write_csv(out / f'confusion_{variant}.csv', matrix_rows)
    write_csv(out / 'summary.csv', rows)
    write_csv(out / 'per_seed.csv', per_seed)
    write_csv(out / 'class_recall.csv', recalls)
    write_csv(out / 'class_recall_per_seed.csv', recall_seeds)

    by_seed = {(run['variant'], run['seed']): run['result']['accuracy'] * 100 for run in runs}
    effects, effect_seeds = [], []
    for effect, coefficients in EFFECTS:
        differences = []
        for seed in protocol['seeds']:
            delta = sum(weight * by_seed[(variant, seed)] for variant, weight in coefficients.items())
            differences.append(delta)
            effect_seeds.append(dict(effect=effect, seed=seed, difference_percentage_points=delta))
        effects.append(dict(effect=effect, n=len(differences),
            mean_difference_percentage_points=statistics.mean(differences),
            sample_std_percentage_points=statistics.stdev(differences)))
    write_csv(out / 'paired_effects.csv', effects)
    write_csv(out / 'paired_effects_per_seed.csv', effect_seeds)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.2))
    metrics = [('train_loss', 'Training loss (train mode)'), ('val_loss', 'Validation loss (eval mode)'),
               ('train_accuracy', 'Training accuracy (%) - train mode'),
               ('val_accuracy', 'Validation accuracy (%) - eval mode')]
    for variant in VARIANTS:
        subset = [run for run in runs if run['variant'] == variant]
        for ax, (metric, title) in zip(axes.flat, metrics):
            values = np.array([[row[metric] for row in run['history']] for run in subset])
            if metric.endswith('accuracy'):
                values *= 100
                ax.set_ylim(0, 100)
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            x = np.arange(1, protocol['epochs'] + 1)
            ax.plot(x, mean, label=variant)
            ax.fill_between(x, mean - sd, mean + sd, alpha=.12)
            ax.set(xlabel='Epoch', title=title, xlim=(1, protocol['epochs']))
            ax.title.set_fontsize(9)
            ax.xaxis.label.set_size(9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.2)
    # Set loss limits only after every line is present; otherwise an early
    # set_ylim(bottom=0) freezes Matplotlib's default upper limit at 1.
    for ax, (metric, _) in zip(axes.flat, metrics):
        if metric.endswith('loss'):
            upper=0.
            for variant in VARIANTS:
                values=np.array([[row[metric] for row in run['history']]
                    for run in runs if run['variant']==variant])
                upper=max(upper,float((values.mean(axis=0)+values.std(axis=0,ddof=1)).max()))
            ax.set_ylim(0,upper*1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(.5, .075),fontsize=8)
    fig.suptitle('Learning curves: mean and sample SD (3 seeds)',fontsize=11)
    fig.text(.5, .015, 'Train-mode values include augmentation/Dropout;\nuse clean eval-mode results for generalization gaps.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .95))
    save(fig, out / 'learning_curves')
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [row['mean_test_accuracy_pct'] for row in rows]
    stds = [row['sample_std_pct'] for row in rows]
    ax.bar(VARIANTS, means, yerr=stds, capsize=5, color=['#527ca5', '#648c66', '#ad875c', '#8c70a0'])
    for index, (mean, sd) in enumerate(zip(means, stds)):
        ax.text(index, mean + sd + 2, f'{mean:.2f} ± {sd:.2f}', ha='center', fontsize=10)
    ax.set(ylim=(0, 100), ylabel='Test accuracy (%)', title='Final test accuracy - 3 seeds per configuration')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, 'Error bars: sample standard deviation, not confidence intervals.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, out / 'test_accuracy')

    report = ['# 正式结果汇总', '', '十二次运行均通过冻结配置、索引、检查点、轮次和测试计数核验。', '',
              '均值与样本标准差基于三个随机种子；标准差不是置信区间。', '',
              '|配置|测试准确率（%）|参数量|平均记录训练秒数|', '|---|---:|---:|---:|']
    for row in rows:
        report.append(f"|{row['variant']}|{row['mean_test_accuracy_pct']:.2f} ± {row['sample_std_pct']:.2f}|"
                      f"{row['parameters']}|{row['mean_training_seconds']:.1f}|")
    report += ['', '记录训练秒数为各轮经过时间之和，可能包含休眠、系统负载和运行中断等干扰。'
               '保留原始值供追溯，不能据此作纯计算效率排名。每轮中位数仅作补充描述，也不能消除这些干扰。', '',
               '## 相同种子的配对差值', '', '|比较|平均差值（百分点）|样本标准差（百分点）|', '|---|---:|---:|']
    for effect in effects:
        report.append(f"|{effect['effect']}|{effect['mean_difference_percentage_points']:.3f}|"
                      f"{effect['sample_std_percentage_points']:.3f}|")
    report += ['', 'interaction = combined − augment − regularized + baseline；它是当前条件下的描述性交互差值。',
               '三次重复只支持有限的描述性比较，不声称统计显著、普遍优势或已分别识别 BN 与 Dropout 的独立贡献。', '',
               '训练曲线中的训练指标在 train 模式计算，受增强与 Dropout 影响，不能直接与验证指标相减作为干净泛化差距。'
               'summary.csv 的泛化差距使用最佳模型在无增强 eval 模式下的训练准确率减测试准确率。']
    (out / '结果汇总.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    manifest = dict(validation='passed', run_count=12, epochs=protocol['epochs'], seeds=protocol['seeds'],
        train_sha256=protocol['train_sha256'], generated_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_time_caveat=ELAPSED_NOTE, inputs={})
    for run in runs:
        manifest['inputs'][f"{run['variant']}_{run['seed']}"] = {
            name: sha256(run['folder'] / name)
            for name in ['config.json', 'split.json', 'history.csv', 'test.json', 'best.pt', 'last.pt']}
    (out / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='Optional output directory; defaults to progress/ or analysis/')
    args = parser.parse_args()
    protocol = read_json(ROOT / 'frozen_protocol.json')
    runs = [load_run(variant, seed) for variant in VARIANTS for seed in protocol['seeds']]
    out = args.output_dir or ROOT / ('progress' if args.progress else 'analysis')
    if args.progress:
        draw_progress(runs, protocol['epochs'], out)
    else:
        final_analysis(runs, protocol, out)
    print(out)


if __name__ == '__main__':
    main()
