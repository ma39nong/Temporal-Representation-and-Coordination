#!/usr/bin/env python3
"""Source-scoped P1 pilot comparison from sealed real exports; never runs a model."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import statistics

import numpy as np

from trc.analysis.flash.metrics import EVAL_GROUPS, coefficient_metrics, summary
from trc.analysis.flash.offset import decompose_offsets, grouped_offsets
from trc.analysis.flash.diagnosis import coefficient_contributions
from trc.analysis.flash.report import csv_rows
from trc.io import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exports', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--training-root', help='Local small-artifact copy, never model weights')
    p.add_argument('--source-review', help='Explicit current source review metadata; labels are not rejection reasons')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    results, temporal, offsets, coefficients, contributions, identities = [], [], [], [], [], []
    seen_exports, reference, finals = set(), None, {}
    for folder in map(Path, args.exports):
        seal = json.loads((folder / 'seal.json').read_text())
        if any(file_sha256(folder / f) != seal[k] for f, k in
               [('aligned.npz', 'aligned_sha256'), ('metadata.json', 'metadata_sha256')]):
            raise ValueError('Unsealed export')
        m = json.loads((folder / 'metadata.json').read_text())
        if m['data_kind'] != 'real' or m['evaluation_precision'] != 'float32':
            raise ValueError('Real shared-FP32 exports required')
        trial = {'absolute': 'B0', 'state_relative': 'B2'}[m['coordinate_mode']]
        ident = {'trial': trial, 'step': m['checkpoint_step']}
        key = (trial, ident['step'])
        if key in seen_exports:
            raise ValueError('Duplicate trial/checkpoint')
        seen_exports.add(key)
        with np.load(folder / 'aligned.npz', allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        if m['checkpoint_step'] == 500:
            finals[trial] = (m, d)
        if not d['valid'].all() or d['truth'].shape != (505, 50, 58):
            raise ValueError('Exactly the sealed 505 complete windows required')
        windows = [(w['sample_id'], w['current_row'], w['source_recording_id']) for w in m['windows']]
        if reference is None:
            reference = (windows, d)
        else:
            if windows != reference[0]:
                raise ValueError('Paired windows differ')
            for field in ('truth', 'history', 'current_state', 'q01', 'q99', 'scale', 'basis'):
                np.testing.assert_array_equal(d[field], reference[1][field])
            np.testing.assert_allclose(d['oracle'], reference[1]['oracle'], atol=1e-6, rtol=1e-6)
        seen = np.array([w['training_source'] for w in m['windows']], bool)
        if (seen.sum(), (~seen).sum()) != (485, 20):
            raise ValueError('Training/candidate scope mismatch')
        sources = np.array([w['source_recording_id'] for w in m['windows']])
        scopes = {'training_sources': seen, 'candidate_sources': ~seen}
        scopes.update({'source:' + source: sources == source for source in sorted(set(sources))})
        for scope, mask in scopes.items():
            row_id = {**ident, 'scope': scope, 'training_source': bool(seen[mask].all()),
                      'windows': int(mask.sum()), 'sources': len(set(sources[mask]))}
            a, o, pred = (d[k][mask].astype(np.float64) for k in ('truth', 'oracle', 'prediction'))
            for method, error in [('total', pred - a), ('representation', o - a), ('prediction', pred - o)]:
                for group, idx in EVAL_GROUPS.items():
                    e = error[..., idx]
                    results.append({**row_id, 'method': method, 'group': group, **summary(e)})
                    if scope in ('training_sources', 'candidate_sources') or not seen[mask].any():
                        for tick in range(50):
                            temporal.append({**row_id, 'method': method, 'group': group, 'tick': tick,
                                             'seconds': tick / 30, **summary(e[:, tick])})
            offsets.extend({**row_id, **r} for r in grouped_offsets(decompose_offsets(a, pred)))
            if scope in ('training_sources', 'candidate_sources') or not seen[mask].any():
                cg, cp = d['c_gt'][mask], d['c_pred'][mask]
                coefficients.extend({**row_id, **r} for r in coefficient_metrics(cg, cp, d['scale'], d['q01'], d['q99']))
                c = coefficient_contributions(a, o, pred, cg, cp, d['scale'], d['basis'], d['q01'], d['q99'])
                contributions.extend({**row_id, **r} for r in c['rows'])
        identities.append({**ident, 'checkpoint_id': m['checkpoint_id'], 'parameter_sha256': m['checkpoint_parameter_sha256'],
                           'training_contract_sha256': m['training_contract_sha256'], 'export_seal': seal})
    if seen_exports != {(trial, step) for trial in ('B0', 'B2') for step in (0, 100, 250, 500)}:
        raise ValueError('The complete paired 0/100/250/500 protocol is required')
    out.mkdir(parents=True)
    for name, rows in [('metrics', results), ('temporal', temporal), ('offsets', offsets),
                       ('coefficients', coefficients), ('coefficient_contributions', contributions)]:
        csv_rows(out / (name + '.csv'), rows)
    paired = []
    for row in results:
        if row['trial'] != 'B0' or row['method'] != 'total':
            continue
        other = next((r for r in results if r['trial'] == 'B2' and all(
            r[k] == row[k] for k in ('step', 'scope', 'method', 'group'))), None)
        if other:
            paired.append({k: row[k] for k in ('step', 'scope', 'group', 'windows', 'sources')} |
                          {'B0_rmse': row['rmse'], 'B2_rmse': other['rmse'],
                           'B2_minus_B0_rmse': other['rmse'] - row['rmse'],
                           'B2_over_B0_rmse': other['rmse'] / row['rmse'] if row['rmse'] else None})
    csv_rows(out / 'paired_comparison.csv', paired)
    rejected = set()
    if args.source_review:
        review = json.loads(Path(args.source_review).read_text())
        rejected = {r['source_recording_id'] for r in review['current_metadata_locations'] if r['rejected_path_label']}
    (out / 'result_identity.json').write_text(json.dumps({'exports': identities, 'data_kind': 'real',
        'interpretation': 'One-seed pilot; 4 candidate sources/20 windows are exploratory, independence incomplete.',
        'source_review_sha256': file_sha256(args.source_review) if args.source_review else None,
        'report_code_sha256': file_sha256(__file__)}, indent=2) + '\n')
    plot(out, results, temporal, offsets, contributions, rejected)
    tail_cases(out, finals)
    if args.training_root:
        training_report(Path(args.training_root), out)
    print(json.dumps({'exports': len(identities), 'paired_rows': len(paired), 'output': str(out)}))


def plot(out, metrics, temporal, offsets, contributions, rejected_sources=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 180})
    scopes = ['training_sources', 'candidate_sources']
    colors = {'B0': '#3465a4', 'B2': '#c65d25'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, scope in zip(axes, scopes):
        for trial in colors:
            rows = sorted([r for r in metrics if r['scope'] == scope and r['trial'] == trial and
                           r['method'] == 'total' and r['group'] == 'full_action'], key=lambda r: r['step'])
            ax.plot([r['step'] for r in rows], [r['rmse'] for r in rows], 'o-', color=colors[trial], label=trial)
        ax.set(xlabel='Accepted optimizer updates', ylabel='Physical RMSE (rad)', title=scope.replace('_', ' '))
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Real fixed windows; candidate sources are exploratory')
    fig.savefig(out / 'paired_rmse.png'); plt.close(fig)
    steps = sorted({r['step'] for r in metrics})
    final = max(steps)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, scope in enumerate(scopes):
        for trial in colors:
            rows = sorted([r for r in temporal if r['scope'] == scope and r['step'] == final and
                           r['trial'] == trial and r['method'] == 'total' and r['group'] == 'full_action'], key=lambda r: r['tick'])
            axes[0, col].plot([r['tick'] for r in rows], [r['rmse'] for r in rows], color=colors[trial], label=trial)
            off = sorted([r for r in offsets if r['scope'] == scope and r['trial'] == trial and r['group'] == 'full_action'], key=lambda r:r['step'])
            axes[1, col].plot([r['step'] for r in off], [r['offset_mse'] for r in off], 'o-', color=colors[trial], label=trial+' offset')
            axes[1, col].plot([r['step'] for r in off], [r['shape_mse'] for r in off], 'x--', color=colors[trial], label=trial+' shape')
        axes[0, col].set(title=scope.replace('_', ' '), xlabel='Prediction tick', ylabel='RMSE (rad)')
        axes[1, col].set(xlabel='Accepted optimizer updates', ylabel='MSE (rad²)')
        for ax in axes[:, col]: ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Real paired predictions: temporal errors and exact offset/shape decomposition')
    fig.savefig(out / 'temporal_and_offsets.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, scope in zip(axes, scopes):
        for trial in colors:
            rows = sorted([r for r in contributions if r['scope'] == scope and r['trial'] == trial and
                           r['step'] == final and r['group'] == 'full_action'], key=lambda r:r['degree'])
            ax.plot([r['degree'] for r in rows], [r['component_rmse_rad'] for r in rows], 'o-', color=colors[trial], label=trial)
        ax.set(title=scope.replace('_', ' '), xlabel='Legendre degree', ylabel='Standalone decoded component RMSE (rad)')
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Per-degree sensitivity; squared components are not additive')
    fig.savefig(out / 'coefficient_components.png'); plt.close(fig)
    groups = ['left_arm', 'left_hand', 'right_arm', 'right_hand']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(4)
    for ax, scope in zip(axes, scopes):
        for n, trial in enumerate(colors):
            vals = [next(r['rmse'] for r in metrics if r['trial'] == trial and r['step'] == final and
                         r['scope'] == scope and r['method'] == 'total' and r['group'] == group) for group in groups]
            ax.bar(x + (n - .5) * .36, vals, width=.36, color=colors[trial], label=trial)
        ax.set(xticks=x, xticklabels=['Left arm', 'Left hand', 'Right arm', 'Right hand'],
               ylabel='Physical RMSE (rad)', title=scope.replace('_', ' '))
        ax.legend()
    fig.suptitle('Real paired group errors at final pilot checkpoint')
    fig.savefig(out / 'group_errors.png'); plt.close(fig)
    candidates = sorted({r['scope'] for r in metrics if r['scope'].startswith('source:') and not r['training_source']})
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for n, trial in enumerate(colors):
        vals = [next(r['rmse'] for r in metrics if r['trial'] == trial and r['step'] == final and
                     r['scope'] == scope and r['method'] == 'total' and r['group'] == 'full_action') for scope in candidates]
        ax.bar(np.arange(len(candidates))+(n-.5)*.36, vals, width=.36, color=colors[trial], label=trial)
    ax.set(xticks=np.arange(len(candidates)), xticklabels=[s[7:15]+('*' if s[7:] in rejected_sources else '') for s in candidates],
           xlabel='Candidate source (5 windows each)', ylabel='Physical RMSE (rad)',
           title='Four exploratory sources; independence incompletely verified' +
                 ('\n* Current rejected label; reason unverified' if rejected_sources else ''))
    ax.legend(); fig.savefig(out / 'candidate_sources.png'); plt.close(fig)


def training_report(root, out):
    import matplotlib.pyplot as plt
    all_rows, reports = {}, {}
    timestamps_path = root / 'run_timestamps.json'
    timestamps = json.loads(timestamps_path.read_text()) if timestamps_path.exists() else {}
    gpu = []
    if (root / 'gpu_samples.csv').exists():
        with (root / 'gpu_samples.csv').open() as stream:
            gpu = list(csv.DictReader(stream))
    for trial in ('B0', 'B2'):
        files = list((root / 'runs/checkpoints' / ('trc_p1_flash_' + trial.lower())).glob('*/flash_metrics/training.jsonl'))
        if len(files) != 1:
            raise ValueError('One original training log per trial required')
        rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
        if [r['step'] for r in rows] != list(range(1, 501)):
            raise ValueError('Exactly 500 accepted, contiguous updates required')
        if not all(all(np.isfinite(r[k]) for k in ('loss', 'gradient_norm', 'vision_gradient_norm',
                                                  'language_gradient_norm', 'expert_gradient_norm')) for r in rows):
            raise ValueError('Nonfinite training log')
        if not all(r['effective_samples'] == r['step'] * 256 for r in rows):
            raise ValueError('Global batch budget drift')
        all_rows[trial] = rows
        steady = [r for r in rows if r['step'] not in (1, 4)]  # first compilation in each process
        seconds = statistics.median(r['seconds'] + r['data_wait_seconds'] for r in steady)
        audit = root / 'runs/execution' / trial
        saved = json.loads((audit / 'saved_state_3.json').read_text())
        restored = json.loads((audit / 'restored_state.json').read_text())
        if any(saved[k] != restored[k] for k in ('params_sha256', 'opt_state_sha256', 'completed_step')):
            raise ValueError('Checkpoint restore identity mismatch')
        reports[trial] = {'accepted_updates': 500, 'sample_positions': 128000,
                         'first20_mean_loss': statistics.mean(r['loss'] for r in rows[:20]),
                         'last20_mean_loss': statistics.mean(r['loss'] for r in rows[-20:]),
                         'median_compute_seconds_excluding_compile': statistics.median(r['seconds'] for r in steady),
                         'median_compute_and_io_seconds_excluding_compile': seconds,
                         'steady_windows_per_second_compute_and_io': 256 / seconds,
                         'gpu0_jax_peak_bytes': max(r['peak_device_bytes'] for r in rows),
                         'restored_step3_byte_identical': True, 'training_log_sha256': file_sha256(files[0]),
                         'initial_state': json.loads((audit / 'initial_state.json').read_text())}
        if trial in timestamps and gpu:
            begin, end = timestamps[trial]['start_epoch'], timestamps[trial]['end_epoch']
            samples = [r for r in gpu if begin <= datetime.datetime.fromisoformat(r['utc']).timestamp() <= end]
            reports[trial]['nvidia_smi_sampled_peak_reserved_mib_by_device'] = {
                str(device): max(float(r['used_mib']) for r in samples if int(r['index']) == device) for device in range(8)}
    if any(reports['B0']['initial_state'][k] != reports['B2']['initial_state'][k]
           for k in ('params_sha256', 'opt_state_sha256')):
        raise ValueError('Paired initial states differ')
    csv_rows(out / 'training_steps.csv', [{'trial': t, **row} for t, rows in all_rows.items() for row in rows])
    (out / 'training_summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for trial, color in [('B0', '#3465a4'), ('B2', '#c65d25')]:
        rows = all_rows[trial]; x = np.array([r['step'] for r in rows])
        for ax, key in zip(axes, ('loss', 'flow_matching_loss', 'polynomial_consistency_loss')):
            y = np.array([r[key] for r in rows])
            ax.plot(x, y, color=color, alpha=.18, linewidth=.6)
            ax.plot(x[19:], np.convolve(y, np.ones(20) / 20, mode='valid'), color=color, label=trial)
            ax.set(title=key.replace('_', ' '), xlabel='Accepted optimizer updates', ylabel='Training objective')
            ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Real training logs: raw steps and trailing 20-step means')
    fig.savefig(out / 'training_loss.png'); plt.close(fig)


def tail_cases(out, finals):
    """Report actual regressions without assigning operation phases or FK meaning."""
    import matplotlib.pyplot as plt
    from trc.data.action_schema import ORDER
    m, b0 = finals['B0']; _, b2 = finals['B2']
    e0 = b0['prediction'].astype(float) - b0['truth']
    e2 = b2['prediction'].astype(float) - b2['truth']
    seen = np.array([w['training_source'] for w in m['windows']], bool)
    delta = np.mean(e2**2 - e0**2, axis=(1, 2))
    rows = []
    for i, w in enumerate(m['windows']):
        rows.append({'window_index': i, 'source': w['source_recording_id'], 'current_row': w['current_row'],
                     'training_source': bool(seen[i]), 'B0_rmse': float(np.sqrt(np.mean(e0[i]**2))),
                     'B2_rmse': float(np.sqrt(np.mean(e2[i]**2))), 'B2_minus_B0_mse': float(delta[i]),
                     'B0_max': float(np.abs(e0[i]).max()), 'B2_max': float(np.abs(e2[i]).max())})
    csv_rows(out / 'paired_window_metrics.csv', rows)
    train = np.flatnonzero(seen); candidate = np.flatnonzero(~seen)
    worst_train = train[np.argmax(np.max(np.abs(e2[train]), axis=(1, 2)))]
    worst_regression = train[np.argmax(delta[train])]
    worst_candidate = candidate[np.argmax(np.max(np.abs(e2[candidate]), axis=(1, 2)))]
    selections = [('Largest training absolute error', worst_train),
                  ('Largest training-window MSE regression', worst_regression),
                  ('Largest candidate absolute error', worst_candidate)]
    cases = []; fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (label, i) in zip(axes, selections):
        tick, joint = np.unravel_index(np.argmax(np.abs(e2[i])), e2[i].shape)
        for name, value, color, style in [('GT', b0['truth'], '#222222', '-'),
                                         ('Oracle', b0['oracle'], '#777777', '--'),
                                         ('B0', b0['prediction'], '#3465a4', '-'),
                                         ('B2', b2['prediction'], '#c65d25', '-')]:
            ax.plot(np.arange(50), value[i, :, joint], style, color=color, label=name)
        ax.set(title=label+'\n'+ORDER[joint], xlabel='Prediction tick', ylabel='Absolute joint command (rad)')
        ax.legend(); ax.grid(alpha=.2)
        cases.append({'selection': label, **rows[i], 'joint_index': int(joint), 'joint_name': ORDER[joint],
                      'worst_tick': int(tick), 'absolute_gt': float(b0['truth'][i, tick, joint]),
                      'absolute_B0': float(b0['prediction'][i, tick, joint]),
                      'absolute_B2': float(b2['prediction'][i, tick, joint])})
    fig.suptitle('Real Step500 tail cases; no phase or hardware-failure interpretation')
    fig.savefig(out / 'tail_cases.png'); plt.close(fig)
    (out / 'tail_cases.json').write_text(json.dumps(cases, indent=2) + '\n')


if __name__ == '__main__':
    main()
