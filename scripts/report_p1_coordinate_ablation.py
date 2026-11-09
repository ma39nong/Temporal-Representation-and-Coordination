#!/usr/bin/env python3
"""Four-way Step0 diagnostic from sealed native FP32 forward tensors."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from report_p1_transfer import load
from trc.analysis.flash.coordinate_ablation import cross_decode, factorial_effects
from trc.analysis.flash.offset import decompose_offsets
from trc.analysis.flash.metrics import EVAL_GROUPS, summary
from trc.analysis.flash.report import csv_rows
from trc.io import file_sha256
from trc.data.action_schema import ORDER


def main():
    parser = argparse.ArgumentParser()
    for key in ('absolute', 'relative', 'baseline', 'review', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists(): raise FileExistsError(out)
    m0, d0, seal0 = load(args.absolute)
    m3, d3, seal3 = load(args.relative)
    mb, db, sealb = load(args.baseline)
    for m in (m0, m3):
        if m['checkpoint_step'] != 0 or not m['future_perturbation_passed'] or m['optimizer_updates'] != 0:
            raise ValueError('Verified zero-update, no-future Step0 exports required')
    if (m0['coordinate_mode'], m3['coordinate_mode']) != ('absolute', 'state_relative'):
        raise ValueError('Wrong coordinate modes')
    for key in ('checkpoint_parameter_sha256', 'windows', 'normalization_sha256', 'rgb_sequence_sha256', 'source_commit'):
        if not m0[key] == m3[key] == mb[key]: raise ValueError('Pair identity mismatch: '+key)
    if m0['checkpoint_parameter_sha256'] != '40b2c2c4dcdd6fb88dd847d5762d2071fba5028af0942c6dfe457409fe5f5919':
        raise ValueError('Expected original Step8000 parameters')
    for key in ('truth', 'oracle', 'c_gt', 'history', 'current_state', 'basis', 'scale', 'q01', 'q99', 'valid'):
        np.testing.assert_array_equal(d0[key], d3[key])
        np.testing.assert_array_equal(d0[key], db[key])
    np.testing.assert_array_equal(d0['prediction'], db['prediction'])
    np.testing.assert_array_equal(d0['history'][:, -1], d0['current_state'])
    decoded, coeff, shift = cross_decode(d0['c_pred'], d3['c_pred'], d0['history'],
                                        d0['basis'], d0['scale'], d0['q01'], d0['q99'])
    parity = {name: float(np.max(np.abs(decoded[name]-d['prediction'])))
              for name, d in [('D0', d0), ('D3', d3)]}
    if max(parity.values()) > 1e-5: raise ValueError('Native coefficient decode parity failed')
    # Retain the original FP32 predictions exactly. Mixed decoders are the exact
    # affine output change on these results, evaluated in float64 offline.
    predictions = {'D0': d0['prediction'], 'D1': d3['prediction']-shift[:, None],
                   'D2': d0['prediction']+shift[:, None], 'D3': d3['prediction']}
    mixed_parity = {k: float(np.max(np.abs(predictions[k]-decoded[k]))) for k in predictions}
    if max(mixed_parity.values()) > 1e-5: raise ValueError('Mixed decode parity failed')
    gt = d0['truth']
    fac = factorial_effects(predictions, gt)
    windows = m0['windows']; src = np.array([w['source_recording_id'] for w in windows])
    train = np.array([w['training_source'] for w in windows])
    review = json.loads(Path(args.review).read_text())
    failed = {r['source_recording_id'] for r in review['records'] if r['review_entries']}
    bad = np.array([s in failed for s in src])
    scopes = {'training_sources': train, 'candidate_unrejected_quality_unverified': ~train & ~bad,
              'candidate_reviewed_failed': ~train & bad, 'all_candidates_exploratory': ~train}
    if [int(scopes[k].sum()) for k in list(scopes)[:3]] != [485, 10, 10]:
        raise ValueError('Historical cohort changed')
    source_alias = {s: next(w['sample_id'] for w in windows if w['source_recording_id'] == s) for s in sorted(set(src))}
    scopes.update({alias: src == source for source, alias in source_alias.items()})
    metrics, offsets, temporal, window_rows, effects, state_rows, joint_states = [], [], [], [], [], [], []
    for name, pred in predictions.items():
        error = pred-gt; dec = decompose_offsets(gt, pred)
        for scope, mask in scopes.items():
            ident = {'variant': name, 'scope': scope, 'windows': int(mask.sum()), 'sources': len(set(src[mask]))}
            for group, idx in EVAL_GROUPS.items():
                err = np.take(error[mask], idx, axis=-1)
                offsets.append({**ident, 'group': group,
                    'offset_mse': float(dec['offset_mse'][mask][:, idx].mean()),
                    'shape_mse': float(dec['shape_mse'][mask][:, idx].mean())})
                for start, label in [(0, 'all50'), (41, 'last9')]:
                    e = err[:, start:]; hit = (np.abs(e) > .5).any(axis=(1, 2))
                    metrics.append({**ident, 'group': group, 'interval': label, **summary(e),
                        'mse': float(np.mean(e*e)), 'over_0p5_elements': int((np.abs(e) > .5).sum()),
                        'over_0p5_windows': int(hit.sum()), 'over_0p5_sources': len(set(src[mask][hit]))})
                if scope in list(scopes)[:4]:
                    temporal.extend({**ident, 'group': group, 'tick': t, **summary(err[:, t])} for t in range(50))
        for i, w in enumerate(windows):
            window_rows.append({'variant': name, 'window': i, 'source': w['sample_id'], 'current_row': w['current_row'],
                'training': bool(train[i]), 'review_failed': bool(bad[i]), **summary(error[i]),
                'offset_mse': float(dec['offset_mse'][i].mean()), 'shape_mse': float(dec['shape_mse'][i].mean())})
    for scope, mask in scopes.items():
        row = {'scope': scope, 'windows': int(mask.sum()), 'sources': len(set(src[mask]))}
        row.update({k: float(v[mask].mean()) for k, v in fac.items() if isinstance(v, np.ndarray)})
        row.update({k+'_mse': float(v[mask].mean()) for k, v in fac['mse'].items()})
        effects.append(row)
        for j in range(58):
            q = d0['current_state'][mask, j]; z = shift[mask, j]
            bias = (predictions['D0']-gt).mean(1)[mask, j]
            joint_states.append({'scope': scope, 'joint': j, 'joint_name': ORDER[j], 'windows': int(mask.sum()),
                'q0_mean_rad': float(q.mean()), 'q0_std_rad': float(q.std()),
                'q0_p05_rad': float(np.quantile(q, .05)), 'q0_p95_rad': float(np.quantile(q, .95)),
                'shift_mean_rad': float(z.mean()), 'D0_bias_mean_rad': float(bias.mean()),
                'shift_bias_cross_term': float(np.mean(2*z*bias)), 'shift_mse': float(np.mean(z*z)),
                'within_joint_q0_bias_correlation': float(np.corrcoef(q, bias)[0, 1]) if q.std() and bias.std() else None})
        for group, idx in EVAL_GROUPS.items():
            q = d0['current_state'][mask][:, idx]; z = shift[mask][:, idx]
            b0 = (predictions['D0']-gt).mean(1)[mask][:, idx]
            bh = (predictions['D1']-predictions['D0']).mean(1)[mask][:, idx]
            corr = lambda x, y: float(np.corrcoef(x.ravel(), y.ravel())[0, 1]) if x.std() and y.std() else None
            state_rows.append({**row, 'group': group, 'q0_mean_rad': float(q.mean()), 'q0_std_rad': float(q.std()),
                'q0_min_rad': float(q.min()), 'q0_max_rad': float(q.max()), 'shift_rms_rad': float(np.sqrt(np.mean(z*z))),
                'q0_vs_D0_bias_pooled_correlation': corr(q, b0), 'shift_vs_D0_bias_pooled_correlation': corr(z, b0),
                'history_effect_bias_vs_shift_pooled_correlation': corr(bh, z),
                'D2_offset_mse_change_cross_term': float(np.mean(2*b0*z+z*z))})
    out.mkdir(parents=True)
    for name, rows in [('metrics', metrics), ('offsets', offsets), ('temporal', temporal),
                       ('windows', window_rows), ('factorial_effects', effects), ('reference_state', state_rows), ('reference_state_joints', joint_states)]:
        csv_rows(out/(name+'.csv'), rows)
    lookup = {(r['scope'], r['variant']): r for r in metrics if r['group'] == 'full_action' and r['interval'] == 'all50'}
    source_pairs = []
    for source, alias in source_alias.items():
        source_pairs.append({'source_alias': alias, 'training_source': bool(train[src == source][0]),
            'review_failed': source in failed, **{v+'_'+k: lookup[alias, v][k]
                for v in predictions for k in ('mae', 'rmse', 'p95', 'p99', 'max', 'over_0p5_elements')},
            **{v+'_minus_D0_rmse': lookup[alias, v]['rmse']-lookup[alias, 'D0']['rmse'] for v in ('D1', 'D2', 'D3')}})
    csv_rows(out/'source_pairs.csv', source_pairs)
    np.savez(out/'cross_predictions.npz', **predictions, physical_shift=shift)
    np.savez(out/'cross_coefficients.npz', **coeff)
    identity = {'data_kind': 'real', 'method': 'exact_affine_cross_decode_of_sealed_native_FP32_Step0_forwards',
        'research_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'script_sha256': file_sha256(__file__), 'core_sha256': file_sha256(Path(__file__).parents[1]/'src/trc/analysis/flash/coordinate_ablation.py'),
        'original_source_commit': m0['source_commit'], 'checkpoint_parameter_sha256': m0['checkpoint_parameter_sha256'],
        'absolute_export': str(Path(args.absolute).resolve()), 'relative_export': str(Path(args.relative).resolve()),
        'absolute_seal': seal0, 'relative_seal': seal3, 'baseline_seal': sealb,
        'normalization_sha256': m0['normalization_sha256'], 'rgb_sequence_sha256': m0['rgb_sequence_sha256'],
        'review_sha256': file_sha256(args.review), 'windows': windows,
        'native_decoder_max_rad': parity, 'mixed_decoder_max_rad': mixed_parity,
        'D0_matches_original_step8000_exactly': True, 'D3_matches_B2_step0_exactly': True,
        'inference_randomness': 'Original inference discards rng; deterministic history-only Euler NFE1. No noise sampled. Training/flow seed42 unchanged in sealed pair.',
        'native_evaluation': 'FP32/highest, batch1, no padding; original no-future test passed both',
        'postprocessing': 'float64; no GPU forward or optimizer update this round',
        'physical_shift': 'q0 - (q01 + (q99-q01+1e-6)/2); NOT q0',
        'factorial_function_interaction_max_rad': fac['function_interaction_max_rad'],
        'independent_generalization_claim': False}
    (out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    # Source-level direction, no pseudo-independent window confidence interval.
    directions = []
    for cohort, cmask in list(scopes.items())[:3]:
        for a, b in [('D3', 'D0'), ('D1', 'D0'), ('D2', 'D0'), ('D3', 'D2'), ('D3', 'D1')]:
            changes = [float((fac['mse'][a]-fac['mse'][b])[src == s].mean()) for s in sorted(set(src[cmask]))]
            directions.append({'scope': cohort, 'contrast': a+'-'+b, 'sources': len(changes),
                'improved': sum(x < -1e-14 for x in changes), 'degraded': sum(x > 1e-14 for x in changes),
                'unchanged': sum(abs(x) <= 1e-14 for x in changes), 'source_mean_delta_mse': float(np.mean(changes))})
    (out/'directions.json').write_text(json.dumps(directions, indent=2)+'\n')
    plot(out, metrics, offsets, temporal, effects, windows, gt, predictions)
    print(json.dumps({'output': str(out), 'native_parity': parity,
        'main': [r for r in metrics if r['scope'] in list(scopes)[:3] and r['group'] == 'full_action' and r['interval'] == 'all50'],
        'effects': effects[:4], 'directions': directions}, indent=2))


def plot(out, metrics, offsets, temporal, effects, windows, gt, predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scopes = ['training_sources', 'candidate_unrejected_quality_unverified', 'candidate_reviewed_failed']
    titles = ['Training 485', 'Unrejected 10 (success unverified)', 'Reviewed failed 10']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, scope, title in zip(axes, scopes, titles):
        rows = [next(r for r in metrics if r['variant'] == v and r['scope'] == scope and r['group'] == 'full_action' and r['interval'] == 'all50') for v in predictions]
        ax.bar(list(predictions), [r['rmse'] for r in rows]); ax.set(title=title, ylabel='Absolute RMSE (rad)')
    fig.suptitle('REAL Step8000 zero-update cross decoding; exploratory candidates')
    fig.savefig(out/'four_way_rmse.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, scope, title in zip(axes, scopes, titles):
        for v in predictions:
            r = [r for r in temporal if r['variant'] == v and r['scope'] == scope and r['group'] == 'full_action']
            ax.plot([x['tick'] for x in r], [x['rmse'] for x in r], label=v)
        ax.axvspan(41, 49, alpha=.1, color='red'); ax.set(title=title, xlabel='Future tick', ylabel='RMSE (rad)'); ax.legend()
    fig.suptitle('REAL same-window temporal diagnostic; no operation-phase inference')
    fig.savefig(out/'temporal.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, scope, title in zip(axes, scopes, titles):
        for pos, v in enumerate(predictions):
            r = next(r for r in offsets if r['variant'] == v and r['scope'] == scope and r['group'] == 'full_action')
            ax.bar(np.arange(2)+(pos-1.5)*.18, [r['offset_mse'], r['shape_mse']], width=.18, label=v)
        ax.set_xticks([0, 1], ['Offset', 'Centered shape']); ax.set(title=title, ylabel='MSE (rad squared)'); ax.legend()
    fig.suptitle('REAL exact physical decomposition; decoder translation leaves shape unchanged')
    fig.savefig(out/'offset_shape.png', dpi=170); plt.close(fig)


if __name__ == '__main__': main()
