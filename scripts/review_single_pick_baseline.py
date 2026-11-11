"""CPU-only review of existing training-source predictions; no new model calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from trc.analysis.flash.crs import selected_npz_rows
from trc.analysis.flash.metrics import EVAL_GROUPS, summary
from trc.data.action_schema import ORDER


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text())
    paths = {k: (config_path.parent / cfg[k]).resolve()
             for k in ('export', 'training_manifest', 'output')}
    export, out = paths['export'], paths['output']
    if out == export or export in out.parents:
        raise ValueError('Output must be separate from frozen export')
    metadata = json.loads((export / 'metadata.json').read_text())
    manifest = json.loads(paths['training_manifest'].read_text())
    records = {r['source_recording_id']: r for r in manifest['records']}
    if (len(records) != 480 or metadata['action_semantics'] != 'absolute'
            or metadata['action_unit'] != 'rad' or manifest['contract']['joint_names'] != list(ORDER)):
        raise ValueError('Expected frozen all480 Absolute physical baseline')
    indices = [i for i, w in enumerate(metadata['windows']) if w.get('training_source') is True]
    windows = [metadata['windows'][i] for i in indices]
    if len(indices) != 485 or any(w['source_recording_id'] not in records for w in windows):
        raise ValueError('Expected 485 verified training-source windows')
    arrays = {key: selected_npz_rows(export / 'aligned.npz', key, indices)
              for key in ('truth', 'oracle', 'prediction', 'valid')}
    a, p, o = (np.asarray(arrays[key], float) for key in ('truth', 'prediction', 'oracle'))
    if arrays['valid'].dtype != bool or not arrays['valid'].all() or not np.isfinite([a, p, o]).all():
        raise ValueError('This review requires fully valid finite fixed windows')
    e = p - a
    increments = np.diff(e, axis=1)  # rad per adjacent sample, not rad/s
    b = e.mean(axis=1)
    rows = []
    for i, w in enumerate(windows):
        tick, joint = np.unravel_index(np.argmax(np.abs(e[i])), e[i].shape)
        rows.append({'local_window': i, 'export_window': indices[i], **w,
                     **summary(e[i]), 'oracle_rmse': summary(o[i]-a[i])['rmse'],
                     'prediction_to_oracle_rmse': summary(p[i]-o[i])['rmse'], 'offset_mse': float(np.mean(b[i]**2)),
                     'shape_mse': float(np.mean((e[i]-b[i])**2)),
                     'increment_rmse_rad': summary(increments[i])['rmse'],
                     'left_arm_last9_rmse': summary(e[i, -9:, :7])['rmse'],
                     'worst_tick': int(tick), 'worst_joint': ORDER[joint]})
    out.mkdir(parents=True, exist_ok=True)
    def write_csv(name, values):
        with (out / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    write_csv('windows.csv', rows)
    groups = []
    for group, idx in EVAL_GROUPS.items():
        ge = e[..., idx]
        groups.append({'group': group, **summary(ge),
                       'offset_mse': float(np.mean(b[:, idx]**2)),
                       'shape_mse': float(np.mean((ge-b[:, None, idx])**2)),
                       'last9_rmse': summary(ge[:, -9:])['rmse'],
                       'increment_rmse_rad': summary(increments[..., idx])['rmse']})
    write_csv('groups.csv', groups)
    source_rows = []
    for source in sorted({w['source_recording_id'] for w in windows}):
        ix = [i for i, w in enumerate(windows) if w['source_recording_id'] == source]
        source_rows.append({'source_recording_id': source, 'windows': len(ix), **summary(e[ix]),
                            'increment_rmse_rad': summary(increments[ix])['rmse'],
                            'left_arm_last9_rmse': summary(e[ix, -9:, :7])['rmse']})
    write_csv('sources.csv', source_rows)
    cases = []
    for metric in ('rmse', 'increment_rmse_rad', 'left_arm_last9_rmse', 'max'):
        seen = set()
        for row in sorted(rows, key=lambda r: r[metric], reverse=True):
            source = row['source_recording_id']
            if source in seen:
                continue
            seen.add(source)
            record = records[source]
            videos = [key for key, value in record['info']['features'].items()
                      if value.get('dtype') == 'video']
            cases.append({**row, 'selection_metric': metric,
                          'export_time_seconds': row['current_frame'] / 30,
                          'video_paths': [str(Path(record['root']) / record['info']['video_path'].format(
                              episode_chunk=record['episode']//1000, episode_index=record['episode'], video_key=key))
                                          for key in videos],
                          'review_status': 'pending', 'task_result': 'unknown',
                          'operation_phase': 'unknown', 'physical_failure': 'unknown',
                          'time_scope': 'cleaned_export; raw clock and cut seams not verified here'})
            if len(seen) == 3:
                break
    (out / 'review_cases.json').write_text(json.dumps(cases, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    for ax, case in zip(axes.flat, cases):
        i = case['local_window']
        if case['selection_metric'] == 'increment_rmse_rad':
            joint = int(np.argmax(np.mean(increments[i]**2, axis=0)))
        elif case['selection_metric'] == 'left_arm_last9_rmse':
            joint = int(np.argmax(np.mean(e[i, -9:, :7]**2, axis=0)))
        else:
            joint = ORDER.index(case['worst_joint'])
        for data, label in ((a, 'GT'), (o, 'Oracle'), (p, 'Prediction')):
            ax.plot(np.arange(50)/30, data[i, :, joint], label=label)
        ax.set_title(f"{case['selection_metric']}\n{case['sample_id'][:10]} @ {case['current_frame']} / {ORDER[joint]}", fontsize=8)
        ax.set_xlabel('Future time (s)'); ax.set_ylabel('rad')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('REAL training-source diagnostics: selected high-error cases; no task-failure labels')
    fig.tight_layout(); fig.savefig(out / 'review_trajectories.png', dpi=160); plt.close(fig)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'data_kind': 'real', 'scope': 'training-source diagnosis only',
              'new_model_forwards': 0, 'new_training_updates': 0,
              'window_count': len(windows), 'source_count': len(source_rows),
              'total': summary(e), 'oracle': summary(o-a),
              'offset_mse': float(np.mean(b*b)), 'shape_mse': float(np.mean((e-b[:, None])**2)),
              'groups': groups, 'review_case_count': len(cases),
              'checkpoint_parameter_sha256': metadata['checkpoint_parameter_sha256'],
              'normalization_sha256': metadata['normalization_sha256'],
              'metadata_sha256': digest(export / 'metadata.json'),
              'aligned_sha256': digest(export / 'aligned.npz'),
              'training_manifest_sha256': digest(paths['training_manifest']),
              'config_sha256': digest(config_path), 'script_sha256': digest(Path(__file__)),
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'code_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())}
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('window_count', 'source_count', 'total', 'review_case_count')}, indent=2))


if __name__ == '__main__':
    main()
