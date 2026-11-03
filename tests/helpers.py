"""Explicit synthetic fixtures only, always generated in temporary directories."""
import json
from pathlib import Path
import numpy as np
from trc.data.action_schema import expected_names
from trc.io import file_sha256


def dataset(root, *, layout='canonical58', episodes=2, frames=55):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(root)
    (root / 'meta').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    rng = np.random.default_rng(8)
    info = {'codebase_version': 'v2.1', 'fps': 30, 'total_episodes': episodes, 'total_frames': episodes * frames,
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'features': {k: {'shape': [len(expected_names(layout, state))], 'dtype': 'float32',
                             'names': list(expected_names(layout, state))}
                         for k, state in [('action', False), ('observation.state', True)]}}
    (root / 'meta/info.json').write_text(json.dumps(info))
    rows = [{'episode_index': i, 'length': frames, 'source_recording_id': f'synthetic-recording-{i}'} for i in range(episodes)]
    (root / 'meta/episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    (root / 'meta/tasks.jsonl').write_text(json.dumps({'task_index': 0, 'task': 'SYNTHETIC TEST ONLY'}))
    for i in range(episodes):
        state_dim = len(expected_names(layout, True))
        table = pa.table({'action': pa.array(rng.normal(size=(frames, 58)).astype(np.float32).tolist(), type=pa.list_(pa.float32(), 58)),
                          'observation.state': pa.array(rng.normal(size=(frames, state_dim)).astype(np.float32).tolist(), type=pa.list_(pa.float32(), state_dim)),
                          'timestamp': pa.array(np.arange(frames, dtype=float) / 30),
                          'frame_index': pa.array(np.arange(frames)), 'episode_index': pa.array(np.full(frames, i)),
                          'valid': pa.array(np.ones(frames, bool))})
        pq.write_table(table, root / f'data/chunk-000/episode_{i:06d}.parquet')
    return root


def assets(root, *, horizon=50, mode='timestamp_zscore_arms', sources=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    stats = {}
    for key, shape in [('state', (58,)), ('actions', (horizon, 58))]:
        stats[key] = {'mean': np.zeros(shape).tolist(), 'std': np.ones(shape).tolist(),
                      'q01': np.full(shape, -4.).tolist(), 'q99': np.full(shape, 4.).tolist()}
    (root / 'norm_stats.json').write_text(json.dumps({'norm_stats': stats}))
    np.savez(root / 'delta_paper.npz', arm_mean=rng.normal(size=(horizon, 14)),
             arm_std=rng.uniform(0, 1, (horizon, 14)), noise_cholesky=np.empty((0, 0)))
    contract = {'action_dim': 58, 'action_horizon': horizon, 'action_hz': 30, 'action_sequence_keys': ['action'],
                'vector_transforms': {'synthetic': 'explicit-test-transform-identity'},
                'datasets': [{'repo_id': s['repo_id'], 'frames': s['info']['total_frames'],
                              'episodes': s['info']['total_episodes'], 'fps': s['info']['fps'], 'files': s['files']}
                             for s in (sources or [])]}
    norm_hash = file_sha256(root / 'norm_stats.json')
    manifest = {'version': 'sharpa-delta-paper-v1', 'mode': mode, 'beta': 0., 'std_floor': 1e-4,
                'shape': [horizon, 58], 'flatten_order': 'time-major', 'windows': 'complete_only',
                'training_windows': 12, 'source_contract': contract,
                'bundle_sha256': file_sha256(root / 'delta_paper.npz'), 'norm_stats_sha256': norm_hash}
    (root / 'delta_paper.json').write_text(json.dumps(manifest))
    (root / 'norm_metadata.json').write_text(json.dumps({'schema_version': 2, 'action_horizon': horizon,
                                                       'norm_stats_sha256': norm_hash, 'contract': contract}))
    (root / 'expected_contract.json').write_text(json.dumps(contract))
    return contract
