#!/usr/bin/env python3
"""Opt-in comparison against read-only, pinned numerical source definitions.

No OpenPI import or source file copies. AST extraction executes ONLY explicitly
selected, reviewed class definitions, plus the NumPy-only protocol module.
Synthetic statistics exercise arithmetic; they do not certify real T3 assets.
"""
import argparse
import ast
import dataclasses
import types
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from trc.data.action_schema import ORDER, ARMS, canonicalize
from trc.data.chunks import continuous_segments, window_starts
from trc.data.normalization import normalize, arm_bridge
from trc.data.semantics import transform_actions

PINS = {
    'main': '4d0cb07e02181ed3bcdd876cad69446b22cfb075',
    'origin/main': '897cb917097c1d56a3fa7a517bf561b9e878036e',
    'delta-paper-training-test': '2ab68c8b8090284fca6d258f86bf5098d2f1302b',
    'github/feat/trainrtc-correlated-t3': 'bd2ee4077b236e9f64dbf135625269de8bf576e1',
}
PIN = PINS['delta-paper-training-test']


class SourceBlob:
    def __init__(self, root, commit, path):
        self.root, self.commit, self.path = root, commit, path

    def read_text(self):
        return subprocess.check_output(['git', '-C', str(self.root), 'show', f'{self.commit}:{self.path}'], text=True)

    def __str__(self):
        return f'{self.commit}:{self.path}'

sys.dont_write_bytecode = True


def selected_class(path, name, namespace=None):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.bases = []
    ns = {'np': np, 'dataclasses': dataclasses, 'dataclass': dataclasses.dataclass}
    ns.update(namespace or {})
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(path), 'exec'), ns)
    return ns[name]


def compare(root, ref='delta-paper-training-test'):
    if ref not in PINS:
        raise ValueError('Only audited references are allowed')
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', ref], text=True).strip()
    if commit != PINS[ref]:
        raise ValueError('Audited branch moved; review and update provenance before comparison')
    files = ['src/openpi/policies/fr3_sharpa_protocol.py', 'src/openpi/policies/fr3_sharpa_multimodal_policy.py',
             'src/openpi/transforms.py', 'src/openpi/shared/delta_paper.py', 'src/openpi/training/delta_paper.py']
    protocol_path, multi_path, transforms_path, paper_path, windows_path = [SourceBlob(root, commit, f) for f in files]
    protocol = types.ModuleType('audited_protocol')
    sys.modules[protocol.__name__] = protocol
    exec(compile(protocol_path.read_text(), str(protocol_path), 'exec'), protocol.__dict__)
    assert tuple(protocol.ORDER) == ORDER
    checks = ['protocol_joint_names']
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 58)).astype(np.float32)
    s = rng.normal(size=58).astype(np.float32)
    mask = np.zeros(58, bool)
    mask[ARMS] = True
    for name, inverse in [('DeltaActions', False), ('AbsoluteActions', True)]:
        cls = selected_class(transforms_path, name)
        expected = cls(mask)({'state': s.copy(), 'actions': x.copy()})['actions']
        np.testing.assert_allclose(transform_actions(x, s, inverse=inverse), expected, rtol=1e-6, atol=1e-6)
        checks.append(name)
    multi = selected_class(multi_path, 'Fr3SharpaMultimodalInputs', {
        'fr3_sharpa_policy': SimpleNamespace(Fr3SharpaInputs=lambda _: lambda value: value)})
    state116 = np.arange(116, dtype=np.float32)
    unique = np.arange(50 * 58, dtype=np.float32).reshape(50, 58)
    expected = multi(None)({'observation/state': state116, 'actions': unique})
    np.testing.assert_array_equal(canonicalize(state116, 'converter58', state=True), expected['observation/state'])
    np.testing.assert_array_equal(canonicalize(unique, 'converter58'), expected['actions'])
    checks.extend(['converter_action_order', 'converter_state_positions'])
    low = rng.uniform(-2, -1, (50, 58)).astype(np.float32)
    high = rng.uniform(1, 2, (50, 58)).astype(np.float32)
    mean = rng.normal(size=(50, 58)).astype(np.float32)
    std = rng.uniform(0, 1, (50, 58)).astype(np.float32)
    stats = {'q01': low, 'q99': high, 'mean': mean, 'std': std}
    objstats = SimpleNamespace(**stats)
    # Only numerical methods: do not instantiate full Normalize dependency tree.
    for name, inv in [('Normalize', False), ('Unnormalize', True)]:
        helper = next(n for n in ast.parse(transforms_path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'pad_to_dim')
        ns = {'np': np}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(transforms_path), 'exec'), ns)
        cls = selected_class(transforms_path, name, {'pad_to_dim': ns['pad_to_dim']})
        obj = object.__new__(cls)
        fn = getattr(obj, '_unnormalize_quantile' if inv else '_normalize_quantile')
        np.testing.assert_allclose(normalize(x, stats, inverse=inv), fn(x, objstats), rtol=1e-6, atol=1e-6)
        if inv:
            np.testing.assert_allclose(normalize(x, stats, method='zscore', inverse=True), obj._unnormalize(x, objstats), rtol=1e-6, atol=1e-6)
        if not inv:
            np.testing.assert_allclose(normalize(x, stats, method='zscore'), obj._normalize(x, objstats), rtol=1e-6, atol=1e-6)
    checks.extend(['Normalize_quantile', 'Unnormalize_quantile', 'Normalize_zscore', 'Unnormalize_zscore'])
    report = {'upstream_ref': ref, 'upstream_commit': commit, 'input': 'synthetic_unique_and_random_values',
              'statistics': 'synthetic_only', 'passed': checks,
              'not_verified': ['full_LeRobot_batch_and_padding', 'real_T3_asset_identity_and_statistics', 'deployment_runtime'],
              'tolerance': {'rtol': 1e-6, 'atol': 1e-6}}
    if ref in ('main', 'origin/main'):
        report['T3'] = 'not_present_in_this_branch'
        return report
    # Read upstream ARMS assignment as well; do not assume it matches ours.
    tree = ast.parse(paper_path.read_text())
    assign = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ARMS' for t in n.targets))
    ns = {'np': np}
    exec(compile(ast.Module(body=[assign], type_ignores=[]), str(paper_path), 'exec'), ns)
    np.testing.assert_array_equal(ns['ARMS'], ARMS)
    bridge = selected_class(paper_path, 'ArmNormalization', {'ARMS': ns['ARMS'], 'MODES': ('timestamp_zscore_arms',)})
    kwargs = dict(q01=low, q99=high, arm_mean=mean[:, ARMS], arm_std=std[:, ARMS], std_floor=1e-3)
    for inv in (False, True):
        original = bridge(mode='timestamp_zscore_arms', bundle_sha256='synthetic-only', inverse=inv, **kwargs)
        np.testing.assert_allclose(arm_bridge(x, inverse=inv, **kwargs), original({'actions': x.copy()})['actions'], rtol=1e-6, atol=1e-6)
    checks.append('T3_bridge_both_directions')
    ids = np.repeat([0, 1, 2], [51, 49, 55])
    frames = np.concatenate([np.arange(n) for n in (51, 49, 55)])
    class View:
        def select_columns(self, keys): return self
        def with_format(self, mode): return self
        def __getitem__(self, index): return {'episode_index': ids, 'frame_index': frames}
        def select(self, indices): return self
    class Dataset:
        root = None
        meta = None
        hf_dataset = View()
        def __len__(self): return len(ids)
    original = selected_class(windows_path, 'CompleteWindows', {'paper': SimpleNamespace(VERSION='synthetic')})(Dataset(), 50)
    segments = [(0, 51), (51, 100), (100, 155)]
    np.testing.assert_array_equal(original.indices, window_starts(segments))
    checks.append('complete_window_indices_at_episode_boundaries')
    return report



def compare_synthetic_bundle(root):
    """Validate the standalone loader against original load_bundle using fake test assets."""
    import hashlib
    import tempfile
    from trc.data.assets import load_assets
    from trc.io import file_sha256
    source = SourceBlob(root, PIN, 'src/openpi/shared/delta_paper.py')
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ('sha256', 'load_bundle')]
    ns = dict(np=np, hashlib=hashlib, json=json, Path=Path, VERSION='sharpa-delta-paper-v1',
              MANIFEST='delta_paper.json', BUNDLE='delta_paper.npz')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    checks = []
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        stats = {}
        for key, shape in [('state', (58,)), ('actions', (50, 58))]:
            stats[key] = {name: np.full(shape, value).tolist() for name, value in
                          [('mean', 0.), ('std', 1.), ('q01', -2.), ('q99', 2.)]}
        (directory/'norm_stats.json').write_text(json.dumps({'norm_stats': stats}))
        np.savez(directory/'delta_paper.npz', arm_mean=np.zeros((50, 14)), arm_std=np.ones((50, 14)), noise_cholesky=np.empty((0, 0)))
        contract = {'action_dim': 58, 'action_horizon': 50, 'datasets': [], 'vector_transforms': {'synthetic': True}}
        norm_hash = file_sha256(directory/'norm_stats.json')
        metadata = {'version': 'sharpa-delta-paper-v1', 'mode': 'timestamp_zscore_arms', 'beta': 0.,
                    'shape': [50, 58], 'std_floor': 1e-4, 'flatten_order': 'time-major', 'windows': 'complete_only',
                    'training_windows': 2, 'source_contract': contract, 'norm_stats_sha256': norm_hash,
                    'bundle_sha256': file_sha256(directory/'delta_paper.npz')}
        (directory/'delta_paper.json').write_text(json.dumps(metadata))
        (directory/'norm_metadata.json').write_text(json.dumps({'schema_version': 2, 'action_horizon': 50,
                                                               'norm_stats_sha256': norm_hash, 'contract': contract}))
        options = dict(mode='timestamp_zscore_arms', beta=0., horizon=50, std_floor=1e-4)
        expected, arrays = ns['load_bundle'](directory, **options)
        actual = load_assets(directory, **options)
        assert expected == actual.metadata
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, actual.arrays[key])
        checks.append('valid_metadata_and_arrays_exact')
        for key, value in [('mode', 'global_quantile'), ('beta', .5), ('horizon', 49), ('std_floor', .01)]:
            for fn in (ns['load_bundle'], load_assets):
                try:
                    fn(directory, **{**options, key: value})
                except ValueError:
                    pass
                else:
                    raise AssertionError('Expected mismatch rejection')
            checks.append(key + '_mismatch_both_reject')
        with (directory/'delta_paper.npz').open('ab') as stream:
            stream.write(b'corrupt')
        for fn in (ns['load_bundle'], load_assets):
            try:
                fn(directory, **options)
            except ValueError:
                pass
            else:
                raise AssertionError('Expected digest rejection')
        checks.append('corruption_both_reject')
    return {'source_commit': PIN, 'fixture': 'synthetic_only', 'passed': checks}


def compare_real_episode(root, config_path):
    """Compare encoding on one actual local episode; no unmatched stats used."""
    from trc.data.lerobot_adapter import LeRobotAdapter
    from trc.data.action_schema import verify_features
    from trc.config import read_config, dataset_specs
    cfg = read_config(config_path)
    spec = dataset_specs(cfg)[0]
    cfg['layout'] = spec['layout']
    dataset_path = Path(spec['root'])
    dataset = LeRobotAdapter(dataset_path)
    verify_features(dataset.info['features'], cfg['layout'])
    ep = dataset.episodes[0]['episode_index']
    data = dataset.read_episode(ep)
    actions = canonicalize(data['action'], cfg['layout'])
    states = canonicalize(data['observation.state'], cfg['layout'], state=True)
    segments = continuous_segments(data['timestamp'], data['frame_index'], dataset.info['fps'],
                                   tolerance_s=cfg.get('tolerance_s', 1e-4),
                                   source_timestamps_ns=data.get('source_timestamp_ns'))
    horizon = cfg.get('horizon', 50)
    starts = window_starts(segments, horizon)
    if not starts:
        raise ValueError('No complete real window available')
    mask = np.zeros(58, bool)
    mask[ARMS] = True
    path = SourceBlob(root, PIN, 'src/openpi/transforms.py')
    delta = selected_class(path, 'DeltaActions')(mask)
    absolute = selected_class(path, 'AbsoluteActions')(mask)
    checked = sorted(set([starts[0], starts[len(starts)//2], starts[-1]]))
    for start in checked:
        x, anchor = actions[start:start+horizon], states[start]
        target = delta({'state': anchor.copy(), 'actions': x.copy()})['actions']
        np.testing.assert_allclose(transform_actions(x, anchor), target, rtol=1e-6, atol=1e-6)
        physical = absolute({'state': anchor.copy(), 'actions': target.copy()})['actions']
        np.testing.assert_allclose(transform_actions(target, anchor, inverse=True), physical, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(physical, x, rtol=1e-6, atol=1e-6)
    return {'episode_index': ep, 'frames': len(actions), 'layout': cfg['layout'],
            'checked_window_starts': checked, 'passed': ['DeltaActions', 'AbsoluteActions', 'physical_roundtrip'],
            'normalization': 'not_compared_missing_matching_T3_bundle',
            'loader_timing': 'numeric_windows_only_full_training_loader_not_verified'}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--refs', nargs='+', choices=tuple(PINS), default=list(PINS))
    p.add_argument('--check-assets', action='store_true', help='Compare loaders on temporary synthetic bundles')
    p.add_argument('--dataset-config', type=Path, help='Optional real-data config; no statistics invented')
    p.add_argument('--output', type=Path, default=Path('outputs/upstream_comparison.json'))
    a = p.parse_args()
    report = {'status': 'passed', 'branches': [compare(a.source_root.resolve(), ref) for ref in a.refs]}
    if a.check_assets:
        report['synthetic_bundle_comparison'] = compare_synthetic_bundle(a.source_root.resolve())
    if a.dataset_config:
        report['real_episode_encoding'] = compare_real_episode(a.source_root.resolve(), a.dataset_config)
    from trc.io import write_report
    protected = [a.source_root]
    if a.dataset_config:
        from trc.config import read_config, protected_roots
        protected += protected_roots(read_config(a.dataset_config))
    write_report(a.output, report, protected=protected)
    print(json.dumps(report, indent=2))
