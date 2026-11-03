#!/usr/bin/env python3
"""Opt-in comparison against read-only, pinned numerical source definitions.

No OpenPI import or source file copies. AST extraction executes ONLY explicitly
selected, reviewed class definitions, plus the NumPy-only protocol module.
Synthetic statistics exercise arithmetic; they do not certify real T3 assets.
"""
import argparse
import ast
import dataclasses
import importlib.util
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

PIN = '2ab68c8b8090284fca6d258f86bf5098d2f1302b'
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


def compare(root):
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PIN:
        raise ValueError('Expected audited upstream commit ' + PIN)
    files = ['src/openpi/policies/fr3_sharpa_protocol.py', 'src/openpi/policies/fr3_sharpa_multimodal_policy.py',
             'src/openpi/transforms.py', 'src/openpi/shared/delta_paper.py', 'src/openpi/training/delta_paper.py']
    if subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--', *files], text=True):
        raise ValueError('Audited upstream files have local changes')
    protocol_path, multi_path, transforms_path, paper_path, windows_path = [root / f for f in files]
    spec = importlib.util.spec_from_file_location('audited_protocol', protocol_path)
    protocol = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = protocol
    spec.loader.exec_module(protocol)
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
        cls = selected_class(transforms_path, name)
        obj = object.__new__(cls)
        fn = getattr(obj, '_unnormalize_quantile' if inv else '_normalize_quantile')
        np.testing.assert_allclose(normalize(x, stats, inverse=inv), fn(x, objstats), rtol=1e-6, atol=1e-6)
        if not inv:
            np.testing.assert_allclose(normalize(x, stats, method='zscore'), obj._normalize(x, objstats), rtol=1e-6, atol=1e-6)
    checks.extend(['Normalize_quantile', 'Unnormalize_quantile', 'Normalize_zscore'])
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
    return {'upstream_commit': commit, 'input': 'synthetic_unique_and_random_values',
            'statistics': 'synthetic_only', 'passed': checks,
            'not_verified': ['full_LeRobot_batch_and_padding', 'real_T3_asset_identity_and_statistics', 'deployment_runtime'],
            'tolerance': {'rtol': 1e-6, 'atol': 1e-6}}



def compare_real_episode(root, config_path):
    """Compare encoding on one actual local episode; no unmatched stats used."""
    from trc.data.lerobot_adapter import LeRobotAdapter
    from trc.data.action_schema import verify_features
    cfg = json.loads(config_path.read_text())
    dataset_path = (config_path.resolve().parent / cfg['dataset_root']).resolve()
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
    path = root / 'src/openpi/transforms.py'
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
    p.add_argument('--dataset-config', type=Path, help='Optional real-data config; no statistics invented')
    p.add_argument('--output', type=Path, default=Path('outputs/upstream_comparison.json'))
    a = p.parse_args()
    report = compare(a.source_root.resolve())
    if a.dataset_config:
        report['real_episode_encoding'] = compare_real_episode(a.source_root.resolve(), a.dataset_config)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
