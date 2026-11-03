"""Standalone T3 assets/provenance validation adapted from tomato-openpi.

Source: shared/delta_paper.py and training/norm_provenance.py at 2ab68c8.
Apache-2.0; see third_party/openpi. No OpenPI or training runtime imports.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from trc.io import file_sha256
from .normalization import load_stats

MODES = ('global_quantile', 'global_zscore_arms', 'timestamp_zscore_arms')


@dataclass(frozen=True)
class NormalizationAssets:
    metadata: dict
    stats: dict
    arrays: dict
    norm_sha256: str
    bundle_sha256: str
    provenance_status: str


def validate_dataset_binding(contract, sources):
    """Match upstream contract's ordered datasets by repo ID and vector file hashes.

    Source contract carries OpenPI transform fingerprints. Their exact expected
    identity must additionally be pinned by expected_contract for full verification.
    """
    if contract.get('action_sequence_keys') != ['action']:
        raise ValueError('Normalization action field contract mismatch')
    if any(contract.get('action_hz') != s['info']['fps'] for s in sources):
        raise ValueError('Offline windows do not resample; normalization action_hz must match source FPS')
    datasets = contract.get('datasets')
    if not isinstance(datasets, list) or len(datasets) != len(sources):
        raise ValueError('Normalization source dataset count mismatch')
    for recorded, source in zip(datasets, sources):
        if not isinstance(source.get('repo_id'), str) or not source['repo_id']:
            raise ValueError('Explicit real dataset repo_id required for normalization binding')
        info = source['info']
        for key, actual in [('repo_id', source['repo_id']), ('frames', info['total_frames']),
                            ('episodes', info['total_episodes']), ('fps', info['fps'])]:
            if recorded.get(key) != actual:
                raise ValueError(f'Normalization dataset {key} mismatch')
        if recorded.get('files') != source['files']:
            raise ValueError('Normalization vector file fingerprints mismatch')


def load_assets(directory, *, mode, beta, horizon, std_floor, expected_bundle_sha256=None,
                expected_norm_sha256=None, expected_contract=None, sources=None):
    if mode not in MODES or type(horizon) is not int or horizon <= 0:
        raise ValueError('Explicit valid normalization mode and horizon required')
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not np.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError('beta must be finite in [0,1)')
    if isinstance(std_floor, bool) or not isinstance(std_floor, (int, float)) or not np.isfinite(std_floor) or std_floor <= 0:
        raise ValueError('Positive finite std_floor required')
    root = Path(directory)
    manifest = json.loads((root / 'delta_paper.json').read_text())
    if not isinstance(manifest, dict):
        raise ValueError('Paper manifest must be an object')
    expected = {'version': 'sharpa-delta-paper-v1', 'mode': mode, 'beta': beta,
                'shape': [horizon, 58], 'std_floor': std_floor,
                'flatten_order': 'time-major', 'windows': 'complete_only'}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'Paper asset {key} mismatch')
    bundle_hash, norm_hash = file_sha256(root / 'delta_paper.npz'), file_sha256(root / 'norm_stats.json')
    if manifest.get('bundle_sha256') != bundle_hash or (expected_bundle_sha256 and bundle_hash != expected_bundle_sha256):
        raise ValueError('Paper bundle digest mismatch')
    if manifest.get('norm_stats_sha256') != norm_hash or (expected_norm_sha256 and norm_hash != expected_norm_sha256):
        raise ValueError('Normalization digest mismatch')
    audit = json.loads((root / 'norm_metadata.json').read_text())
    if not isinstance(audit, dict):
        raise ValueError('Normalization provenance must be an object')
    contract = manifest.get('source_contract')
    if not isinstance(contract, dict) or audit.get('contract') != contract or audit.get('norm_stats_sha256') != norm_hash:
        raise ValueError('Bundle and normalization provenance mismatch')
    if audit.get('schema_version') != 2 or audit.get('action_horizon') != horizon:
        raise ValueError('Unsupported normalization provenance schema or horizon')
    if contract.get('action_dim') != 58 or contract.get('action_horizon') != horizon:
        raise ValueError('Source contract dimensions mismatch')
    paper_settings = contract.get('vector_transforms', {}).get('delta_paper') if isinstance(contract.get('vector_transforms'), dict) else None
    if paper_settings is not None:
        for key in ('version', 'mode', 'beta', 'std_floor', 'windows'):
            if not isinstance(paper_settings, dict) or paper_settings.get(key) != expected[key]:
                raise ValueError('Source transform paper settings mismatch')
    if not isinstance(contract.get('vector_transforms'), dict) or not contract['vector_transforms']:
        raise ValueError('Missing source vector transform identity')
    if expected_contract is not None and contract != expected_contract:
        raise ValueError('Expected source/transform contract mismatch')
    if sources is not None:
        validate_dataset_binding(contract, sources)
    status = 'verified' if expected_contract is not None and sources is not None else (
        'dataset_bound_transform_unpinned' if sources is not None else 'integrity_only')
    with np.load(root / 'delta_paper.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ('arm_mean', 'arm_std', 'noise_cholesky')}
    for key in ('arm_mean', 'arm_std'):
        if arrays[key].shape != (horizon, 14) or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid {key}')
    if np.any(arrays['arm_std'] < 0):
        raise ValueError('Negative arm standard deviation')
    chol = arrays['noise_cholesky']
    if chol.shape != ((horizon * 58, horizon * 58) if beta else (0, 0)) or not np.isfinite(chol).all():
        raise ValueError('Invalid noise factor shape/values')
    if beta and (np.any(np.diag(chol) <= 0) or not np.allclose(chol, np.tril(chol), atol=1e-7)):
        raise ValueError('Noise factor must be lower triangular with positive diagonal')
    if type(manifest.get('training_windows')) is not int or manifest['training_windows'] < 2:
        raise ValueError('At least two training windows required')
    stats = load_stats(root / 'norm_stats.json')
    for key, shape in [('state', (58,)), ('actions', (horizon, 58))]:
        for field in ('mean', 'std', 'q01', 'q99'):
            value = stats.get(key, {}).get(field)
            if value is None or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid {key}.{field} shape/values')
        if np.any(stats[key]['std'] < 0) or np.any(stats[key]['q99'] < stats[key]['q01']):
            raise ValueError(f'Invalid {key} normalization range')
    return NormalizationAssets(manifest, stats, arrays, norm_hash, bundle_hash, status)
