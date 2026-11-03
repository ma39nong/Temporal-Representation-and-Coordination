"""High-level P0 project reports, shared by the CLI and Python callers."""
import json
from pathlib import Path
import numpy as np
from trc.config import dataset_specs
from trc.io import object_sha256
from trc.data.assets import load_assets
from trc.data.manifest import build_manifest
from trc.data.pipeline import ActionCodec
from trc.data.validation import audit_dataset
from trc.data.windows import iter_windows


def project_assets(cfg, manifest=None):
    if not cfg.get('normalization'):
        raise ValueError('Matching normalization assets are not configured')
    options = dict(cfg['normalization'])
    directory = options.pop('asset_dir')
    contract_path = options.pop('expected_contract_path', None)
    contract = json.loads(Path(contract_path).read_text()) if contract_path else None
    for key in ('mode', 'beta', 'std_floor'):
        if key not in options:
            raise ValueError(f'normalization.{key} required; no inferred experiment settings')
    return load_assets(directory, horizon=cfg.get('horizon', 50), expected_contract=contract,
                       sources=manifest['sources'] if manifest else None, **options)


def asset_report(assets):
    return {'status': assets.provenance_status, 'norm_sha256': assets.norm_sha256,
            'bundle_sha256': assets.bundle_sha256, 'mode': assets.metadata['mode'],
            'shape': assets.metadata['shape'], 'beta': assets.metadata['beta'],
            'std_floor': assets.metadata['std_floor']}


def audit_project(cfg):
    specs = dataset_specs(cfg)
    if not specs:
        return audit_dataset(cfg)
    reports = []
    for spec in specs:
        report = audit_dataset({**cfg, 'dataset_root': spec['root'], 'layout': spec['layout']})
        reports.append({'dataset_id': spec['id'], 'report': report})
    result = {'report_version': 2, 'datasets': reports, 'errors': [], 'warnings': [],
              'real_data_read': any(r['report']['real_data_read'] for r in reports),
              'real_data_audit_completed': all(r['report']['real_data_audit_completed'] for r in reports),
              'observed_frames': sum(r['report'].get('observed_frames', 0) for r in reports),
              'observed_episodes': sum(r['report'].get('observed_episodes', 0) for r in reports)}
    manifest = None
    try:
        manifest = build_manifest(cfg)
        result['data_sha256'] = manifest['data_sha256']
        result['source_provenance_complete'] = manifest['source_provenance_complete']
        if not manifest['source_provenance_complete']:
            result['warnings'].append('Original recording provenance missing; formal recording split blocked')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        result['errors'].append('manifest:' + str(exc))
    result['normalization'] = {'status': 'unavailable'}
    if cfg.get('normalization'):
        try:
            assets = project_assets(cfg, manifest)
            result['normalization'] = asset_report(assets)
            if assets.provenance_status != 'verified':
                result['warnings'].append('Normalization transform contract is not fully verified')
        except (OSError, KeyError, TypeError, ValueError) as exc:
            result['errors'].append('normalization:' + str(exc))
    # Count windows with the exact same validity/time rules as the public iterator.
    if not result['errors'] and all(r['report']['status'] == 'passed_with_warnings' for r in reports):
        try:
            result['eligible_windows'] = sum(1 for _ in iter_windows(cfg, manifest=manifest))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            result['errors'].append('windows:' + str(exc))
    result['vector_contract_ready'] = bool(manifest and manifest['source_provenance_complete'] and
                                       result['normalization']['status'] == 'verified' and
                                       all(r['report']['status'] == 'passed_with_warnings' for r in reports) and
                                       result.get('eligible_windows', 0) > 0 and not result['errors'])
    result['formal_data_ready'] = False
    result['not_verified'] = ['RGB_video_sync', 'full_training_batch_equivalence', 'deployment_runtime']
    result['status'] = 'failed' if result['errors'] or any(
        r['report']['status'] != 'passed_with_warnings' for r in reports) else 'passed_with_warnings'
    result['config_sha256'] = object_sha256(cfg)
    return result


def check_roundtrip(cfg, *, max_windows=3):
    if type(max_windows) is not int or max_windows <= 0:
        raise ValueError('max_windows must be a positive integer')
    manifest = build_manifest(cfg)
    assets = project_assets(cfg, manifest)
    specs = {s['id']: s for s in dataset_specs(cfg)}
    samples = []
    for window in iter_windows(cfg, manifest=manifest):
        codec = ActionCodec(assets, specs[window.dataset_id]['layout'])
        encoded = codec.encode(window.storage_actions, window.storage_anchor)
        decoded = codec.decode(encoded['actions'], encoded['anchor_state'])
        np.testing.assert_allclose(decoded, window.physical_actions, rtol=1e-5, atol=1e-5)
        samples.append({'dataset_id': window.dataset_id, 'episode_index': window.episode_index,
                        'frame_index': window.frame_index, 'max_abs_error_rad': float(np.max(np.abs(decoded-window.physical_actions)))})
        if len(samples) >= max_windows:
            break
    if not samples:
        raise ValueError('No eligible full windows')
    return {'status': 'passed', 'data_sha256': manifest['data_sha256'], 'assets': asset_report(assets),
            'samples': samples, 'rtol': 1e-5, 'atol': 1e-5,
            'scope': 'offline_roundtrip_only_not_deployment_or_full_training_batch'}
