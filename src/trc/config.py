"""Strict portable configuration (JSON, a subset of YAML 1.2)."""
import json
import math
from pathlib import Path


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def read_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be an object')
    allowed = {'dataset_root', 'dataset_id', 'repo_id', 'layout', 'horizon', 'tolerance_s', 'norm_stats_path',
               'datasets', 'normalization', 'source_mapping_path', 'validity_column'}
    if set(cfg) - allowed:
        raise ValueError(f'Unknown configuration keys: {sorted(set(cfg) - allowed)}')
    positive_int(cfg.get('horizon', 50), 'horizon')
    tolerance = cfg.get('tolerance_s', 1e-4)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance_s must be finite and nonnegative')
    def resolve(value):
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Paths must be nonempty strings or null')
        return str((path.parent / Path(value).expanduser()).resolve())
    for key in ('dataset_root', 'norm_stats_path', 'source_mapping_path'):
        if key in cfg:
            cfg[key] = resolve(cfg[key])
    for key in ('dataset_id', 'repo_id'):
        if key in cfg and (not isinstance(cfg[key], str) or not cfg[key]):
            raise ValueError(f'{key} must be a nonempty string')
    if cfg.get('dataset_root') and cfg.get('datasets'):
        raise ValueError('Choose dataset_root or datasets, not both')
    datasets = cfg.get('datasets', [])
    if not isinstance(datasets, list):
        raise ValueError('datasets must be a list')
    for spec in datasets:
        if not isinstance(spec, dict) or set(spec) - {'id', 'repo_id', 'root', 'layout'}:
            raise ValueError('Invalid dataset specification')
        if not all(isinstance(spec.get(k), str) and spec[k] for k in ('id', 'repo_id', 'root', 'layout')):
            raise ValueError('Each dataset requires id, repo_id, root and layout')
        spec['root'] = resolve(spec['root'])
        if spec['layout'] not in ('canonical58', 'converter58'):
            raise ValueError('Unknown dataset layout')
    if cfg.get('layout', 'canonical58') not in ('canonical58', 'converter58'):
        raise ValueError('Unknown layout')
    if cfg.get('validity_column') is not None and (not isinstance(cfg['validity_column'], str) or not cfg['validity_column']):
        raise ValueError('validity_column must be a nonempty field name')
    norm = cfg.get('normalization')
    if norm is not None:
        allowed_norm = {'asset_dir', 'mode', 'beta', 'std_floor', 'expected_bundle_sha256', 'expected_norm_sha256', 'expected_contract_path'}
        if not isinstance(norm, dict) or set(norm) - allowed_norm:
            raise ValueError('Invalid normalization configuration')
        for key in ('asset_dir', 'expected_contract_path'):
            if key in norm:
                norm[key] = resolve(norm[key])
        if not norm.get('asset_dir'):
            raise ValueError('normalization.asset_dir required')
    return cfg


def dataset_specs(cfg):
    if cfg.get('datasets'):
        return cfg['datasets']
    if cfg.get('dataset_root'):
        return [{'id': cfg.get('dataset_id', 'dataset'), 'repo_id': cfg.get('repo_id'), 'root': cfg['dataset_root'], 'layout': cfg.get('layout', 'canonical58')}]
    return []


def protected_roots(cfg):
    roots = [Path(s['root']) for s in dataset_specs(cfg)]
    for value in (cfg.get('norm_stats_path'), cfg.get('source_mapping_path')):
        if value:
            roots.append(Path(value))
    if cfg.get('normalization'):
        roots.append(Path(cfg['normalization']['asset_dir']))
        if cfg['normalization'].get('expected_contract_path'):
            roots.append(Path(cfg['normalization']['expected_contract_path']))
    return roots
