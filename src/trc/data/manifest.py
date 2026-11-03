"""Ordered datasets, content identity and recording-level split without leakage."""
import json
from pathlib import Path
from trc.config import dataset_specs
from trc.io import file_sha256, object_sha256
from .action_schema import verify_features
from .lerobot_adapter import LeRobotAdapter


def fingerprint_source(spec):
    adapter = LeRobotAdapter(spec['root'])
    verify_features(adapter.info['features'], spec['layout'])
    paths = [adapter.root / 'meta' / name for name in ('info.json', 'episodes.jsonl', 'tasks.jsonl')]
    parquet = sorted((adapter.root / 'data').rglob('*.parquet'))
    if not parquet:
        raise ValueError('Dataset contains no parquet files')
    # Matches upstream norm_provenance.build_contract vector file selection.
    files = {p.relative_to(adapter.root).as_posix(): file_sha256(p) for p in [*paths, *parquet]}
    return {'id': spec['id'], 'repo_id': spec['repo_id'], 'root': str(adapter.root),
            'layout': spec['layout'], 'info': adapter.info, 'files': files,
            'content_sha256': object_sha256(files)}


def build_manifest(cfg):
    specs = dataset_specs(cfg)
    if not specs:
        raise ValueError('No real datasets configured')
    ids = [s['id'] for s in specs]
    roots = [Path(s['root']).resolve() for s in specs]
    if len(set(ids)) != len(ids) or len(set(roots)) != len(roots):
        raise ValueError('Dataset IDs and resolved roots must be unique')
    sources = [fingerprint_source(s) for s in specs]
    if len({s['content_sha256'] for s in sources}) != len(sources):
        raise ValueError('Duplicate dataset content under different roots')
    if len({s['info']['fps'] for s in sources}) != 1:
        raise ValueError('Collection FPS mismatch')
    camera_reference = None
    for s in sources:
        cameras = {k: v for k, v in s['info']['features'].items() if k.startswith('observation.images.')}
        if camera_reference is None:
            camera_reference = cameras
        if cameras != camera_reference:
            raise ValueError('Collection camera contracts differ')
    mapping = {}
    if cfg.get('source_mapping_path'):
        mapping = json.loads(Path(cfg['source_mapping_path']).read_text())
        if not isinstance(mapping, dict):
            raise ValueError('Source mapping must be a dataset-ID keyed object')
    if set(mapping) - set(ids):
        raise ValueError('Source mapping includes unknown dataset IDs')
    used_mappings, records = set(), []
    for source in sources:
        adapter = LeRobotAdapter(source['root'])
        source_map = mapping.get(source['id'], {})
        if not isinstance(source_map, dict):
            raise ValueError('Each dataset mapping must be episode-ID keyed')
        for row in sorted(adapter.episodes, key=lambda r: r['episode_index']):
            ep = row['episode_index']
            key = str(ep)
            extra = source_map.get(key, {})
            if not isinstance(extra, dict) or set(extra) - {'source_recording_id', 'segment_id'}:
                raise ValueError('Mapping allows only source_recording_id and segment_id')
            if key in source_map:
                used_mappings.add((source['id'], key))
            for k, value in extra.items():
                if k in row and value != row[k]:
                    raise ValueError('Mapping conflicts with original episode provenance')
                if not isinstance(value, str) or not value:
                    raise ValueError('Provenance IDs must be nonempty strings')
            provenance = {**row, **extra}
            if provenance.get('source_recording_id') is not None and (not isinstance(provenance['source_recording_id'], str) or not provenance['source_recording_id']):
                raise ValueError('source_recording_id must be a nonempty string')
            records.append({'id': f"{source['id']}:{ep}", 'global_episode_index': len(records),
                            'dataset_id': source['id'], 'episode_index': ep, 'frames': row['length'],
                            'source_recording_id': provenance.get('source_recording_id'),
                            'segment_id': provenance.get('segment_id')})
    provided = {(sid, key) for sid, rows in mapping.items() for key in rows}
    if provided != used_mappings:
        raise ValueError('Source mapping includes unknown dataset/episode IDs')
    identity = {'datasets': [{k: s[k] for k in ('id', 'repo_id', 'layout', 'content_sha256')} for s in sources],
                'records': records, 'horizon': cfg.get('horizon', 50),
                'tolerance_s': cfg.get('tolerance_s', 1e-4), 'validity_column': cfg.get('validity_column')}
    report = {'schema_version': 1, 'sources': sources, 'records': records,
              'data_identity': identity, 'data_sha256': object_sha256(identity),
              'source_mapping_sha256': file_sha256(cfg['source_mapping_path']) if cfg.get('source_mapping_path') else None,
              'source_provenance_complete': all(bool(r['source_recording_id']) for r in records)}
    report['manifest_sha256'] = object_sha256(report)
    return report


def recording_split(manifest, *, seed=42, fractions=None):
    fractions = fractions or {'train': .8, 'validation': .2, 'test': 0.}
    import math
    if set(fractions) != {'train', 'validation', 'test'} or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in fractions.values()
    ) or not math.isclose(sum(fractions.values()), 1):
        raise ValueError('Split fractions must be nonnegative and sum to one')
    if type(seed) is not int:
        raise ValueError('Integer split seed required')
    if not manifest['source_provenance_complete']:
        raise ValueError('Recording provenance is missing; refusing episode-level leakage fallback')
    groups = sorted({r['source_recording_id'] for r in manifest['records']},
                    key=lambda s: object_sha256({'seed': seed, 'source_recording_id': s}))
    active = [k for k, v in fractions.items() if v > 0]
    if len(groups) < len(active):
        raise ValueError('Not enough original recordings for requested nonempty splits')
    # Reserve one group for each requested split, then assign remaining proportionally.
    counts = {k: (1 if k in active else 0) for k in fractions}
    remaining = len(groups) - len(active)
    for _ in range(remaining):
        key = max(active, key=lambda k: fractions[k] * len(groups) - counts[k])
        counts[key] += 1
    assignments, start = {}, 0
    for label in fractions:
        for group in groups[start:start+counts[label]]:
            assignments[group] = label
        start += counts[label]
    result = {'schema_version': 1, 'data_sha256': manifest['data_sha256'], 'seed': seed,
              'fractions': fractions, 'recording_counts': counts, 'recording_assignments': assignments,
              'episode_assignments': {r['id']: assignments[r['source_recording_id']] for r in manifest['records']}}
    result['sha256'] = object_sha256(result)
    return result
