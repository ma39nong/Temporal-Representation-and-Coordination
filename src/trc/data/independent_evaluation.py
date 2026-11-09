"""Small reviewed-recording gate on the existing LeRobot reader.

Human review is evidence, not an automatic proof of task success/independence.
No model selection or prediction-dependent window selection happens here.
"""
import numpy as np
from trc.io import object_sha256, file_sha256
from .lerobot_adapter import LeRobotAdapter
from .action_schema import canonicalize
from .validation import audit_episode
from trc.config import dataset_specs


def evidence_files(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Review requires evidence files with pinned hashes')
    for row in rows:
        if file_sha256(row['path']) != row['sha256']:
            raise ValueError('Review evidence hash mismatch')


def validate_reviews(manifest, reviews, exclusion):
    if not manifest['source_provenance_complete']:
        raise ValueError('Original recording provenance required')
    if not exclusion.get('complete_training_and_diagnostic_registry'):
        raise ValueError('Full training and previously analysed source registry required')
    evidence_files(exclusion['evidence_files'])
    forbidden_ids = set(exclusion['source_recording_ids'])
    forbidden_hashes = set(exclusion['original_recording_sha256'])
    registry_hash = object_sha256(exclusion)
    sources = {r['source_recording_id'] for r in manifest['records']}
    if len(reviews) != len(sources) or {r['source_recording_id'] for r in reviews} != sources:
        raise ValueError('Exactly one review per original recording required')
    seen_hashes = set()
    for row in reviews:
        digest = row['original_recording_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Raw recording SHA256 required')
        lineage = set(row['ancestor_recording_ids']) | {row['source_recording_id']}
        if lineage & forbidden_ids or digest in forbidden_hashes or digest in seen_hashes:
            raise ValueError('Training/diagnostic or duplicate recording overlap')
        seen_hashes.add(digest)
        if row['checked_exclusion_sha256'] != registry_hash or row['lineage_verified'] is not True:
            raise ValueError('Reviewer must check recording lineage against current registry')
        if row['quality_passed'] is not True or row['complete_task'] is not True:
            raise ValueError('Only reviewed complete, numerically valid tasks may be locked')
        if row['task_outcome'] not in ('success', 'failure'):
            raise ValueError('Unverified success cannot enter independent success evaluation')
        if not row['initial_position_description'] or not row['truss_configuration_description']:
            raise ValueError('Initial-position and truss configuration annotations required')
        evidence_files(row['evidence_files'])
    return registry_hash


def iter_numeric_candidates(collection, manifest):
    """Existing audit enforces 30Hz/58D/segments; require full [-7,51] support.

    Returns physical numeric inputs only. RGB loading remains in the existing
    pinned NativeDataset inference runtime, subject to camera/preprocessing review.
    """
    records = {r['id']: r for r in manifest['records']}
    for spec in dataset_specs(collection):
        adapter = LeRobotAdapter(spec['root'])
        if adapter.info['fps'] != 30: raise ValueError('FLASH requires verified 30Hz data')
        for ep in adapter.episodes:
            key = f"{spec['id']}:{ep['episode_index']}"
            validity = collection.get('validity_column')
            d = adapter.read_episode(ep['episode_index'], extra_columns=[validity] if validity else [])
            report = audit_episode(d, fps=30, layout=spec['layout'], horizon=50,
                                   tolerance_s=collection.get('tolerance_s', 1e-4))
            allowed = {'discontinuous_time_or_frame_or_source_mapping', 'frame_index_not_zero_based_contiguous'}
            if set(report['errors']) - allowed or len(d['action']) != ep['length']:
                raise ValueError('Invalid independent evaluation episode')
            if not np.all(d['episode_index'] == ep['episode_index']): raise ValueError('Mixed episode IDs')
            valid = np.ones(len(d['action']), bool)
            if validity:
                valid = np.asarray(d.get(validity))
                if valid.dtype != bool or valid.shape != (len(d['action']),):
                    raise ValueError('Explicit boolean validity field required')
            actions = canonicalize(d['action'], spec['layout'])
            states = canonicalize(d['observation.state'], spec['layout'], state=True)
            for first, end in report['segments']:
                for row in range(first+7, end-51):
                    if not valid[row-7:row+52].all(): continue
                    yield {'dataset_id': spec['id'], 'episode_index': ep['episode_index'], 'current_row': row,
                           'source_recording_id': records[key]['source_recording_id'],
                           'history': states[row-7:row+1].copy(), 'q0': states[row].copy(),
                           'target_ext': actions[row-2:row+52].copy(), 'truth': actions[row:row+50].copy(),
                           'timestamp': float(d['timestamp'][row])}


def prepare_plan(collection, manifest, reviews, exclusion, protocol):
    registry_hash = validate_reviews(manifest, reviews, exclusion)
    if protocol['windows_per_recording'] != 5:
        raise ValueError('Predeclared five uniformly spaced valid windows per recording required')
    for source in manifest['sources']:
        cameras = {k: v for k, v in source['info']['features'].items() if k.startswith('observation.images.')}
        if len(cameras) != 3 or cameras != protocol['expected_camera_features']:
            raise ValueError('Three verified matching camera contracts required')
    evidence_files(protocol['preprocessing_evidence_files'])
    by_source = {}
    for w in iter_numeric_candidates(collection, manifest):
        by_source.setdefault(w['source_recording_id'], []).append(
            {k: w[k] for k in ('dataset_id', 'episode_index', 'current_row', 'source_recording_id', 'timestamp')})
    selected = []
    for review in sorted(reviews, key=lambda r: r['source_recording_id']):
        choices = by_source.get(review['source_recording_id'], [])
        if len(choices) < 5: raise ValueError('Insufficient complete history/target windows')
        for i in np.linspace(0, len(choices)-1, 5, dtype=int):
            selected.append({**choices[i], 'evaluation_role': 'independent_success' if review['task_outcome'] == 'success' else 'failure_mode_only'})
    if not any(r['task_outcome'] == 'success' for r in reviews):
        raise ValueError('No reviewed independent successful recordings available')
    result = {'status': 'REVIEWED_PLAN_PREPARED_NOT_FINAL_LOCK', 'formal_independent_test_locked': False,
              'data_sha256': manifest['data_sha256'],
              'manifest_sha256': manifest['manifest_sha256'], 'exclusion_sha256': registry_hash,
              'reviews_sha256': object_sha256(reviews), 'protocol_sha256': object_sha256(protocol), 'windows': selected,
              'requires_before_final_lock': ['hash exact RGB frames with original inference preprocessing',
                  'freeze checkpoint/PRNG/FP32 configuration and confirm video timestamp alignment',
                  'archive signed protocol plus raw-recording lineage review before model evaluation'],
              'restriction': 'Final test is never used for coordinate tuning or model selection; source-level reporting; failure cases separate.'}
    result['plan_sha256'] = object_sha256(result)
    return result


def read_planned_numeric(collection, manifest, plan):
    if object_sha256({k: v for k, v in plan.items() if k != 'plan_sha256'}) != plan['plan_sha256']:
        raise ValueError('Prepared plan identity changed')
    if manifest['data_sha256'] != plan['data_sha256']:
        raise ValueError('Dataset changed after plan preparation')
    wanted = {(w['dataset_id'], w['episode_index'], w['current_row']): w for w in plan['windows']}
    if len(wanted) != len(plan['windows']): raise ValueError('Duplicate planned windows')
    found = {}
    for w in iter_numeric_candidates(collection, manifest):
        key = (w['dataset_id'], w['episode_index'], w['current_row'])
        if key in wanted:
            if w['source_recording_id'] != wanted[key]['source_recording_id']: raise ValueError('Recording identity changed')
            found[key] = w
    if set(found) != set(wanted): raise ValueError('Planned window no longer valid')
    for key in wanted:
        yield found[key]
