"""JSON-safe, explicit audit status: data validity and semantic certainty are separate."""
import numpy as np
from .action_schema import GROUPS, canonicalize, verify_features
from .chunks import continuous_segments, window_starts
from .lerobot_adapter import LeRobotAdapter
from .normalization import load_stats


def audit_episode(data, *, fps, layout, horizon=50, tolerance_s=1e-4):
    errors, warnings = [], []
    result = {'errors': errors, 'warnings': warnings, 'segments': [], 'complete_windows': 0}
    required = ('action', 'observation.state', 'timestamp', 'frame_index', 'episode_index')
    missing = [k for k in required if k not in data]
    if missing:
        errors.append('missing_fields:' + ','.join(missing))
        return result
    lengths = {k: len(np.atleast_1d(data[k])) for k in required}
    n = lengths['timestamp']
    result['frames'] = n
    if len(set(lengths.values())) != 1:
        errors.append('field_length_mismatch')
        return result
    if n < horizon:
        warnings.append('short_episode')
    for key, state in [('action', False), ('observation.state', True)]:
        x = np.asarray(data[key])
        result[key + '_dtype'] = str(x.dtype)
        if x.dtype != np.dtype('float32'):
            errors.append(key + ':expected_float32')
        try:
            if x.ndim != 2:
                raise ValueError('Expected frame × dimension array')
            canonicalize(x, layout, state=state)
        except (TypeError, ValueError) as exc:
            errors.append(key + ':' + str(exc))
    t = np.asarray(data['timestamp'])
    frames, ids = np.asarray(data['frame_index']), np.asarray(data['episode_index'])
    if t.shape != (n,) or frames.shape != (n,) or ids.shape != (n,):
        errors.append('scalar_columns_must_be_1d')
        return result
    if not np.issubdtype(frames.dtype, np.integer) or not np.issubdtype(ids.dtype, np.integer):
        errors.append('frame_and_episode_indices_must_be_integer')
    if len(np.unique(ids)) != 1:
        errors.append('mixed_episode_ids')
    if not np.array_equal(frames, np.arange(n)):
        errors.append('frame_index_not_zero_based_contiguous')
    try:
        segments = continuous_segments(t, frames, fps, tolerance_s=tolerance_s,
                                       source_timestamps_ns=data.get('source_timestamp_ns'))
    except (TypeError, ValueError) as exc:
        errors.append('time:' + str(exc))
        return result
    result['segments'] = [list(s) for s in segments]
    if n and abs(float(t[0])) > tolerance_s:
        errors.append('timestamp_not_zero_based')
    dt = np.diff(t.astype(float))
    if len(dt):
        result['interval_s'] = {'min': float(dt.min()), 'max': float(dt.max()), 'median': float(np.median(dt))}
        result['observed_fps'] = float(1 / np.median(dt)) if np.median(dt) > 0 else None
        if np.any(dt <= 0):
            errors.append('nonmonotonic_timestamps')
    if len(segments) > 1:
        errors.append('discontinuous_time_or_frame_or_source_mapping')
    if 'source_timestamp_ns' not in data:
        warnings.append('original_time_mapping_unavailable_in_parquet')
    for key in ('observation.arm_engaged', 'observation.engaged'):
        if key in data:
            result[key + '_inactive_rows'] = int(np.sum(~np.asarray(data[key], dtype=bool).reshape(n, -1).all(axis=1)))
            warnings.append('engagement_flags_are_not_automatically_used_as_validity_masks')
    if not errors:
        result['complete_windows'] = len(window_starts(segments, horizon))
        result['group_dimensions'] = {k: len(v) for k, v in GROUPS.items()}
    return result


def audit_dataset(config):
    report = {'report_version': 1, 'real_data_read': False, 'real_data_audit_completed': False,
              'errors': [], 'warnings': [], 'episodes': [], 'normalization': {'status': 'unavailable'}}
    root = config.get('dataset_root')
    if not root:
        report['status'] = 'blocked'
        report['warnings'].append('真实数据审计尚未完成：未配置 dataset_root；合成测试不能替代真实审计。')
        return report
    try:
        adapter = LeRobotAdapter(root)
        report['metadata'] = adapter.info
        layout = config.get('layout')
        verify_features(adapter.info['features'], layout)
        fps = adapter.info['fps']
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid metadata FPS')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        report['errors'].append(str(exc))
        report['status'] = 'blocked' if isinstance(exc, OSError) else 'failed'
        return report
    ids, total, successful_reads = [], 0, 0
    for meta in adapter.episodes:
        ep = meta.get('episode_index')
        ids.append(ep)
        row = {'episode_index': ep, 'declared_length': meta.get('length'),
               'provenance': {k: v for k, v in meta.items() if k.startswith(('source_', 'segment_', 'subtask_', 'milestone_'))}}
        try:
            data = adapter.read_episode(ep)
            successful_reads += 1
            row.update(audit_episode(data, fps=fps, layout=layout,
                                    horizon=config.get('horizon', 50), tolerance_s=config.get('tolerance_s', 1e-4)))
            if row.get('frames') != meta.get('length'):
                row['errors'].append('declared_episode_length_mismatch')
            if 'episode_index' in data and not np.all(data['episode_index'] == ep):
                row['errors'].append('episode_metadata_id_mismatch')
            total += row.get('frames', 0)
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
            row.update(errors=[str(exc)], warnings=[])
        report['episodes'].append(row)
    if len(ids) != len(set(ids)):
        report['errors'].append('duplicate_episode_metadata')
    if len(ids) != adapter.info.get('total_episodes'):
        report['errors'].append('total_episode_count_mismatch')
    if total != adapter.info.get('total_frames'):
        report['errors'].append('total_frame_count_mismatch')
    if not ids:
        report['errors'].append('empty_dataset')
    norm_path = config.get('norm_stats_path')
    if norm_path:
        try:
            stats = load_stats(norm_path)
            report['normalization'] = {'status': 'loaded_unmatched', 'shapes': {
                k: {f: list(v.shape) for f, v in value.items()} for k, value in stats.items()}}
            report['warnings'].append('统计已读取但未验证dataset/checkpoint/T3 bundle身份；不能据此宣称物理恢复已验证。')
        except (OSError, KeyError, TypeError, ValueError) as exc:
            report['warnings'].append('normalization:' + str(exc))
    elif config.get('normalization'):
        report['normalization']['status'] = 'checked_at_project_scope'
    else:
        report['warnings'].append('缺少匹配归一化参数；物理动作恢复和真实T3一致性未完成。')
    report['warnings'].append('RGB元信息已读取；未解码视频或验证视频帧同步。')
    report['real_data_read'] = successful_reads > 0
    report['real_data_audit_completed'] = bool(ids) and successful_reads == len(ids)
    report['observed_frames'] = total
    report['observed_episodes'] = len(ids)
    report['status'] = 'failed' if report['errors'] or any(e['errors'] for e in report['episodes']) else 'passed_with_warnings'
    return report
