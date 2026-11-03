"""Lazy offline action chunks with explicit validity, time and source provenance."""
from dataclasses import dataclass
import numpy as np
from trc.config import dataset_specs
from .action_schema import canonicalize, verify_features
from .chunks import continuous_segments, window_starts
from .lerobot_adapter import LeRobotAdapter
from .validation import audit_episode


@dataclass(frozen=True)
class ActionWindow:
    dataset_id: str
    episode_index: int
    frame_index: int
    timestamps: np.ndarray
    storage_actions: np.ndarray
    storage_anchor: np.ndarray
    physical_actions: np.ndarray
    canonical_anchor: np.ndarray
    source_recording_id: str | None
    source_timestamps_ns: np.ndarray | None


def iter_windows(cfg, *, manifest=None):
    records = {r['id']: r for r in manifest['records']} if manifest else {}
    for spec in dataset_specs(cfg):
        adapter = LeRobotAdapter(spec['root'])
        verify_features(adapter.info['features'], spec['layout'])
        for meta in adapter.episodes:
            ep = meta['episode_index']
            validity = cfg.get('validity_column')
            data = adapter.read_episode(ep, extra_columns=[validity] if validity else [])
            report = audit_episode(data, fps=adapter.info['fps'], layout=spec['layout'],
                                   horizon=cfg.get('horizon', 50), tolerance_s=cfg.get('tolerance_s', 1e-4))
            # Time/frame discontinuities are explicitly represented by segments.
            permitted = {'discontinuous_time_or_frame_or_source_mapping', 'frame_index_not_zero_based_contiguous'}
            if set(report['errors']) - permitted:
                raise ValueError(f"Invalid episode {spec['id']}:{ep}: {report['errors']}")
            if report.get('frames') != meta['length'] or not np.all(data['episode_index'] == ep):
                raise ValueError('Window metadata length/episode identity mismatch')
            n = len(data['timestamp'])
            mask = np.ones(n, bool)
            if validity:
                if validity not in data:
                    raise ValueError('Configured validity field is missing')
                values = np.asarray(data[validity])
                if values.dtype != np.dtype(bool) or values.shape != (n,):
                    raise ValueError('Validity must be an explicit one-dimensional boolean field')
                mask = values
            segments = report['segments']
            horizon = cfg.get('horizon', 50)
            for start in window_starts(segments, horizon):
                end = start + horizon
                if not mask[start:end].all():
                    continue
                storage, anchor = data['action'][start:end], data['observation.state'][start]
                source = records.get(f"{spec['id']}:{ep}", meta).get('source_recording_id')
                yield ActionWindow(spec['id'], ep, int(data['frame_index'][start]), data['timestamp'][start:end].copy(),
                                   storage.copy(), anchor.copy(), canonicalize(storage, spec['layout']),
                                   canonicalize(anchor, spec['layout'], state=True), source,
                                   data['source_timestamp_ns'][start:end].copy() if 'source_timestamp_ns' in data else None)
