"""Complete offline windows. No padding, resampling or crossing segment boundaries."""
import numpy as np


def continuous_segments(timestamps, frame_indices, fps, *, tolerance_s=1e-4, source_timestamps_ns=None):
    t, frames = np.asarray(timestamps), np.asarray(frame_indices)
    if t.ndim != 1 or frames.shape != t.shape or not np.isfinite(t).all():
        raise ValueError('Finite aligned time and frame vectors required')
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError('Invalid FPS or tolerance')
    if not len(t):
        return []
    dt = np.diff(t)
    bad = (dt <= 0) | (np.abs(dt - 1 / fps) > tolerance_s) | (np.diff(frames) != 1)
    if source_timestamps_ns is not None:
        source = np.asarray(source_timestamps_ns)
        if source.shape != t.shape or not np.isfinite(source).all():
            raise ValueError('Invalid source time mapping')
        ds = np.diff(source.astype(np.float64)) / 1e9
        bad |= (ds <= 0) | (np.abs(ds - 1 / fps) > tolerance_s)
    cuts = [0, *(np.flatnonzero(bad) + 1).tolist(), len(t)]
    return list(zip(cuts[:-1], cuts[1:]))


def window_starts(segments, horizon=50):
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError('Positive integer horizon required')
    return [i for start, end in segments for i in range(start, end - horizon + 1)]


def extract_chunk(actions, start, segments, horizon=50):
    if start not in window_starts(segments, horizon):
        raise ValueError('Incomplete chunk or discontinuity boundary')
    return np.asarray(actions)[start:start + horizon].copy()
