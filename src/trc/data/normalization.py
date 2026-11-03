"""Offline numerical API; requires explicit matching statistics, never estimates them."""
import json
from pathlib import Path
import numpy as np
from .action_schema import ARMS


def load_stats(path):
    if not path:
        raise ValueError('Missing normalization parameters; physical recovery unavailable')
    raw = json.loads(Path(path).read_text())
    return {key: {k: np.asarray(v) for k, v in value.items() if v is not None}
            for key, value in raw['norm_stats'].items()}


def normalize(x, stats, *, method='quantile', inverse=False):
    x = np.asarray(x)
    if stats is None:
        raise ValueError('Missing normalization parameters')
    if x.ndim < 1 or x.shape[-1] != 58 or not np.isfinite(x).all():
        raise ValueError('Expected finite 58D actions')
    keys = ('q01', 'q99') if method == 'quantile' else ('mean', 'std')
    if method not in ('quantile', 'zscore') or any(k not in stats for k in keys):
        raise ValueError('Missing or unknown normalization parameters')
    a, b = (np.asarray(stats[k]) for k in keys)
    if a.shape != b.shape or a.shape not in ((58,), x.shape[-2:]) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Normalization shape/finite mismatch')
    if method == 'quantile':
        if np.any(b < a):
            raise ValueError('Reversed quantile bounds')
        scale = b - a + 1e-6
        return (x + 1) / 2 * scale + a if inverse else (x - a) / scale * 2 - 1
    if np.any(b < 0):
        raise ValueError('Negative standard deviation')
    return x * (b + 1e-6) + a if inverse else (x - a) / (b + 1e-6)


def arm_bridge(x, *, q01, q99, arm_mean, arm_std, std_floor, inverse=False):
    """T3 bridge AFTER global quantile on input; BEFORE unquantile on output.

    Arrays must come from the matching bundle. This API does not certify bundle
    identity; full upstream manifest validation remains required for real T3 use.
    """
    x = np.asarray(x)
    q01, q99, mean, std = map(np.asarray, (q01, q99, arm_mean, arm_std))
    if x.ndim < 2 or x.shape[-1] != 58 or q01.shape != x.shape[-2:] or q99.shape != q01.shape:
        raise ValueError('Full horizon quantile arrays required')
    if mean.shape != (x.shape[-2], 14) or std.shape != mean.shape:
        raise ValueError('Expected horizon × 14 arm statistics')
    if not all(np.isfinite(a).all() for a in (x, q01, q99, mean, std)) or np.any(std < 0) or np.any(q99 < q01):
        raise ValueError('Invalid normalization statistics')
    if not np.isfinite(std_floor) or std_floor <= 0:
        raise ValueError('Explicit positive std_floor required')
    result = x.copy()
    lo, scale = q01[:, ARMS], q99[:, ARMS] - q01[:, ARMS] + 1e-6
    scale_std = np.maximum(std, std_floor)
    if inverse:
        raw = x[..., ARMS] * scale_std + mean
        result[..., ARMS] = (raw - lo) / scale * 2 - 1
    else:
        raw = (x[..., ARMS] + 1) / 2 * scale + lo
        result[..., ARMS] = (raw - mean) / scale_std
    return result
