"""Absolute command ↔ mixed arm delta / hand absolute, relative to anchor state."""
import numpy as np
from .action_schema import ARMS, canonicalize


def transform_actions(actions, anchor_state, *, inverse=False):
    x = canonicalize(actions, 'canonical58').astype(np.result_type(np.asarray(actions).dtype, np.float32), copy=False)
    s = canonicalize(anchor_state, 'canonical58')
    if x.ndim < 2 or s.shape != x.shape[:-2] + (58,):
        raise ValueError('One anchor state per action chunk required')
    # All future arm commands share the same measured anchor, including tick 0.
    x[..., ARMS] += (1 if inverse else -1) * np.expand_dims(s[..., ARMS], -2)
    return x
