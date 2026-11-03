"""Explicit, metadata-checked storage layouts; no dimension-based guessing."""
import numpy as np

HAND = ('thumb_CMC_FE', 'thumb_CMC_AA', 'thumb_MCP_FE', 'thumb_MCP_AA', 'thumb_IP',
        'index_MCP_FE', 'index_MCP_AA', 'index_PIP', 'index_DIP',
        'middle_MCP_FE', 'middle_MCP_AA', 'middle_PIP', 'middle_DIP',
        'ring_MCP_FE', 'ring_MCP_AA', 'ring_PIP', 'ring_DIP',
        'pinky_CMC', 'pinky_MCP_FE', 'pinky_MCP_AA', 'pinky_PIP', 'pinky_DIP')
JOINTS = {f'{side}_arm': tuple(f'{side}_fr3_joint{i}' for i in range(1, 8))
          for side in ('left', 'right')}
JOINTS.update({f'{side}_hand': tuple(f'{side}_{j}' for j in HAND) for side in ('left', 'right')})
GROUPS = {'left_arm': list(range(7)), 'left_hand': list(range(7, 29)),
          'right_arm': list(range(29, 36)), 'right_hand': list(range(36, 58))}
ORDER = tuple(j for group in GROUPS for j in JOINTS[group])
ARMS = np.array(GROUPS['left_arm'] + GROUPS['right_arm'])
LAYOUTS = ('canonical58', 'converter58')


def expected_names(layout, state=False):
    if layout == 'canonical58':
        return ORDER
    if layout != 'converter58':
        raise ValueError('Explicit known layout required')
    groups = ('left_arm', 'right_arm', 'left_hand', 'right_hand')
    if state:
        return tuple(f'{j}.{kind}' for g in groups for kind in ('position', 'velocity') for j in JOINTS[g])
    return tuple(f'{j}.{"position" if g.endswith("arm") else "command"}' for g in groups for j in JOINTS[g])


def verify_features(features, layout):
    for key, state in [('action', False), ('observation.state', True)]:
        f = features.get(key, {})
        names = expected_names(layout, state)
        if f.get('shape') != [len(names)] or f.get('dtype') != 'float32' or f.get('names') != list(names):
            raise ValueError(f'{key}: shape/dtype/joint names do not match explicit {layout}')


def canonicalize(values, layout, state=False):
    x = np.asarray(values)
    dim = 116 if layout == 'converter58' and state else 58
    if layout not in LAYOUTS or x.ndim == 0 or x.shape[-1] != dim:
        raise ValueError(f'Invalid {layout} vector dimension; expected {dim}')
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite numeric joint values')
    if layout == 'canonical58':
        return x.copy()
    indices = np.r_[0:7, 28:50, 14:21, 72:94] if state else np.r_[0:7, 14:36, 7:14, 36:58]
    return x[..., indices].copy()


def split_groups(canonical_action):
    x = canonicalize(canonical_action, 'canonical58')
    return {name: x[..., indices] for name, indices in GROUPS.items()}
