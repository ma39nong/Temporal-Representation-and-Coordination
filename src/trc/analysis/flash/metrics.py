"""Masked, pooled physical errors. Prediction and representation are not additive norms."""
import numpy as np
from trc.data.action_schema import GROUPS, ORDER

EVAL_GROUPS = {**GROUPS, 'all_arms': GROUPS['left_arm']+GROUPS['right_arm'],
               'all_hands': GROUPS['left_hand']+GROUPS['right_hand'], 'full_action': list(range(58))}


def summary(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError('Nonempty finite errors required')
    a = np.abs(x)
    return {'mae': float(a.mean()), 'rmse': float(np.sqrt(np.mean(x*x))),
            'p95': float(np.quantile(a, .95)), 'p99': float(np.quantile(a, .99)), 'max': float(a.max()), 'count': int(x.size)}


def evaluate(truth, oracle, prediction, valid=None):
    arrays = [np.asarray(a, dtype=np.float64) for a in (truth, oracle, prediction)]
    if any(a.ndim != 3 or a.shape[1:] != (50, 58) or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Aligned N × 50 × 58 arrays required')
    mask = np.ones(arrays[0].shape, bool) if valid is None else np.asarray(valid)
    if mask.dtype != bool or mask.shape != arrays[0].shape or not mask.any():
        raise ValueError('Explicit nonempty aligned boolean mask required')
    if any(not np.isfinite(a[mask]).all() for a in arrays):
        raise ValueError('Nonfinite valid samples')
    a, o, p = arrays
    residuals = {'representation': o-a, 'prediction': p-o, 'total': p-a}
    groups, temporal, joints, windows = [], [], [], []
    for method, delta in residuals.items():
        for group, idx in EVAL_GROUPS.items():
            sel = delta[..., idx]; m = mask[..., idx]
            groups.append({'method': method, 'group': group, **summary(sel[m])})
            for tick in range(50):
                mt = m[:, tick]
                if mt.any():
                    temporal.append({'method': method, 'group': group, 'tick': tick, 'seconds': tick/30, **summary(sel[:, tick][mt])})
        for joint in range(58):
            m = mask[..., joint]
            if m.any():
                joints.append({'method': method, 'joint_index': joint, 'joint_name': ORDER[joint], **summary(delta[..., joint][m])})
        for i in range(len(a)):
            m = mask[i]
            if m.any():
                scores = np.where(m, np.abs(delta[i]), -np.inf)
                tick, joint = np.unravel_index(np.argmax(scores), scores.shape)
                windows.append({'method': method, 'window_index': i, 'worst_tick': int(tick), 'worst_joint_index': int(joint),
                                'worst_joint': ORDER[joint], **summary(delta[i][m])})
    r, e, total = (residuals[k][mask] for k in ('representation', 'prediction', 'total'))
    decomposition = {'representation_mse': float(np.mean(r*r)), 'prediction_mse': float(np.mean(e*e)),
                     'cross_term': float(2*np.mean(r*e)), 'total_mse': float(np.mean(total*total)),
                     'identity': 'MSE_total = MSE_repr + MSE_pred + 2 mean(residual_repr * residual_pred)'}
    return {'groups': groups, 'temporal': temporal, 'joints': joints, 'windows': windows, 'mse_decomposition': decomposition}


def coefficient_metrics(gt, predicted, scale, q01, q99):
    gt, predicted = np.asarray(gt), np.asarray(predicted)
    if gt.shape != predicted.shape or gt.ndim != 3 or gt.shape[1:] != (7, 58):
        raise ValueError('Scaled coefficient arrays N × 7 × 58 required')
    scale = np.asarray(scale)
    if scale.shape != (7,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Seven positive coefficient scales required')
    delta = predicted-gt
    # Report each degree separately: no single mixed-scale coefficient score.
    raw_normalized = delta * scale[None, :, None]
    physical = raw_normalized * (np.asarray(q99)-np.asarray(q01)+1e-6)[None, None, :]/2
    rows = []
    for degree in range(7):
        for group, idx in EVAL_GROUPS.items():
            for space, value in [('scaled_training', delta), ('raw_normalized', raw_normalized), ('raw_physical_rad', physical)]:
                rows.append({'degree': degree, 'group': group, 'space': space, **summary(value[:, degree, idx])})
    return rows


def joint_time_diagnostics(truth, prediction, valid):
    """Joint increments only; no unsupported FK, grasp-stage or failure score."""
    a, p = np.asarray(truth, float), np.asarray(prediction, float)
    m = np.asarray(valid, bool)[:, 1:] & np.asarray(valid, bool)[:, :-1]
    da, dp = np.diff(a, axis=1)*30, np.diff(p, axis=1)*30
    rows = []
    for group, idx in EVAL_GROUPS.items():
        for tick in range(49):
            good = m[:, tick, idx]
            # Advanced indexing orders dimensions differently; reshape is immaterial for pooling.
            if good.any():
                av, pv = da[:, tick, idx][good], dp[:, tick, idx][good]
                rows.append({'group': group, 'interval_start_tick': tick, 'unit': 'rad_per_second',
                             'gt_mean_absolute_joint_speed': float(np.abs(av).mean()),
                             'pred_mean_absolute_joint_speed': float(np.abs(pv).mean()), **summary(pv-av)})
    return rows
