"""Exact cross-decoding of sealed zero-update FLASH predictions; no GT input.

The two expensive network calls already exist in the paired Step0 export.
D1/D2 only change the final affine decoder, so do not require another forward.
All arithmetic here is float64 post-processing of the recorded FP32 tensors.
"""
import numpy as np
from trc.models.flash_coordinates import FlashCoordinates
from trc.data.action_schema import GROUPS


def cross_decode(c_absolute, c_relative_restored, history, basis, scale, q01, q99):
    c0, c3, h, B, s, lo, hi = [np.asarray(x, dtype=np.float64) for x in
                               (c_absolute, c_relative_restored, history, basis, scale, q01, q99)]
    if c0.shape != c3.shape or c0.ndim != 3 or c0.shape[1:] != (7, 58):
        raise ValueError('Paired N x 7 x 58 native coefficients required')
    if h.shape != (len(c0), 8, 58) or B.shape != (50, 7):
        raise ValueError('Aligned causal history and actual 50-point basis required')
    if not all(np.isfinite(x).all() for x in (c0, c3, h, B)) or not len(c0):
        raise ValueError('Finite nonempty inputs required')
    adapter = FlashCoordinates('state_relative', lo, hi, s, tuple(GROUPS))
    # This branch relies on the actual constant Legendre column, not a guessed grid.
    np.testing.assert_array_equal(B[:, 0], np.ones(50))
    anchor = adapter.anchor_coefficients(h)
    coeff = {'D0': c0, 'D1': c3-anchor, 'D2': c0+anchor, 'D3': c3}
    span = hi-lo+1e-6
    predictions = {name: (np.einsum('tk,nkj,k->ntj', B, c, s)+1)*span/2+lo
                   for name, c in coeff.items()}
    shift = h[:, -1] - (lo+span/2)
    np.testing.assert_allclose(predictions['D2']-predictions['D0'], shift[:, None] + np.zeros((1, 50, 1)), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(predictions['D3']-predictions['D1'], shift[:, None] + np.zeros((1, 50, 1)), atol=1e-12, rtol=1e-12)
    return predictions, coeff, shift


def factorial_effects(predictions, truth):
    """Error interaction differs from function interaction for squared loss."""
    p = {k: np.asarray(predictions[k], float) for k in ('D0', 'D1', 'D2', 'D3')}
    gt = np.asarray(truth, float)
    if any(x.shape != gt.shape for x in p.values()) or not np.isfinite(gt).all():
        raise ValueError('Aligned finite evaluation labels required')
    e = {k: x-gt for k, x in p.items()}
    mse = {k: np.mean(x*x, axis=(1, 2)) for k, x in e.items()}
    interaction = mse['D3']-mse['D1']-mse['D2']+mse['D0']
    dh, dout = p['D1']-p['D0'], p['D2']-p['D0']
    expected = 2*np.mean(dh*dout, axis=(1, 2))
    np.testing.assert_allclose(interaction, expected, atol=1e-12, rtol=1e-10)
    return {'mse': mse, 'history_effect_output_absolute': mse['D1']-mse['D0'],
            'history_effect_output_relative': mse['D3']-mse['D2'],
            'output_effect_history_absolute': mse['D2']-mse['D0'],
            'output_effect_history_relative': mse['D3']-mse['D1'],
            'mse_interaction': interaction, 'mse_interaction_cross_term': expected,
            'function_interaction_max_rad': float(np.max(np.abs(p['D3']-p['D1']-p['D2']+p['D0'])))}
