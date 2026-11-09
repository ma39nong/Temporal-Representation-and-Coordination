import unittest
import numpy as np
from numpy.polynomial.legendre import legvander
from trc.analysis.flash.coordinate_ablation import cross_decode, factorial_effects


class CoordinateAblation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.c0 = rng.normal(size=(2, 7, 58))
        self.c3 = rng.normal(size=(2, 7, 58))
        self.h = rng.normal(size=(2, 8, 58))
        self.B = legvander(np.linspace(-1, 1, 54)[2:52], 6)
        self.s = np.linspace(.1, 1, 7)
        self.lo = np.linspace(-3, -1, 58)
        self.hi = np.linspace(1, 4, 58)

    def run_decode(self):
        return cross_decode(self.c0, self.c3, self.h, self.B, self.s, self.lo, self.hi)

    def test_actual_affine_shift_is_not_q0(self):
        p, c, shift = self.run_decode()
        np.testing.assert_allclose(shift, self.h[:, -1]-(self.lo+self.hi+1e-6)/2)
        self.assertGreater(np.max(np.abs(shift-self.h[:, -1])), .1)
        np.testing.assert_allclose(p['D2']-p['D0'], np.broadcast_to(shift[:, None], p['D0'].shape), atol=1e-13)
        np.testing.assert_array_equal(c['D1'][:, 1:], c['D3'][:, 1:])
        np.testing.assert_allclose(np.diff(p['D2'], axis=1), np.diff(p['D0'], axis=1), atol=1e-13)

    def test_no_function_interaction_but_squared_error_interaction(self):
        p, _, _ = self.run_decode()
        f = factorial_effects(p, np.zeros_like(p['D0']))
        self.assertLess(f['function_interaction_max_rad'], 1e-13)
        self.assertGreater(np.max(np.abs(f['mse_interaction'])), .01)

    def test_decoder_uses_no_labels_or_future(self):
        p, _, _ = self.run_decode()
        a = factorial_effects(p, np.zeros_like(p['D0']))
        b = factorial_effects(p, np.ones_like(p['D0']))
        np.testing.assert_allclose(a['mse_interaction'], b['mse_interaction'], atol=1e-12)
        self.assertFalse(np.array_equal(a['mse']['D0'], b['mse']['D0']))

    def test_cross_decoders_match_explicit_euler_paths(self):
        from trc.models.flash_coordinates import FlashCoordinates
        from trc.data.action_schema import GROUPS
        adapter = FlashCoordinates('state_relative', self.lo, self.hi, self.s, tuple(GROUPS))
        anchor = adapter.anchor_coefficients(self.h)
        source = self.c0
        velocity = lambda c: np.tanh(c) + .1*c*c
        absolute = source+velocity(source)
        relative_raw = source-anchor+velocity(source-anchor)
        _, coeff, _ = cross_decode(absolute, relative_raw+anchor, self.h, self.B, self.s, self.lo, self.hi)
        for key, expected in [('D0', absolute), ('D1', relative_raw),
                              ('D2', absolute+anchor), ('D3', relative_raw+anchor)]:
            np.testing.assert_allclose(coeff[key], expected, atol=1e-13)

    def test_invalid_inputs(self):
        self.c0[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): self.run_decode()
        self.c0[0, 0, 0] = 0
        self.B[:, 0] = .5
        with self.assertRaises(AssertionError): self.run_decode()
