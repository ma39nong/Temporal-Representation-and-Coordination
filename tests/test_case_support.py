"""Regression tests for actual case-boundary mistakes, independent of Parquet."""
import unittest
from trc.data.qualification import flash_window_cuts


class CaseSupportTests(unittest.TestCase):
    def test_history_only_cut_rejects_case(self):
        self.assertEqual(flash_window_cuts(678, [161, 671, 675]), [675])

    def test_gt_and_extended_target_support(self):
        self.assertEqual(flash_window_cuts(626, [650, 659]), [650, 659])
        self.assertEqual(flash_window_cuts(100, [150, 151, 152]), [150, 151])

    def test_boundary_at_first_row_allowed(self):
        self.assertEqual(flash_window_cuts(100, [93]), [])
        self.assertEqual(flash_window_cuts(848, [552]), [])

    def test_bad_indices_rejected(self):
        for current, cuts in [(True, []), (1.5, []), (10, [3.1]), (10, [8, 7]), (10, [8, 8])]:
            with self.subTest(current=current, cuts=cuts), self.assertRaises(ValueError):
                flash_window_cuts(current, cuts)
