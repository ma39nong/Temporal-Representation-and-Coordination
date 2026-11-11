import copy
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'readout_fairness_audit', Path(__file__).parents[1]/'scripts/audit_readout_fairness.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ReadoutFairnessAuditTests(unittest.TestCase):
    def fixture(self):
        record = dict(id='r', source_recording_id='raw', rows=180,
                      root='/synthetic/no-block-one/A', tasks={'0': 'single-pick'})
        proof = dict(id='r', source_recording_id='raw', rows=180, cuts=[80], legal_windows=64)
        return {'records': [record]}, {'records': [proof], 'legal_windows': 64}

    def test_boundary_exclusion_is_exact_and_not_source_removal(self):
        m, c = self.fixture()
        summary, rows = audit.window_audit(m, c)
        self.assertEqual(summary['compact_windows'], 122)
        self.assertEqual(summary['continuous_windows'], 64)
        self.assertEqual(summary['excluded'], 58)
        self.assertEqual(summary['sources_without_windows'], 0)
        self.assertEqual(rows[0]['recording_id'], 'raw')

    def test_short_adjacent_segments_do_not_double_count_exclusions(self):
        m, c = self.fixture()
        c['records'][0].update(cuts=[80, 90], legal_windows=54)
        c['legal_windows'] = 54
        summary, _ = audit.window_audit(m, c)
        self.assertEqual(summary['excluded'], 68)
        self.assertNotEqual(summary['excluded'], 2*58)

    def test_continuous_task_root_rejected(self):
        m, c = self.fixture()
        m['records'][0]['root'] = '/synthetic/no-block-more/A'
        with self.assertRaises(ValueError):
            audit.window_audit(m, c)

    def test_same_recording_cannot_count_as_two_sources(self):
        m, c = self.fixture()
        r = copy.deepcopy(m['records'][0]); r['id'] = 'second'
        p = copy.deepcopy(c['records'][0]); p['id'] = 'second'
        m['records'].append(r); c['records'].append(p); c['legal_windows'] *= 2
        with self.assertRaises(ValueError):
            audit.window_audit(m, c)
