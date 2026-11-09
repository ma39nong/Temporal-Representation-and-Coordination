"""Synthetic review gates and FLASH [-7,51] window support; no real data claims."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from trc.data.independent_evaluation import validate_reviews, iter_numeric_candidates, prepare_plan, read_planned_numeric
from trc.data.manifest import build_manifest
from trc.io import object_sha256, file_sha256
from helpers import dataset


class IndependentReviews(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name); evidence = root/'synthetic-review.txt'
        evidence.write_text('SYNTHETIC UNIT TEST EVIDENCE ONLY')
        self.evidence = [{'path': str(evidence), 'sha256': file_sha256(evidence)}]
        self.ex = {'complete_training_and_diagnostic_registry': True, 'source_recording_ids': ['old-training', 'old-diagnostic'],
                   'original_recording_sha256': ['a'*64], 'evidence_files': self.evidence}
        self.manifest = {'source_provenance_complete': True, 'records': [{'source_recording_id': 'new'}]}
        self.reviews = [{'source_recording_id': 'new', 'original_recording_sha256': 'b'*64,
            'ancestor_recording_ids': [], 'checked_exclusion_sha256': object_sha256(self.ex), 'lineage_verified': True,
            'quality_passed': True, 'complete_task': True, 'task_outcome': 'success',
            'initial_position_description': 'synthetic initial state', 'truss_configuration_description': 'synthetic truss',
            'evidence_files': self.evidence}]

    def test_review_identity_and_outcome_required(self):
        validate_reviews(self.manifest, self.reviews, self.ex)
        for key, val in [('task_outcome', 'unknown'), ('quality_passed', False), ('complete_task', False),
                         ('lineage_verified', False), ('checked_exclusion_sha256', 'outdated')]:
            r = copy.deepcopy(self.reviews); r[0][key] = val
            with self.subTest(key=key), self.assertRaises(ValueError): validate_reviews(self.manifest, r, self.ex)

    def test_lineage_and_hash_leakage_rejected(self):
        for key, val in [('ancestor_recording_ids', ['old-training']), ('ancestor_recording_ids', ['old-diagnostic']),
                         ('original_recording_sha256', 'a'*64)]:
            r = copy.deepcopy(self.reviews); r[0][key] = val
            with self.subTest(key=key), self.assertRaises(ValueError): validate_reviews(self.manifest, r, self.ex)

    def test_evidence_drift_and_incomplete_registry(self):
        ex = copy.deepcopy(self.ex); ex['complete_training_and_diagnostic_registry'] = False
        with self.assertRaises(ValueError): validate_reviews(self.manifest, self.reviews, ex)
        Path(self.evidence[0]['path']).write_text('changed')
        with self.assertRaises(ValueError): validate_reviews(self.manifest, self.reviews, self.ex)

    def test_new_data_missing_is_explicitly_blocked(self):
        import subprocess, sys
        repo = Path(__file__).resolve().parents[1]; out = Path(self.tmp.name)/'blocked.json'
        subprocess.run([sys.executable, str(repo/'scripts/prepare_independent_evaluation.py'), '--config',
                        str(repo/'configs/p1_6_independent.example.json'), '--output', str(out)], check=True, capture_output=True)
        report = json.loads(out.read_text())
        self.assertFalse(report['real_data_read']); self.assertFalse(report['formal_independent_test_locked'])

    def test_numeric_history_target_alignment_and_prepared_plan(self):
        try: import pyarrow
        except ImportError: self.skipTest('optional pyarrow not installed')
        root = dataset(Path(self.tmp.name)/'dataset', episodes=1, frames=100)
        info_path = root/'meta/info.json'; info = json.loads(info_path.read_text())
        cameras = {f'observation.images.cam{i}': {'dtype': 'video', 'shape': [16, 16, 3]} for i in range(3)}
        info['features'].update(cameras); info_path.write_text(json.dumps(info))
        cfg = {'datasets': [{'id': 'new', 'repo_id': 'synthetic/new', 'root': str(root), 'layout': 'canonical58'}]}
        m = build_manifest(cfg); self.reviews[0]['source_recording_id'] = 'synthetic-recording-0'
        protocol = {'windows_per_recording': 5, 'expected_camera_features': cameras, 'preprocessing_evidence_files': self.evidence}
        plan = prepare_plan(cfg, m, self.reviews, self.ex, protocol)
        self.assertEqual(plan['status'], 'REVIEWED_PLAN_PREPARED_NOT_FINAL_LOCK')
        all_rows = list(iter_numeric_candidates(cfg, m))
        self.assertEqual([w['current_row'] for w in all_rows], list(range(7, 49)))
        selected = list(read_planned_numeric(cfg, m, plan)); self.assertEqual(len(selected), 5)
        for w in selected:
            self.assertEqual(w['history'].shape, (8, 58)); self.assertEqual(w['target_ext'].shape, (54, 58))
            np.testing.assert_array_equal(w['history'][-1], w['q0'])
            np.testing.assert_array_equal(w['target_ext'][2:52], w['truth'])
        plan['windows'][0]['current_row'] += 1
        with self.assertRaises(ValueError): list(read_planned_numeric(cfg, m, plan))

    def test_full_support_never_crosses_time_gap_or_invalid_history(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError: self.skipTest('optional pyarrow not installed')
        root = dataset(Path(self.tmp.name)/'dataset', episodes=1, frames=100)
        cfg = {'datasets': [{'id': 'new', 'repo_id': 'synthetic/new', 'root': str(root), 'layout': 'canonical58'}],
               'validity_column': 'valid'}
        path = root/'data/chunk-000/episode_000000.parquet'; table = pq.read_table(path)
        valid = np.ones(100, bool); valid[30] = False
        masked = table.set_column(table.schema.get_field_index('valid'), 'valid', pa.array(valid))
        pq.write_table(masked, path)
        rows = list(iter_numeric_candidates(cfg, build_manifest(cfg)))
        self.assertEqual([w['current_row'] for w in rows], list(range(38, 49)))
        timestamps = np.arange(100)/30; timestamps[50:] += 1
        broken = table.set_column(table.schema.get_field_index('timestamp'), 'timestamp', pa.array(timestamps))
        pq.write_table(broken, path)
        self.assertEqual(list(iter_numeric_candidates(cfg, build_manifest(cfg))), [])
