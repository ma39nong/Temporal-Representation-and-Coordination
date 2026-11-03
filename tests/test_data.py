import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from trc.data.action_schema import ORDER, GROUPS, expected_names, canonicalize, verify_features, split_groups
from trc.data.chunks import continuous_segments, window_starts, extract_chunk
from trc.data.normalization import load_stats, normalize, arm_bridge
from trc.data.semantics import transform_actions
from trc.data.validation import audit_episode, audit_dataset


def episode(n=55):
    return {'action': np.zeros((n, 58), np.float32), 'observation.state': np.zeros((n, 58), np.float32),
            'timestamp': np.arange(n) / 30, 'frame_index': np.arange(n), 'episode_index': np.zeros(n, int)}


class Contracts(unittest.TestCase):
    def audit(self, data):
        return audit_episode(data, fps=30, layout='canonical58')

    def test_valid_dimension(self):
        self.assertEqual(self.audit(episode())['complete_windows'], 6)

    def test_invalid_dimension(self):
        data = episode()
        data['action'] = np.zeros((55, 54), np.float32)
        self.assertTrue(self.audit(data)['errors'])

    def test_dtype(self):
        data = episode()
        data['action'] = data['action'].astype(np.float64)
        self.assertIn('action:expected_float32', self.audit(data)['errors'])

    def test_mapping(self):
        raw = np.arange(58, dtype=np.float32)
        x = canonicalize(raw, 'converter58')
        parts = split_groups(x)
        np.testing.assert_array_equal(parts['right_arm'], raw[7:14])
        np.testing.assert_array_equal(parts['left_hand'], raw[14:36])
        np.testing.assert_array_equal(parts['right_hand'], raw[36:58])
        np.testing.assert_array_equal(canonicalize(np.arange(116), 'converter58', state=True),
                                      np.r_[0:7, 28:50, 14:21, 72:94])

    def test_feature_identity(self):
        features = {k: {'shape': [len(expected_names('canonical58', st))], 'dtype': 'float32',
                         'names': list(expected_names('canonical58', st))}
                    for k, st in [('action', False), ('observation.state', True)]}
        verify_features(features, 'canonical58')
        features['action']['names'][0] = 'wrong'
        with self.assertRaises(ValueError): verify_features(features, 'canonical58')

    def test_nan_inf(self):
        for value in (float('nan'), float('inf')):
            data = episode()
            data['action'][1, 2] = value
            self.assertTrue(self.audit(data)['errors'])

    def test_nonmonotonic(self):
        data = episode()
        data['timestamp'][5] = data['timestamp'][4]
        self.assertIn('nonmonotonic_timestamps', self.audit(data)['errors'])

    def test_gap(self):
        data = episode(110)
        data['timestamp'][55:] += 1
        report = self.audit(data)
        self.assertEqual(report['segments'], [[0, 55], [55, 110]])
        self.assertEqual(len(window_starts(report['segments'])), 12)
        with self.assertRaises(ValueError): extract_chunk(data['action'], 10, report['segments'])

    def test_source_time_gap(self):
        data = episode()
        data['source_timestamp_ns'] = (data['timestamp'] * 1e9).astype(np.int64)
        data['source_timestamp_ns'][20:] += 10**9
        self.assertTrue(self.audit(data)['errors'])

    def test_episode_boundaries(self):
        self.assertEqual(window_starts([(0, 51), (51, 100), (100, 155)]), [0, 1, 100, 101, 102, 103, 104, 105])
        data = episode()
        data['episode_index'][20:] = 1
        self.assertIn('mixed_episode_ids', self.audit(data)['errors'])

    def test_short_and_missing(self):
        self.assertIn('short_episode', self.audit(episode(10))['warnings'])
        data = episode()
        del data['action']
        self.assertTrue(self.audit(data)['errors'])

    def test_missing_norm(self):
        with self.assertRaises(ValueError): load_stats(None)
        with self.assertRaises(ValueError): normalize(np.zeros((50, 58)), None)
        with self.assertRaises(ValueError): normalize(np.zeros((50, 58)), {'q01': np.zeros(58)})

    def test_semantic_roundtrip(self):
        x = np.arange(50 * 58, dtype=np.float32).reshape(50, 58)
        s = np.arange(58, dtype=np.float32)
        delta = transform_actions(x, s)
        np.testing.assert_array_equal(delta[:, 7:29], x[:, 7:29])
        np.testing.assert_array_equal(delta[:, 0], x[:, 0] - s[0])
        np.testing.assert_array_equal(transform_actions(delta, s, inverse=True), x)

    def test_normalization_roundtrip_synthetic(self):
        x = np.random.default_rng(1).normal(size=(50, 58)).astype(np.float32)
        stats = {'q01': np.full(58, -2.), 'q99': np.full(58, 2.), 'mean': np.zeros(58), 'std': np.ones(58)}
        for mode in ('quantile', 'zscore'):
            np.testing.assert_allclose(normalize(normalize(x, stats, method=mode), stats, method=mode, inverse=True), x, atol=1e-6)
        kwargs = dict(q01=np.full((50, 58), -2.), q99=np.full((50, 58), 2.), arm_mean=np.zeros((50, 14)), arm_std=np.zeros((50, 14)), std_floor=.01)
        np.testing.assert_allclose(arm_bridge(arm_bridge(x, **kwargs), inverse=True, **kwargs), x, atol=1e-5)
        self.assertGreater(normalize(np.full((50, 58), 10.), stats).max(), 1)  # upstream does not clip

    def test_no_real_data(self):
        report = audit_dataset({'dataset_root': None})
        self.assertEqual(report['status'], 'blocked')
        self.assertFalse(report['real_data_read'])
        self.assertIn('真实数据审计尚未完成', report['warnings'][0])
        json.dumps(report, allow_nan=False)

    def test_bad_time(self):
        for fps in (0, -1, float('nan')):
            with self.assertRaises(ValueError): continuous_segments(np.arange(3), np.arange(3), fps)
        data = episode()
        data['timestamp'][0] = float('nan')
        self.assertTrue(self.audit(data)['errors'])

    def test_optional_parquet_integration(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('optional pyarrow not installed')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'meta').mkdir()
            (root / 'data/chunk-000').mkdir(parents=True)
            info = {'codebase_version': 'v2.1', 'fps': 30, 'total_episodes': 1, 'total_frames': 55,
                    'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                    'features': {k: {'shape': [58], 'dtype': 'float32', 'names': list(ORDER)}
                                 for k in ('action', 'observation.state')}}
            (root / 'meta/info.json').write_text(json.dumps(info))
            (root / 'meta/episodes.jsonl').write_text(json.dumps({'episode_index': 0, 'length': 55}))
            data = episode()
            table = pa.table({k: pa.array(v.tolist(), type=pa.list_(pa.float32(), 58)) if v.ndim == 2 else pa.array(v)
                              for k, v in data.items()})
            pq.write_table(table, root / 'data/chunk-000/episode_000000.parquet')
            report = audit_dataset({'dataset_root': temp, 'layout': 'canonical58'})
            self.assertEqual(report['status'], 'passed_with_warnings')
            self.assertTrue(report['real_data_read'])  # actual file read, synthetic content
            self.assertEqual(report['observed_frames'], 55)
            self.assertEqual(report['episodes'][0]['complete_windows'], 6)
            info['total_frames'] = 56
            (root / 'meta/info.json').write_text(json.dumps(info))
            self.assertEqual(audit_dataset({'dataset_root': temp, 'layout': 'canonical58'})['status'], 'failed')


if __name__ == '__main__': unittest.main()
