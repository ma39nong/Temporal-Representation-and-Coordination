import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from helpers import dataset, assets
from trc.config import read_config
from trc.cli import main
from trc.data.manifest import build_manifest, recording_split
from trc.data.windows import iter_windows
from trc.project import audit_project, project_assets, check_roundtrip
from trc.io import file_sha256, write_report

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


@unittest.skipIf(pq is None, 'optional pyarrow not installed')
class Project(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = dataset(self.root/'dataset', layout='converter58')
        self.cfg = {'horizon': 50, 'datasets': [{'id': 'a', 'repo_id': 'synthetic/a', 'root': str(self.data), 'layout': 'converter58'}]}

    def with_assets(self):
        manifest = build_manifest(self.cfg)
        assets(self.root/'assets', sources=manifest['sources'])
        self.cfg['normalization'] = dict(asset_dir=str(self.root/'assets'), mode='timestamp_zscore_arms', beta=0.,
                                          std_floor=1e-4, expected_contract_path=str(self.root/'assets/expected_contract.json'))
        return manifest

    def test_malformed_metadata_returns_failed_report(self):
        for name in ('info.json', 'episodes.jsonl'):
            with self.subTest(name=name):
                p = self.data/'meta'/name
                original = p.read_text()
                p.write_text('[]')
                self.assertEqual(audit_project(self.cfg)['status'], 'failed')
                p.write_text(original)

    def test_full_flow(self):
        self.with_assets()
        report = audit_project(self.cfg)
        self.assertEqual(report['status'], 'passed_with_warnings')
        self.assertTrue(report['vector_contract_ready'])
        self.assertFalse(report['formal_data_ready'])
        self.assertEqual(report['eligible_windows'], 12)
        self.assertEqual(check_roundtrip(self.cfg)['status'], 'passed')

    def test_dataset_content_drift_rejected(self):
        manifest = self.with_assets()
        p = self.data/'meta/tasks.jsonl'
        p.write_text(p.read_text()+'\n')
        changed = build_manifest(self.cfg)
        self.assertNotEqual(changed['data_sha256'], manifest['data_sha256'])
        with self.assertRaisesRegex(ValueError, 'fingerprints'): project_assets(self.cfg, changed)

    def test_wrong_sampling_frequency_rejected(self):
        manifest = self.with_assets()
        p = self.root/'assets/delta_paper.json'
        doc = json.loads(p.read_text()); doc['source_contract']['action_hz'] = 20; p.write_text(json.dumps(doc))
        p = self.root/'assets/norm_metadata.json'; doc = json.loads(p.read_text()); doc['contract']['action_hz'] = 20; p.write_text(json.dumps(doc))
        p = self.root/'assets/expected_contract.json'; doc = json.loads(p.read_text()); doc['action_hz'] = 20; p.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'action_hz'): project_assets(self.cfg, manifest)

    def test_dataset_repo_identity(self):
        self.with_assets()
        self.cfg['datasets'][0]['repo_id'] = 'synthetic/wrong'
        with self.assertRaisesRegex(ValueError, 'repo_id'): project_assets(self.cfg, build_manifest(self.cfg))

    def test_recording_split_keeps_shared_origin(self):
        manifest = build_manifest(self.cfg)
        record = dict(manifest['records'][0]); record['id'] = 'another-crop'; manifest['records'].append(record)
        split = recording_split(manifest)
        self.assertEqual(split['episode_assignments']['a:0'], split['episode_assignments']['another-crop'])
        self.assertEqual(split, recording_split(manifest))
        self.assertEqual(set(split['episode_assignments'].values()), {'train', 'validation'})
        with self.assertRaises(ValueError): recording_split(manifest, fractions={'train': .8, 'validation': .3, 'test': 0})

    def test_missing_source_provenance_does_not_guess(self):
        p = self.data/'meta/episodes.jsonl'
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        for row in rows: del row['source_recording_id']
        p.write_text('\n'.join(json.dumps(r) for r in rows))
        manifest = build_manifest(self.cfg)
        with self.assertRaisesRegex(ValueError, 'provenance'): recording_split(manifest)
        mapping = {'a': {str(r['episode_index']): {'source_recording_id': f"known-{r['episode_index']}"} for r in rows}}
        m = self.root/'mapping.json'; m.write_text(json.dumps(mapping)); self.cfg['source_mapping_path'] = str(m)
        self.assertTrue(build_manifest(self.cfg)['source_provenance_complete'])
        mapping['a']['999'] = {'source_recording_id': 'wrong'}; m.write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError, 'unknown'): build_manifest(self.cfg)

    def test_duplicate_root_and_duplicate_content(self):
        other = dict(self.cfg['datasets'][0]); other['id'] = 'b'
        self.cfg['datasets'].append(other)
        with self.assertRaisesRegex(ValueError, 'unique'): build_manifest(self.cfg)
        import shutil
        shutil.copytree(self.data, self.root/'duplicate'); other['root'] = str(self.root/'duplicate')
        with self.assertRaisesRegex(ValueError, 'Duplicate dataset content'): build_manifest(self.cfg)

    def test_validity_mask(self):
        path = self.data/'data/chunk-000/episode_000000.parquet'
        import pyarrow as pa
        table = pq.read_table(path)
        table = table.set_column(table.schema.get_field_index('valid'), 'valid', pa.array([False]+[True]*54))
        pq.write_table(table, path)
        self.cfg['validity_column'] = 'valid'
        windows = list(iter_windows(self.cfg))
        self.assertEqual(len(windows), 11)
        self.assertEqual(windows[0].frame_index, 1)
        self.cfg['validity_column'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'missing'): list(iter_windows(self.cfg))

    def test_windows_stop_at_time_gap(self):
        self.data = dataset(self.root/'long', layout='converter58', frames=110)
        self.cfg['datasets'][0]['root'] = str(self.data)
        path = self.data/'data/chunk-000/episode_000000.parquet'
        import pyarrow as pa
        table = pq.read_table(path)
        t = np.arange(110)/30; t[55:] += 1.
        table = table.set_column(table.schema.get_field_index('timestamp'), 'timestamp', pa.array(t))
        pq.write_table(table, path)
        windows = list(iter_windows(self.cfg))
        starts = [w.frame_index for w in windows if w.episode_index == 0]
        self.assertEqual(starts, list(range(6))+list(range(55, 61)))
        self.assertEqual(len(windows), 73)

    def test_no_cross_episode_and_no_time_shift(self):
        windows = list(iter_windows(self.cfg))
        self.assertEqual(len(windows), 12)
        for w in windows:
            self.assertEqual(w.storage_actions.shape, (50, 58))
            self.assertEqual(w.storage_anchor.shape, (116,))
            self.assertAlmostEqual(float(w.timestamps[0]), w.frame_index/30)
            self.assertLessEqual(w.frame_index+50, 55)

    def test_cli_and_readonly_output_guard(self):
        self.with_assets()
        path = self.root/'config.json'; path.write_text(json.dumps(self.cfg))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for command in ('audit', 'manifest', 'split', 'assets-check', 'roundtrip'):
                self.assertEqual(main([command, '--config', str(path), '--output', str(self.root/f'{command}.json')]), 0)
            before = file_sha256(self.data/'meta/info.json')
            self.assertEqual(main(['audit', '--config', str(path), '--output', str(self.data/'meta/info.json')]), 2)
            self.assertEqual(before, file_sha256(self.data/'meta/info.json'))
            before_cfg = path.read_text()
            self.assertEqual(main(['audit', '--config', str(path), '--output', str(path)]), 2)
            self.assertEqual(path.read_text(), before_cfg)


class Configuration(unittest.TestCase):
    def test_relative_paths_and_invalid_config(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'cfg.json'
            p.write_text(json.dumps({'dataset_root': '../some-data', 'horizon': 50}))
            self.assertEqual(read_config(p)['dataset_root'], str((p.parent/'../some-data').resolve()))
            for cfg in ({'typo': 1}, {'horizon': True}, {'horizon': -1}, {'tolerance_s': float('nan')}, {'dataset_root': 'x', 'datasets': [{}]}):
                p.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError): read_config(p)

    def test_report_cannot_write_through_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'data').mkdir(); (root/'alias').symlink_to(root/'data')
            with self.assertRaises(ValueError): write_report(root/'alias/report.json', {}, protected=[root/'data'])
