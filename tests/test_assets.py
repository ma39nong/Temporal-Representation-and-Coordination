import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from helpers import assets
from trc.data.assets import load_assets
from trc.data.pipeline import ActionCodec


class Assets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contract = assets(self.root)
        self.options = dict(mode='timestamp_zscore_arms', beta=0., horizon=50, std_floor=1e-4)

    def test_integrity_is_not_dataset_binding(self):
        loaded = load_assets(self.root, **self.options)
        self.assertEqual(loaded.provenance_status, 'integrity_only')
        with self.assertRaises(ValueError): ActionCodec(loaded, 'canonical58')

    def test_corrupted_bundle(self):
        with (self.root / 'delta_paper.npz').open('ab') as f: f.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'digest'): load_assets(self.root, **self.options)

    def test_corrupted_stats(self):
        with (self.root / 'norm_stats.json').open('a') as f: f.write(' ')
        with self.assertRaisesRegex(ValueError, 'digest'): load_assets(self.root, **self.options)

    def test_wrong_experiment(self):
        for key, value in [('mode', 'global_quantile'), ('horizon', 49), ('beta', .5), ('std_floor', .001)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_assets(self.root, **{**self.options, key: value})

    def test_wrong_provenance(self):
        p = self.root / 'norm_metadata.json'
        audit = json.loads(p.read_text()); audit['contract']['action_horizon'] = 1
        p.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, 'provenance'): load_assets(self.root, **self.options)

    def test_transform_pin(self):
        contract = {**self.contract, 'vector_transforms': {'different': 'transform'}}
        with self.assertRaisesRegex(ValueError, 'contract'): load_assets(self.root, expected_contract=contract, **self.options)

    def test_expected_hash_pin(self):
        with self.assertRaises(ValueError): load_assets(self.root, expected_bundle_sha256='0'*64, **self.options)

    def test_codec_roundtrip_synthetic_both_layouts_and_batches(self):
        for mode in ('timestamp_zscore_arms', 'global_zscore_arms', 'global_quantile'):
            assets(self.root, mode=mode)
            loaded = load_assets(self.root, **{**self.options, 'mode': mode})
            rng = np.random.default_rng(2)
            x = rng.normal(size=(2, 50, 58)).astype(np.float32)
            for layout, dim in [('canonical58', 58), ('converter58', 116)]:
                codec = ActionCodec(loaded, layout, require_verified=False)
                self.assertLess(codec.roundtrip(x, rng.normal(size=(2, dim)).astype(np.float32)), 1e-5)
                with self.assertRaises(ValueError): codec.encode(x[:, :49], np.zeros((2, dim)))

    def test_bad_arrays_even_with_fresh_digest(self):
        from trc.io import file_sha256
        for kind in ('arm_shape', 'negative_std', 'nan'):
            with self.subTest(kind=kind):
                assets(self.root)
                std = np.ones((50, 14)); mean = np.zeros((50, 14))
                if kind == 'arm_shape': mean = np.zeros((49, 14))
                if kind == 'negative_std': std[0, 0] = -1
                if kind == 'nan': mean[0, 0] = float('nan')
                np.savez(self.root/'delta_paper.npz', arm_mean=mean, arm_std=std, noise_cholesky=np.empty((0, 0)))
                p = self.root/'delta_paper.json'; doc = json.loads(p.read_text())
                doc['bundle_sha256'] = file_sha256(self.root/'delta_paper.npz'); p.write_text(json.dumps(doc))
                with self.assertRaises(ValueError): load_assets(self.root, **self.options)

    def test_malformed_stats_json(self):
        from trc.data.normalization import load_stats
        for payload in ({'norm_stats': []}, {'norm_stats': {'actions': []}}, {'norm_stats': {'actions': {'q01': ['wrong']}}}):
            (self.root/'norm_stats.json').write_text(json.dumps(payload))
            with self.assertRaises(ValueError): load_stats(self.root/'norm_stats.json')

    def test_nonzero_beta_noise_factor_validation(self):
        from trc.io import file_sha256
        self.contract = assets(self.root, horizon=1)
        manifest_path = self.root/'delta_paper.json'
        manifest = json.loads(manifest_path.read_text()); manifest['beta'] = .5
        options = {**self.options, 'horizon': 1, 'beta': .5}
        for valid in (True, False):
            chol = np.eye(58)
            if not valid: chol[0, 1] = .2
            np.savez(self.root/'delta_paper.npz', arm_mean=np.zeros((1, 14)), arm_std=np.ones((1, 14)), noise_cholesky=chol)
            manifest['bundle_sha256'] = file_sha256(self.root/'delta_paper.npz')
            manifest_path.write_text(json.dumps(manifest))
            if valid:
                self.assertEqual(load_assets(self.root, **options).arrays['noise_cholesky'].shape, (58, 58))
            else:
                with self.assertRaisesRegex(ValueError, 'triangular'): load_assets(self.root, **options)
