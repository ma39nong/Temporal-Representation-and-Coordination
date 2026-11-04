"""Synthetic arithmetic tests only; no fixtures are presented as scientific results."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from trc.analysis.flash.metrics import evaluate, coefficient_metrics, joint_time_diagnostics
from trc.analysis.flash.report import build_report
from trc.io import file_sha256


class FlashAttribution(unittest.TestCase):
    def setUp(self):
        self.a=np.zeros((2,50,58));self.o=self.a+.1;self.p=self.a+.3

    def test_known_error_and_nonadditive_mse(self):
        result=evaluate(self.a,self.o,self.p)
        full=[r for r in result['groups'] if r['group']=='full_action']
        self.assertAlmostEqual(full[0]['rmse'],.1)
        self.assertAlmostEqual(full[1]['rmse'],.2)
        self.assertAlmostEqual(full[2]['rmse'],.3)
        d=result['mse_decomposition']
        self.assertAlmostEqual(d['total_mse'],d['representation_mse']+d['prediction_mse']+d['cross_term'])
        self.assertNotAlmostEqual(d['total_mse'],d['representation_mse']+d['prediction_mse'])

    def test_cancellation(self):
        result=evaluate(self.a,self.o,self.a)
        d=result['mse_decomposition']
        self.assertEqual(d['total_mse'],0)
        self.assertLess(d['cross_term'],0)

    def test_actual_joint_groups(self):
        p=self.a.copy();p[:,:,29:36]=1
        r=evaluate(self.a,self.a,p)
        rows={v['group']:v for v in r['groups'] if v['method']=='total'}
        self.assertEqual(rows['right_arm']['mae'],1)
        self.assertEqual(rows['left_arm']['mae'],0)
        self.assertEqual(rows['all_arms']['mae'],.5)
        self.assertEqual(len(r['temporal']),3*7*50)

    def test_mask_applies_identically(self):
        p=self.p.copy();p[0,0,0]=float('nan');mask=np.ones(p.shape,bool);mask[0,0,0]=False
        r=evaluate(self.a,self.o,p,mask)
        self.assertEqual([v for v in r['groups'] if v['group']=='full_action'][0]['count'],5799)
        with self.assertRaises(ValueError):evaluate(self.a,self.o,p)
        with self.assertRaises(ValueError):evaluate(self.a,self.o,self.p,np.zeros(p.shape,bool))

    def test_alignment_shape(self):
        with self.assertRaises(ValueError):evaluate(np.zeros((2,54,58)),self.o,self.p)

    def test_degree_scale_is_explicit(self):
        rows=coefficient_metrics(np.zeros((2,7,58)),np.ones((2,7,58)),np.arange(1,8),np.zeros(58),np.ones(58)*2)
        indexed={(r['degree'],r['group'],r['space']):r for r in rows}
        self.assertEqual(indexed[(6,'full_action','raw_normalized')]['mae'],7)
        self.assertEqual(indexed[(0,'full_action','scaled_training')]['mae'],1)
        self.assertEqual(len(rows),7*7*3)

    def test_time_differences_do_not_imply_coordination_failure(self):
        rows=joint_time_diagnostics(self.a,self.p,np.ones(self.a.shape,bool))
        self.assertTrue(all(r['rmse']==0 for r in rows)) # constant bias has no speed error
        p=self.a.copy();p[:,1:]=1
        rows=joint_time_diagnostics(self.a,p,np.ones(self.a.shape,bool))
        self.assertEqual(rows[0]['rmse'],30)

    def test_report_rejects_synthetic_and_tampered_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);export=root/'export';export.mkdir()
            metadata={'data_kind':'synthetic','action_semantics':'absolute','action_unit':'rad'}
            (export/'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'real'):build_report(export,root/'result',make_figures=False)
            metadata.update(data_kind='real',horizon_ticks=[0,49],K=6,windows=[{},{}])
            (export/'metadata.json').write_text(json.dumps(metadata))
            np.savez(export/'aligned.npz',truth=self.a)
            (export/'seal.json').write_text(json.dumps({'aligned_sha256':'wrong','metadata_sha256':'wrong'}))
            with self.assertRaisesRegex(ValueError,'seal'):build_report(export,root/'result',make_figures=False)
