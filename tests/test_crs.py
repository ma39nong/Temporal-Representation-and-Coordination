import tempfile
import unittest
from pathlib import Path
import numpy as np
from trc.analysis.flash.crs import selected_npz_rows,mix_actions,oracle_weights,StateReference,source_balanced_weights,fit_threshold


class CRS(unittest.TestCase):
    def test_allowlist_does_not_deserialize_excluded_nan_rows(self):
        with tempfile.TemporaryDirectory() as t:
            a=np.ones((5,8,58),np.float32);a[1:4]=np.nan;p=Path(t)/'data.npz';np.savez(p,history=a)
            selected=selected_npz_rows(p,'history',[4,0]);self.assertTrue(np.isfinite(selected).all())
            np.testing.assert_array_equal(selected,a[[4,0]])
            with self.assertRaises(ValueError):selected_npz_rows(p,'history',[0,0])
            with self.assertRaises(ValueError):selected_npz_rows(p,'history',[5])

    def test_chunk_scalar_and_derivative(self):
        a=np.zeros((2,50,58));b=np.broadcast_to(np.arange(50)[None,:,None],a.shape)
        x=mix_actions(a,b,[0,.25]);np.testing.assert_array_equal(x[0],a[0]);np.testing.assert_allclose(x[1],b[1]*.25)
        with self.assertRaises(ValueError):mix_actions(a,b,np.zeros((2,50)))
        with self.assertRaises(ValueError):mix_actions(a,b,1.1)

    def test_oracle_binary_is_not_bound_for_all_convex_fusions(self):
        a=np.ones((2,50,58))*-1;b=-a;gt=np.zeros_like(a)
        binary,continuous=oracle_weights(gt,a,b)
        np.testing.assert_array_equal(binary,[0,0]);np.testing.assert_allclose(continuous,[.5,.5])
        self.assertEqual(float(np.mean(mix_actions(a,b,continuous)**2)),0)

    def test_selector_features_have_no_label_or_identity_arguments(self):
        h=np.ones((3,8,58));ref=StateReference(h);scores=ref.score(h)
        self.assertEqual(set(scores),{'state_z','state_nn','history_motion'})
        np.testing.assert_array_equal(scores['state_nn'],np.zeros(3))
        with self.assertRaises(ValueError):ref.score(h[:,:7])

    def test_source_and_cohort_balance(self):
        w=source_balanced_weights(['a','a','b','c'],['training','training','training','development'])
        np.testing.assert_allclose(w,[.125,.125,.25,.5])

    def test_threshold_retains_nominal_constraint_using_training_only(self):
        threshold=fit_threshold([0,1,10],[.01,.01,1],[1,1,.01],[.25,.25,.5],[True,True,False])
        np.testing.assert_array_equal(np.array([0,1,10])>threshold,[False,False,True])
