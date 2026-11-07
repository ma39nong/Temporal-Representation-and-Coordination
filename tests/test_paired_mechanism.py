import unittest
import numpy as np
from trc.analysis.flash.diagnosis import paired_component_gain


class PairedMechanism(unittest.TestCase):
    def test_nonorthogonal_components_and_oracle_interaction(self):
        rng=np.random.default_rng(7)
        a=rng.normal(size=(3,5,2));o=a+.03
        c0=rng.normal(size=(3,7,5,2));c2=rng.normal(size=c0.shape)
        p0=o+c0.sum(1)+1e-7;p2=o+c2.sum(1)-2e-7
        r=paired_component_gain(a,o,p0,p2,c0,c2)
        self.assertAlmostEqual(r['signed_final_mse_gain'].sum()+r['roundoff_gain'],
                               np.mean((p0-a)**2)-np.mean((p2-a)**2),places=12)
        self.assertGreater(r['decode_remainder_max_rad'],0)

    def test_cancellation_allows_negative_gain(self):
        a=np.zeros((1,5,2));c0=np.zeros((1,2,5,2));c2=c0.copy()
        c0[:,0]=2;c0[:,1]=-1;c2[:,0]=1
        r=paired_component_gain(a,a,a+1,a+1,c0,c2)
        np.testing.assert_allclose(r['signed_final_mse_gain'],[1,-1])
        self.assertEqual(r['total_mse_gain'],0)

    def test_bad_decode_shape_and_nan_rejected(self):
        a=np.zeros((1,5,2));c=np.zeros((1,2,5,2))
        for bad in (a+1,np.full_like(a,np.nan),np.zeros((2,5,2))):
            with self.assertRaises(ValueError):paired_component_gain(a,a,bad,a,c,c)
