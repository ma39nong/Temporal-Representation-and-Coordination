import unittest
import numpy as np
from trc.models.velocity_readout import adapter_shapes,initialize_adapter,adapter_count,residual,strict_extend_flat


class VelocityReadoutTests(unittest.TestCase):
    def test_counts_and_zero_identity(self):
        self.assertEqual(adapter_count('none'),0)
        self.assertEqual(adapter_count('unified'),11902)
        self.assertEqual(adapter_count('order_specific'),11440)
        h=np.random.default_rng(1).normal(size=(2,7,1024)).astype(np.float32)
        for mode in ('unified','order_specific'):
            p=initialize_adapter(mode)
            self.assertGreater(np.linalg.norm(p['adapter_down']),0)
            np.testing.assert_array_equal(residual(h,p['adapter_down'],p['adapter_up'],mode),np.zeros((2,7,58)))

    def test_order_routing_and_shared_control(self):
        h=np.ones((1,7,1024),np.float32)
        for mode in ('unified','order_specific'):
            p=initialize_adapter(mode);u=p['adapter_up'];u[...] = 1
            if mode=='order_specific':u[3]=2
            y=residual(h,p['adapter_down'],u,mode)
            np.testing.assert_allclose(y[:,0],y[:,1])
            if mode=='order_specific':np.testing.assert_allclose(y[:,3],2*y[:,0],rtol=1e-6)

    def test_strict_original_coverage(self):
        old={('head','kernel'):np.ones((2,3),np.float32)}
        p=initialize_adapter('order_specific');ref={**old,**{('action_out_proj',k):v for k,v in p.items()}}
        ext=strict_extend_flat(ref,old,'order_specific')
        self.assertIs(ext[('head','kernel')],old[('head','kernel')])
        for bad in ({},{**old,('unexpected',):np.zeros(1)}, {('head','kernel'):np.ones((2,3),np.float64)}):
            with self.assertRaises(ValueError):strict_extend_flat(ref,bad,'order_specific')

    def test_layout_and_seed(self):
        a=initialize_adapter('unified',42);b=initialize_adapter('unified',42)
        np.testing.assert_array_equal(a['adapter_down'],b['adapter_down'])
        with self.assertRaises(ValueError):adapter_shapes('bad')
        with self.assertRaises(ValueError):residual(np.zeros((1,50,1024)),a['adapter_down'],a['adapter_up'],'unified')
