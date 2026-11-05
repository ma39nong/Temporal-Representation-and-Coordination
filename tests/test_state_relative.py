import unittest
import numpy as np
from trc.models.flash_coordinates import FlashCoordinates
from trc.data.action_schema import GROUPS
from trc.data.normalization import normalize
from trc.analysis.flash.offset import decompose_offsets,grouped_offsets,anchor_correction

class StateRelative(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(42);self.h=rng.normal(size=(3,8,58));self.a=rng.normal(size=(3,50,58));self.c=rng.normal(size=(3,7,58))
  self.lo=np.linspace(-2,.2,58);self.hi=self.lo+np.linspace(.1,4,58);self.s=np.arange(1,8)/10
  self.adapter=FlashCoordinates('state_relative',self.lo,self.hi,self.s,tuple(GROUPS))
 def test_physical_roundtrip_and_derivative(self):
  r=self.adapter.actions(self.a,self.h)
  np.testing.assert_allclose(self.adapter.actions(r,self.h,inverse=True),self.a,atol=1e-14)
  np.testing.assert_allclose(np.diff(r,axis=1),np.diff(self.a,axis=1),atol=1e-14)
 def test_coefficients_only_zero_degree_changes(self):
  c=self.adapter.coefficients(self.c,self.h)
  np.testing.assert_array_equal(c[:,1:],self.c[:,1:])
  np.testing.assert_allclose(self.adapter.coefficients(c,self.h,inverse=True),self.c,atol=1e-14)
 def test_normalization_order_and_no_offset_in_relative_space(self):
  stats={'q01':self.lo,'q99':self.hi};r=self.adapter.actions(self.a,self.h)
  x=self.adapter.normalize_coordinates(r);absolute=normalize(self.a,stats);anchor=normalize(self.h[:,-1],stats)
  np.testing.assert_allclose(x,absolute-anchor[:,None],atol=2e-14)
  self.assertGreater(np.max(np.abs(normalize(r,stats)-x)),.1)
  np.testing.assert_allclose(self.adapter.normalize_coordinates(x,inverse=True),r,atol=1e-14)
 def test_50_step_decode_invariance(self):
  from numpy.polynomial.legendre import legvander
  B=legvander(np.linspace(-1,1,50),6)
  norm_abs=np.einsum('tk,nkj,k->ntj',B,self.c,self.s)
  c_rel=self.adapter.coefficients(self.c,self.h)
  norm_rel=np.einsum('tk,nkj,k->ntj',B,c_rel,self.s)
  absolute=normalize(norm_abs,{'q01':self.lo,'q99':self.hi},inverse=True)
  recovered=self.adapter.actions(self.adapter.normalize_coordinates(norm_rel,inverse=True),self.h,inverse=True)
  np.testing.assert_allclose(recovered,absolute,atol=1e-13)
 def test_same_reference_preserves_velocity_noise_and_mixture(self):
  target=self.c*.9;source=self.c+.17
  s=self.adapter.coefficients(source,self.h);t=self.adapter.coefficients(target,self.h)
  np.testing.assert_allclose(t-s,target-source,atol=1e-14)
  tau=.37
  np.testing.assert_allclose((1-tau)*s+tau*t,self.adapter.coefficients((1-tau)*source+tau*target,self.h),atol=1e-14)
 def test_absolute_path_is_exact_and_does_not_force_command0(self):
  adapter=FlashCoordinates('absolute',self.lo,self.hi,self.s,tuple(GROUPS))
  np.testing.assert_array_equal(adapter.actions(self.a,self.h),self.a)
  np.testing.assert_array_equal(adapter.coefficients(self.c,self.h),self.c)
  self.assertGreater(np.max(np.abs(self.adapter.actions(self.a,self.h)[:,0])),.1)
 def test_partial_compatibility_never_transforms_other_groups(self):
  adapter=FlashCoordinates('state_relative',self.lo,self.hi,self.s,('left_arm','right_arm'))
  r=adapter.actions(self.a,self.h);np.testing.assert_array_equal(r[...,7:29],self.a[...,7:29])
  norm=adapter.normalize_coordinates(r);absolute=normalize(self.a,{'q01':self.lo,'q99':self.hi})
  np.testing.assert_allclose(norm[...,7:29],absolute[...,7:29])
 def test_reject_undefined_mode_group_stats_and_history(self):
  for mode,groups in [('delta',tuple(GROUPS)),('state_relative',()),('state_relative',('finger',))]:
   with self.assertRaises(ValueError):FlashCoordinates(mode,self.lo,self.hi,self.s,groups)
  with self.assertRaises(ValueError):FlashCoordinates('absolute',self.lo,self.lo-1,self.s,tuple(GROUPS))
  with self.assertRaises(ValueError):self.adapter.actions(self.a,self.h[:,:7])
  with self.assertRaises(ValueError):self.adapter.coefficients(self.c[:,:6],self.h)
  broken=self.a.copy();broken[0,0,0]=np.nan
  with self.assertRaises(ValueError):self.adapter.actions(broken,self.h)

class Offsets(unittest.TestCase):
 def test_exact_offset_shape_orthogonality(self):
  a=np.zeros((2,50,58));shape=np.linspace(-1,1,50)[None,:,None];p=a+2+shape
  d=decompose_offsets(a,p);r=grouped_offsets(d)[-1]
  self.assertAlmostEqual(r['offset_mse'],4)
  self.assertAlmostEqual(r['shape_mse'],float(np.mean(shape**2)))
  np.testing.assert_allclose(d['residual'].mean(1),0,atol=1e-14)
 def test_zero_error_has_no_defined_fraction(self):
  a=np.zeros((1,50,58));self.assertIsNone(grouped_offsets(decompose_offsets(a,a))[-1]['offset_fraction'])
 def test_b1_is_causal_but_not_gt0_and_keeps_derivatives(self):
  p=np.broadcast_to(np.arange(50)[None,:,None],(1,50,58));h=np.ones((1,8,58))*3
  b=anchor_correction(p,h);np.testing.assert_array_equal(b[:,0],h[:,-1]);np.testing.assert_array_equal(np.diff(b,axis=1),np.diff(p,axis=1))
 def test_bad_physical_actions_rejected(self):
  with self.assertRaises(ValueError):decompose_offsets(np.zeros((1,49,58)),np.zeros((1,49,58)))
