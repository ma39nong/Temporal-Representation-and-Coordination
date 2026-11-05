import unittest
import numpy as np
from trc.analysis.flash.diagnosis import coefficient_contributions,temporal_structure,tick_zero_alignment

class Diagnosis(unittest.TestCase):
 def test_exact_degree_contribution_and_repair(self):
  b=np.ones((50,7));b[:,1]=np.linspace(-1,1,50);c=np.zeros((2,7,58));p=c.copy();p[:,1]=.2
  gt=np.zeros((2,50,58));pred=np.einsum('tk,nkj->ntj',b,p)
  result=coefficient_contributions(gt,gt,pred,c,p,np.ones(7),b,-np.ones(58),np.ones(58)-1e-6)
  row=next(r for r in result['rows'] if r['degree']==1 and r['group']=='full_action')
  self.assertAlmostEqual(row['oracle_repair_final_rmse_rad'],0)
  self.assertGreater(row['oracle_repair_final_mse_reduction'],0)
 def test_cancellation_is_not_positive_importance(self):
  b=np.ones((50,7));c=np.zeros((1,7,58));p=c.copy();p[:,0]=1;p[:,1]=-1
  a=np.zeros((1,50,58));r=coefficient_contributions(a,a,a,c,p,np.ones(7),b,-np.ones(58),np.ones(58)-1e-6)
  rows=[x for x in r['rows'] if x['group']=='full_action']
  self.assertTrue(all(abs(x['signed_allocated_prediction_mse'])<1e-10 for x in rows))
  self.assertGreater(rows[0]['self_mse'],0)
  self.assertLess(rows[0]['oracle_repair_final_mse_reduction'],0)
 def test_invalid_decode_is_rejected(self):
  a=np.zeros((1,50,58));c=np.zeros((1,7,58))
  with self.assertRaises(AssertionError):coefficient_contributions(a,a,a+1,c,c,np.ones(7),np.ones((50,7)),-np.ones(58),np.ones(58))
 def test_temporal_and_tick0(self):
  a=np.zeros((2,50,58));p=a.copy();p[:,30:]=1
  _,rows=temporal_structure(a,p)
  self.assertEqual(rows[-1]['largest_error_tick'],30)
  h=np.zeros((2,8,58));s=np.zeros((2,58));r=tick_zero_alignment(a,p,h,s)
  self.assertEqual(r[0]['rmse'],0)
  h[:,-1]=1
  with self.assertRaises(AssertionError):tick_zero_alignment(a,p,h,s)
