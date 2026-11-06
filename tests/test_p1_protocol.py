import copy,unittest,runpy
from pathlib import Path
import numpy as np
from trc.experiments.p1_protocol import validate_protocol
from trc.analysis.flash.reference import lag_profile


def fixture():
 return {'shared':{'num_train_steps':500,'batch_size':256,'fsdp_devices':8,'train_dtype':'bfloat16','model_seed':42,'data_seed':42001,'augmentation_seed':42002,'flow_seed':42},
 'trials':{'B0':{'coordinate_mode':'absolute'},'B2':{'coordinate_mode':'state_relative'}},
 'evaluation':{'dtype':'float32','matmul_precision':'highest','batch_size':1,'num_steps':1,'padding':False}}

class Protocol(unittest.TestCase):
 def test_matched_protocol(self):self.assertEqual(len(validate_protocol(fixture())),64)
 def test_reject_unfair_trial_override(self):
  c=fixture();c['trials']['B2']['learning_rate']=1e-3
  with self.assertRaises(ValueError):validate_protocol(c)
 def test_reject_wrong_evaluation_and_budget(self):
  for key,value in [('dtype','bfloat16'),('batch_size',2),('padding',True)]:
   c=fixture();c['evaluation'][key]=value
   with self.assertRaises(ValueError):validate_protocol(c)
  c=fixture();c['shared']['num_train_steps']=8000
  with self.assertRaises(ValueError):validate_protocol(c)
 def test_reject_loss_override(self):
  c=fixture();c['shared']['consistency_weight']=2
  with self.assertRaises(ValueError):validate_protocol(c)

class Entrypoint(unittest.TestCase):
 def test_spawn_import_does_not_parse_args_or_start_training(self):
  # Spawn workers import __mp_main__; they must not rerun preflight/launch.
  script=Path(__file__).resolve().parents[1]/'scripts/prepare_p1_training_source.py'
  namespace=runpy.run_path(str(script),run_name='__mp_main__')
  self.assertTrue(callable(namespace['main']))

class Lag(unittest.TestCase):
 def test_positive_lag_means_feedback_follows_command(self):
  rng=np.random.default_rng(0);a=rng.normal(size=(120,58));s=np.roll(a,2,axis=0)
  rows=lag_profile(a,s,list(range(7)),max_lag=5);best=min(rows,key=lambda r:r['position_rmse_rad'])
  self.assertEqual(best['lag_frames'],2);self.assertEqual(best['position_rmse_rad'],0)
  self.assertAlmostEqual(best['increment_correlation'],1)
 def test_static_signals_do_not_identify_dynamic_lag(self):
  a=np.ones((60,58));self.assertTrue(all(r['increment_correlation'] is None for r in lag_profile(a,a,list(range(7)))))
 def test_bad_inputs(self):
  with self.assertRaises(ValueError):lag_profile(np.zeros((10,58)),np.zeros((10,58)),[0],10)
