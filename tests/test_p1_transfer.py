import copy,json,runpy,unittest
from pathlib import Path
from trc.experiments.p1_transfer import validate_transfer_protocol
ROOT=Path(__file__).resolve().parents[1]


def fixture():
    c= {'shared':{'num_train_steps':500,'batch_size':256,'fsdp_devices':8,'train_dtype':'bfloat16','model_seed':42,'data_seed':42001,'augmentation_seed':42002,'flow_seed':42},'trials':{'B0':{'coordinate_mode':'absolute'},'B2':{'coordinate_mode':'state_relative'}},'evaluation':{'dtype':'float32','matmul_precision':'highest','batch_size':1,'num_steps':1,'padding':False},'expected_optimizer':{'b1':.9,'b2':.95,'eps':1e-8,'weight_decay':1e-10,'clip_gradient_norm':1.0}}
    c['experiment_kind']='step8000_transfer';c['optimizer_state_policy']='reset_both';c['shared']['num_train_steps']=250
    c['expected_schedule']={'warmup_steps':25,'peak_lr':2.5e-6,'decay_steps':250,'decay_lr':2.5e-6}
    c['initial_parameter_sha256']='40b2c2c4dcdd6fb88dd847d5762d2071fba5028af0942c6dfe457409fe5f5919'
    return c


class Transfer(unittest.TestCase):
    def test_explicit_transfer_is_valid_but_old_gate_stays_strict(self):
        self.assertEqual(len(validate_transfer_protocol(fixture())),64)
        from trc.experiments.p1_protocol import validate_protocol
        with self.assertRaises(ValueError):validate_protocol(fixture())

    def test_reject_asymmetric_or_wrong_initialization_and_budget(self):
        for key,val in [('optimizer_state_policy','restore_only_B0'),('initial_parameter_sha256','wrong')]:
            c=fixture();c[key]=val
            with self.assertRaises(ValueError):validate_transfer_protocol(c)
        for key,val in [('num_train_steps',500),('data_seed',42003),('batch_size',128)]:
            c=fixture();c['shared'][key]=val
            with self.assertRaises(ValueError):validate_transfer_protocol(c)
        c=fixture();c['trials']['B2']['learning_rate']=1e-5
        with self.assertRaises(ValueError):validate_transfer_protocol(c)

    def test_prepare_import_does_not_start_training(self):
        for name in ('prepare_p1_transfer_source.py','init_transfer_tracking_source.py'):
            ns=runpy.run_path(str(ROOT/'scripts'/name),run_name='__mp_main__')
            self.assertTrue(callable(ns['main']))
