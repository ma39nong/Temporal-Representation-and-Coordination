import unittest
import numpy as np
from trc.data.qualification import state_stratum,inverse_compact_time,coverage_cell
from trc.data.action_schema import GROUPS


class Qualification(unittest.TestCase):
    def setUp(self):
        self.limits=[{'group':g,'training_LOO_nn_q95':.1,'training_LOO_nn_q99':.3} for g in [*GROUPS,'full_action']]
    def test_same_reference_required_for_near(self):
        ref=np.zeros((2,58));ref[0,29:]=1;ref[1,:29]=1
        r=state_stratum(np.zeros(58),ref,np.zeros(58),np.ones(58),self.limits)
        self.assertEqual(r['nearest_distances']['left_arm'],0);self.assertEqual(r['nearest_distances']['right_arm'],0)
        self.assertNotEqual(r['class'],'near')
    def test_near_far_and_ambiguous(self):
        for value,expected in [(0,'near'),(.2,'ambiguous'),(1,'far')]:
            self.assertEqual(state_stratum(np.full(58,value),np.zeros((2,58)),np.zeros(58),np.ones(58),self.limits)['class'],expected)
    def test_invalid_state_not_force_classified(self):
        with self.assertRaises(ValueError):state_stratum(np.full(58,np.nan),np.zeros((1,58)),np.zeros(58),np.ones(58),self.limits)
    def test_inverse_mapping_keeps_seam(self):
        mapping=[{'source_start_ns':30,'source_end_ns':50,'removed_ns':20},{'source_start_ns':80,'source_end_ns':100,'removed_ns':20}]
        raw,cuts=inverse_compact_time(np.array([10,20,30,40,50,60,70]),mapping)
        np.testing.assert_array_equal(raw,[10,20,50,60,70,100,110]);self.assertEqual(cuts,[2,5])
    def test_bad_mapping_rejected(self):
        with self.assertRaises(ValueError):inverse_compact_time([1,2],[{'source_start_ns':3,'source_end_ns':5,'removed_ns':3}])
    def test_uncertain_success_cannot_fill_cell(self):
        for status in ('uncertain','failure'):
            self.assertIsNone(coverage_cell(status,'verified','verified','verified','verified','near','near','single_pick'))
    def test_all_gates_and_task_type_required(self):
        self.assertEqual(coverage_cell('success','verified','verified','verified','verified','far','different','single_pick'),'D')
        self.assertIsNone(coverage_cell('success','verified','unknown','verified','verified','near','near','single_pick'))
        self.assertIsNone(coverage_cell('success','verified','verified','verified','verified','near','near','continuous_multi'))


class ProvenanceProof(unittest.TestCase):
    @staticmethod
    def adapter():
        import importlib.util
        from pathlib import Path
        p=Path(__file__).resolve().parents[1]/'scripts/audit_recording_qualification.py'
        spec=importlib.util.spec_from_file_location('qualification_source_script',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

    def fixture(self):
        source='source';meta={'length':100,'source_start_ns':10,'source_end_ns':20}
        bag={'recording_id':source,'bag_path':'/raw/A/episode1','files':[{'name':'episode1.db3','exists':True,'bytes':100}], 'state_evidence':{'source_bag':'/capture/A/episode1'}}
        job={'source':'/capture/A/episode1','original_sha256_verified':True,'verification':{'status':'passed'},'readiness':{'volumes':[{'name':'episode1.db3','bytes':100}]}}
        docs={'summary.json':{'path':'summary','sha256':'test','document':{'jobs':[job],'dataset_verification_passed':True}},'trim_report.json':{'path':'trim','sha256':'test','document':{'timestamps':'compact_shared_time_mapping','mapping':[]}}}
        proof={'source_episode_name':'episode1','proof_files':docs}
        return source,meta,{'bags':[bag],'exports':[{'root':'/export','episodes':[{'source_recording_id':source,**meta}],'source_proofs':[proof]}]}

    def test_complete_link_uses_prior_hash_verification(self):
        source,meta,raw=self.fixture();self.assertEqual(self.adapter().lineage(source,meta,raw)['status'],'verified')
    def test_same_id_with_different_crop_is_not_verified(self):
        source,meta,raw=self.fixture();meta={**meta,'length':99};self.assertEqual(self.adapter().lineage(source,meta,raw)['status'],'unknown')
    def test_missing_mapping_or_raw_size_change_not_verified(self):
        source,meta,raw=self.fixture();raw['bags'][0]['files'][0]['bytes']=101
        self.assertEqual(self.adapter().lineage(source,meta,raw)['status'],'unknown')

        source,meta,raw=self.fixture();del raw['exports'][0]['source_proofs'][0]['proof_files']['trim_report.json']
        self.assertEqual(self.adapter().lineage(source,meta,raw)['status'],'unknown')
