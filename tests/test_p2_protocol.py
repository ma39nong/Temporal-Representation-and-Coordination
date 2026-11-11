import copy,json,unittest
from pathlib import Path
from trc.data.flash_sampling import ContinuousFlashIndex
from trc.data.qualification import flash_window_cuts
from trc.experiments.p2_protocol import validate_order_protocol


class ContinuousSamplerTests(unittest.TestCase):
    def fixture(self):
        records=[{'id':'r','rows':180,'source_recording_id':'raw'}]
        manifest={'records':[{'id':'r','rows':180,'source_recording_id':'raw','cuts':[80], 'legal_windows':64}],'legal_windows':64}
        return records,manifest

    def test_all_positions_are_unique_and_continuous(self):
        r,m=self.fixture();a=ContinuousFlashIndex(r,m,42);b=ContinuousFlashIndex(r,m,42)
        values=[a.index(i) for i in range(a.total)]
        self.assertEqual(len({v['start'] for v in values}),64)
        self.assertTrue(all(not flash_window_cuts(v['start'],[80]) for v in values))
        self.assertEqual(values,[b.index(i) for i in range(b.total)])
        self.assertEqual(a.index(64),b.index(64))

    def test_missing_or_misbound_proof_rejected(self):
        r,m=self.fixture()
        for mutate in [lambda x:x['records'].clear(),lambda x:x['records'][0].update(source_recording_id='other'),
                       lambda x:x['records'][0].update(legal_windows=63)]:
            bad=copy.deepcopy(m);mutate(bad)
            with self.assertRaises(ValueError):ContinuousFlashIndex(r,bad,42)

    def test_real_protocol_rejects_per_trial_overrides(self):
        cfg=json.loads((Path(__file__).parents[1]/'configs/p2_2_velocity_readout.json').read_text())
        self.assertEqual(len(validate_order_protocol(cfg)),64)
        for key,value in [('freeze_vlm',False),('preserve_frozen_parameter_dtype',False),('batch_size',128)]:
            bad=copy.deepcopy(cfg);bad['shared'][key]=value
            with self.assertRaises(ValueError):validate_order_protocol(bad)
        cfg['trials']['B3']['learning_rate']=1e-3
        with self.assertRaises(ValueError):validate_order_protocol(cfg)
