import unittest
from copy import deepcopy
from trc.models.task_condition import TaskCondition,causal_condition,compact_prompt,parse_compact_prompt,controller_request,task_prompt
from trc.data.causal_supervision import event_export_interval,reviewed_progress,validate_window


class CausalSupervisionTest(unittest.TestCase):
    def clock(self):return {'frames':400,'export_start_ns':1_000_000_000,'cut_rows':[200],
        'mapping':[{'source_start_ns':'7666666667','source_end_ns':'8666666667','removed_ns':'1000000000'}]}
    def review(self):return {'recording_id':'r','goal':{'review_status':'verified','task_goal':'harvest_all','target_count':None},
        'coverage':{'review_status':'verified','reset_at':0,'complete_until':12,'evidence':'reviewed prefix and reset',
                    'deleted_intervals_reviewed':True},
        'events':[{'recording_id':'r','event_id':'e','event_type':'pick_success','object_id':'fruit1',
            'event_timestamp':[1.,1.2],'timestamp_domain':'export_seconds','evidence':'prefix-visible detach',
            'review_status':'verified','verified_by':'human','causal_availability_verified':True,
            'confirmation_available_at':[1.2,1.5],'confirmation_timestamp_domain':'export_seconds'}]}

    def test_compact_all_and_unknown_roundtrip(self):
        c=causal_condition(task_goal='harvest_all',target_count=None,observation_time=0)
        self.assertEqual(compact_prompt(c),'Right truss left pick basket */? ?0')
        self.assertEqual(parse_compact_prompt(compact_prompt(c),observation_time=0),c)
        self.assertIsNone(c.remaining_count)
        with self.assertRaises(ValueError):task_prompt(c,'goal_progress')
        self.assertEqual(controller_request(c),'continue_or_review')
        with self.assertRaises(ValueError):causal_condition(task_goal='harvest_all',target_count=3,observation_time=0)

    def test_numeric_progress_and_default_single(self):
        c=causal_condition(target_count=1,observation_time=0,reset_at=0,reset_evidence='e',coverage_until=0)
        self.assertEqual(c.remaining_count,1)
        for n in (1,2,3,9):
            c=TaskCondition('pick_tomatoes',n,1,True,'harvest',False,4,('e',))
            self.assertEqual(parse_compact_prompt(compact_prompt(c),observation_time=4,evidence=c.evidence),c)
            self.assertEqual(c.remaining_count,n-1)
        with self.assertRaises(ValueError):compact_prompt(TaskCondition('pick_tomatoes',10,None,False,None,False,0))
        with self.assertRaises(ValueError):parse_compact_prompt('Right truss left pick basket 3/0 h0 EXTRA',observation_time=0)

    def test_interval_upper_bound_and_confirmation_delay(self):
        r=self.review();clock=self.clock()
        # Frame40=1.333: event happened, but confirmation only at1.5. No future leak.
        self.assertEqual(reviewed_progress(r,40,clock)['condition'].completed_count,0)
        self.assertEqual(reviewed_progress(r,45,clock)['condition'].completed_count,1)
        # Modifying future events must not change the current snapshot.
        extra=deepcopy(r['events'][0]);extra.update(event_id='future',object_id='fruit2',event_timestamp=[2,2.5],confirmation_available_at=3)
        r['events'].append(extra)
        self.assertEqual(reviewed_progress(r,45,clock)['condition'].completed_count,1)

    def test_offline_or_pending_is_not_progress(self):
        for change in ({'review_status':'pending_human'},{'verified_by':None},
                       {'causal_availability_verified':False},{'confirmation_available_at':None}):
            r=self.review();r['events'][0].update(change)
            c=reviewed_progress(r,45,self.clock())['condition']
            self.assertIsNone(c.completed_count)
        # A zero count is only valid if a COMPLETE reviewed prefix is separately supplied.
        r=self.review();r['coverage']['review_status']='pending_human'
        self.assertIsNone(reviewed_progress(r,45,self.clock())['condition'].completed_count)

    def test_raw_clock_inverse_and_deleted_event_rejected(self):
        clock=self.clock();row={'event_timestamp':['9000000000','9100000000'],'timestamp_domain':'raw_recording_ns'}
        self.assertEqual(event_export_interval(row,clock),(7.,7.1))
        row['event_timestamp']=['7800000000','7900000000']
        with self.assertRaises(ValueError):event_export_interval(row,clock)
        with self.assertRaises(ValueError):event_export_interval({'event_timestamp':[-1,1],'timestamp_domain':'export_seconds'},clock)

    def test_cut_and_episode_support_guard(self):
        clock=self.clock()
        for frame in (0,6,149,200,206,349,399):
            with self.assertRaises(ValueError):validate_window(frame,clock)
        for frame in (7,148,207,348):validate_window(frame,clock)
        with self.assertRaises(ValueError):event_export_interval({'event_timestamp':[6,7],'timestamp_domain':'export_seconds'},clock)

    def test_gap_coverage_is_required_for_global_count(self):
        r=self.review();r['coverage']['deleted_intervals_reviewed']=False
        c=reviewed_progress(r,207,self.clock())['condition']
        self.assertFalse(c.progress_valid);self.assertIsNone(c.completed_count)

    def test_future_pending_raw_event_does_not_invalidate_present(self):
        r=self.review();extra=deepcopy(r['events'][0]);extra.update(event_id='rawfuture',object_id='future',
            timestamp_domain='raw_recording_ns',event_timestamp=['12000000000','12100000000'],review_status='pending_human')
        r['events'].append(extra)
        self.assertEqual(reviewed_progress(r,45,self.clock())['condition'].completed_count,1)

    def test_wrong_recording_and_availability_rejected(self):
        r=self.review();r['events'][0]['recording_id']='other'
        with self.assertRaises(ValueError):reviewed_progress(r,45,self.clock())
        r=self.review();r['events'][0]['confirmation_available_at']=1.
        with self.assertRaises(ValueError):reviewed_progress(r,45,self.clock())
        r=self.review();r['events'][0]['confirmation_timestamp_domain']='raw_recording_ns'
        with self.assertRaises(ValueError):reviewed_progress(r,45,self.clock())


if __name__=='__main__':unittest.main()
