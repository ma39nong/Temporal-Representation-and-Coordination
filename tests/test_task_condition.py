import unittest
from dataclasses import dataclass
import numpy as np
from trc.models.task_condition import (ConfirmedEvent, TaskCondition, causal_condition,
    task_prompt, controller_request, checked_tokens, condition_observation)


class TaskConditionTest(unittest.TestCase):
    def snapshot(self, **kw):
        return causal_condition(**{'target_count':2,'observation_time':5.,'reset_at':0.,
            'reset_evidence':'reviewed reset','coverage_until':5.,**kw})

    def test_unknown_is_not_zero(self):
        c=causal_condition(target_count=2,observation_time=5)
        self.assertFalse(c.progress_valid);self.assertIsNone(c.completed_count)
        self.assertIn('unknown',task_prompt(c,'goal_progress'))

    def test_only_visible_success_counts(self):
        events=[ConfirmedEvent('1','pick_success',1,2,'confirmation','fruit1'),
                ConfirmedEvent('2','placement',2,3,'placement','fruit1'),
                ConfirmedEvent('3','retry',3,3,'retry'),
                ConfirmedEvent('4','pick_success',4,6,'delayed confirmation','fruit2'),
                ConfirmedEvent('5','pick_success',6,6,'future confirmation','fruit3')]
        self.assertEqual(self.snapshot(events=events).completed_count,1)
        self.assertEqual(self.snapshot(events=events,observation_time=6,coverage_until=6).completed_count,3)

    def test_repeated_object_not_recounted(self):
        events=[ConfirmedEvent('1','pick_success',1,1,'e','fruit'),ConfirmedEvent('2','pick_success',2,2,'e','fruit')]
        self.assertEqual(self.snapshot(events=events).completed_count,1)
        with self.assertRaises(ValueError):self.snapshot(events=[events[0],events[0]])

    def test_phase_is_confirmed_and_causal(self):
        e=ConfirmedEvent('1','phase',1,6,'confirmation',phase='harvest')
        self.assertIsNone(self.snapshot(events=[e]).current_phase)
        self.assertEqual(self.snapshot(events=[e],observation_time=6,coverage_until=6).current_phase,'harvest')

    def test_missing_or_stale_coverage_invalidates_count(self):
        for kw in ({'reset_evidence':None},{'coverage_until':4},{'reset_at':6}):
            self.assertFalse(self.snapshot(**kw).progress_valid)

    def test_reset_excludes_previous_task(self):
        e=ConfirmedEvent('old','pick_success',1,1,'e','fruit')
        self.assertEqual(self.snapshot(reset_at=2,events=[e]).completed_count,0)

    def test_goal_switch_preserves_real_count(self):
        e=ConfirmedEvent('e','pick_success',1,1,'e','fruit')
        c=self.snapshot(target_count=1,events=[e]);self.assertEqual(controller_request(c),'request_stop')
        self.assertEqual(controller_request(self.snapshot(target_count=3,events=[e])),'continue_or_review')
        self.assertEqual(controller_request(self.snapshot(terminate_request=True)),'request_stop')

    def test_invalid_counts_and_clock(self):
        for n in (True,0,-1,1.5):
            with self.assertRaises(ValueError):self.snapshot(target_count=n)
        with self.assertRaises(ValueError):self.snapshot(observation_time=float('nan'))
        with self.assertRaises(ValueError):ConfirmedEvent('a','pick_success',2,1,'e','fruit')
        with self.assertRaises(ValueError):ConfirmedEvent('a','attempt',1,1,'e')
        with self.assertRaises(ValueError):ConfirmedEvent('a','pick_success',1,1,'e')
        with self.assertRaises(ValueError):TaskCondition('pick_tomatoes',1,0,False,None,False,1)

    def test_disabled_is_identity_and_no_future_argument(self):
        ob=object();self.assertIs(condition_observation(ob),ob)
        with self.assertRaises(TypeError):condition_observation(ob,target=np.zeros((50,58)))

    def test_tokens_and_budget_fail_closed(self):
        def encode(text,add_bos):return [1]+list(text.encode())
        a,m=checked_tokens('Goal',np.zeros(58),encode,512)
        self.assertEqual(a.shape,(512,));self.assertEqual(m.dtype,bool)
        with self.assertRaises(ValueError):checked_tokens('Goal',np.zeros(58),encode,8)
        for state in (np.zeros(54),np.full(58,np.inf)):
            with self.assertRaises(ValueError):checked_tokens('Goal',state,encode)

    def test_batch_conditions_and_modes(self):
        @dataclass
        class Ob:
            state: np.ndarray
            tokenized_prompt: object=None
            tokenized_prompt_mask: object=None
        ob=Ob(np.zeros((2,58)))
        cs=[self.snapshot(target_count=i) for i in (1,3)]
        for mode in ('language_goal','explicit_goal','goal_progress'):
            a=condition_observation(ob,cs,mode=mode,encode=lambda text,add_bos:list(text.encode()),max_len=512)
            self.assertEqual(a.tokenized_prompt.shape,(2,512))
            self.assertFalse(np.array_equal(a.tokenized_prompt[0],a.tokenized_prompt[1]))
            self.assertIs(a.state,ob.state)
        with self.assertRaises(ValueError):condition_observation(ob,cs[:1],mode='goal_progress')


if __name__=='__main__':unittest.main()
