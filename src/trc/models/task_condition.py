"""Causal task metadata through FLASH's existing language prefix; no new weights.

No robot commands, event detector, offline-phase oracle, or stop action is defined.
Timestamps share the observation clock. Evidence is supplied by an audited caller;
this module checks causality and consistency, not the truth of a physical event.
"""
from dataclasses import dataclass, replace
import math
from typing import Literal

import numpy as np

PHASES = ('pickup', 'harvest', 'return', 'adjust')


def _time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite timestamp required')


def _count(value, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError('Integer count out of range')


@dataclass(frozen=True)
class ConfirmedEvent:
    event_id: str
    kind: Literal['pick_success', 'placement', 'retry', 'phase']
    occurred_at: float
    available_at: float
    evidence: str
    object_id: str | None = None
    phase: str | None = None

    def __post_init__(self):
        _time(self.occurred_at); _time(self.available_at)
        if self.available_at < self.occurred_at or not self.event_id or not self.evidence:
            raise ValueError('Event requires consistent clocks, unique ID and confirmation evidence')
        if self.kind not in ('pick_success', 'placement', 'retry', 'phase'):
            raise ValueError('An attempt or offline final outcome is not a confirmed pick event')
        if self.kind in ('pick_success', 'placement') and not self.object_id:
            raise ValueError('Confirmed physical object identity required for counting')
        if self.kind == 'phase' and self.phase not in PHASES:
            raise ValueError('Unknown phase')
        if self.kind != 'phase' and self.phase is not None:
            raise ValueError('Phase needs its own confirmed event')


@dataclass(frozen=True)
class TaskCondition:
    task_goal: str
    target_count: int | None
    completed_count: int | None
    progress_valid: bool
    current_phase: str | None
    terminate_request: bool
    observation_time: float
    evidence: tuple[str, ...] = ()

    def __post_init__(self):
        _time(self.observation_time)
        if self.target_count is not None: _count(self.target_count, True)
        if self.task_goal not in ('pick_tomatoes', 'harvest_all'):
            raise ValueError('Only the reviewed tomato task vocabulary is supported')
        if self.task_goal == 'harvest_all' and self.target_count is not None:
            raise ValueError('Harvest-all does not imply a known numeric target')
        if type(self.progress_valid) is not bool or type(self.terminate_request) is not bool:
            raise ValueError('Boolean flags required')
        if self.progress_valid:
            _count(self.completed_count)
            if not self.evidence: raise ValueError('Reliable progress requires confirmation evidence')
        elif self.completed_count is not None or self.current_phase is not None:
            raise ValueError('Unknown progress cannot contain a count or phase')
        if self.current_phase is not None and self.current_phase not in PHASES:
            raise ValueError('Unknown phase')

    @property
    def remaining_count(self):
        if self.target_count is None or not self.progress_valid: return None
        return max(0, self.target_count-self.completed_count)


def causal_condition(*, target_count, observation_time, task_goal='pick_tomatoes', events=(), reset_at=None,
                     reset_evidence=None, coverage_until=None, terminate_request=False):
    """Count unique successful objects only with a complete confirmed event stream.

    A reviewed task reset establishes count zero. coverage_until is the last
    observation time for which the caller guarantees no confirmation is missing.
    Delayed/future confirmations are invisible, even for offline data preparation.
    Offline full-video phase labels must NOT be passed as confirmed online events.
    """
    _time(observation_time)
    events = tuple(events)
    if any(not isinstance(e, ConfirmedEvent) for e in events):
        raise ValueError('ConfirmedEvent required')
    if len({e.event_id for e in events}) != len(events):
        raise ValueError('Duplicate event IDs')
    if reset_at is not None: _time(reset_at)
    if coverage_until is not None: _time(coverage_until)
    valid = (reset_at is not None and reset_at <= observation_time and bool(reset_evidence)
             and coverage_until is not None and coverage_until >= observation_time)
    if not valid:
        return TaskCondition(task_goal, target_count, None, False, None,
                             terminate_request, observation_time)
    visible = sorted((e for e in events if reset_at <= e.occurred_at <= observation_time
                      and e.available_at <= observation_time), key=lambda e: (e.available_at, e.event_id))
    objects = {e.object_id for e in visible if e.kind == 'pick_success'}
    phases = [e.phase for e in sorted(visible, key=lambda e: (e.occurred_at, e.event_id))
              if e.kind == 'phase']
    return TaskCondition(task_goal, target_count, len(objects), True,
                         phases[-1] if phases else None, terminate_request, observation_time,
                         (reset_evidence, *(e.evidence for e in visible)))


def controller_request(condition):
    """Decision only. A stop request needs a separate validated safe controller."""
    return 'request_stop' if (condition.terminate_request or
        (condition.progress_valid and condition.target_count is not None and condition.completed_count >= condition.target_count)) else 'continue_or_review'


def task_prompt(condition, mode):
    if mode == 'compact_v1': return compact_prompt(condition)
    if condition.task_goal != 'pick_tomatoes' or condition.target_count is None:
        raise ValueError('Harvest-all/unknown target requires the versioned compact_v1 protocol')
    if mode not in ('language_goal', 'explicit_goal', 'goal_progress'):
        raise ValueError('Unknown condition mode')
    # These are language variants of the SAME existing prefix channel.
    if mode == 'language_goal':
        return f'Right hand holds truss; left hand picks {condition.target_count} tomatoes into left basket.'
    goal = f'Right hand holds truss; left hand picks into left basket. Target count: {condition.target_count}.'
    if mode == 'explicit_goal': return goal
    count = str(condition.completed_count) if condition.progress_valid else 'unknown'
    phase = condition.current_phase or 'unknown'
    return (goal + f' Completed count: {count}. Progress valid: {int(condition.progress_valid)}.'
            f' Phase: {phase}. Terminate request: {int(condition.terminate_request)}.')


def checked_tokens(prompt, normalized_state, encode, max_len=256):
    """Original pi0.5 formatting with a strict pre-truncation token-budget gate.

    encode is the explicitly supplied original SentencePiece encode method.
    No tokenizer, model dependency, or private path is imported here.
    """
    state = np.asarray(normalized_state)
    if state.shape != (58,) or not np.isfinite(state).all():
        raise ValueError('Finite normalized canonical58 state required')
    if type(max_len) is not int or max_len <= 0: raise ValueError('Positive token budget required')
    clean = prompt.strip().replace('_', ' ').replace('\n', ' ')
    bins = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
    text = f"Task: {clean}, State: {' '.join(map(str, bins))};\nAction: "
    tokens = list(encode(text, add_bos=True))
    if len(tokens) > max_len:
        raise ValueError(f'Condition would truncate state/action prefix: {len(tokens)} > {max_len}')
    length = len(tokens)
    return (np.asarray(tokens + [0] * (max_len-length), np.int32),
            np.arange(max_len) < length)


def condition_observation(observation, conditions=(), *, mode='disabled', encode=None, max_len=256):
    """Change only language tokens; history and targets never enter this interface."""
    if mode == 'disabled': return observation
    states = np.asarray(observation.state)
    if states.ndim != 2 or len(conditions) != len(states):
        raise ValueError('One causal condition per batch observation required')
    if any(not isinstance(c, TaskCondition) for c in conditions): raise ValueError('TaskCondition required')
    pairs = [checked_tokens(task_prompt(c, mode), s, encode, max_len) for c,s in zip(conditions, states)]
    return replace(observation, tokenized_prompt=np.stack([p[0] for p in pairs]),
                   tokenized_prompt_mask=np.stack([p[1] for p in pairs]))


COMPACT_TEMPLATE_VERSION = 'truss-progress-v1'
PHASE_CODES = {None: '?', 'pickup': 'p', 'harvest': 'h', 'return': 'r', 'adjust': 'a'}


def compact_prompt(condition):
    """Lossless typed task macro, NOT a claim that Step8000 learned its grammar.

    Fixed macro: right hand manipulates/holds truss; left picks into LEFT basket.
    harvest_all additionally requires returning empty stem. Fields are target /
    confirmed count, then phase+terminate bit. '*' means all, '?' means unknown.
    Current pilot admits single-digit counts; larger values explicitly fail.
    Same serializer must be called by training, validation and inference.
    """
    for value in (condition.target_count, condition.completed_count):
        if value is not None and value > 9:
            raise ValueError('compact_v1 pilot count domain is 0..9; no truncation')
    target = '*' if condition.task_goal == 'harvest_all' else (
        '?' if condition.target_count is None else str(condition.target_count))
    count = '?' if not condition.progress_valid else str(condition.completed_count)
    return f'Right truss left pick basket {target}/{count} {PHASE_CODES[condition.current_phase]}{int(condition.terminate_request)}'


def parse_compact_prompt(text, *, observation_time, evidence=()):
    """Decode metadata to check that shortening did not silently lose any field."""
    import re
    match = re.fullmatch(r'Right truss left pick basket ([1-9*?])/([0-9?]) ([?phra])([01])', text)
    if match is None: raise ValueError('Unknown compact task grammar')
    target,count,phase,stop=match.groups()
    return TaskCondition('harvest_all' if target == '*' else 'pick_tomatoes',
        int(target) if target.isdigit() else None, int(count) if count.isdigit() else None,
        count != '?', {v:k for k,v in PHASE_CODES.items()}[phase], stop == '1',
        observation_time, tuple(evidence))
