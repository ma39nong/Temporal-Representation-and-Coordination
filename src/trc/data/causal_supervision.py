"""Reviewed event intervals -> causal FLASH conditions; no visual label inference."""
import math
from trc.models.task_condition import ConfirmedEvent, causal_condition
from .qualification import flash_window_cuts, inverse_compact_time


def _interval(value):
    values = [value, value] if isinstance(value, (int, float)) and not isinstance(value, bool) else value
    if not isinstance(values, (list, tuple)) or len(values) != 2 or any(
        isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        raise ValueError('Finite observable time interval required')
    if values[0] > values[1]: raise ValueError('Reversed interval')
    return values


def event_export_interval(event, clock):
    """Never interpolate across deletion gaps; raw ns must retain integer precision."""
    if event['timestamp_domain'] == 'export_seconds':
        a,b = _interval(event['event_timestamp'])
        if a < 0 or b > (clock['frames']-1)/30: raise ValueError('Event outside recording')
        if any(a < c/30 <= b for c in clock['cut_rows']):
            raise ValueError('Event interval crosses cleaning boundary')
        return a,b
    if event['timestamp_domain'] != 'raw_recording_ns': raise ValueError('Unknown timestamp domain')
    values=event['event_timestamp']
    values=[values,values] if isinstance(values,(int,str)) else values
    if len(values)!=2 or any(not isinstance(v,(str,int)) or isinstance(v,bool) for v in values):
        raise ValueError('Integer raw nanoseconds required')
    a,b=map(int,values)
    if a>b:raise ValueError('Reversed raw interval')
    for row in clock['mapping']:
        left,right=int(row['source_start_ns']),int(row['source_end_ns'])
        if a<right and b>=left:raise ValueError('Event intersects deleted interval')
    offset=sum(int(r['removed_ns']) for r in clock['mapping'] if int(r['source_end_ns'])<=a)
    start=clock['export_start_ns']
    result=((a-offset-start)/1e9,(b-offset-start)/1e9)
    return event_export_interval({**event,'timestamp_domain':'export_seconds','event_timestamp':result},clock)


def validate_window(current_frame, clock):
    if current_frame-7 < 0 or current_frame+51 >= clock['frames']:
        raise ValueError('FLASH support outside episode')
    if flash_window_cuts(current_frame,clock['cut_rows']):raise ValueError('FLASH support crosses cleaning boundary')


def _started_by(event, now, clock):
    if event['timestamp_domain']=='export_seconds':
        return _interval(event['event_timestamp'])[0]<=now
    if event['timestamp_domain']!='raw_recording_ns':raise ValueError('Unknown event clock')
    value=event['event_timestamp']
    start=value[0] if isinstance(value,(list,tuple)) else value
    if not isinstance(start,(int,str)) or isinstance(start,bool):raise ValueError('Integer raw nanoseconds required')
    raw_now,_=inverse_compact_time([clock['export_start_ns']+round(now*1e9)],clock['mapping'])
    return int(start)<=int(raw_now[0])


def reviewed_progress(review, current_frame, clock):
    """Use upper interval bounds; confirmation delays are not guessed.

    Even human-reviewed offline phases are not input until their online
    availability is separately established. Missing event coverage is unknown.
    """
    validate_window(current_frame,clock)
    now=current_frame/30;events=[]; unresolved_count=False
    for row in review['events']:
        if row['recording_id']!=review['recording_id']:raise ValueError('Mixed recording IDs')
        if row.get('review_status')!='verified' or row.get('verified_by') in (None,'','assistant'):
            if row.get('review_status')!='rejected' and row['event_type']=='pick_success':
                if _started_by(row,now,clock):
                    unresolved_count=True
            continue
        if not row.get('causal_availability_verified') or row.get('confirmation_available_at') is None:
            if row['event_type']=='pick_success' and _started_by(row,now,clock):unresolved_count=True
            continue
        a,b=event_export_interval(row,clock)
        if row.get('confirmation_timestamp_domain')!='export_seconds':
            raise ValueError('Explicit confirmation clock domain required')
        available=_interval(row['confirmation_available_at'])[1]
        if available<b:raise ValueError('Confirmation predates event interval upper bound')
        if row['event_type'] not in ('pick_success','placement','retry','phase'):continue
        if not row.get('evidence'):raise ValueError('Confirmed event evidence required')
        events.append(ConfirmedEvent(row['event_id'],row['event_type'],b,available,
            str(row['evidence']),row.get('object_id'),row.get('phase')))
    coverage=review.get('coverage',{})
    reset=coverage.get('reset_at') if coverage.get('review_status')=='verified' else None
    complete=coverage.get('complete_until') if reset is not None else None
    # Missing events in a removed interval make the global count unreliable.
    if unresolved_count or (any(c<=current_frame for c in clock['cut_rows']) and not coverage.get('deleted_intervals_reviewed')):
        complete=None
    goal=review['goal']
    if goal['review_status']!='verified':raise ValueError('Task goal needs original instruction evidence')
    condition=causal_condition(task_goal=goal['task_goal'],target_count=goal['target_count'],observation_time=now,
        events=events,reset_at=reset,reset_evidence=coverage.get('evidence'),coverage_until=complete)
    return {'condition':condition,'remaining_count':condition.remaining_count,
            'visible_confirmed_event_ids':[e.event_id for e in events if e.available_at<=now]}
