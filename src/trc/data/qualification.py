"""Frozen state strata and explicit eligibility; no model or success inference."""
import numpy as np
from .action_schema import GROUPS


def state_stratum(q,reference,mean,scale,statistics):
    q=np.asarray(q,float);ref=np.asarray(reference,float);mu=np.asarray(mean,float);sd=np.asarray(scale,float)
    if q.shape!=(58,) or ref.ndim!=2 or ref.shape[1]!=58 or mu.shape!=(58,) or sd.shape!=(58,) or not np.isfinite(q).all() or not np.isfinite(ref).all() or not np.isfinite([mu,sd]).all() or np.any(sd<=0):raise ValueError('Finite frozen 58D reference required')
    delta=(ref-q)/sd;dist={g:np.sqrt(np.mean(delta[:,idx]**2,axis=1)) for g,idx in {**GROUPS,'full_action':list(range(58))}.items()}
    limits={s['group']:s for s in statistics};near=np.logical_and.reduce([d<=limits[g]['training_LOO_nn_q95'] for g,d in dist.items()])
    far=[g for g,d in dist.items() if d.min()>limits[g]['training_LOO_nn_q99']]
    result={'class':'near' if near.any() else ('far' if far else 'ambiguous'),'nearest_distances':{g:float(d.min()) for g,d in dist.items()},
            'far_groups':far,'common_near_reference_index':int(np.flatnonzero(near)[0]) if near.any() else None}
    return result


def inverse_compact_time(compact_ns,mapping):
    """Invert documented shared-time deletions; right side of a seam chosen.

    Returns raw-clock timestamps and explicit row boundaries. No interpolation
    across deleted intervals; callers must exclude history/target crossing seams.
    """
    t=np.asarray(compact_ns,dtype=np.int64)
    if t.ndim!=1 or (len(t)>1 and np.any(np.diff(t)<=0)):raise ValueError('Increasing compact timestamps required')
    bounds=[];removed=[];total=0;end=None
    for entry in mapping:
        a,b,d=(int(entry[k]) for k in ('source_start_ns','source_end_ns','removed_ns'))
        if b<=a or d!=b-a or (end is not None and a<end):raise ValueError('Ordered disjoint deletion intervals required')
        bounds.append(a-total);total+=d;removed.append(total);end=b
    count=np.searchsorted(np.asarray(bounds,dtype=np.int64),t,side='right');offset=np.r_[np.int64(0),np.asarray(removed,dtype=np.int64)][count]
    cuts=sorted({int(i) for i in np.searchsorted(t,np.asarray(bounds,dtype=np.int64),side='left') if 0<i<len(t)})
    return t+offset,cuts


def coverage_cell(success_status,data_status,lineage_status,training_isolation,prior_use,state_class,scene_class,task_type):
    if (success_status,data_status,lineage_status,training_isolation,prior_use,task_type)!=('success','verified','verified','verified','verified','single_pick'):return None
    return {('near','near'):'A',('near','different'):'B',('far','near'):'C',('far','different'):'D'}.get((state_class,scene_class))
