"""Post-hoc error structure: exact linear contributions, not architecture claims."""
import numpy as np
from .metrics import EVAL_GROUPS, summary


def coefficient_contributions(truth, oracle, prediction, c_gt, c_pred, scale, basis, q01, q99):
    """Exact additive decoded residuals with explicit interaction and oracle repair.

    Scaled coefficient error is mapped to physical actions by the ORIGINAL
    exported basis/scale. Cross-degree cancellation makes standalone energies
    nonadditive. Replacing one predicted degree with ground-truth coefficients
    is an oracle sensitivity experiment, not an implementable inference method.
    """
    a,o,p=(np.asarray(v,np.float64) for v in (truth,oracle,prediction))
    gt,cp,s,b,low,high=(np.asarray(v,np.float64) for v in (c_gt,c_pred,scale,basis,q01,q99))
    if a.ndim!=3 or a.shape[1:]!=(50,58) or p.shape!=a.shape or o.shape!=a.shape:
        raise ValueError('Aligned physical trajectories required')
    if gt.shape!=cp.shape or gt.shape!=(len(a),7,58) or s.shape!=(7,) or b.shape!=(50,7) or low.shape!=(58,) or high.shape!=(58,):
        raise ValueError('Original K6 basis/scales and coefficient arrays required')
    if not all(np.isfinite(v).all() for v in (a,o,p,gt,cp,s,b,low,high)) or np.any(s<=0) or np.any(high<low):
        raise ValueError('Finite valid inputs required')
    component=np.einsum('tk,nkj,k,j->nktj',b,cp-gt,s,(high-low+1e-6)/2)
    e=component.sum(axis=1)
    np.testing.assert_allclose(e,p-o,atol=3e-5,rtol=2e-4)
    rows=[];pairs=[]
    for group,idx in EVAL_GROUPS.items():
        total=(p-a)[...,idx];pred=e[...,idx]
        baseline=float(np.mean(total*total))
        for degree in range(7):
            # Explicit axis avoids advanced-index dimension transposition.
            selected=np.take(component[:,degree],idx,axis=-1)
            allocated=float(np.mean(selected*pred))
            repaired=total-selected
            rows.append({'group':group,'degree':degree,'component_rmse_rad':float(np.sqrt(np.mean(selected*selected))),
                         'self_mse':float(np.mean(selected*selected)),
                         'signed_allocated_prediction_mse':allocated,
                         'cross_with_other_degrees':float(2*np.mean(selected*(pred-selected))),
                         'oracle_repair_final_rmse_rad':float(np.sqrt(np.mean(repaired*repaired))),
                         'oracle_repair_final_mse_reduction':baseline-float(np.mean(repaired*repaired))})
        for k in range(7):
            for l in range(k+1,7):
                ck=np.take(component[:,k],idx,axis=-1);cl=np.take(component[:,l],idx,axis=-1)
                pairs.append({'group':group,'degree_k':k,'degree_l':l,'mse_cross_term':float(2*np.mean(ck*cl))})
        allocated_sum=sum(v['signed_allocated_prediction_mse'] for v in rows if v['group']==group)
        np.testing.assert_allclose(allocated_sum,np.mean(pred*pred),atol=1e-10,rtol=1e-8)
    return {'rows':rows,'pair_interactions':pairs,'decode_residual_max_rad':float(np.max(np.abs(e-(p-o))))}


def temporal_structure(truth,prediction):
    delta=np.asarray(prediction,np.float64)-np.asarray(truth,np.float64)
    rows=[];overview=[]
    for group,idx in EVAL_GROUPS.items():
        x=np.take(delta,idx,axis=-1)
        for start,end,label in [(0,10,'early_0_9'),(10,30,'middle_10_29'),(30,50,'late_30_49')]:
            rows.append({'group':group,'interval':label,**summary(x[:,start:end])})
        by_tick=np.sqrt(np.mean(x*x,axis=(0,2)))
        per_window=np.mean(x*x,axis=(1,2));top=max(1,int(np.ceil(len(x)*.1)))
        energy=float(per_window.sum())
        overview.append({'group':group,'tick0_rmse':float(by_tick[0]),'tick49_rmse':float(by_tick[-1]),
                         'increasing_tick_transitions':int(np.sum(np.diff(by_tick)>0)),
                         'largest_error_tick':int(np.argmax(by_tick)),
                         'top10percent_windows_mse_share':float(np.sort(per_window)[-top:].sum()/energy) if energy else 0.,
                         'top_window_count':top,'all_window_count':len(x)})
    return rows,overview


def tick_zero_alignment(truth,prediction,history,current_state):
    a,p,h,s=(np.asarray(v,np.float64) for v in (truth,prediction,history,current_state))
    if h.shape!=(len(a),8,58) or s.shape!=(len(a),58):raise ValueError('Measured aligned history/current state required')
    np.testing.assert_array_equal(h[:,-1],s)
    rows=[]
    for group,idx in EVAL_GROUPS.items():
        for kind,error in [('gt_command_minus_measured_state',a[:,0]-s),('prediction_minus_gt_command',p[:,0]-a[:,0]),
                           ('prediction_minus_measured_state',p[:,0]-s)]:
            rows.append({'group':group,'comparison':kind,**summary(error[:,idx])})
    return rows


def conditioning_ranges(truth,history,state,state_q01,state_q99,action_q01,action_q99):
    """Quantile exceedance is distribution context, never an invalid-frame mask."""
    rows=[]
    for name,value,low,high in [('current_state',state,state_q01,state_q99),('measured_history',history,action_q01,action_q99),
                                ('gt_future_actions',truth,action_q01,action_q99)]:
        value,low,high=map(np.asarray,(value,low,high))
        normalized=2*(value-low)/(high-low+1e-6)-1
        for group,idx in EVAL_GROUPS.items():
            x=np.take(normalized,idx,axis=-1)
            rows.append({'quantity':name,'group':group,'outside_q01_q99_fraction':float(np.mean((x < -1)|(x > 1))),
                         'normalized_min':float(x.min()),'normalized_max':float(x.max())})
    return rows
