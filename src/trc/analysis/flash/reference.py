"""Offline command-feedback lag diagnostics; not causal delay identification."""
import numpy as np


def lag_profile(actions,states,indices,max_lag=10):
    a,s=map(lambda x:np.asarray(x,np.float64),(actions,states));idx=np.asarray(indices)
    if a.ndim!=2 or a.shape!=s.shape or a.shape[1]!=58 or len(a)<=2*max_lag+2 or max_lag<0:
        raise ValueError('Long aligned canonical58 position trajectories required')
    if not np.isfinite(a).all() or not np.isfinite(s).all():raise ValueError('Finite trajectories required')
    lo,hi=max_lag,len(a)-max_lag;command=a[lo:hi,idx];da=np.diff(command,axis=0);da-=da.mean(0)
    rows=[]
    for lag in range(-max_lag,max_lag+1):
        measured=s[lo+lag:hi+lag,idx];ds=np.diff(measured,axis=0);ds-=ds.mean(0)
        denominator=float(np.sqrt(np.sum(da*da)*np.sum(ds*ds)))
        rows.append({'lag_frames':lag,'position_rmse_rad':float(np.sqrt(np.mean((command-measured)**2))),
                     'increment_correlation':float(np.sum(da*ds)/denominator) if denominator>1e-12 else None})
    return rows
