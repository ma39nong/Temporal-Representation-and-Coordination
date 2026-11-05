"""Physical mean-offset decomposition, distinct from Legendre allocations."""
import numpy as np
from .metrics import EVAL_GROUPS, summary


def decompose_offsets(truth, prediction):
    a,p=map(lambda x:np.asarray(x,np.float64),(truth,prediction))
    if a.ndim!=3 or a.shape[1:]!=(50,58) or p.shape!=a.shape or not len(a):
        raise ValueError('Aligned nonempty (N,50,58) physical actions required')
    if not np.isfinite(a).all() or not np.isfinite(p).all():raise ValueError('Finite actions required')
    error=p-a
    bias=error.mean(axis=1)
    residual=error-bias[:,None,:]
    total=np.mean(error**2,axis=1);offset=bias**2;shape=np.mean(residual**2,axis=1)
    np.testing.assert_allclose(offset+shape,total,rtol=1e-12,atol=1e-14)
    return {'bias':bias,'residual':residual,'total_mse':total,'offset_mse':offset,'shape_mse':shape}


def grouped_offsets(result):
    rows=[]
    for group,idx in EVAL_GROUPS.items():
        total,offset,shape=(float(np.mean(result[k][...,idx])) for k in ('total_mse','offset_mse','shape_mse'))
        rows.append({'group':group,'total_mse':total,'offset_mse':offset,'shape_mse':shape,
                     'offset_fraction':offset/total if total else None,'total_rmse':total**.5,
                     'offset_rmse':offset**.5,'shape_rmse':shape**.5})
    return rows


def anchor_correction(prediction, measured_history):
    """B1: causal-input offline control; assumes current state available, not safe deployment.

    Translate the entire prediction by q0-P0, preserving its time derivatives.
    No GT argument or fitted correction uses future labels. Command/measurement
    mismatch and actual sensor/control latency remain evaluation limitations.
    """
    p,h=map(lambda x:np.asarray(x,np.float64),(prediction,measured_history))
    if p.ndim!=3 or p.shape[1:]!=(50,58) or h.shape!=(len(p),8,58):raise ValueError('Prediction/history shape mismatch')
    if not np.isfinite(p).all() or not np.isfinite(h).all():raise ValueError('Finite inputs required')
    return p+(h[:,-1]-p[:,0])[:,None]


def compare_constant_coefficient(bias,c_gt,c_pred,scale,basis,q01,q99):
    """Compare physical b (total error) to C0 component (prediction-oracle)."""
    b,cg,cp,s,B,lo,hi=map(lambda x:np.asarray(x,np.float64),(bias,c_gt,c_pred,scale,basis,q01,q99))
    if b.ndim!=2 or b.shape[1]!=58 or cg.shape!=cp.shape or cg.shape!=(len(b),7,58) or B.shape!=(50,7):
        raise ValueError('Aligned K6 coefficients and physical biases required')
    e0=np.mean(B[:,0])*s[0]*(cp[:,0]-cg[:,0])*(hi-lo+1e-6)/2
    rows=[]
    for group,idx in EVAL_GROUPS.items():
        x,y=b[:,idx],e0[:,idx]
        covariance=float(np.mean((x-x.mean())*(y-y.mean())))
        denominator=float(x.std()*y.std())
        rows.append({'group':group,'physical_bias_rmse_rad':float(np.sqrt(np.mean(x*x))),
                     'coefficient0_component_rmse_rad':float(np.sqrt(np.mean(y*y))),
                     'bias_minus_coefficient0_rmse_rad':float(np.sqrt(np.mean((x-y)**2))),
                     'pooled_bias_c0_correlation':covariance/denominator if denominator else None,
                     'bias_times_c0_mean':float(np.mean(x*y))})
    return rows
