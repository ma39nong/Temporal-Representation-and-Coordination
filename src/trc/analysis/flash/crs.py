"""Chunk-shared physical fusion and causal, non-neural reference rules.

Oracle functions are OFFLINE labels/bounds, never selector inputs. This module
does not import OpenPI, execute a model, or fit a neural gate.
"""
import zipfile
import numpy as np


def selected_npz_rows(path, key, indices):
    """Read only allowlisted NPY rows, without materializing excluded episodes."""
    idx=np.asarray(indices)
    if idx.ndim!=1 or not np.issubdtype(idx.dtype,np.integer) or len(set(idx.tolist()))!=len(idx):
        raise ValueError('Distinct integer row indices required')
    with zipfile.ZipFile(path) as z, z.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
        else:raise ValueError('Unsupported NPY version')
        if fortran or dtype.hasobject or len(shape)<1 or np.any(idx<0) or np.any(idx>=shape[0]):
            raise ValueError('Numeric C-order arrays and valid row indices required')
        start=f.tell();row_size=int(np.prod(shape[1:]))*dtype.itemsize;values=[]
        for i in idx:
            f.seek(start+int(i)*row_size)
            b=f.read(row_size)
            if len(b)!=row_size:raise ValueError('Truncated array row')
            values.append(np.frombuffer(b,dtype=dtype).reshape(shape[1:]).copy())
    return np.stack(values) if len(idx) else np.empty((0,*shape[1:]),dtype=dtype)


def mix_actions(absolute,relative,weight):
    a,b=np.asarray(absolute,float),np.asarray(relative,float)
    if a.ndim!=3 or a.shape[1:]!=(50,58) or b.shape!=a.shape or not np.isfinite([a,b]).all():
        raise ValueError('Aligned finite physical (N,50,58) endpoints required')
    w=np.asarray(weight,float)
    if w.ndim==0:w=np.full(len(a),float(w))
    if w.shape!=(len(a),) or not np.isfinite(w).all() or np.any((w<0)|(w>1)):
        raise ValueError('One bounded scalar per entire chunk required')
    return a+w[:,None,None]*(b-a)


def oracle_weights(truth,absolute,relative):
    """Binary endpoint bound and stronger convex-fusion bound, OFFLINE ONLY."""
    a=np.asarray(absolute,float);b=np.asarray(relative,float);gt=np.asarray(truth,float)
    mix_actions(a,b,0)
    if gt.shape!=a.shape or not np.isfinite(gt).all():raise ValueError('Aligned labels required')
    e=a-gt;d=b-a;den=np.mean(d*d,axis=(1,2));num=-np.mean(e*d,axis=(1,2))
    binary=(np.mean((b-gt)**2,axis=(1,2))<np.mean(e*e,axis=(1,2))).astype(float)
    continuous=np.clip(np.divide(num,den,out=np.zeros_like(num),where=den>0),0,1)
    return binary,continuous


class StateReference:
    """Fit only nominal-fold states; score accepts history, never ID/GT/role."""
    def __init__(self,history):
        h=np.asarray(history,float)
        if h.ndim!=3 or h.shape[1:]!=(8,58) or not len(h) or not np.isfinite(h).all():
            raise ValueError('Finite causal histories required')
        q=h[:,-1];self.mean=q.mean(0);self.scale=np.maximum(q.std(0),.05)
        self.reference=(q-self.mean)/self.scale

    def score(self,history):
        h=np.asarray(history,float)
        if h.ndim!=3 or h.shape[1:]!=(8,58) or not np.isfinite(h).all():raise ValueError('Causal8x58 history required')
        z=(h[:,-1]-self.mean)/self.scale
        nn=np.sqrt(np.min(np.mean((z[:,None]-self.reference[None])**2,axis=-1),axis=1))
        return {'state_z':np.sqrt(np.mean(z*z,axis=1)), 'state_nn':nn,
                'history_motion':np.sqrt(np.mean((np.diff(h,axis=1)/self.scale)**2,axis=(1,2)))}


def source_balanced_weights(sources,roles):
    s,r=np.asarray(sources),np.asarray(roles);out=np.zeros(len(s),float)
    if s.shape!=r.shape or set(r)!={'training','development'}:raise ValueError('Two declared fitting cohorts required')
    for role in ('training','development'):
        members=set(s[r==role])
        for source in members:
            m=(s==source)&(r==role);out[m]=.5/len(members)/int(m.sum())
    return out


def fit_threshold(score,loss0,loss3,weights,nominal_mask):
    """Training-only threshold; require nominal RMSE <=1.1 x D0 in fitting.

    Constraint is not a guarantee on validation. Only high-score->D3 is allowed;
    the direction and tolerance are fixed before inspecting any outcomes.
    """
    x,l0,l3,w=map(lambda a:np.asarray(a,float),(score,loss0,loss3,weights))
    nominal=np.asarray(nominal_mask,bool)
    if any(v.shape!=x.shape for v in (l0,l3,w,nominal)) or not np.isfinite([x,l0,l3,w]).all() or not nominal.any():raise ValueError('Aligned finite fitting vectors required')
    values=np.unique(x);cuts=np.r_[-np.inf,(values[:-1]+values[1:])/2,np.inf]
    best=(float(np.dot(w,l0)),np.inf)
    for threshold in cuts:
        loss=np.where(x>threshold,l3,l0)
        if loss[nominal].mean()>1.21*l0[nominal].mean()+1e-15:continue
        objective=float(np.dot(w,loss))
        if objective<best[0]-1e-15:best=(objective,float(threshold))
    return best[1]
