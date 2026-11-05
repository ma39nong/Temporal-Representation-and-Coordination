"""Invertible FLASH coordinate adapter, NumPy core and explicit JAX backend.

Absolute mode preserves original tensors. State-relative mode subtracts the
ideal constant polynomial AFTER the original history/target fits. Refitting
centered history would change ridge HAF initialization and is deliberately
not done here. No new statistics, trainable parameters or future references.
"""
from dataclasses import dataclass
import numpy as np
from trc.data.action_schema import GROUPS


@dataclass(frozen=True)
class FlashCoordinates:
    mode: str
    q01: object
    q99: object
    scale: object
    compatible_groups: tuple

    def __post_init__(self):
        if self.mode not in ('absolute','state_relative'):raise ValueError('Explicit coordinate mode required')
        if not self.compatible_groups or len(set(self.compatible_groups))!=len(self.compatible_groups) or any(g not in GROUPS for g in self.compatible_groups):
            raise ValueError('Explicit compatible joint groups required')
        lo,hi,s=(np.array(x,dtype=np.float64,copy=True) for x in (self.q01,self.q99,self.scale))
        if lo.shape!=(58,) or hi.shape!=(58,) or s.shape!=(7,) or not all(np.isfinite(x).all() for x in (lo,hi,s)) or np.any(hi<lo) or np.any(s<=0):
            raise ValueError('Matching finite global58 Quantile and K6 scale required')
        mask=np.zeros(58)
        for group in self.compatible_groups:mask[GROUPS[group]]=1
        for name,value in [('q01',lo),('q99',hi),('scale',s),('_mask',mask)]:
            value.flags.writeable=False;object.__setattr__(self,name,value)

    @staticmethod
    def _array(x,xp):
        x=xp.asarray(x)
        if xp is np and (not np.issubdtype(x.dtype,np.floating) or not np.isfinite(x).all()):
            raise ValueError('Finite floating point values required')
        return x

    def reference(self,measured_history,*,xp=np):
        h=self._array(measured_history,xp)
        if h.ndim<2 or h.shape[-2:]!=(8,58):raise ValueError('Eight causal measured history positions required')
        return h[...,-1,:]

    def _constants(self,xp,dtype):
        return tuple(xp.asarray(v,dtype=dtype) for v in (self.q01,self.q99,self.scale,self._mask))

    def anchor_coefficients(self,measured_history,*,xp=np):
        q=self.reference(measured_history,xp=xp);lo,hi,s,mask=self._constants(xp,q.dtype)
        one=xp.asarray([1,0,0,0,0,0,0],dtype=q.dtype)
        value=(2*(q-lo)/(hi-lo+1e-6)-1)*mask if self.mode=='state_relative' else xp.zeros_like(q)
        return value[...,None,:]*one[:,None]/s[0]

    def coefficients(self,coeff,measured_history,*,inverse=False,xp=np):
        c=self._array(coeff,xp);h=self._array(measured_history,xp)
        if c.shape!=h.shape[:-2]+(7,58):raise ValueError('Aligned seven 58D coefficient tokens required')
        self.reference(h,xp=xp)
        if self.mode=='absolute':return c
        anchor=self.anchor_coefficients(h,xp=xp)
        return c+anchor if inverse else c-anchor

    def actions(self,values,measured_history,*,inverse=False,xp=np):
        x=self._array(values,xp);q=self.reference(measured_history,xp=xp)
        if x.ndim<2 or x.shape[-1]!=58 or x.shape[:-2]!=q.shape[:-1]:raise ValueError('Aligned physical trajectories required')
        if self.mode=='absolute':return x
        mask=xp.asarray(self._mask,dtype=x.dtype);anchor=(q*mask)[...,None,:]
        return x+anchor if inverse else x-anchor

    def normalize_coordinates(self,physical_coordinates,*,inverse=False,xp=np):
        """Selected relative joints use slope only; NEVER apply F to A-q0.

        Selected: R_norm=2*R/(q99-q01+eps). Unselected/absolute: original F.
        Hence R_norm=F(A)-F(q0), while F(A-q0) has an extra F(0).
        """
        x=self._array(physical_coordinates,xp)
        if x.shape[-1]!=58:raise ValueError('58D coordinates required')
        lo,hi,_,mask=self._constants(xp,x.dtype);span=hi-lo+1e-6
        relative=mask if self.mode=='state_relative' else xp.zeros_like(mask)
        offset=(-2*lo/span-1)*(1-relative)
        return (x-offset)*span/2 if inverse else 2*x/span+offset
