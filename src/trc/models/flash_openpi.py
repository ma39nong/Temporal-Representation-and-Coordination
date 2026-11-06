"""Opt-in original OpenPI FLASH extension. Import ONLY in explicit source runtime.

The research NumPy package never imports this module implicitly. Architecture,
VLM, polynomial fits, parameter shapes, noise/loss weights and Euler integration
are inherited from the pinned source. Only the coefficient coordinate changes.
"""
from dataclasses import dataclass,fields
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from openpi.models.pi0_flash import Pi0Flash,Pi0FlashConfig
from trc.data.action_schema import GROUPS
from .flash_coordinates import FlashCoordinates


@dataclass(frozen=True)
class CoordinateFlashConfig(Pi0FlashConfig):
    coordinate_mode: str = 'absolute'

    def __post_init__(self):
        super().__post_init__()
        if self.coordinate_mode not in ('absolute','state_relative'):raise ValueError('Unknown coordinate mode')

    @classmethod
    def from_base(cls,base,mode):
        return cls(**{f.name:getattr(base,f.name) for f in fields(Pi0FlashConfig)},coordinate_mode=mode)

    def create(self,rng):return CoordinateFlash(self,rngs=nnx.Rngs(rng))


class CoordinateFlash(Pi0Flash):
    def __init__(self,config,rngs):
        super().__init__(config,rngs)
        self.coordinate_mode=config.coordinate_mode  # Static string, not a trainable variable.

    def coordinates(self):
        return FlashCoordinates(self.coordinate_mode,self.action_q01,self.action_q99,
                                np.asarray(self.polynomial.scale),tuple(GROUPS))

    def loss_from_cache(self,rng,prefix_mask,kv_cache,*,history,target_ext,train=True):
        if self.coordinate_mode=='absolute':
            return super().loss_from_cache(rng,prefix_mask,kv_cache,history=history,target_ext=target_ext,train=train)
        if history.shape[1:]!=(8,58) or target_ext.shape[1:]!=(54,58):raise ValueError('Original FLASH history/target shapes required')
        noise_rng,time_rng=jax.random.split(rng)
        source=self.polynomial.fit_history(self.normalize(history))
        target=self.polynomial.fit_target(self.normalize(target_ext),boundary_constraint=self.boundary_constraint)
        if train and self.history_noise_std:
            source=source+self.history_noise_std*jax.random.normal(noise_rng,source.shape)
        adapter=self.coordinates()
        relative_source=adapter.coefficients(source,history,xp=jnp)
        relative_target=adapter.coefficients(target,history,xp=jnp)
        time=jax.random.uniform(time_rng,(history.shape[0],))
        mixture=(1-time[:,None,None])*source+time[:,None,None]*target
        relative_mixture=adapter.coefficients(mixture,history,xp=jnp)
        velocity=self.coefficient_velocity(relative_mixture,time,prefix_mask,kv_cache)
        fm=jnp.mean((velocity-(target-source))**2,axis=(1,2))
        at_zero=self.coefficient_velocity(relative_source,jnp.zeros_like(time),prefix_mask,kv_cache)
        consistency=jnp.mean((relative_source+at_zero-relative_target)**2,axis=(1,2))
        return {'loss':fm+self.consistency_weight*consistency,'flow_matching_loss':fm,'polynomial_consistency_loss':consistency}

    def sample_from_cache(self,prefix_mask,kv_cache,*,history,num_steps=1):
        if self.coordinate_mode=='absolute':
            return super().sample_from_cache(prefix_mask,kv_cache,history=history,num_steps=num_steps)
        if type(num_steps) is not int or num_steps<1:raise ValueError('Positive static num_steps required')
        if history.shape[1:]!=(8,58):raise ValueError('Eight measured history positions required')
        absolute=self.sample_coefficients_from_cache(prefix_mask,kv_cache,history=history,num_steps=num_steps)
        return self.denormalize(self.polynomial.reconstruct(absolute))

    def sample_coefficients_from_cache(self,prefix_mask,kv_cache,*,history,num_steps=1):
        """Direct coefficients in ORIGINAL Absolute training coordinates for evaluation."""
        if type(num_steps) is not int or num_steps<1:raise ValueError('Positive static num_steps required')
        if history.shape[1:]!=(8,58):raise ValueError('Eight measured history positions required')
        adapter=self.coordinates()
        coeff=adapter.coefficients(self.polynomial.fit_history(self.normalize(history)),history,xp=jnp)
        for step in range(num_steps):
            time=jnp.full((history.shape[0],),step/num_steps)
            coeff+=self.coefficient_velocity(coeff,time,prefix_mask,kv_cache)/num_steps
        return adapter.coefficients(coeff,history,inverse=True,xp=jnp)
