"""Opt-in pinned OpenPI FLASH extension; original velocity computation inherited.

Only the final linear head is extended. Original kernel/bias retain their exact
checkpoint paths. No coordinate, loss, polynomial or attention change occurs.
"""
from dataclasses import dataclass, fields
import jax
import jax.numpy as jnp
from flax import nnx, traverse_util
from openpi.models.pi0_flash import Pi0Flash, Pi0FlashConfig
from .velocity_readout import MODES, initialize_adapter, strict_extend_flat


class ResidualVelocityHead(nnx.Module):
    def __init__(self, original, mode, seed=42003, enabled=True):
        if mode not in ('unified','order_specific'):
            raise ValueError('Residual head requires an adapter mode')
        if original.kernel.value.shape != (1024,58):
            raise ValueError('Actual native velocity head is not 1024->58')
        # Retain the exact original variables, names, and native linear operation.
        for name in ('kernel','bias','dtype','precision','use_bias','dot_general','in_features','out_features'):
            setattr(self, name, getattr(original,name))
        self.mode, self.enabled = mode, enabled
        params=initialize_adapter(mode,seed)
        self.adapter_down=nnx.Param(jnp.asarray(params['adapter_down']))
        self.adapter_up=nnx.Param(jnp.asarray(params['adapter_up']))

    def __call__(self, hidden):
        base=nnx.Linear.__call__(self,hidden)
        if not self.enabled:
            return base
        if hidden.ndim!=3 or hidden.shape[1:]!=(7,1024):
            raise ValueError('Seven coefficient token hidden states required')
        z=jnp.einsum('bkh,hr->bkr',hidden.astype(jnp.float32),self.adapter_down.value,
                     precision=jax.lax.Precision.HIGHEST)
        expr='bkr,rd->bkd' if self.mode=='unified' else 'bkr,krd->bkd'
        delta=jnp.einsum(expr,z,self.adapter_up.value,precision=jax.lax.Precision.HIGHEST)
        return base.astype(jnp.float32)+delta


@dataclass(frozen=True)
class ReadoutFlashConfig(Pi0FlashConfig):
    readout_mode: str = 'none'
    adapter_seed: int = 42003
    adapter_enabled: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.readout_mode not in MODES or not self.freeze_vlm:
            raise ValueError('P2-2 requires frozen VLM and explicit readout mode')

    @classmethod
    def from_base(cls,base,mode,**overrides):
        kwargs={f.name:getattr(base,f.name) for f in fields(Pi0FlashConfig)}
        kwargs.update(freeze_vlm=True,**overrides)
        return cls(**kwargs,readout_mode=mode)

    def create(self,rng):
        return ReadoutFlash(self,rngs=nnx.Rngs(rng))

    def warm_start(self,params,expected_sha256):
        """Strict original Step8000 load; only two whitelisted leaves are new."""
        from openpi.training.flash_checkpoint import parameter_digest
        if parameter_digest(params)!=expected_sha256:
            raise ValueError('Step8000 parameter identity differs')
        shaped=nnx.eval_shape(self.create,jax.random.key(0))
        reference=traverse_util.flatten_dict(nnx.state(shaped,nnx.Param).to_pure_dict())
        flat=strict_extend_flat(reference,traverse_util.flatten_dict(params),self.readout_mode,self.adapter_seed)
        return self.load(traverse_util.unflatten_dict(flat),remove_extra_params=False)


class ReadoutFlash(Pi0Flash):
    def __init__(self,config,rngs):
        super().__init__(config,rngs)
        self.readout_mode=config.readout_mode
        if config.readout_mode!='none':
            self.action_out_proj=ResidualVelocityHead(self.action_out_proj,config.readout_mode,
                        seed=config.adapter_seed,enabled=config.adapter_enabled)
