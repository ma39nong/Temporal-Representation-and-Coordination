"""Small synthetic JAX head check; separate from real-model acceptance."""
import json,sys
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import nnx
from trc.models.flash_order_readout import ResidualVelocityHead

if jax.default_backend()!='cpu':raise ValueError('Use CPU for this small check')
h=jnp.ones((2,7,1024));original=nnx.Linear(1024,58,rngs=nnx.Rngs(12))
base=original(h);results={}
for mode in ('unified','order_specific'):
    head=ResidualVelocityHead(original,mode)
    initial=head(h)
    np.testing.assert_array_equal(initial,base)
    head.adapter_up.value=jnp.ones_like(head.adapter_up.value)*.01
    if mode=='order_specific':
        head.adapter_up.value=jnp.zeros_like(head.adapter_up.value).at[2].set(.01)
        delta=np.asarray(head(h)-base)
        np.testing.assert_array_equal(np.delete(delta,2,axis=1),0)
        assert np.linalg.norm(delta[:,2])>0
    head.enabled=False
    np.testing.assert_array_equal(head(h),base)
    head.enabled=True
    def loss(m):return jnp.sum(m(h)[:,2])
    grad=nnx.grad(loss)(head)
    assert np.linalg.norm(np.asarray(grad.adapter_up.value))>0
    if mode=='order_specific':np.testing.assert_array_equal(np.delete(np.asarray(grad.adapter_up.value),2,axis=0),0)
    results[mode]={'zero_init_exact':True,'nonzero_adapter_disabled_exact':True,'single_order_output_gradient_routing_correct':True}
Path(sys.argv[1]).write_text(json.dumps({'data_kind':'synthetic','backend':'cpu','checks':results},indent=2))
print(json.dumps(results))
