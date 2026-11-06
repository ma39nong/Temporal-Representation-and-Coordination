#!/usr/bin/env python3
"""Real B0/B2 integration forward/loss checks. NO gradients or optimizer updates."""
import hashlib,json,site,subprocess,sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from openpi.models import model as model_lib
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.flash_native import NativeDataset,bind_stats,preflight
from openpi.training.flash_checkpoint import parameter_digest
from scripts.run_training import load_experiment
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads(Path(sys.argv[1]).read_text());site.addsitedir(cfg['research_runtime'])
from trc.models import flash_openpi
from trc.models.flash_openpi import CoordinateFlashConfig
out=Path(cfg['output']).resolve()
if out.exists() or out.is_relative_to(Path.cwd().resolve()):raise ValueError('New research output required')
if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=cfg['source_commit']:raise ValueError('Source mismatch')
exp=bind_stats(load_experiment(cfg['experiment_module']));contract=json.loads(Path(cfg['training_contract']).read_text())
if sha(cfg['training_contract'])!=cfg['training_contract_sha256'] or preflight(exp)!=contract['dataset'] or str(exp.model)!=contract['model']:raise ValueError('Contract mismatch')
params=model_lib.restore_params(Path(cfg['checkpoint'])/'params',restore_type=np.ndarray);parameter_sha=parameter_digest(params);params=jax.device_put(params)
base=replace(exp.model,dtype='float32');original=base.load(params,remove_extra_params=False)
model0=CoordinateFlashConfig.from_base(base,'absolute').load(params,remove_extra_params=False);model2=CoordinateFlashConfig.from_base(base,'state_relative').load(params,remove_extra_params=False)
for m in (original,model0,model2):m.eval()
states=[nnx.state(m,nnx.Param).to_pure_dict() for m in (original,model0,model2)];leaves=[jax.tree.leaves(v) for v in states]
layout=lambda x:[(v.shape,str(v.dtype)) for v in x]
assert layout(leaves[0])==layout(leaves[1])==layout(leaves[2])
assert jax.tree.structure(states[0])==jax.tree.structure(states[1])==jax.tree.structure(states[2])
shared_arrays=all(a is b and b is c for a,b,c in zip(*leaves))
assert shared_arrays,'Parameter arrays must be shared unchanged for parity test'
dataset=NativeDataset(exp,tokenizer=PaligemmaTokenizer(max_len=exp.model.max_token_len));inputs=json.loads(Path(cfg['paired_inputs']).read_text())
if sha(cfg['paired_inputs'])!=cfg['paired_inputs_sha256']:raise ValueError('Plan mismatch')
dataset.records={r['id']:r for r in inputs['records']};dataset.numeric={};original_image=dataset.image
hashes=[]
def checked(index,role):
 a=original_image(index,role);h=hashlib.sha256(a.tobytes()).hexdigest()
 assert h==inputs['rgb_sha256'][f"{index['id']}:{index['start']}:{role}"];hashes.append(h);return a
dataset.image=checked
samples=[dataset.observation(inputs['windows'][i]) for i in cfg['window_indices']]
obs=jax.tree.map(lambda *x:jnp.concatenate(x,axis=0),*[v[0] for v in samples]);h=jnp.asarray(np.concatenate([v[1] for v in samples]));target=jnp.asarray(np.concatenate([v[2] for v in samples]))
@nnx.jit
def forward(m,obs,h,target,key):
 ob=model_lib.preprocess_observation(None,obs,train=False);mask,cache=m.encode_condition(ob)
 p=m.sample_from_cache(mask,cache,history=h,num_steps=1)
 loss=m.loss_from_cache(key,mask,cache,history=h,target_ext=target,train=True)
 return p,loss
p,loss=jax.block_until_ready(forward(original,obs,h,target,jax.random.key(42)))
p0,l0=jax.block_until_ready(forward(model0,obs,h,target,jax.random.key(42)));p2,l2=jax.block_until_ready(forward(model2,obs,h,target,jax.random.key(42)))
maxdiff=lambda a,b:float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
@nnx.jit
def coefficient_parity(m,obs,h):
 ob=model_lib.preprocess_observation(None,obs,train=False);mask,cache=m.encode_condition(ob)
 c=m.sample_coefficients_from_cache(mask,cache,history=h,num_steps=1)
 return m.denormalize(m.polynomial.reconstruct(c)),m.sample_from_cache(mask,cache,history=h,num_steps=1)
cp0,ap0=jax.block_until_ready(coefficient_parity(model0,obs,h));cp2,ap2=jax.block_until_ready(coefficient_parity(model2,obs,h))
checks={'B0_prediction_vs_original_rad':maxdiff(p,p0),'B0_loss_vs_original':max(maxdiff(loss[k],l0[k]) for k in loss),'B0_direct_coeff_decode_rad':maxdiff(cp0,ap0),'B2_direct_coeff_decode_rad':maxdiff(cp2,ap2)}
# Compare fixed eval batch2 with original API at batch1; no padding, no temporal mask change.
single=[]
for i in range(len(samples)):
 ob=jax.tree.map(lambda x:x[i:i+1],obs)
 pred,_=jax.block_until_ready(forward(original,ob,h[i:i+1],target[i:i+1],jax.random.key(42)));single.append(np.asarray(pred[0]))
checks['original_batch2_vs_singletons_rad']=maxdiff(np.stack(single),p)
assert all(checks[k]<1e-5 for k in ('B0_prediction_vs_original_rad','B0_direct_coeff_decode_rad','B2_direct_coeff_decode_rad','original_batch2_vs_singletons_rad'))
assert checks['B0_loss_vs_original']<1e-5
assert all(np.isfinite(np.asarray(v)).all() for v in [p2,*l2.values()])
out.mkdir(parents=True)
result={'training_adapter_passed':True,'training_updates':0,'B2_is_trained':False,'checks':checks,'same_parameter_layout':True,'same_parameter_array_objects':shared_arrays,'parameter_elements':sum(int(np.prod(v.shape)) for v in leaves[0]),'parameter_sha256':parameter_sha,
 'source_commit':cfg['source_commit'],'adapter_sha256':sha(flash_openpi.__file__),'script_sha256':sha(__file__),'batch_size':len(samples),'window_indices':cfg['window_indices'],'rgb_verified':len(hashes),'evaluation_dtype':'float32','loss_noise_seed':42,
 'B0_forward_losses':{k:np.asarray(v).tolist() for k,v in l0.items()},'B2_untrained_forward_losses':{k:np.asarray(v).tolist() for k,v in l2.items()},'loss_interpretation':'Finite functional check only; not performance comparison or training.'}
(out/'adapter_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
