#!/usr/bin/env python3
"""Real two-window forward/coordinate checks in the explicitly selected original runtime.

B2 uses the EXISTING Absolute weights only as an untrained functional smoke test.
No optimizer/gradient updates, deployment, future-conditioned inference or B2
accuracy claim. Install the research wheel in a separate --target directory.
"""
import hashlib,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from openpi.models import model as model_lib
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.flash_native import NativeDataset,bind_stats,preflight
from openpi.training.flash_checkpoint import parameter_digest
from scripts.run_training import load_experiment
from trc.models.flash_coordinates import FlashCoordinates
from trc.data.action_schema import GROUPS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory(root):return {str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns] for p in root.rglob('*') if p.is_file()}
cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(cfg['output']).resolve();checkpoint=Path(cfg['checkpoint']).resolve()
if out.exists() or out.is_relative_to(Path.cwd().resolve()) or out.is_relative_to(checkpoint):raise ValueError('New research output outside originals required')
if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=cfg['source_commit']:raise ValueError('Source version mismatch')
if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Dirty source')
if sha(cfg['training_contract'])!=cfg['training_contract_sha256']:raise ValueError('Contract mismatch')
contract=json.loads(Path(cfg['training_contract']).read_text());experiment=bind_stats(load_experiment(cfg['experiment_module']));verified=preflight(experiment)
if verified!=contract['dataset'] or str(experiment.model)!=contract['model']:raise ValueError('Model/data/stats mismatch')
completion=json.loads((checkpoint/'_CHECKPOINT_METADATA').read_text())
if not completion.get('commit_timestamp_nsecs'):raise ValueError('Incomplete checkpoint')
before=inventory(checkpoint);params=model_lib.restore_params(checkpoint/'params',restore_type=np.ndarray);parameter_sha=parameter_digest(params)
model=experiment.model.load(jax.device_put(params),remove_extra_params=False);model.eval();del params
inputs=json.loads(Path(cfg['paired_inputs']).read_text())
if sha(cfg['paired_inputs'])!=cfg['paired_inputs_sha256']:raise ValueError('Paired input mismatch')
reference=np.load(cfg['reference_arrays'],allow_pickle=False)
if sha(cfg['reference_arrays'])!=cfg['reference_arrays_sha256']:raise ValueError('Reference mismatch')
dataset=NativeDataset(experiment,tokenizer=PaligemmaTokenizer(max_len=model.max_token_len));dataset.records={r['id']:r for r in inputs['records']};dataset.numeric={}
original_image=dataset.image
rgb_verified=[]
def image_checked(index,role):
 pixels=original_image(index,role)
 expected=inputs['rgb_sha256'][f"{index['id']}:{index['start']}:{role}"]
 assert hashlib.sha256(pixels.tobytes()).hexdigest()==expected
 rgb_verified.append(expected)
 return pixels
dataset.image=image_checked
indices=cfg['window_indices'];samples=[dataset.observation(inputs['windows'][i]) for i in indices]
obs=jax.tree.map(lambda *x:jnp.concatenate(x,axis=0),*[v[0] for v in samples]);history=jnp.asarray(np.concatenate([v[1] for v in samples]));target=jnp.asarray(np.concatenate([v[2] for v in samples]))
np.testing.assert_array_equal(np.asarray(target[:,2:52]),reference['truth'][indices]);np.testing.assert_array_equal(np.asarray(history),reference['history'][indices])
coordinates=FlashCoordinates(cfg['coordinates']['mode'],model.action_q01,model.action_q99,np.asarray(model.polynomial.scale),tuple(cfg['coordinates']['compatible_groups']))
absolute=FlashCoordinates('absolute',model.action_q01,model.action_q99,np.asarray(model.polynomial.scale),tuple(GROUPS))
@nnx.jit
def original_p0_graph(m,observation,history):
 observation=model_lib.preprocess_observation(None,observation,train=False)
 prefix_mask,kv_cache=m.encode_condition(observation)
 source=m.polynomial.fit_history(m.normalize(history))
 coeff=source+m.coefficient_velocity(source,jnp.zeros((1,)),prefix_mask,kv_cache)
 prediction=m.denormalize(m.polynomial.reconstruct(coeff))
 direct=m.sample_from_cache(prefix_mask,kv_cache,history=history,num_steps=1)
 return coeff,prediction,direct
replayed=[]
for row in range(len(indices)):
 ob=jax.tree.map(lambda x:x[row:row+1],obs)
 _,replay,_=jax.block_until_ready(original_p0_graph(model,ob,history[row:row+1]))
 replayed.append(np.asarray(replay[0]))
replayed=np.stack(replayed)
@nnx.jit
def forward(m,obs,h):
 obs=model_lib.preprocess_observation(None,obs,train=False);mask,cache=m.encode_condition(obs)
 source=m.polynomial.fit_history(m.normalize(h));rel=coordinates.coefficients(source,h,xp=jnp)
 c0=absolute.coefficients(source,h,xp=jnp);cp0=c0+m.coefficient_velocity(c0,jnp.zeros((h.shape[0],)),mask,cache)
 cpr=rel+m.coefficient_velocity(rel,jnp.zeros((h.shape[0],)),mask,cache)
 cp2=coordinates.coefficients(cpr,h,inverse=True,xp=jnp)
 p0=m.denormalize(m.polynomial.reconstruct(cp0));p2=m.denormalize(m.polynomial.reconstruct(cp2))
 native=m.sample_from_cache(mask,cache,history=h,num_steps=1)
 return source,rel,cp0,cpr,p0,p2,native,mask,cache
source,relative_source,cp0,cpr,p0,p2,native,mask,cache=jax.block_until_ready(forward(model,obs,history))
# Future supervision is only used AFTER the causal forward for mathematical/loss checks.
target_coeff=model.polynomial.fit_target(model.normalize(target),boundary_constraint=model.boundary_constraint)
relative_target=coordinates.coefficients(target_coeff,history,xp=jnp)
physical_relative=coordinates.actions(target,history,xp=jnp)
normalized_relative=coordinates.normalize_coordinates(physical_relative,xp=jnp)
refit_target=model.polynomial.fit_target(normalized_relative,boundary_constraint=model.boundary_constraint)
refit_history=model.polynomial.fit_history(model.normalize(history)-model.normalize(history[:,-1])[:,None])
@nnx.jit
def losses(m,source,target,relative_source,relative_target,mask,cache):
 t=jnp.full((source.shape[0],),.37)
 # Same noise realization in both coordinate systems (as in original source-noise convention).
 noise=m.history_noise_std*jax.random.normal(jax.random.key(0),source.shape)
 desired=target-(source+noise);results=[]
 for s,g in ((source+noise,target),(relative_source+noise,relative_target)):
  mixture=(1-t[:,None,None])*s+t[:,None,None]*g
  v=m.coefficient_velocity(mixture,t,mask,cache);v0=m.coefficient_velocity(s,jnp.zeros_like(t),mask,cache)
  fm=jnp.mean((v-desired)**2);consistency=jnp.mean((s+v0-g)**2)
  results.append(jnp.stack([fm,consistency,fm+m.consistency_weight*consistency]))
 return jnp.stack(results)
loss_values=np.asarray(jax.block_until_ready(losses(model,source,target_coeff,relative_source,relative_target,mask,cache)))
def maxdiff(x,y):return float(np.max(np.abs(np.asarray(x,dtype=float)-np.asarray(y,dtype=float))))
recovered=coordinates.actions(physical_relative,history,inverse=True,xp=jnp)
recovered_coeff=coordinates.coefficients(relative_target,history,inverse=True,xp=jnp)
reconstructed=model.denormalize(model.polynomial.reconstruct(target_coeff))
rel_reconstructed=coordinates.actions(coordinates.normalize_coordinates(model.polynomial.reconstruct(relative_target),inverse=True,xp=jnp),history,inverse=True,xp=jnp)
checks={'physical_roundtrip_rad':maxdiff(recovered,target),'coefficient_roundtrip':maxdiff(recovered_coeff,target_coeff),
 'high_degree_change':maxdiff(relative_target[:,1:],target_coeff[:,1:]),'decode_50_rad':maxdiff(rel_reconstructed,reconstructed),
 'target_refit_decode_normalized':maxdiff(model.polynomial.reconstruct(relative_target),model.polynomial.reconstruct(refit_target)),
 'target_minus_source_invariance':maxdiff(relative_target-relative_source,target_coeff-source),
 'derivative_invariance_rad_per_s':maxdiff(jnp.diff(physical_relative,axis=1)*30,jnp.diff(target,axis=1)*30),
 'B0_original_API_rad':maxdiff(p0,native),'absolute_coeff_identity':maxdiff(absolute.coefficients(source,history,xp=jnp),source),
 'history_naive_refit_vs_preserved_source':maxdiff(refit_history,relative_source)}
for k in ('physical_roundtrip_rad','decode_50_rad','B0_original_API_rad'):assert checks[k]<2e-5,(k,checks[k])
assert checks['high_degree_change']==0 and checks['absolute_coeff_identity']==0
assert np.isfinite(loss_values).all() and np.isfinite(np.asarray(p2)).all()
# Perturb FUTURE labels only: causal forward accepts no target at all.
changed=target.at[:,3:].add(123);assert not np.array_equal(np.asarray(changed),np.asarray(target))
assert inventory(checkpoint)==before
out.mkdir(parents=True)
np.savez(out/'real_batch_check.npz',B0_original_singleton_graph=replayed,history=np.asarray(history),target=np.asarray(target),B0_prediction=np.asarray(p0),B2_untrained_functional_prediction=np.asarray(p2),source=np.asarray(source),relative_source=np.asarray(relative_source),target_coeff=np.asarray(target_coeff),relative_target=np.asarray(relative_target))
result={'status':'real_batch_forward_completed','training_updates':0,'B2_is_trained':False,'B2_accuracy_result':False,'batch_size':len(indices),'window_indices':indices,
 'coordinate_mode':coordinates.mode,'input_shapes':{'history':list(history.shape),'target':list(target.shape),'prediction':list(p0.shape),'tokens':list(cpr.shape)},'parameter_sha256':parameter_sha,
 'source_commit':cfg['source_commit'],'training_contract_sha256':cfg['training_contract_sha256'],'checks':checks,'checkpoint_unchanged':True,'rgb_raw_pixels_verified':len(rgb_verified),'B0_p0_graph_vs_batched_graph_max_rad':maxdiff(replayed,p0),
 'loss_columns':['flow_matching','consistency','combined'],'same_time_noise_forward_losses':loss_values.tolist(),'future_targets_not_in_inference_signature':True,
 'script_sha256':sha(__file__),'adapter_sha256':sha(__import__('trc.models.flash_coordinates',fromlist=['x']).__file__),
 'config':cfg,'runtime':{'jax':jax.__version__,'device':str(jax.devices()[0]),'peak_bytes':int(jax.devices()[0].memory_stats()['peak_bytes_in_use'])}}
(out/'forward_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
