#!/usr/bin/env python3
"""Read-only original-runtime replay: prepared inputs, staged tensors, three APIs.

No fitting new statistics, parameter mutation, optimizer or training. Launch
with the EXACT upstream runtime wrapper. Compiler policy is recorded, never
changed after JAX initialization. Optional paired training only after parity.
"""
import hashlib,json,os,subprocess,sys
from dataclasses import replace
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.flash_native import NativeDataset,bind_stats,preflight
from openpi.training.flash_checkpoint import parameter_digest
from scripts.run_training import load_experiment


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest_array(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def inventory(root):return {str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns] for p in root.rglob('*') if p.is_file()}
def diff(x,y):
 d=np.asarray(x,np.float64)-np.asarray(y,np.float64)
 return {'max':float(np.abs(d).max()),'rmse':float(np.sqrt(np.mean(d*d)))}

cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(cfg['output']).resolve();checkpoint=Path(cfg['checkpoint']).resolve()
if out.exists() or out.is_relative_to(Path.cwd().resolve()) or out.is_relative_to(checkpoint):raise ValueError('New separate output required')
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if commit!=cfg['source_commit'] or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Source mismatch/dirty')
if sha(cfg['training_contract'])!=cfg['training_contract_sha256']:raise ValueError('Contract hash mismatch')
contract=json.loads(Path(cfg['training_contract']).read_text());exp=bind_stats(load_experiment(cfg['experiment_module']));verified=preflight(exp)
if verified!=contract['dataset'] or str(exp.model)!=contract['model']:raise ValueError('Actual configuration mismatch')
inputs=json.loads(Path(cfg['paired_inputs']).read_text())
if sha(cfg['paired_inputs'])!=cfg['paired_inputs_sha256']:raise ValueError('Window plan mismatch')
dataset=NativeDataset(exp,tokenizer=PaligemmaTokenizer(max_len=exp.model.max_token_len));dataset.records={r['id']:r for r in inputs['records']};dataset.numeric={}
original_image=dataset.image;rgb_checks=[]
def checked_image(index,role):
 pixels=original_image(index,role);key=f"{index['id']}:{index['start']}:{role}"
 if digest_array(pixels)!=inputs['rgb_sha256'][key]:raise ValueError('Raw RGB mismatch')
 rgb_checks.append(key);return pixels
dataset.image=checked_image
samples=[dataset.observation(inputs['windows'][i]) for i in cfg['window_indices']]
before=inventory(checkpoint);params=model_lib.restore_params(checkpoint/'params',restore_type=np.ndarray);parameter_sha=parameter_digest(params)
inference_config=replace(exp.model,dtype=cfg['evaluation_precision']) if cfg.get('evaluation_precision') else exp.model
m=inference_config.load(jax.device_put(params),remove_extra_params=False);m.eval();del params
out.mkdir(parents=True)
@nnx.jit
def stages(m,observation,history):
 ob=model_lib.preprocess_observation(None,observation,train=False)
 tokens,pmask,ar=m.embed_prefix(ob)
 _,cache=m.PaliGemma.llm([tokens,None],mask=make_attn_mask(pmask,ar),positions=jnp.cumsum(pmask,axis=1)-1)
 normalized=m.normalize(history);source=m.polynomial.fit_history(normalized)
 velocity=m.coefficient_velocity(source,jnp.zeros((1,)),pmask,cache);coeff=source+velocity
 reconstructed=m.polynomial.reconstruct(coeff)
 values={'normalized_history':normalized,'source_coeff':source,'prefix_tokens':tokens,'prefix_mask':pmask,
         'expert_velocity':velocity,'predicted_coeff':coeff,'normalized_actions':reconstructed,'physical_actions':m.denormalize(reconstructed)}
 for i,x in enumerate(jax.tree.leaves(cache)):values[f'condition_cache_{i}']=x
 for key,image in ob.images.items():values['preprocessed_'+key]=image
 return values
@nnx.jit
def p0(m,obs,h):
 ob=model_lib.preprocess_observation(None,obs,train=False);mask,cache=m.encode_condition(ob)
 source=m.polynomial.fit_history(m.normalize(h));c=source+m.coefficient_velocity(source,jnp.zeros((1,)),mask,cache)
 return c,m.denormalize(m.polynomial.reconstruct(c)),m.sample_from_cache(mask,cache,history=h,num_steps=1)
@nnx.jit
def original(m,obs,h,key):return m.sample_actions(key,obs,history=h,num_steps=1)
# The research Absolute mode is an identity transform; use the actual class from
# a caller-specified research wheel, loaded without altering source dependencies.
# The launcher supplies the research package as a separate explicit runtime path.
if cfg.get('research_runtime'):
 import site
 site.addsitedir(cfg['research_runtime'])
from trc.models.flash_coordinates import FlashCoordinates
from trc.data.action_schema import GROUPS
adapter=FlashCoordinates('absolute',m.action_q01,m.action_q99,np.asarray(m.polynomial.scale),tuple(GROUPS))
@nnx.jit
def research(m,obs,h):
 ob=model_lib.preprocess_observation(None,obs,train=False);mask,cache=m.encode_condition(ob)
 source=adapter.coefficients(m.polynomial.fit_history(m.normalize(h)),h,xp=jnp)
 c=source+m.coefficient_velocity(source,jnp.zeros((1,)),mask,cache)
 c=adapter.coefficients(c,h,inverse=True,xp=jnp)
 return m.denormalize(m.polynomial.reconstruct(c))
reports=[]
for ix,(obs,h,target) in zip(cfg['window_indices'],samples):
 prepared={'history':h,'state':obs.state,'prompt':obs.tokenized_prompt,'prompt_mask':obs.tokenized_prompt_mask,'target_ext':target}
 for k,v in obs.images.items():prepared['image_'+k]=v;prepared['mask_'+k]=obs.image_masks[k]
 np.savez_compressed(out/f'window{ix}_prepared.npz',**prepared)
 h=jnp.asarray(h)
 trace=jax.block_until_ready(stages(m,obs,h));trace={k:np.asarray(v) for k,v in trace.items()}
 np.savez_compressed(out/f'window{ix}_stages.npz',**{k:v.astype(np.float32) if str(v.dtype)=='bfloat16' else v for k,v in trace.items()},basis=np.asarray(m.polynomial.basis_ext),scale=np.asarray(m.polynomial.scale),q01=np.asarray(m.action_q01),q99=np.asarray(m.action_q99))
 outputs={};repeat=[]
 for name,fn in [('original',lambda key:original(m,obs,h,key)),('p0',lambda key:p0(m,obs,h)[1]),('research',lambda key:research(m,obs,h))]:
  values=[]
  for seed in cfg.get('seeds',[0,1,42]):values.append(np.asarray(jax.block_until_ready(fn(jax.random.key(seed)))))
  outputs[name]=values[0]
  repeat.append({'entry':name,'different_seed_max_rad':max(diff(v,values[0])['max'] for v in values)})
 np.savez(out/f'window{ix}_outputs.npz',**outputs)
 reports.append({'window':ix,'prepared_hashes':{k:digest_array(v) for k,v in prepared.items()},'stage_hashes':{k:digest_array(v) for k,v in trace.items()},
                 'within_process_repeats':repeat,'entry_parity':{name:diff(value,outputs['original']) for name,value in outputs.items()},
                 'staged_vs_original':diff(trace['physical_actions'],outputs['original'])})
 print(json.dumps(reports[-1]),flush=True)
if inventory(checkpoint)!=before:raise ValueError('Checkpoint changed')
result={'status':'replay_completed','source_commit':commit,'parameter_sha256':parameter_sha,'normalization_sha256':verified['normalization_sha256'],
 'configuration':cfg,'model_config':str(exp.model),'inference_model_config':str(inference_config),'checkpoint_unchanged':True,'raw_rgb_verified':rgb_checks,'windows':reports,
 'randomness':'sample_actions ignores rng; train=False, source is measured history, NFE=1, no sampling noise',
 'environment':{k:os.environ.get(k) for k in ('XLA_FLAGS','JAX_DEFAULT_MATMUL_PRECISION','NVIDIA_TF32_OVERRIDE','LD_LIBRARY_PATH','LD_PRELOAD','JAX_ENABLE_X64')},
 'runtime':{'jax':jax.__version__,'numpy':np.__version__,'devices':[str(x) for x in jax.devices()]},
 'script_sha256':sha(__file__)}
(out/'trace_report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'completed','output':str(out)}),flush=True)
