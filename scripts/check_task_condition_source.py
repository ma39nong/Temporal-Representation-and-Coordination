"""Explicit pinned FLASH CPU smoke; real inputs, synthetic task metadata only.

No optimizer update or robot command. Resources come from a private JSON config.
The checkpoint test saves/restores the native velocity head plus condition schema;
all unchanged body weights remain bound by the complete Step8000 parameter SHA.
"""
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from trc.models.task_condition import causal_condition, condition_observation


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(path):
    cfg = json.loads(Path(path).read_text())
    out = Path(cfg['output'])
    if out.exists(): raise FileExistsError(out)
    import jax
    import jax.numpy as jnp
    from flax import nnx, traverse_util
    import optax
    import torch
    from openpi.models import model as model_lib
    from openpi.models.pi0_flash import Pi0FlashConfig
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.training.flash_native import NativeDataset
    from openpi.training.flash_checkpoint import parameter_digest, is_vlm_parameter
    if jax.default_backend() != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU only; no interference with active GPU jobs')
    torch.set_num_threads(1)
    for relative, expected in json.loads(Path(cfg['source_manifest']).read_text())['files'].items():
        if sha(Path(cfg['source_root'])/relative) != expected: raise ValueError('Source snapshot changed')
    if sha(cfg['training_contract']) != cfg['training_contract_sha256']: raise ValueError('Contract identity')
    if sha(cfg['norm_stats']) != cfg['normalization_sha256']: raise ValueError('Statistics identity')
    node = ast.parse(json.loads(Path(cfg['training_contract']).read_text())['model'], mode='eval').body
    if not isinstance(node, ast.Call) or node.func.id not in ('Pi0FlashConfig', 'CoordinateFlashConfig'):
        raise ValueError('Unrecognized model contract')
    kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    if kw.pop('coordinate_mode', 'absolute') != 'absolute': raise ValueError('Absolute required')
    mc = Pi0FlashConfig(**{**kw, 'dtype':'float32', 'freeze_vlm':True})
    params = model_lib.restore_params(Path(cfg['checkpoint'])/'params', restore_type=np.ndarray)
    if parameter_digest(params) != cfg['parameter_sha256']: raise ValueError('Original Step8000 identity')
    m = mc.load(jax.device_put(params), remove_extra_params=False); m.eval()
    out.mkdir(parents=True)
    print('Step8000 loaded and full parameter SHA verified', flush=True)
    data = NativeDataset.__new__(NativeDataset)
    data.config = SimpleNamespace(data=SimpleNamespace(extra_delta_transform=False))
    record = next(r for r in json.loads(Path(cfg['training_manifest']).read_text())['records']
                  if r['source_recording_id'] == cfg['recording_id'])
    data.records={record['id']:record}; data.numeric={}
    data.stats=normalize.load(Path(cfg['norm_stats']).parent)
    data.tokenizer=PaligemmaTokenizer(max_len=mc.max_token_len)
    obs, history, target = data.observation({'id':record['id'], 'start':cfg['current_frame']})
    with np.load(cfg['reference_arrays']) as ref:
        np.testing.assert_array_equal(history, ref['history'])
        np.testing.assert_array_equal(target[:,2:52], ref['truth'])
    if cfg['crossing_cuts']: raise ValueError('Window crosses cuts')
    @nnx.jit
    def prefix(model, ob):
        return model.encode_condition(model_lib.preprocess_observation(None, ob, train=False))
    @nnx.jit
    def infer(model, mask, cache):
        return model.sample_from_cache(mask, cache, history=jnp.asarray(history), num_steps=1)
    h,t = jnp.asarray(history),jnp.asarray(target)
    source=m.polynomial.fit_history(m.normalize(h))
    def run(ob):
        started=time.monotonic()
        mask,cache=jax.block_until_ready(prefix(m, ob))
        velocity=jax.block_until_ready(m.coefficient_velocity(source,jnp.zeros((1,),jnp.float32),mask,cache))
        actions=jax.block_until_ready(infer(m,mask,cache))
        loss=jax.block_until_ready(m.loss_from_cache(jax.random.key(42),mask,cache,history=h,target_ext=t,train=True))
        if actions.shape != (1,50,58) or not np.isfinite(np.asarray(actions)).all(): raise ValueError('Actions invalid')
        return mask,cache,velocity,actions,loss,time.monotonic()-started
    original=run(obs)
    disabled=condition_observation(obs, mode='disabled')
    assert disabled is obs
    replay=run(disabled)
    delta=lambda a,b:float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
    checks={'disabled_velocity_max_difference':delta(original[2],replay[2]),
            'disabled_physical_max_difference_rad':delta(original[3],replay[3])}
    if any(v>1e-5 for v in checks.values()): raise ValueError('Disabled replay failure')
    result={'data_scope':cfg.get('data_scope', 'ONE REAL single-pick training window; all new goal/progress metadata SYNTHETIC FUNCTIONAL TEST'),
            'new_trainable_parameters':0, 'optimizer_updates':0, 'source_commit':cfg['source_commit'],
            'parameter_sha256':cfg['parameter_sha256'], 'normalization_sha256':cfg['normalization_sha256'],
            'source_manifest_sha256':sha(cfg['source_manifest']), 'script_sha256':sha(__file__),
            'condition_module_sha256':sha(cfg['condition_module']), 'checks':checks,
            'recording_id':cfg['recording_id'], 'current_frame':cfg['current_frame'],
            'FP32':True, 'backend':'cpu', 'batch_size':1, 'action_shape':[1,50,58],
            'history_span_seconds':7/30, 'variants':{}}
    last=None
    for mode in cfg.get('condition_modes', ('language_goal','explicit_goal','goal_progress')):
        scenarios=cfg.get('condition_scenarios', [dict(label=str(n),target_count=n,observation_time=0,reset_at=0,reset_evidence='SYNTHETIC reset for interface smoke',coverage_until=0) for n in (1,2,3)])
        for scenario in scenarios:
            args={k:v for k,v in scenario.items() if k!='label'}
            count=scenario['label']
            c=causal_condition(**args)
            ob=condition_observation(obs,[c],mode=mode,encode=data.tokenizer._tokenizer.encode,max_len=mc.max_token_len)
            r=run(ob);last=(ob,r,c)
            result['variants'][f'{mode}_{count}']={'token_count':int(ob.tokenized_prompt_mask.sum()),
                'loss':{k:np.asarray(v).tolist() for k,v in r[4].items()},
                'physical_change_from_original_max_rad':delta(original[3],r[3]),
                'wall_seconds_including_first_compilation':r[5]}
            print(mode,count,'forward/loss passed',flush=True)
    ob,r,c=last
    @nnx.jit
    def backward(model):
        def loss(mm):return jnp.mean(mm.loss_from_cache(jax.random.key(42),r[0],r[1],history=h,target_ext=t,train=True)['loss'])
        return nnx.value_and_grad(loss,argnums=nnx.DiffState(0,mc.get_trainable_filter()))(model)
    loss,gr=jax.block_until_ready(backward(m))
    flat=traverse_util.flatten_dict(gr.to_pure_dict())
    if any(is_vlm_parameter(k) for k in flat):raise ValueError('VLM gradient leak')
    if not all(np.isfinite(np.asarray(v)).all() for v in flat.values()):raise ValueError('Nonfinite gradients')
    result['gradient']={'loss':float(loss),'norm':float(optax.global_norm(gr)),
        'leaves':len(flat),'elements':sum(v.size for v in flat.values()),'VLM_gradient_leaves':0,
        'scope':'Full existing Expert, conditioned real batch; no new neural module'}
    # Since conditioning adds zero weights, persist its exact configuration and
    # native head, bind unchanged body to full SHA; never claim a new full model.
    st=nnx.state(m.action_out_proj,nnx.Param)
    leaves,definition=jax.tree.flatten(st)
    np.savez(out/'native_head.npz',**{f'leaf{i}':np.asarray(v) for i,v in enumerate(leaves)})
    (out/'condition.json').write_text(json.dumps(asdict(c),indent=2))
    with np.load(out/'native_head.npz') as saved:
        restored=jax.tree.unflatten(definition,[jnp.asarray(saved[f'leaf{i}']) for i in range(len(leaves))])
    nnx.update(m.action_out_proj,restored)
    from trc.models.task_condition import TaskCondition
    d=json.loads((out/'condition.json').read_text());d['evidence']=tuple(d['evidence'])
    again=condition_observation(obs,[TaskCondition(**d)],mode=mode,encode=data.tokenizer._tokenizer.encode,max_len=mc.max_token_len)
    np.testing.assert_array_equal(ob.tokenized_prompt,again.tokenized_prompt)
    after=run(again)
    result['checkpoint']={'scope':'native velocity head + condition JSON; unchanged full body bound by SHA',
        'prediction_max_difference_rad':delta(r[3],after[3]),'head_sha256':sha(out/'native_head.npz')}
    if result['checkpoint']['prediction_max_difference_rad']>1e-5:raise ValueError('Restored output differs')
    if parameter_digest(nnx.state(m,nnx.Param).to_pure_dict())!=cfg['parameter_sha256']:raise ValueError('Weights mutated')
    (out/'acceptance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__': main(sys.argv[1])
