#!/usr/bin/env python3
"""Zero-update FullFT evaluation in original runtime, on fixed audited raw windows.

Run in the configured upstream checkout (read only). This script and outputs
live in a separate research directory. Reuses NativeDataset and Pi0FlashConfig;
never imports an alternative FLASH expert or starts an optimizer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import os
from dataclasses import replace
sys.dont_write_bytecode=True
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def inventory(root):
    return {str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns] for p in root.rglob('*') if p.is_file()}


def main(config_path):
    import jax
    import jax.numpy as jnp
    from flax import nnx,traverse_util
    from openpi.models import model as model_lib
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training.flash_native import NativeDataset,bind_stats,preflight
    from openpi.training.flash_checkpoint import parameter_digest,is_vlm_parameter
    from scripts.run_training import load_experiment
    cfg=json.loads(Path(config_path).read_text());out=Path(cfg['output']).resolve()
    if out.is_relative_to(Path.cwd().resolve()) or out.exists():raise ValueError('New output outside original checkout required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=cfg['source_commit']:raise ValueError('Source commit mismatch')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Dirty original source')
    checkpoint=Path(cfg['checkpoint']);before=inventory(checkpoint)
    completion=json.loads((checkpoint/'_CHECKPOINT_METADATA').read_text())
    if not completion.get('commit_timestamp_nsecs') or not (checkpoint/'params/_METADATA').exists():raise ValueError('Incomplete checkpoint')
    if int(checkpoint.name)!=cfg['expected_step']:raise ValueError('Step mismatch')
    if sha(cfg['training_contract'])!=cfg['training_contract_sha256']:raise ValueError('Training contract pin mismatch')
    contract=json.loads(Path(cfg['training_contract']).read_text())
    experiment=bind_stats(load_experiment(cfg['experiment_module']))
    verified=preflight(experiment)
    training_model=experiment.model
    if 'coordinate_mode' in cfg:
        import site
        site.addsitedir(cfg['research_runtime'])
        from trc.models import flash_openpi
        from trc.models.flash_openpi import CoordinateFlashConfig
        if sha(flash_openpi.__file__)!=cfg['expected_adapter_sha256']:raise ValueError('Training adapter identity mismatch')
        if cfg.get('evaluation_precision')!='float32' or os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest':raise ValueError('Paired P1 evaluation requires shared FP32/highest policy')
        if not all(flag in os.environ.get('XLA_FLAGS','') for flag in ('--xla_gpu_autotune_level=0','--xla_gpu_deterministic_ops=true')):raise ValueError('Pinned XLA policy required')
        training_model=CoordinateFlashConfig.from_base(training_model,cfg['coordinate_mode'])
    if verified!=contract['dataset'] or str(training_model)!=contract['model']:raise ValueError('Actual model/data/normalization contract mismatch')
    inference_model=replace(training_model,dtype=cfg['evaluation_precision']) if cfg.get('evaluation_precision') else training_model
    if experiment.model.freeze_vlm or experiment.data.extra_delta_transform:raise ValueError('Expected FullFT Absolute model')
    # Full checkpoint supplies every leaf, including vision and language. No partial/random load.
    params=model_lib.restore_params(checkpoint/'params',restore_type=np.ndarray)
    flat=traverse_util.flatten_dict(params)
    counts={'vision':0,'language':0,'expert_and_projections':0};elements={k:0 for k in counts}
    for key,value in flat.items():
        group='vision' if key[:2]==('PaliGemma','img') else ('language' if is_vlm_parameter(key) else 'expert_and_projections')
        counts[group]+=1;elements[group]+=int(np.size(value))
    if not all(counts.values()):raise ValueError('Incomplete VLM/expert checkpoint')
    parameter_hash=parameter_digest(params)
    model=inference_model.load(jax.device_put(params),remove_extra_params=False);model.eval();del params,flat
    data=NativeDataset(experiment,tokenizer=PaligemmaTokenizer(max_len=experiment.model.max_token_len))
    inputs=json.loads(Path(cfg['paired_inputs']).read_text())
    reference=np.load(cfg['reference_arrays'],allow_pickle=False)
    if sha(cfg['paired_inputs'])!=cfg['paired_inputs_sha256'] or sha(cfg['reference_arrays'])!=cfg['reference_arrays_sha256']:raise ValueError('Paired input pin mismatch')
    # Reuse original NativeDataset.window/image/observation, extend only in-memory
    # records to include explicitly held-out sources. Current stats stay unchanged.
    data.records={r['id']:r for r in inputs['records']};data.numeric={}
    train_sources=set(data.manifest['train_sources'])
    original_image=data.image
    rgb_hashes=[]
    def checked_image(index,role):
        pixels=original_image(index,role)
        h=hashlib.sha256(pixels.tobytes()).hexdigest()
        expected=inputs['rgb_sha256'][f"{index['id']}:{index['start']}:{role}"]
        if h!=expected:raise ValueError('Current decoded RGB differs from historical paired observation')
        rgb_hashes.append(h)
        return pixels
    data.image=checked_image
    @nnx.jit
    def predict(m,obs,history):
        obs=model_lib.preprocess_observation(None,obs,train=False)
        mask,cache=m.encode_condition(obs)
        source=m.polynomial.fit_history(m.normalize(history))
        coeff=(m.sample_coefficients_from_cache(mask,cache,history=history,num_steps=1) if hasattr(m,'sample_coefficients_from_cache') else source+m.coefficient_velocity(source,jnp.zeros((1,)),mask,cache))
        prediction=m.denormalize(m.polynomial.reconstruct(coeff))
        direct=m.sample_from_cache(mask,cache,history=history,num_steps=1)
        return coeff,prediction,direct
    @nnx.jit
    def oracle(m,target):
        c=m.polynomial.fit_target(m.normalize(target),boundary_constraint=m.boundary_constraint)
        return c,m.denormalize(m.polynomial.reconstruct(c))
    out.mkdir(parents=True);rows=[];values=[];times=[];max_api=0.;max_truth=0.;max_history=0.;future_test=False
    for i,w in enumerate(inputs['windows']):
        obs,history,target=data.observation(w)
        np.testing.assert_array_equal(target[0,2:52],reference['truth'][i])
        np.testing.assert_array_equal(history[0],reference['history'][i])
        tick=time.perf_counter()
        cp,p,d=jax.block_until_ready(predict(model,obs,jnp.asarray(history)))
        times.append(time.perf_counter()-tick)
        cp,p,d=map(np.asarray,(cp,p,d));max_api=max(max_api,float(np.max(np.abs(p-d))))
        np.testing.assert_allclose(p,d,rtol=1e-5,atol=2e-5)
        cg,o=oracle(model,jnp.asarray(target));cg,o=map(np.asarray,(cg,o))
        if i==0:
            # Future perturbation changes labels, never obs/current state/history/prediction.
            ident,start=w['id'],w['start'];actions,states,tasks=data.numeric[ident]
            changed_a,changed_s=actions.copy(),states.copy();changed_a[start+1:]+=.123;changed_s[start+1:]+=.456
            data.numeric[ident]=(changed_a,changed_s,tasks)
            obs2,h2,t2=data.observation(w)
            for x,y in zip(jax.tree.leaves(obs),jax.tree.leaves(obs2)):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(history,h2)
            if np.array_equal(target,t2):raise ValueError('Perturbation must change future supervision')
            _,p2,_=jax.block_until_ready(predict(model,obs2,jnp.asarray(h2)))
            np.testing.assert_allclose(p,p2,rtol=1e-6,atol=1e-6)
            data.numeric[ident]=(actions,states,tasks);future_test=True
        values.append((target[0,2:52],o[0],p[0],cg[0],cp[0],history[0],data.numeric[w['id']][1][w['start']]))
        record=data.records[w['id']];seen=record['source_recording_id'] in train_sources
        rows.append({'sample_id':w['id'],'current_row':w['start'],'current_frame':w['start'],'episode_index':record['episode'],
                     'source_recording_id':record['source_recording_id'],'source_path':record['path'],
                     'training_source':seen,'scope':'training_source_diagnostic' if seen else 'candidate_unseen_source','fps':30})
        if (i+1)%25==0:print(json.dumps({'evaluated':i+1,'total':len(inputs['windows']),'forward_s':times[-1]}),flush=True)
    a,o,p,cg,cp,h,s=map(np.stack,zip(*values))
    if not all(np.isfinite(x).all() for x in (a,o,p,cg,cp,h,s)):raise ValueError('Nonfinite export')
    np.savez(out/'aligned.npz',truth=a,oracle=o,prediction=p,c_gt=cg,c_pred=cp,history=h,current_state=s,
             scale=np.asarray(model.polynomial.scale),basis=np.asarray(model.polynomial.basis_ext[2:52]),
             q01=np.asarray(model.action_q01),q99=np.asarray(model.action_q99),valid=np.ones(a.shape,bool))
    if inventory(checkpoint)!=before:raise ValueError('Checkpoint changed during evaluation')
    meta={'data_kind':'real','inference_mode':'fresh_fullft_checkpoint_paired_raw_inputs','evaluation_scope':'mixed_485_training_source_20_candidate_unseen',
          'checkpoint_id':str(checkpoint),'checkpoint_step':cfg['expected_step'],'checkpoint_parameter_sha256':parameter_hash,
          'checkpoint_files_unchanged':True,'checkpoint_completion':completion,'parameter_leaf_counts':counts,'parameter_elements':elements,
          'strict_load_remove_extra_params':False,'source_commit':cfg['source_commit'],'upstream_runtime_commit':cfg['source_commit'],
          'freeze_vlm':False,'coordinate_mode':cfg.get('coordinate_mode','absolute'),'evaluation_precision':inference_model.dtype,'optimizer_updates':0,'action_semantics':'absolute','action_unit':'rad','K':6,'coefficient_tokens':7,
          'history_ticks':[-7,0],'target_ext_ticks':[-2,51],'horizon_ticks':[0,49],'c_pred_origin':'direct_model_tensor',
          'window_count':len(rows),'windows':rows,'dataset':verified,'training_protocol':{k:contract[k] for k in ('model_seed','data_seed','flow_seed','augmentation_seed','batch_size','optimizer','schedule','sampler') if k in contract},'training_contract_sha256':sha(cfg['training_contract']),
          'normalization_sha256':verified['normalization_sha256'],'historical_plan_sha256':inputs['historical_plan_sha256'],
          'paired_gt_history_exact':True,'rgb_raw_pixels_exact':True,'rgb_sequence_sha256':hashlib.sha256(''.join(rgb_hashes).encode()).hexdigest(),
          'input_preprocessing':'original NativeDataset.resize_image/state quantile/tokenizer with CURRENT checkpoint stats',
          'future_perturbation_passed':future_test,'original_sample_api_max_difference':max_api,
          'runtime':{'jax':jax.__version__,'numpy':np.__version__,'device':str(jax.devices()[0]),
                     'peak_bytes_in_use':int(jax.devices()[0].memory_stats()['peak_bytes_in_use'])},
          'median_forward_seconds_excluding_compile':float(np.median(times[1:]))}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'experiment_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'seal.json').write_text(json.dumps({'aligned_sha256':sha(out/'aligned.npz'),'metadata_sha256':sha(out/'metadata.json'),
                                          'bridge_sha256':sha(__file__) if '__file__' in globals() and Path(__file__).is_file() else cfg['script_sha256']},indent=2)+'\n')
    print(json.dumps({'status':'completed','step':cfg['expected_step'],'windows':len(rows),'output':str(out)}),flush=True)

if __name__=='__main__':main(sys.argv[1])
