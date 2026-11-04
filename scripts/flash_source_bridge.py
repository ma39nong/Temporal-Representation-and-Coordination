#!/usr/bin/env python3
"""Run with an explicitly configured OpenPI runtime; reuse original FLASH math.

This script never trains or saves checkpoints. It is isolated from trc imports;
ordinary trc tests/metrics require neither OpenPI nor JAX.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cached(config, out):
    """Authenticate saved real predictions, raw numeric inputs and historical source split."""
    import jax
    import jax.numpy as jnp
    from openpi.shared.flash_polynomial import FlashPolynomial, normalize_actions, unnormalize_actions
    from openpi.training.flash_data import load_episode, make_batch
    data = Path(config['data_dir'])
    manifest = json.loads((data/'manifest.json').read_text())
    split = json.loads((data/'split.json').read_text())
    norm = json.loads((data/'normalization.json').read_text())
    plan = json.loads(Path(config['plan']).read_text())
    audit = json.loads(Path(config['audit']).read_text())
    reference = json.loads(Path(config['checkpoint_reference']).read_text())
    if reference['checkpoint_pre_save_hashes']['parameters'] != audit['parameter_sha256'] or reference['checkpoint']['step'] != audit['optimizer_step']:
        raise ValueError('Historical checkpoint report does not match inference audit')
    if reference['dataset_sha256'] != manifest['sha256'] or reference['split_sha256'] != split['sha256']:
        raise ValueError('Historical training data/split identity mismatch')
    for document in (manifest, split, plan):
        if digest({k:v for k,v in document.items() if k != 'sha256'}) != document['sha256']:
            raise ValueError('Manifest/split/plan hash mismatch')
    if audit['prediction_file_sha256'] != sha(config['predictions']) or audit['plan_sha256'] != plan['sha256']:
        raise ValueError('Predictions are not bound to this audited window plan')
    if plan['dataset_sha256'] != manifest['sha256'] or plan['split_sha256'] != split['sha256']:
        raise ValueError('Plan dataset/split mismatch')
    if plan['normalization_snapshot_sha256'] != digest(norm):
        raise ValueError('Normalization snapshot mismatch')
    if audit.get('checkpoint_files_unchanged') is not True or audit.get('optimizer_updates') != 0:
        raise ValueError('Require zero-update audited original inference')
    train_records = {r['id']:r for r in manifest['records'] if split['assignments'][r['id']] == 'train'}
    if set(norm['train_sample_ids']) != set(train_records):
        raise ValueError('Statistics fitting scope mismatch')
    if norm['train_cache_sha256'] != [train_records[i]['sha256'] for i in norm['train_sample_ids']]:
        raise ValueError('Statistics input hash identities mismatch')
    roots = {r['root']:r for r in manifest['roots']}
    records = {r['id']:{**r,'info':roots[r['root']]['info']} for r in manifest['records']}
    train_sources = {r['source_recording_id'] for r in train_records.values()}
    eval_sources = {records[w['id']]['source_recording_id'] for w in plan['windows']}
    if train_sources & eval_sources or any(split['assignments'][w['id']] != 'validation' for w in plan['windows']):
        raise ValueError('Source leakage or non-heldout evaluation')
    low, high = np.asarray(norm['q01'],np.float32), np.asarray(norm['q99'],np.float32)
    if low.shape != (58,) or high.shape != (58,) or not np.isfinite([low,high]).all() or np.any(high<low):
        raise ValueError('Invalid FLASH quantile stats')
    arrays = np.load(config['predictions'],allow_pickle=False)
    truth, oracle, pred, history = [arrays[k] for k in ('truth','oracle','prediction','history')]
    windows = plan['windows']
    if truth.shape != (len(windows),54,58) or pred.shape != truth.shape or oracle.shape != truth.shape or history.shape != (len(windows),8,58):
        raise ValueError('Saved shape/window count mismatch')
    if len({(w['id'],w['start']) for w in windows}) != len(windows):
        raise ValueError('Duplicate evaluation windows')
    if audit['windows'] != len(windows): raise ValueError('Audit window count mismatch')
    # Validate every evaluated numeric window against original loader with real timestamps/IDs.
    episodes = {}; rows=[]
    for i, w in enumerate(windows):
        ident = w['id']; record = records[ident]
        if ident not in episodes: episodes[ident] = load_episode(record,data)
        actual = make_batch([episodes[ident]],[(ident,w['start'])])
        np.testing.assert_array_equal(truth[i],actual['target_ext'][0])
        np.testing.assert_array_equal(history[i],actual['history'][0])
        rows.append({**actual['provenance'][0], 'source_recording_id':record['source_recording_id'],
                     'dataset_root':record['root'], 'scope':'historical_held_out_sources', 'fps':record['info']['fps']})
    # Perturb only future labels and future measured states. Current state/history
    # must remain unchanged; targets must change. Tests original batch constructor.
    first=windows[0]; original=episodes[first['id']]; altered={**original,'actions':original['actions'].copy(),'states':original['states'].copy()}
    start=first['start']; altered['actions'][start+1:]+=0.123; altered['states'][start+1:]+=0.456
    before=make_batch([original],[(first['id'],start)]); after=make_batch([altered],[(first['id'],start)])
    np.testing.assert_array_equal(before['history'],after['history']);np.testing.assert_array_equal(before['state'],after['state'])
    if np.array_equal(before['target_ext'],after['target_ext']):raise ValueError('Future perturbation test failed')
    original_cfg=reference['configuration']
    for key,value in [('action_dim',58),('action_horizon',50),('poly_degree',6),('history_steps',8),('history_reg_lambda',.1),('boundary_constraint',True)]:
        if original_cfg.get(key) != value:raise ValueError('Historical FLASH target/history configuration mismatch')
    np.testing.assert_array_equal(np.asarray(original_cfg['action_q01'],np.float32),low)
    np.testing.assert_array_equal(np.asarray(original_cfg['action_q99'],np.float32),high)
    polynomial=FlashPolynomial()
    ngt=normalize_actions(jnp.asarray(truth),jnp.asarray(low),jnp.asarray(high))
    cgt=np.asarray(polynomial.fit_target(ngt,boundary_constraint=True))
    reconstructed=np.asarray(unnormalize_actions(polynomial.reconstruct(jnp.asarray(cgt),extended=True),low,high))
    oracle_diff=float(np.max(np.abs(reconstructed-oracle)))
    np.testing.assert_allclose(reconstructed,oracle,atol=2e-5,rtol=1e-5)
    # Old exports did not persist C_pred. Recover scaled coefficients by inversion of
    # the exact original extended decode matrix; never apply target boundary projection.
    npred=np.asarray(normalize_actions(pred,low,high),np.float64)
    raw=np.einsum('kh,nhd->nkd',np.linalg.pinv(np.asarray(polynomial.basis_ext,dtype=np.float64)),npred)
    cpred=(raw/np.asarray(polynomial.scale)[None,:,None]).astype(np.float32)
    pred_reconstructed=np.asarray(unnormalize_actions(polynomial.reconstruct(jnp.asarray(cpred),extended=True),low,high))
    inversion_error=float(np.max(np.abs(pred_reconstructed-pred)))
    np.testing.assert_allclose(pred_reconstructed,pred,rtol=1e-5,atol=2e-5)
    np.savez(out/'aligned.npz',truth=truth[:,2:52],oracle=oracle[:,2:52],prediction=pred[:,2:52],
             c_gt=cgt,c_pred=cpred,scale=np.asarray(polynomial.scale),q01=low,q99=high,
             history=history,valid=np.ones((len(windows),50,58),bool))
    metadata={'data_kind':'real','evaluation_scope':'historical_held_out_sources',
              'inference_mode':'authenticated_saved_original_inference_not_new_checkpoint_restore',
              'checkpoint_id':config['checkpoint_id'],'checkpoint_parameter_sha256':audit['parameter_sha256'],
              'checkpoint_step':audit['optimizer_step'],'checkpoint_reference_sha256':sha(config['checkpoint_reference']),'freeze_vlm':True,'window_count':len(windows),
              'sources':len(eval_sources),'normalization_sha256':sha(data/'normalization.json'),
              'plan_sha256':plan['sha256'],'dataset_sha256':manifest['sha256'],'split_sha256':split['sha256'],
              'prediction_file_sha256':sha(config['predictions']),'raw_window_checks':len(windows),
              'oracle_recompute_max_abs_rad':oracle_diff,'coefficient_inverse_decode_max_abs_rad':inversion_error,
              'c_pred_origin':'recovered_from_54_frame_decode_by_linear_inverse_not_direct_model_tensor',
              'action_semantics':'absolute','action_unit':'rad','horizon_ticks':[0,49],
              'target_ext_ticks':[-2,51],'history_ticks':[-7,0], 'K':6,'coefficient_tokens':7,
              'constraints':'target_C1_projection_only; no_inference_clip_or_endpoint_projection',
              'upstream_runtime_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'upstream_math_file_sha256':sha('src/openpi/shared/flash_polynomial.py'),
              'original_inference_script_sha256':sha('scripts/flash/evaluate_p25.py'),
              'future_condition_audit':'original evaluator predict(mask,cache,history) has no target argument; original audit reused',
              'future_label_state_perturbation':'original_make_batch_history_and_current_state_unchanged_targets_changed',
              'runtime':{'jax':jax.__version__,'numpy':np.__version__},'windows':rows}
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')


def native(config,out):
    """Fresh full-model inference via original native dataset and model restore APIs.

    Config must name the exact experiment module, step directory and matching
    training contract. Missing or partial weights are blocking, never random init.
    """
    import dataclasses
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training.flash_native import NativeDataset, bind_stats, preflight
    from openpi.training.flash_checkpoint import parameter_digest
    from scripts.run_training import load_experiment
    from openpi.shared.flash_polynomial import normalize_actions
    stepdir=Path(config['checkpoint'])
    if not (stepdir/'params').is_dir(): raise FileNotFoundError('Complete native params checkpoint is not available')
    if config.get('expected_training_contract_sha256') != sha(config['training_contract']):
        raise ValueError('Pin the exact actual training contract before loading weights')
    contract=json.loads(Path(config['training_contract']).read_text())
    cfg=bind_stats(load_experiment(config['experiment_module']))
    checked=preflight(cfg)
    if contract['dataset'] != checked or contract['model'] != str(cfg.model):
        raise ValueError('Experiment configuration differs from actual training contract')
    if cfg.model.freeze_vlm or cfg.data.extra_delta_transform or cfg.model.action_dim!=58 or cfg.model.action_horizon!=50:
        raise ValueError('Expected native Absolute FLASH FullFT configuration')
    if int(stepdir.name) != config['expected_step']: raise ValueError('Checkpoint step identity mismatch')
    model=cfg.model.load(model_lib.restore_params(stepdir/'params'),remove_extra_params=False)
    model.eval()
    dataset=NativeDataset(cfg,tokenizer=PaligemmaTokenizer(max_len=cfg.model.max_token_len))
    plan=json.loads(Path(config['plan']).read_text()) if config.get('plan') else dataset.validation
    if isinstance(plan,dict): plan=plan['windows']
    if len({(w['id'],w['start']) for w in plan}) != len(plan): raise ValueError('Duplicate eval windows')
    before_checkpoint={str(p.relative_to(stepdir)):[p.stat().st_size,p.stat().st_mtime_ns] for p in stepdir.rglob('*') if p.is_file()}
    samples=[]; rows=[]
    allowed_sources=set(dataset.manifest['validation_sources'] if checked['evaluation_scope']=='held_out_sources' else dataset.manifest['train_sources'])
    for w in plan:
        if w['id'] not in dataset.records or dataset.records[w['id']]['source_recording_id'] not in allowed_sources:
            raise ValueError('Window does not match declared evaluation scope')
        # Original observation has only current RGB/prompt/state; target goes to oracle below.
        obs,history,target=dataset.observation(w)
        obs=model_lib.preprocess_observation(None,obs,train=False)
        mask,cache=model.encode_condition(obs)
        source=model.polynomial.fit_history(model.normalize(jnp.asarray(history)))
        cpred=source+model.coefficient_velocity(source,jnp.zeros((1,)),mask,cache)
        predicted=model.denormalize(model.polynomial.reconstruct(cpred))
        # Must equal original inference API, no additional clipping/projection.
        direct=model.sample_from_cache(mask,cache,history=jnp.asarray(history),num_steps=1)
        np.testing.assert_allclose(predicted,direct,rtol=1e-5,atol=2e-5)
        cgt=model.polynomial.fit_target(model.normalize(jnp.asarray(target)),boundary_constraint=model.boundary_constraint)
        oracle=model.denormalize(model.polynomial.reconstruct(cgt))
        samples.append(tuple(np.asarray(a)[0] for a in (target[:,2:52],oracle,predicted,cgt,cpred)))
        r=dataset.records[w['id']]
        rows.append({'sample_id':w['id'],'current_row':w['start'],'episode_index':r['episode'],
                     'source_recording_id':r['source_recording_id'],'source_path':r['path'],'fps':30})
    if not samples: raise ValueError('No evaluation windows')
    after_checkpoint={str(p.relative_to(stepdir)):[p.stat().st_size,p.stat().st_mtime_ns] for p in stepdir.rglob('*') if p.is_file()}
    if before_checkpoint != after_checkpoint:raise ValueError('Checkpoint inventory changed during evaluation')
    a,o,p,cg,cp=map(np.stack,zip(*samples))
    np.savez(out/'aligned.npz',truth=a,oracle=o,prediction=p,c_gt=cg,c_pred=cp,
             scale=np.asarray(model.polynomial.scale),q01=np.asarray(model.action_q01),q99=np.asarray(model.action_q99),valid=np.ones(a.shape,bool))
    meta={'data_kind':'real','evaluation_scope':checked['evaluation_scope'],'inference_mode':'fresh_native_checkpoint',
          'checkpoint_id':str(stepdir),'checkpoint_step':config['expected_step'],
          'checkpoint_parameter_sha256':parameter_digest(nnx.state(model).to_pure_dict()),'freeze_vlm':False,
          'action_semantics':'absolute','action_unit':'rad','K':6,'coefficient_tokens':7,'c_pred_origin':'direct_model_tensor',
          'window_count':len(rows),'windows':rows,'dataset':checked,'model_config':dataclasses.asdict(cfg.model),
          'upstream_runtime_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'checkpoint_files_unchanged':True,'optimizer_updates':0,'target_ext_ticks':[-2,51],'history_ticks':[-7,0],'horizon_ticks':[0,49]}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.output).resolve()
    source=Path.cwd().resolve()
    if out.is_relative_to(source): raise ValueError('Write exports in research outputs, not original repository')
    out.mkdir(parents=True,exist_ok=False)
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=cfg['expected_source_commit']:
        raise ValueError('Original FLASH source commit mismatch')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Original tracked source has uncommitted changes')
    (cached if cfg['mode']=='cached' else native)(cfg,out)
    print(json.dumps({'status':'passed','output':str(out)}))
