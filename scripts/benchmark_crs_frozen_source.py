#!/usr/bin/env python3
"""Explicit pinned-source runtime only. One allowed development input; zero updates."""
import argparse,json,time,os,site
from pathlib import Path
from dataclasses import replace
import numpy as np
from evaluate_fullft_paired_source import model_kwargs_from_contract,sha,inventory


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    c=json.loads(Path(args.config).read_text());out=Path(c['output'])
    if out.exists():raise FileExistsError(out)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.models.pi0_flash import Pi0FlashConfig
    from openpi.training.flash_checkpoint import parameter_digest
    from openpi.training.flash_native import NativeDataset
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    import torch
    site.addsitedir(c['research_runtime'])  # Explicit opt-in pinned research adapter runtime.
    from trc.models.flash_openpi import CoordinateFlashConfig
    torch.set_num_threads(1)
    if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest':raise ValueError('FP32 highest required')
    snap=json.loads(Path(c['source_snapshot_manifest']).read_text())
    for name,value in snap['files'].items():
        if sha(Path.cwd()/name)!=value:raise ValueError('Pinned source changed')
    if sha(c['training_contract'])!=c['training_contract_sha256']:raise ValueError('Contract changed')
    contract=json.loads(Path(c['training_contract']).read_text())
    if sha(c['norm_stats_path'])!=contract['dataset']['normalization_sha256']:raise ValueError('Statistics changed')
    if sha(c['input'])!=c['input_sha256'] or sha(c['reference'])!=c['reference_sha256']:raise ValueError('Input changed')
    base=Pi0FlashConfig(**model_kwargs_from_contract(contract['model']))
    config=replace(CoordinateFlashConfig.from_base(base,'state_relative'),dtype='float32')
    checkpoint=Path(c['checkpoint']);before=inventory(checkpoint)
    params=model_lib.restore_params(checkpoint/'params',restore_type=np.ndarray)
    digest=parameter_digest(params)
    if digest!=c['expected_parameter_sha256']:raise ValueError('Frozen weights changed')
    model=config.load(jax.device_put(params),remove_extra_params=False);model.eval();del params
    inp=json.loads(Path(c['input']).read_text())
    if len(inp['windows'])!=1 or inp['windows'][0]['evaluation_role']!='development_success':raise ValueError('One allowed development input only')
    data=NativeDataset.__new__(NativeDataset);data.stats=normalize.load(Path(c['norm_stats_path']).parent)
    data.records={r['id']:r for r in inp['records']};data.numeric={};data.tokenizer=PaligemmaTokenizer(max_len=base.max_token_len)
    original=data.image
    def image(w,role):
        import hashlib
        pixels=original(w,role)
        if hashlib.sha256(pixels.tobytes()).hexdigest()!=inp['rgb_sha256'][f"{w['id']}:{w['start']}:{role}"]:raise ValueError('RGB mismatch')
        return pixels
    data.image=image
    obs,h,target=data.observation(inp['windows'][0]);history=jnp.asarray(h)
    ref=np.load(c['reference'],allow_pickle=False)
    np.testing.assert_array_equal(h[0],ref['history']);np.testing.assert_array_equal(target[0,2:52],ref['truth'])
    # No target appears in any inference function signature.
    @nnx.jit
    def prefix(m,o):return m.encode_condition(model_lib.preprocess_observation(None,o,train=False))
    # Separate closures keep branch choice static for NNX/JAX.
    @nnx.jit
    def d0(m,o,h):
        mask,cache=m.encode_condition(model_lib.preprocess_observation(None,o,train=False))
        s=m.polynomial.fit_history(m.normalize(h))
        return m.denormalize(m.polynomial.reconstruct(s+m.coefficient_velocity(s,jnp.zeros((1,)),mask,cache)))
    @nnx.jit
    def d3(m,o,h):
        mask,cache=m.encode_condition(model_lib.preprocess_observation(None,o,train=False))
        return m.sample_from_cache(mask,cache,history=h,num_steps=1)
    @nnx.jit
    def shared(m,o,h):
        mask,cache=m.encode_condition(model_lib.preprocess_observation(None,o,train=False))
        s=m.polynomial.fit_history(m.normalize(h));q=m.coordinates().anchor_coefficients(h,xp=jnp);r=s-q
        c0=s+m.coefficient_velocity(s,jnp.zeros((1,)),mask,cache)
        c3=r+m.coefficient_velocity(r,jnp.zeros((1,)),mask,cache)+q
        return tuple(m.denormalize(m.polynomial.reconstruct(c)) for c in (c0,c3))
    call={'D0_single':lambda:d0(model,obs,history),'D3_single':lambda:d3(model,obs,history),
          'prefix_only':lambda:prefix(model,obs),'shared_prefix_pair':lambda:shared(model,obs,history)}
    # Truly sequential calls, so the naive timing includes two VLM executions.
    def naive():
        a=jax.block_until_ready(d0(model,obs,history));b=jax.block_until_ready(d3(model,obs,history));return a,b
    call['naive_pair']=naive
    results={};parity={}
    for name,f in call.items():
        for _ in range(3):value=jax.block_until_ready(f())
        if name in ('D0_single','D3_single'):
            key='d0' if name=='D0_single' else 'd3';parity[name]=float(np.max(np.abs(np.asarray(value)[0]-ref[key])))
        elif name in ('shared_prefix_pair','naive_pair'):
            parity[name]=max(float(np.max(np.abs(np.asarray(v)[0]-ref[k]))) for v,k in zip(value,('d0','d3')))
        times=[]
        for _ in range(20):
            start=time.perf_counter();jax.block_until_ready(f());times.append(time.perf_counter()-start)
        results[name]={'median_ms':float(np.median(times)*1000),'p95_ms':float(np.quantile(times,.95)*1000),'repetitions':20,'warmups':3}
        print(name,results[name],flush=True)
    if max(parity.values())>1e-5:raise ValueError(f'Frozen replay parity failed: {parity}')
    if before!=inventory(checkpoint):raise ValueError('Checkpoint modified')
    gpu=jax.devices()[0];stats=gpu.memory_stats()
    result={'timing':results,'max_absolute_parity_rad':parity,'gpu':str(gpu),'device_kind':gpu.device_kind,
            'jax_version':jax.__version__,'parameters_sha256':digest,'checkpoint_unchanged':True,
            'memory_stats':{k:int(v) for k,v in stats.items()},'memory_note':'Peak allocation across all compiled benchmark variants, not standalone deployment peak.',
            'input_scope':'one development_success window; no prior locked-test numeric data','script_sha256':sha(__file__),
            'input_sha256':c['input_sha256'],'reference_sha256':c['reference_sha256'],'units':'Batch1 FP32 synchronous wall time, excludes CPU data reading and JIT compilation; no robot timing claim'}
    out.mkdir(parents=True);(out/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
