"""FP32 evaluation of readout pilots on the frozen 485 single-pick train windows.

Exports all 485 with continuous-support flags; no old candidate source is run.
Main scientific metrics use the predeclared common 423-window subset.
"""
import argparse
import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--paired-config',required=True,type=Path)
    p.add_argument('--continuous-indices',required=True,type=Path)
    p.add_argument('--trial',choices=['B1','B2','B3'],required=True)
    p.add_argument('--step',type=int,choices=[0,50,100,250],required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--expected-vlm-sha256',required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest':
        raise ValueError('Strict FP32/highest evaluation required')
    import jax,jax.numpy as jnp,numpy as np,torch
    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.models.pi0_flash import Pi0FlashConfig
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.training.flash_native import NativeDataset
    from openpi.training.flash_checkpoint import parameter_digest
    from trc.models.flash_order_readout import ReadoutFlashConfig
    torch.set_num_threads(1)
    cfg=json.loads(args.config.read_text());paired=json.loads(args.paired_config.read_text())
    protocol=json.loads(Path(cfg['protocol']).read_text())
    for path,digest in [(cfg['norm_stats'],protocol['data']['normalization_sha256']),
                        (cfg['training_contract'],cfg['training_contract_sha256']),
                        (paired['paired_inputs'],paired['paired_inputs_sha256']),
                        (paired['reference_arrays'],paired['reference_arrays_sha256']),
                        (args.continuous_indices,protocol['data']['evaluation_indices_file_sha256'])]:
        if sha(path)!=digest:raise ValueError('Evaluation input identity changed: '+str(path))
    for relative,digest in json.loads(Path(cfg['source_manifest']).read_text())['files'].items():
        if sha(Path(cfg['source_root'])/relative)!=digest:raise ValueError('Pinned source changed')
    contract=json.loads(Path(cfg['training_contract']).read_text())
    node=ast.parse(contract['model'],mode='eval').body
    if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id not in ('Pi0FlashConfig','CoordinateFlashConfig'):
        raise ValueError('Unknown original model')
    kwargs={v.arg:ast.literal_eval(v.value) for v in node.keywords}
    if kwargs.pop('coordinate_mode','absolute')!='absolute':raise ValueError('Absolute only')
    modelcfg=ReadoutFlashConfig.from_base(Pi0FlashConfig(**{**kwargs,'dtype':'float32','freeze_vlm':True}),protocol['trials'][args.trial]['mode'])
    checkpoint=Path(cfg['output'])/args.trial/'checkpoints'/str(args.step)
    if not json.loads((checkpoint/'_CHECKPOINT_METADATA').read_text()).get('commit_timestamp_nsecs'):
        raise ValueError('Incomplete checkpoint')
    params=model_lib.restore_params(checkpoint/'params',restore_type=np.ndarray)
    digest=parameter_digest(params)
    frozen_digest=parameter_digest(params,vlm_only=True)
    if frozen_digest!=args.expected_vlm_sha256:
        raise ValueError('Frozen VLM bytes changed')
    model=modelcfg.load(jax.device_put(params),remove_extra_params=False);model.eval();del params
    inputs=json.loads(Path(paired['paired_inputs']).read_text());reference=np.load(paired['reference_arrays'])
    manifest=json.loads(Path(cfg['training_manifest']).read_text());train=set(manifest['train_sources'])
    data=NativeDataset.__new__(NativeDataset)
    data.config=SimpleNamespace(data=SimpleNamespace(extra_delta_transform=False))
    data.records={r['id']:r for r in inputs['records']};data.numeric={}
    data.stats=normalize.load(Path(cfg['norm_stats']).parent)
    data.tokenizer=PaligemmaTokenizer(max_len=modelcfg.max_token_len)
    original_image=data.image
    def checked_image(index,role):
        pixels=original_image(index,role)
        if hashlib.sha256(pixels.tobytes()).hexdigest()!=inputs['rgb_sha256'][f"{index['id']}:{index['start']}:{role}"]:
            raise ValueError('Original RGB differs')
        return pixels
    data.image=checked_image
    @nnx.jit
    def predict(m,obs,h):
        obs=model_lib.preprocess_observation(None,obs,train=False)
        mask,cache=m.encode_condition(obs)
        source=m.polynomial.fit_history(m.normalize(h))
        cp=source+m.coefficient_velocity(source,jnp.zeros((1,),jnp.float32),mask,cache)
        return cp,m.denormalize(m.polynomial.reconstruct(cp))
    @nnx.jit
    def native_predict(m,obs,h):
        return m.sample_actions(jax.random.key(42),obs,history=h,num_steps=1)
    @nnx.jit
    def oracle(m,target):
        c=m.polynomial.fit_target(m.normalize(target),boundary_constraint=m.boundary_constraint)
        return c,m.denormalize(m.polynomial.reconstruct(c))
    continuous=set(json.loads(args.continuous_indices.read_text())['indices'])
    selected=[(i,w) for i,w in enumerate(inputs['windows']) if data.records[w['id']]['source_recording_id'] in train]
    if len(selected)!=485:raise ValueError('Require exactly frozen 485 training-source windows')
    values=[];rows=[];times=[];native_times=[];api_difference=0.
    args.output.mkdir(parents=True)
    for i,w in selected:
        obs,h,t=data.observation(w)
        np.testing.assert_array_equal(h[0],reference['history'][i]);np.testing.assert_array_equal(t[0,2:52],reference['truth'][i])
        tick=time.perf_counter();cp,pred=jax.block_until_ready(predict(model,obs,jnp.asarray(h)))
        times.append(time.perf_counter()-tick)
        if len(rows)<25:
            tick=time.perf_counter();direct=jax.block_until_ready(native_predict(model,obs,jnp.asarray(h)))
            native_times.append(time.perf_counter()-tick)
            difference=float(np.max(np.abs(np.asarray(pred)-np.asarray(direct))))
            api_difference=max(api_difference,difference)
            if difference>1e-5:raise ValueError('Coefficient decoding differs from original sampling API')
        cg,o=oracle(model,jnp.asarray(t))
        values.append((t[0,2:52],np.asarray(o)[0],np.asarray(pred)[0],np.asarray(cg)[0],np.asarray(cp)[0],h[0]))
        rows.append(dict(original_export_index=i,id=w['id'],current_frame=w['start'],
                         source_recording_id=data.records[w['id']]['source_recording_id'],
                         training_source=True,continuous_support=i in continuous))
        if len(rows)%25==0:print(json.dumps({'windows':len(rows),'total':485,'forward_seconds':times[-1]}),flush=True)
    a,o,pred,cg,cp,h=map(np.stack,zip(*values))
    if not all(np.isfinite(x).all() for x in (a,o,pred,cg,cp,h)):raise FloatingPointError('Nonfinite evaluation')
    if sum(r['continuous_support'] for r in rows)!=423:raise ValueError('Primary subset changed')
    np.savez(args.output/'aligned.npz',truth=a,oracle=o,prediction=pred,c_gt=cg,c_pred=cp,history=h,
             scale=np.asarray(model.polynomial.scale),basis=np.asarray(model.polynomial.basis_ext[2:52]),
             q01=np.asarray(model.action_q01),q99=np.asarray(model.action_q99),valid=np.ones(a.shape,bool))
    meta=dict(data_kind='real',trial=args.trial,step=args.step,windows=rows,coordinate_mode='absolute',
              checkpoint=str(checkpoint),parameter_sha256=digest,frozen_vlm_sha256=frozen_digest,
              native_sample_api_max_difference_rad=api_difference,training_source_diagnostic_only=True,
              original_training_contract_sha256=cfg['training_contract_sha256'],protocol_sha256=sha(cfg['protocol']),
              normalization_sha256=sha(cfg['norm_stats']),evaluation_dtype='float32',matmul_precision='highest',
              inference_inputs='current images/state/prompt and legal measured history; no target argument',
              inference_seconds_excluding_compile=times[1:],native_sample_seconds_excluding_compile=native_times[1:],
              native_benchmark_scope='first 25 fixed training windows, first compile excluded; offline Batch1, not robot control latency',
              gpu_memory=jax.devices()[0].memory_stats(),
              primary_windows=423,legacy_windows=485,script_sha256=sha(__file__),research_commit=cfg['research_commit'])
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    (args.output/'seal.json').write_text(json.dumps({'aligned_sha256':sha(args.output/'aligned.npz'),'metadata_sha256':sha(args.output/'metadata.json')},indent=2))
    print(json.dumps({'completed':True,'trial':args.trial,'step':args.step,'windows':485}),flush=True)


if __name__=='__main__':main()
