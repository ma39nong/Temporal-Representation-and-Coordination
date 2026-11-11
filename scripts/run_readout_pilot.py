"""Pinned FLASH step + native loader/checkpoints; preserve frozen FP32 weights.

Explicit precheck (updates 1..5) and resume (6..250), identical three-arm protocol.
No training starts merely by importing this module. Runtime paths are configured.
"""
import argparse
import ast
from dataclasses import replace
import functools
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_step(path, expected, namespace):
    """Execute only the original sealed step function, without trainer.main hooks."""
    if sha(path) != expected:
        raise ValueError('Original training step source changed')
    module = ast.parse(Path(path).read_text())
    functions = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'step']
    if len(functions) != 1:
        raise ValueError('Expected exactly one original FLASH step')
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['step']


def state_digest(tree):
    import jax
    import numpy as np
    h = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        a = np.asarray(jax.device_get(value))
        h.update(str(path).encode()); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        h.update(memoryview(np.ascontiguousarray(a).view(np.uint8)))
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--trial', choices=['B1','B2','B3'], required=True)
    p.add_argument('--phase', choices=['precheck','resume'], required=True)
    args = p.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import torch
    from flax import nnx, traverse_util
    from openpi.models import model as model_lib
    from openpi.models.pi0_flash import Pi0FlashConfig
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.training import checkpoints, sharding, optimizer
    from openpi.training.flash_checkpoint import parameter_digest
    from openpi.training.flash_loader import PrefetchedBatches
    from openpi.training.utils import TrainState
    from trc.data.flash_sampling import ContinuousFlashIndex
    from trc.models.flash_order_data import CutAwareNativeDataset
    from trc.models.flash_order_readout import ReadoutFlashConfig
    from trc.experiments.p2_protocol import validate_order_protocol

    cfg = json.loads(args.config.read_text())
    protocol = json.loads(Path(cfg['protocol']).read_text())
    protocol_identity = validate_order_protocol(protocol)
    if jax.default_backend() != 'gpu' or jax.device_count() != 8:
        raise ValueError('Exactly eight GPUs required')
    torch.set_num_threads(1)
    for relative, digest in json.loads(Path(cfg['source_manifest']).read_text())['files'].items():
        if sha(Path(cfg['source_root'])/relative) != digest:
            raise ValueError('Pinned source changed: '+relative)
    for path, digest in [(cfg['norm_stats'],protocol['data']['normalization_sha256']),
                         (cfg['training_manifest'],protocol['data']['training_manifest_file_sha256']),
                         (cfg['cuts'],protocol['data']['cut_manifest_sha256']),
                         (cfg['training_contract'],cfg['training_contract_sha256'])]:
        if sha(path) != digest:
            raise ValueError('Input identity mismatch: '+path)
    trial = args.trial
    run = Path(cfg['output'])/trial
    run.mkdir(parents=True, exist_ok=True)
    identity = dict(config_sha256=sha(args.config), protocol_sha256=protocol_identity,
                    runner_sha256=sha(__file__), trial=trial, research_commit=cfg['research_commit'])
    contract_file = run/'execution_contract.json'
    if args.phase == 'precheck':
        if contract_file.exists():
            raise FileExistsError('Do not overwrite an attempted experiment')
        contract_file.write_text(json.dumps(identity, indent=2))
    else:
        if json.loads(contract_file.read_text()) != identity:
            raise ValueError('Code/config changed since precheck')
        for peer in ('B1','B2','B3'):
            gate=json.loads((Path(cfg['output'])/peer/'precheck_complete.json').read_text())
            if not gate['passed']:
                raise ValueError('All three prechecks must pass before formal continuation')

    node = ast.parse(json.loads(Path(cfg['training_contract']).read_text())['model'],mode='eval').body
    if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id not in ('Pi0FlashConfig','CoordinateFlashConfig'):
        raise ValueError('Invalid original model contract')
    kwargs = {n.arg:ast.literal_eval(n.value) for n in node.keywords}
    if kwargs.pop('coordinate_mode','absolute') != 'absolute':
        raise ValueError('Absolute FLASH only')
    base = Pi0FlashConfig(**{**kwargs,'dtype':'bfloat16','freeze_vlm':True})
    modelcfg = ReadoutFlashConfig.from_base(base, protocol['trials'][trial]['mode'])
    shared=protocol['shared']
    config = SimpleNamespace(model=modelcfg, trainable_filter=modelcfg.get_trainable_filter(),
        policy_metadata={'algorithm':'flash'}, augmentation_seed=shared['augmentation_seed'],
        batch_size=256, num_workers=16, data_seed=shared['data_seed'],
        data=SimpleNamespace(extra_delta_transform=False, assets=SimpleNamespace(asset_id='normal480_flash_absolute_k6_s1_all_sources_v2')))
    data = CutAwareNativeDataset.__new__(CutAwareNativeDataset)
    data.config=config; data.directory=Path(cfg['asset_directory'])
    data.cut_manifest_path=Path(cfg['cuts'])
    data.manifest=json.loads(Path(cfg['training_manifest']).read_text())
    data.records={r['id']:r for r in data.manifest['records']}
    data.train=list(data.records.values())
    if len(data.train)!=480 or len({r['source_recording_id'] for r in data.train})!=480 or any('/no-block-one/' not in r['root'] for r in data.train):
        raise ValueError('Require original 480 unique single-pick recordings')
    data.safe_index=ContinuousFlashIndex(data.train,json.loads(Path(cfg['cuts']).read_text()),shared['data_seed'])
    data.ends=np.asarray(data.safe_index.ends); data.numeric={};data.permutation=None;data.epoch=None
    data.stats=normalize.load(Path(cfg['norm_stats']).parent);data.tokenizer=None
    np.testing.assert_array_equal(np.asarray(modelcfg.action_q01),data.stats['actions'].q01)
    np.testing.assert_array_equal(np.asarray(modelcfg.action_q99),data.stats['actions'].q99)
    data.validation=[]
    sample_hash=hashlib.sha256()
    for i in range(250*256):
        sample_hash.update(json.dumps(data.index(i),sort_keys=True).encode()+b'\n')
    if sample_hash.hexdigest()!=cfg['sampler_sha256']:
        raise ValueError('Actual native sampler order changed')

    step=load_step(cfg['original_step_file'],cfg['original_step_sha256'],
                   dict(nnx=nnx,jax=jax,jnp=jnp,np=np,optax=optax,replace=replace,model_lib=model_lib))
    schedule=optimizer.CosineDecaySchedule(**shared['schedule'])
    tx=optimizer.AdamW(**shared['optimizer']).create(schedule.create())
    mesh=sharding.make_mesh(8)
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    batch_layout=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    abstract=nnx.eval_shape(modelcfg.create,jax.random.key(shared['model_seed']))
    graphdef, shaped_params=nnx.split(abstract)
    def build(params):
        return TrainState(step=jnp.asarray(0,jnp.int32),params=params,model_def=graphdef,tx=tx,
                          opt_state=tx.init(params.filter(config.trainable_filter)),ema_decay=None,ema_params=None)
    shaped=jax.eval_shape(build,shaped_params)
    layout=sharding.fsdp_sharding(shaped,mesh)
    manager,resuming=checkpoints.initialize_checkpoint_dir(run/'checkpoints',keep_period=1,
                                overwrite=False,resume=args.phase=='resume')
    try:
        if resuming:
            state=checkpoints.restore_state(manager,shaped,data)
            if int(state.step)!=5 or state_digest((state.params,state.opt_state))!=json.loads((run/'precheck_complete.json').read_text())['state_sha256']:
                raise ValueError('Restart model/Adam differs from accepted Step5')
            (run/'resume_verified.json').write_text(json.dumps({'step':5,'exact_model_adam_restore':True}))
        else:
            original=model_lib.restore_params(Path(cfg['checkpoint'])/'params',restore_type=np.ndarray)
            digest=parameter_digest(original)
            if digest!=protocol['initialization']['parameter_sha256']:
                raise ValueError('Original full Step8000 parameters differ')
            # One real window: native FP32 and new zero-residual head, no target in inference.
            data.tokenizer=PaligemmaTokenizer(max_len=modelcfg.max_token_len)
            rec=next(r for r in data.train if r['source_recording_id']==cfg['recording_id'])
            obs,hist,target=data.observation({'id':rec['id'],'start':cfg['current_frame']})
            reference=np.load(cfg['reference_arrays'])
            np.testing.assert_array_equal(hist,reference['history']);np.testing.assert_array_equal(target[:,2:52],reference['truth'])
            fp32=replace(base,dtype='float32')
            native=fp32.load(original,remove_extra_params=False);native.eval()
            extended=ReadoutFlashConfig.from_base(fp32,modelcfg.readout_mode).warm_start(original,digest);extended.eval()
            @nnx.jit
            def infer(m,ob,h):
                ob=model_lib.preprocess_observation(None,ob,train=False)
                mask,cache=m.encode_condition(ob)
                source=m.polynomial.fit_history(m.normalize(h))
                velocity=m.coefficient_velocity(source,jnp.zeros((h.shape[0],),jnp.float32),mask,cache)
                return velocity,m.denormalize(m.polynomial.reconstruct(source+velocity))
            with jax.default_matmul_precision('highest'):
                nv,na=jax.device_get(infer(native,obs,jnp.asarray(hist)))
                ev,ea=jax.device_get(infer(extended,obs,jnp.asarray(hist)))
            differences=dict(velocity=float(np.max(np.abs(nv-ev))),physical_rad=float(np.max(np.abs(na-ea))))
            if max(differences.values())>1e-5:
                raise ValueError('FP32 Step0 zero-residual parity failed')
            np.savez(run/'step0_parity.npz',native_velocity=nv,extended_velocity=ev,native_action=na,extended_action=ea)
            (run/'step0_parity.json').write_text(json.dumps({'passed':True,**differences,'parameter_sha256':digest},indent=2))
            del native,extended,infer
            jax.clear_caches()
            # Warm start uses original NumPy parameter buffers; no FP32->BF16 weight cast.
            warm=modelcfg.warm_start(original,digest)
            flat=traverse_util.flatten_dict(nnx.state(warm,nnx.Param).to_pure_dict())
            original_paths=set(traverse_util.flatten_dict(original))
            if parameter_digest(traverse_util.unflatten_dict({k:v for k,v in flat.items() if k in original_paths}))!=digest:
                raise ValueError('Warm start changed original parameter bytes')
            _,params=nnx.split(warm)
            params=jax.device_put(params,layout.params)
            state=jax.jit(build,out_shardings=layout)(params)
            jax.block_until_ready(state)
            if any(np.any(np.asarray(jax.device_get(v))!=0) for v in jax.tree.leaves(state.opt_state)):
                raise ValueError('Adam initialization was not zero')
            del original,warm,params,flat
            data.tokenizer=None
            checkpoints.save_state(manager,state,data,0);manager.wait_until_finished()
            print(json.dumps({'stage':'initialized','trial':trial,'sampler_sha256':sample_hash.hexdigest(),'parameter_sha256':digest}),flush=True)
        update=jax.jit(functools.partial(step,config),in_shardings=(replicated,layout,batch_layout),
                       out_shardings=(layout,replicated),donate_argnums=(1,))
        start=int(state.step);end=5 if args.phase=='precheck' else 250
        with PrefetchedBatches(data,start,end) as batches:
            for completed in range(start,end):
                tick=time.perf_counter();host=next(batches)
                batch=jax.device_put(host,batch_layout); del host
                wait=time.perf_counter()-tick; tick=time.perf_counter()
                with sharding.set_mesh(mesh):
                    candidate,metrics=update(jax.random.key(shared['flow_seed']),state,batch)
                metrics=jax.device_get(metrics)
                if not bool(metrics['finite']):
                    raise FloatingPointError('Nonfinite update rejected; stop this experiment')
                state=candidate
                row={k:float(v) for k,v in metrics.items() if k!='finite'}
                row.update(step=int(state.step),seconds=time.perf_counter()-tick,data_wait_seconds=wait,
                    learning_rate=float(schedule.create()(completed)),effective_samples=int(state.step)*256,
                    peak_device_bytes=[(d.memory_stats() or {}).get('peak_bytes_in_use') for d in jax.devices()])
                with (run/'training.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
                if int(state.step) in (5,50,100,250):
                    checkpoints.save_state(manager,state,data,int(state.step));manager.wait_until_finished()
        if args.phase=='precheck':
            before=state_digest((state.params,state.opt_state))
            restored=checkpoints.restore_state(manager,shaped,data)
            if state_digest((restored.params,restored.opt_state))!=before:
                raise ValueError('Full sharded checkpoint/Adam restore mismatch')
            report=dict(passed=True,step=5,state_sha256=before,global_batch=256,devices=8,
                        train_dtype='bfloat16',frozen_parameter_dtype='original preserved',**identity)
        else:
            report=dict(passed=True,step=int(state.step),**identity)
        (run/(args.phase+'_complete.json')).write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    finally:
        manager.close()


if __name__=='__main__':
    main()
