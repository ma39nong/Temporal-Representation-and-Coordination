"""CPU-only real-batch numerical/gradient checks; no paired GPU pilot launcher.

Pinned runtime and checkpoint are explicit config resources. Head-only ten-step
checks are labeled as such, never substituted for the requested full pilot.
"""
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(path):
    cfg=json.loads(Path(path).read_text());out=Path(cfg['output']).resolve()
    if out.exists():raise FileExistsError(out)
    import jax,jax.numpy as jnp,optax
    from flax import nnx,traverse_util
    import torch
    from openpi.models import model as model_lib
    from openpi.models.pi0_flash import Pi0FlashConfig
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.training.flash_native import NativeDataset
    from openpi.training.flash_checkpoint import parameter_digest,is_vlm_parameter
    from trc.models.flash_order_readout import ReadoutFlashConfig
    from trc.models.velocity_readout import adapter_count
    if jax.default_backend()!='cpu' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only; leave occupied GPUs untouched')
    torch.set_num_threads(1)
    for relative,expected in json.loads(Path(cfg['source_manifest']).read_text())['files'].items():
        if sha(Path(cfg['source_root'])/relative)!=expected:raise ValueError('Pinned source differs: '+relative)
    contract=json.loads(Path(cfg['training_contract']).read_text())
    if sha(cfg['training_contract'])!=cfg['training_contract_sha256']:raise ValueError('Training contract differs')
    node=ast.parse(contract['model'],mode='eval').body
    if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id not in ('Pi0FlashConfig','CoordinateFlashConfig'):raise ValueError('Model contract repr differs')
    kwargs={v.arg:ast.literal_eval(v.value) for v in node.keywords}
    if kwargs.pop('coordinate_mode','absolute') != 'absolute':raise ValueError('Absolute original initialization required')
    original=Pi0FlashConfig(**{**kwargs,'dtype':'float32','freeze_vlm':True})
    params=model_lib.restore_params(Path(cfg['checkpoint'])/'params',restore_type=np.ndarray)
    digest=parameter_digest(params)
    if digest!=cfg['parameter_sha256']:raise ValueError('Wrong original weights')
    oldflat=traverse_util.flatten_dict(params)
    native=original.load(jax.device_put(params),remove_extra_params=False);native.eval()
    out.mkdir(parents=True)
    print(json.dumps({'stage':'weights_loaded','parameter_sha256':digest}),flush=True)
    data=NativeDataset.__new__(NativeDataset)
    data.config=SimpleNamespace(data=SimpleNamespace(extra_delta_transform=False))
    records=json.loads(Path(cfg['training_manifest']).read_text())['records']
    record=next(r for r in records if r['source_recording_id']==cfg['recording_id'])
    data.records={record['id']:record};data.numeric={};data.stats=normalize.load(Path(cfg['norm_stats']).parent)
    if sha(cfg['norm_stats'])!=cfg['normalization_sha256']:raise ValueError('Stats identity differs')
    data.tokenizer=PaligemmaTokenizer(max_len=original.max_token_len)
    index={'id':record['id'],'start':cfg['current_frame']}
    obs,history,target=data.observation(index)
    reference=np.load(cfg['reference_arrays'])
    np.testing.assert_array_equal(history,reference['history']);np.testing.assert_array_equal(target[:,2:52],reference['truth'])
    if cfg['crossing_cuts']:raise ValueError('Only verified continuous window allowed')
    @nnx.jit
    def condition(m,ob):
        ob=model_lib.preprocess_observation(None,ob,train=False)
        return m.encode_condition(ob)
    mask,cache=jax.block_until_ready(condition(native,obs))
    h,t=jnp.asarray(history),jnp.asarray(target)
    source=native.polynomial.fit_history(native.normalize(h))
    target_c=native.polynomial.fit_target(native.normalize(t),boundary_constraint=native.boundary_constraint)
    zero=jnp.zeros((1,),jnp.float32)
    # Capture the actual native Expert hidden, without changing source files.
    captured=[];head=native.action_out_proj
    def capture(x):
        captured.append(x)
        return head(x)
    native.action_out_proj=capture
    velocity=jax.block_until_ready(native.coefficient_velocity(source,zero,mask,cache))
    native.action_out_proj=head
    hidden=captured[0]
    baseline=native.denormalize(native.polynomial.reconstruct(source+velocity))
    result={'backend':'cpu','real_batch_size':1,'dtype':'float32','VLM_frozen':True,
        'checkpoint_parameter_sha256':digest,'normalization_sha256':cfg['normalization_sha256'],
        'source_commit':cfg['source_commit'],'source_manifest_sha256':sha(cfg['source_manifest']),
        'script_sha256':sha(__file__),'adapter_module_sha256':sha(cfg['adapter_module']),
        'window':index,'history_truth_exact':True,'hidden_shape':list(hidden.shape),'velocity_shape':list(velocity.shape),
        'physical_action_shape':list(baseline.shape),'trials':{},'paired_250_steps_completed':False,
        'smoke_scope':'10 optimizer steps on native readout only with cached REAL Expert hidden; NOT full Expert/globalBatch256 smoke or paired pilot'}
    native_loss=jax.block_until_ready(native.loss_from_cache(jax.random.key(42),mask,cache,history=h,target_ext=t,train=True))
    # Separate models preserve every original parameter array. No repeated VLM prefix.
    for trial,mode in [('B1','none'),('B2','unified'),('B3','order_specific')]:
        modelcfg=ReadoutFlashConfig.from_base(original,mode)
        model=modelcfg.warm_start(params,digest);model.eval()
        flat=traverse_util.flatten_dict(nnx.state(model,nnx.Param).to_pure_dict())
        added={k:v for k,v in flat.items() if k not in oldflat}
        original_state={k:v for k,v in flat.items() if k in oldflat}
        if parameter_digest(traverse_util.unflatten_dict(original_state))!=digest:raise ValueError('Original model parameters changed')
        if sum(v.size for v in added.values())!=adapter_count(mode):raise ValueError('Added parameter count differs')
        v=jax.block_until_ready(model.coefficient_velocity(source,zero,mask,cache))
        a=jax.block_until_ready(model.sample_from_cache(mask,cache,history=h,num_steps=1))
        diff=lambda x,y:float(np.max(np.abs(np.asarray(x,dtype=float)-np.asarray(y,dtype=float))))
        checks={'native_velocity_max_difference':diff(v,velocity),'physical_action_max_difference_rad':diff(a,baseline)}
        if mode!='none':
            model.action_out_proj.enabled=False
            off=jax.block_until_ready(model.coefficient_velocity(source,zero,mask,cache))
            checks['disabled_velocity_max_difference']=diff(off,velocity)
            model.action_out_proj.enabled=True
        loss=jax.block_until_ready(model.loss_from_cache(jax.random.key(42),mask,cache,history=h,target_ext=t,train=True))
        checks['native_loss_max_difference']=max(diff(loss[k],native_loss[k]) for k in loss)
        if any(x>1e-5 for x in checks.values()):raise ValueError('FP32 native parity failed: '+str(checks))
        full_gradient=None
        if cfg.get('full_expert_backward',True):
            @nnx.jit
            def backward(m):
                def objective(mm):
                    return jnp.mean(mm.loss_from_cache(jax.random.key(42),mask,cache,history=h,target_ext=t,train=True)['loss'])
                return nnx.value_and_grad(objective,argnums=nnx.DiffState(0,modelcfg.get_trainable_filter()))(m)
            ll,gg=jax.block_until_ready(backward(model))
            gf=traverse_util.flatten_dict(gg.to_pure_dict())
            if any(is_vlm_parameter(k) for k in gf):raise ValueError('VLM entered backward filter')
            if not all(np.isfinite(np.asarray(v)).all() for v in gf.values()):raise FloatingPointError('Nonfinite Expert gradient')
            full_gradient={'loss':float(ll),'leaves':len(gf),'elements':sum(v.size for v in gf.values()),
                           'norm':float(optax.global_norm(gg)),'VLM_gradient_leaves':0}
            del gg,gf,backward
        # Exact readout-only gradients on real hidden/velocity targets, plus Adam
        # save/restore. This isolates new algebra even while GPUs are occupied.
        readout=model.action_out_proj;tx=optax.adamw(2.5e-6,b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10)
        state=nnx.state(readout,nnx.Param);opt=tx.init(state);trace=[];target_velocity=target_c-source
        @nnx.jit
        def step(head,opt):
            def objective(hh):return jnp.mean((hh(hidden).astype(jnp.float32)-target_velocity)**2)
            ll,gg=nnx.value_and_grad(objective)(head)
            norm=optax.global_norm(gg);st=nnx.state(head,nnx.Param)
            updates,opt=tx.update(gg,opt,st);nnx.update(head,optax.apply_updates(st,updates))
            return opt,ll,norm,gg
        for it in range(10):
            opt,ll,norm,gg=jax.block_until_ready(step(readout,opt))
            grad=traverse_util.flatten_dict(gg.to_pure_dict())
            if not all(np.isfinite(np.asarray(v)).all() for v in grad.values()) or not np.isfinite(float(ll)):raise FloatingPointError('Readout smoke nonfinite')
            trace.append({'step':it+1,'loss':float(ll),'gradient_norm':float(norm),
                'down_gradient_norm':float(np.linalg.norm(np.asarray(grad.get(('adapter_down',),np.zeros(1))))),
                'up_gradient_norm':float(np.linalg.norm(np.asarray(grad.get(('adapter_up',),np.zeros(1)))))})
        if mode!='none' and (trace[0]['up_gradient_norm']<=0 or max(x['down_gradient_norm'] for x in trace[1:])<=0):raise ValueError('Adapter gradient blocked')
        # Check degree routing: loss on k=3 produces no gradient on other U slices.
        routing=None
        if mode=='order_specific':
            def one_order(hh):return jnp.sum(hh(hidden)[:,3])
            gg=nnx.grad(one_order)(readout);u=np.asarray(gg.adapter_up.value)
            other=np.delete(u,3,axis=0)
            routing={'selected_order_up_grad_norm':float(np.linalg.norm(u[3])), 'other_order_up_grad_max':float(np.max(np.abs(other)))}
            if routing['other_order_up_grad_max']!=0 or routing['selected_order_up_grad_norm']==0:raise ValueError('Order gradient misrouting')
        tree=(nnx.state(readout,nnx.Param),opt)
        leaves,definition=jax.tree.flatten(tree)
        destination=out/(trial+'_readout_adam_step10.npz')
        np.savez(destination,**{f'leaf{i}':np.asarray(v) for i,v in enumerate(leaves)})
        with np.load(destination) as saved:
            restored=jax.tree.unflatten(definition,[jnp.asarray(saved[f'leaf{i}']) for i in range(len(leaves))])
        for left,right in zip(leaves,jax.tree.leaves(restored)):np.testing.assert_array_equal(left,np.asarray(right))
        nnx.update(readout,restored[0])
        result['trials'][trial]={'mode':mode,'added_parameters':sum(v.size for v in added.values()),'checks':checks,
          'initial_loss':{k:np.asarray(v).tolist() for k,v in loss.items()},'full_expert_backward':full_gradient,
          'readout_only_smoke':trace,'order_gradient_routing':routing,'readout_adam_checkpoint_restore_exact':True,
          'checkpoint_sha256':sha(destination),'readout_checkpoint_scope':'head + Adam; frozen Expert/VLM identified by original Step8000 SHA'}
        if cfg.get('full_expert_smoke_steps',0):
            # Fresh initialization: readout-only diagnostics above must not alter
            # the starting point of this separate full-Expert functional smoke.
            fresh=modelcfg.warm_start(params,digest);fresh.train()
            train_filter=modelcfg.get_trainable_filter()
            full_tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-6,b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10))
            full_opt=full_tx.init(nnx.state(fresh,train_filter))
            if any(np.any(np.asarray(v)!=0) for v in jax.tree.leaves(full_opt)):raise ValueError('Adam not reset')
            @nnx.jit
            def full_step(m,adam,key):
                def objective(mm):return jnp.mean(mm.loss_from_cache(key,mask,cache,history=h,target_ext=t,train=True)['loss'])
                ll,gr=nnx.value_and_grad(objective,argnums=nnx.DiffState(0,train_filter))(m)
                finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(gr)])) & jnp.isfinite(ll)
                st=nnx.state(m,train_filter)
                updates,adam=full_tx.update(gr,adam,st);nnx.update(m,optax.apply_updates(st,updates))
                return adam,ll,optax.global_norm(gr),finite
            steps=[]
            for completed in range(cfg['full_expert_smoke_steps']):
                full_opt,ll,gn,finite=jax.block_until_ready(full_step(fresh,full_opt,jax.random.fold_in(jax.random.key(42),completed)))
                if not bool(finite):raise FloatingPointError('Full Expert smoke nonfinite; discard this isolated smoke run')
                steps.append({'step':completed+1,'loss':float(ll),'gradient_norm':float(gn)})
            st=nnx.state(fresh,train_filter);leaves,definition=jax.tree.flatten((st,full_opt))
            checkpoint=out/(trial+'_expert_adam_step5.npz')
            np.savez(checkpoint,**{f'leaf{i}':np.asarray(v) for i,v in enumerate(leaves)})
            with np.load(checkpoint) as saved:
                restored=jax.tree.unflatten(definition,[jnp.asarray(saved[f'leaf{i}']) for i in range(len(leaves))])
            for left,right in zip(leaves,jax.tree.leaves(restored)):np.testing.assert_array_equal(left,np.asarray(right))
            before=jax.block_until_ready(fresh.sample_from_cache(mask,cache,history=h,num_steps=1))
            nnx.update(fresh,restored[0])
            after=jax.block_until_ready(fresh.sample_from_cache(mask,cache,history=h,num_steps=1))
            if diff(before,after)!=0:raise ValueError('Full Expert checkpoint changes prediction')
            frozen=nnx.state(fresh,nnx.All(nnx.Param,modelcfg.get_freeze_filter())).to_pure_dict()
            if parameter_digest(frozen,vlm_only=True)!=parameter_digest(params,vlm_only=True):raise ValueError('VLM changed')
            cp=fresh.polynomial.fit_history(fresh.normalize(h))+fresh.coefficient_velocity(source,zero,mask,cache)
            oracle=fresh.denormalize(fresh.polynomial.reconstruct(target_c))
            np.savez(out/(trial+'_expert_smoke_predictions.npz'),truth=target[:,2:52],oracle=np.asarray(oracle),prediction=np.asarray(after),c_gt=np.asarray(target_c),c_pred=np.asarray(cp))
            result['trials'][trial]['full_expert_cpu_smoke']={'steps':steps,'batch':1,'cached_real_prefix':True,
                'VLM_parameters_unchanged':True,'checkpoint_restore_exact':True,'prediction_restore_max_difference_rad':diff(before,after),
                'checkpoint_path':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,'checkpoint_sha256':sha(checkpoint),
                'scope':'five full-Expert Adam updates on one continuous REAL window; CPU FP32, no image augmentation; not the 8GPU/global256 BF16 pilot'}
            del fresh,full_opt,full_step,leaves,restored,st,frozen
        (out/'acceptance.partial.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'trial':trial,**result['trials'][trial]},indent=2),flush=True)
        del model,readout,opt,step,flat,original_state,added
    np.savez(out/'baseline_real_batch.npz',history=history,truth=target[:,2:52],hidden=np.asarray(hidden),
        source=np.asarray(source),target_coefficients=np.asarray(target_c),velocity=np.asarray(velocity),prediction=np.asarray(baseline))
    (out/'acceptance.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main(sys.argv[1])
