#!/usr/bin/env python3
"""Step8000 full-parameter warm-start; default CPU preflight, explicit training only.

Run through the pinned original launcher; source and assets stay read-only.
The original FSDP optimizer/sampler/checkpoint trainer is reused, not copied.
"""
import argparse,hashlib,json,os,site,subprocess
from pathlib import Path
from dataclasses import replace,asdict
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--start-training',choices=['B0','B2']);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());site.addsitedir(cfg['research_runtime'])
    from trc.experiments.p1_transfer import validate_transfer_protocol as validate_protocol
    shared_hash=validate_protocol(cfg)
    def sha(path):
     h=hashlib.sha256()
     with Path(path).open('rb') as f:
      for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
     return h.hexdigest()
    root=Path.cwd().resolve();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if commit!=cfg['source_commit'] or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Pinned clean source required')
    replay=json.loads(Path(cfg['replay_gate']).read_text());forward=json.loads(Path(cfg['forward_gate']).read_text())
    if sha(cfg['replay_gate'])!=cfg['replay_gate_sha256'] or sha(cfg['forward_gate'])!=cfg['forward_gate_sha256']:raise ValueError('Gate identity mismatch')
    if replay['identity']['source_commit']!=commit or forward['source_commit']!=commit:raise ValueError('Gate/source version differs')
    if replay.get('evaluation_policy')!=cfg['evaluation']:raise ValueError('Evaluation policy differs from verified replay')
    if replay['status']!='PARITY_PASSED' or not forward['training_adapter_passed']:raise ValueError('Parity/adapter gates not passed')
    from trc.models import flash_openpi
    if forward['adapter_sha256']!=sha(flash_openpi.__file__):raise ValueError('Untested adapter code')
    from scripts.run_training import load_experiment
    from openpi.training.flash_native import bind_stats,preflight,NativeDataset
    from flax import nnx,traverse_util
    import jax,numpy as np
    base=bind_stats(load_experiment(cfg['experiment_module']));verified=preflight(base)
    if replay['identity']['normalization_sha256']!=verified['normalization_sha256']:raise ValueError('Gate statistics differ from training')
    if verified!=cfg['expected_dataset']:raise ValueError('Data/window/statistics identity mismatch')
    if asdict(base.optimizer)!=cfg['expected_optimizer']:raise ValueError('Optimizer drift')
    from openpi.training.optimizer import CosineDecaySchedule
    from trc.models.flash_warmstart import StrictFlashWarmStart
    from openpi.models import model as model_lib
    from openpi.training.flash_checkpoint import parameter_digest
    initial=model_lib.restore_params(cfg['initial_params'],restore_type=np.ndarray)
    if parameter_digest(initial)!=cfg['initial_parameter_sha256']:raise ValueError('Wrong Step8000 parameters')
    initial_layout={str(k):{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in traverse_util.flatten_dict(initial).items()}
    del initial
    output=Path(cfg['output_root']).resolve();base_weights=Path(cfg['initial_params']).resolve()
    protected=[root,base_weights,*[Path(s.root).resolve() for s in base.data.lerobot_datasets]]
    if any(output.is_relative_to(x) for x in protected):raise ValueError('Outputs must be outside source, data and initial weights')
    # Data assets stay at the ORIGINAL bound location; only future checkpoints/logs go to output.
    if int(base_weights.parent.name)!=8000:raise ValueError('Explicit Step8000 params required')
    weight_files={str(f.relative_to(base_weights)):sha(f) for f in sorted(base_weights.rglob('*')) if f.is_file()}
    if not weight_files:raise ValueError('No initial weights')
    weight_identity=hashlib.sha256(json.dumps(weight_files,sort_keys=True).encode()).hexdigest()
    # Full parameter content hash, rather than path spelling, pins this warm start.
    configs={};layouts={}
    for name,trial in cfg['trials'].items():
     model=flash_openpi.CoordinateFlashConfig.from_base(base.model,trial['coordinate_mode'])
     c=replace(base,model=model,freeze_filter=model.get_freeze_filter(),name='trc_p15_flash_'+name.lower(),exp_name=cfg['run_id'],
      weight_loader=StrictFlashWarmStart(cfg['initial_params'],cfg['initial_parameter_sha256']),lr_schedule=CosineDecaySchedule(**cfg['expected_schedule']),
      batch_size=cfg['shared']['batch_size'],num_train_steps=cfg['shared']['num_train_steps'],fsdp_devices=cfg['shared']['fsdp_devices'],
      seed=cfg['shared']['flow_seed'],model_seed=cfg['shared']['model_seed'],data_seed=cfg['shared']['data_seed'],augmentation_seed=cfg['shared']['augmentation_seed'],
      checkpoint_base_dir=str(output/'checkpoints'),resume=False,overwrite=False,save_interval=cfg['shared']['num_train_steps'],keep_period=cfg['shared']['num_train_steps'],
      policy_metadata={**base.policy_metadata,'coordinate_mode':trial['coordinate_mode'],'research_shared_protocol':shared_hash,'research_adapter_sha256':sha(flash_openpi.__file__)})
     if c.model.dtype!=cfg['shared']['train_dtype'] or c.model.freeze_vlm:raise ValueError('FullFT training precision/freeze drift')
     if c.checkpoint_dir.exists():
      if a.start_training is None or a.start_training==name:raise FileExistsError(c.checkpoint_dir)
      # A completed matched peer may already exist when launching the second trial.
      peer=json.loads((c.checkpoint_dir/'flash_training_contract.json').read_text())
      marker=json.loads((c.checkpoint_dir/'0'/'_CHECKPOINT_METADATA').read_text())
      expected={'dataset':verified,'model':str(c.model),'batch_size':c.batch_size,'model_seed':c.model_seed,'data_seed':c.data_seed,'flow_seed':c.seed,'augmentation_seed':c.augmentation_seed,'optimizer':str(c.optimizer),'schedule':str(c.lr_schedule)}
      if any(peer[k]!=v for k,v in expected.items()) or not marker.get('commit_timestamp_nsecs'):raise ValueError('Existing peer is not completed under the matching protocol')
     if preflight(c)!=verified:raise ValueError('Representation changed data contract')
     shaped=nnx.eval_shape(c.model.create,jax.random.key(c.model_seed))
     flat=traverse_util.flatten_dict(nnx.state(shaped,nnx.Param).to_pure_dict())
     if set(traverse_util.flatten_dict(nnx.state(shaped,c.trainable_filter).to_pure_dict()))!=set(flat):raise ValueError('FullFT optimizer coverage differs')
     layouts[name]={str(k):{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in flat.items()};configs[name]=c
    if layouts['B0']!=layouts['B2'] or layouts['B0']!=initial_layout:raise ValueError('Warm-start/paired parameter layout differs')
    sampler_hashes={}
    for name in ('B0','B2'):
     dataset=NativeDataset(configs[name],tokenizer=None);sampler=hashlib.sha256()
     for i in range(cfg['shared']['num_train_steps']*cfg['shared']['batch_size']):
      sampler.update(json.dumps(dataset.index(i),sort_keys=True).encode()+b'\n')
     sampler_hashes[name]=sampler.hexdigest()
    if sampler_hashes['B0']!=sampler_hashes['B2']:raise ValueError('Sampler sequences differ')
    report={'status':'P1 TRANSFER INITIALIZATION READY','training_started':False,'source_commit':commit,'shared_protocol_sha256':shared_hash,
     'dataset':verified,'planned_checkpoint_dirs':{k:str(v.checkpoint_dir) for k,v in configs.items()},'entry_sha256':sha(__file__),'same_parameter_layout':True,'parameter_elements':sum(int(np.prod(v['shape'])) for v in layouts['B0'].values()),
     'initial_parameter_files_identity':weight_identity,'initial_weight_files':weight_files,'same_initializer':'Strict Step8000 full parameters, original init_train_state creates fresh Adam; actual byte equality checked during initialization',
     'sampler_windows':cfg['shared']['num_train_steps']*cfg['shared']['batch_size'],'sampler_sequence_sha256':sampler_hashes['B0'],'paired_sampler_hashes':sampler_hashes,
     'optimizer':str(base.optimizer),'schedule':str(configs['B0'].lr_schedule),'configs':{k:str(v) for k,v in configs.items()},'evaluation':cfg['evaluation'],
     'replay_gate_sha256':sha(cfg['replay_gate']),'forward_gate_sha256':sha(cfg['forward_gate']),'adapter_sha256':sha(flash_openpi.__file__)}
    report_path=Path(cfg['preflight_output']).resolve()
    if any(report_path.is_relative_to(x) for x in protected):raise ValueError('New research preflight output required')
    if report_path.exists():
     saved=json.loads(report_path.read_text())
     material=('source_commit','shared_protocol_sha256','dataset','parameter_elements','initial_parameter_files_identity','sampler_sequence_sha256','adapter_sha256','entry_sha256','replay_gate_sha256','forward_gate_sha256','planned_checkpoint_dirs')
     if any(saved[k]!=report[k] for k in material):raise ValueError('Existing preflight differs')
    else:
     report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('configs','initial_weight_files')},indent=2),flush=True)
    if a.start_training:
     if jax.default_backend()!='gpu' or jax.device_count()!=cfg['shared']['fsdp_devices']:raise ValueError('Explicit matched GPU mesh required for training')
     # No automatic launch on a successful preflight. This branch is explicitly selected by the human later.
     for directory in ('wandb','tmp'):(output/directory).mkdir(parents=True,exist_ok=True)
     os.environ['WANDB_DIR']=str(output/'wandb');os.environ['TMPDIR']=str(output/'tmp');os.environ.setdefault('WANDB_MODE','offline')
     os.environ['FLASH_INPUT_PIPELINE']='multiprocess'
     from scripts.train_flash import main
     main(configs[a.start_training])


if __name__=='__main__':
    main()
