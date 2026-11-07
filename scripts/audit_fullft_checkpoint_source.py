#!/usr/bin/env python3
"""Read-only host restore audit in an explicitly supplied original FLASH runtime.

No optimizer construction/update, no checkpoint manager and no GPU allocation.
Output must be a new file outside the source checkout and checkpoint run.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def inventory(root):
    return {str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns] for p in root.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--contract-sha256',required=True);p.add_argument('--experiment-module',required=True)
    args=p.parse_args();run=Path(args.run).resolve();out=Path(args.output).resolve()
    if out.exists() or out.is_relative_to(run) or out.is_relative_to(Path.cwd()):raise ValueError('New external audit output required')
    import jax
    import orbax.checkpoint as ocp
    from scripts.run_training import load_experiment
    from openpi.training.flash_native import bind_stats,preflight
    from openpi.shared import normalize
    if jax.default_backend()!='cpu':raise ValueError('CPU-only audit required')
    contract_path=run/'flash_training_contract.json'
    if sha(contract_path)!=args.contract_sha256:raise ValueError('Contract differs')
    contract=json.loads(contract_path.read_text());config=bind_stats(load_experiment(args.experiment_module))
    if preflight(config)!=contract['dataset'] or str(config.model)!=contract['model']:raise ValueError('Model/data contract differs')
    records=[]
    for step in (4000,6000,8000):
        root=run/str(step);before=inventory(root)
        committed=json.loads((root/'_CHECKPOINT_METADATA').read_text())
        if not committed.get('commit_timestamp_nsecs'):raise ValueError('Uncommitted checkpoint')
        assets=root/'assets';norm_files=list(assets.rglob('norm_stats.json'))
        if len(norm_files)!=1 or sha(norm_files[0])!=contract['dataset']['normalization_sha256']:raise ValueError('Checkpoint statistics differ')
        stats=normalize.load(norm_files[0].parent)
        for name,stat in stats.items():
            for field in ('mean','std','q01','q99'):
                value=np.asarray(getattr(stat,field))
                if not np.isfinite(value).all():raise ValueError('Invalid normalization')
        with ocp.PyTreeCheckpointer() as ckpt:
            item=ckpt.metadata(root/'train_state')
            restored=ckpt.restore(root/'train_state',ocp.args.PyTreeRestore(item=item,
                restore_args=jax.tree.map(lambda _:ocp.ArrayRestoreArgs(restore_type=np.ndarray),item)))
        leaves=jax.tree_util.tree_flatten_with_path(restored)[0]
        digest=hashlib.sha256();counts=[];moments={'mu':{'leaves':0,'elements':0},'nu':{'leaves':0,'elements':0}};total=0
        for keys,value in leaves:
            if value is None:continue
            a=np.asarray(value);path='/'.join(str(getattr(k,'key',getattr(k,'idx',k))) for k in keys)
            if not np.isfinite(a).all():raise ValueError('Nonfinite restored optimizer/step')
            digest.update(path.encode());digest.update(str(a.dtype).encode());digest.update(str(a.shape).encode());digest.update(memoryview(np.ascontiguousarray(a).view(np.uint8)))
            total+=a.size
            if a.shape==() and ('step' in path or 'count' in path):counts.append({'path':path,'value':int(a)})
            for kind in moments:
                if kind in path.split('/'):
                    moments[kind]['leaves']+=1;moments[kind]['elements']+=a.size
                    if kind=='nu' and np.any(a<0):raise ValueError('Negative Adam second moment')
        if not counts or any(v['value']!=step for v in counts):raise ValueError('Optimizer count / checkpoint step mismatch')
        if any(v['elements']==0 for v in moments.values()):raise ValueError('Missing optimizer moments')
        saved_assets={str(f.relative_to(assets)):sha(f) for f in assets.rglob('*.json')}
        if inventory(root)!=before:raise ValueError('Checkpoint changed during audit')
        records.append({'step':step,'commit_metadata':committed,'train_state_restored_to_host':True,'train_state_sha256':digest.hexdigest(),
            'restored_leaf_count':len(leaves),'restored_elements':total,'step_counters':counts,'moments':moments,
            'checkpoint_files_unchanged':True,'normalization_loaded':True,'assets_sha256':saved_assets,
            'checkpoint_inventory_sha256':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()})
        print(json.dumps({'audited_step':step,'moments':moments,'counts':counts}),flush=True)
        del restored,item,leaves,value,a;gc.collect()
    logs=run/'flash_metrics';evidence={}
    for file in sorted(logs.glob('*.json')):
        evidence[file.name]={'sha256':sha(file),'content':json.loads(file.read_text())}
    training=[json.loads(l) for l in (logs/'training.jsonl').read_text().splitlines() if l.strip()]
    result={'contract':contract,'contract_sha256':sha(contract_path),'records':records,'runtime_backend':'cpu',
        'training_updates_executed':0,'evidence':evidence,'training_log_sha256':sha(logs/'training.jsonl'),
        'training_log_last':training[-1],'audit_script_sha256':sha(__file__)}
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
