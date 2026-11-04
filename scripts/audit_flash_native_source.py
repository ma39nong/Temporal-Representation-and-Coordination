#!/usr/bin/env python3
"""Read-only minimal native FLASH input acceptance; execute in original runtime.

Can be streamed to a remote interpreter via stdin. No models/checkpoints are
created, no datasets changed. Emits a single JSON report on stdout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.dont_write_bytecode=True


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

p=argparse.ArgumentParser();p.add_argument('--asset-dir',required=True);p.add_argument('--training-contract',required=True)
a=p.parse_args()
from openpi.training.flash_native import read_numeric
from openpi.shared import normalize
from openpi.shared.flash_polynomial import normalize_actions,unnormalize_actions
from openpi.policies.fr3_sharpa_protocol import ORDER
import pyarrow.parquet as pq
root=Path(a.asset_dir);m=json.loads((root/'flash_manifest.json').read_text());seal=json.loads((root/'flash_seal.json').read_text())
plan=json.loads((root/'flash_validation.json').read_text());contract=json.loads(Path(a.training_contract).read_text())
assert digest({k:v for k,v in m.items() if k!='sha256'})==m['sha256']==seal['manifest_sha256']
assert sha(root/'norm_stats.json')==seal['norm_sha256'] and digest(plan)==seal['validation_sha256']
assert contract['dataset']['manifest_sha256']==m['sha256']
assert contract['dataset']['normalization_sha256']==seal['norm_sha256']
assert contract['dataset']['validation_sha256']==seal['validation_sha256']
assert m['contract']['action_semantics']=='absolute' and m['contract']['horizon']==50 and m['contract']['target']==54
assert m['contract']['joint_names']==list(ORDER)
stats=normalize.load(root);q=stats['actions'];q01=np.asarray(q.q01,np.float32);q99=np.asarray(q.q99,np.float32);records=[];max_inverse=0.;missing_source_time=0
for record in m['records']:
 path=Path(record['path']);st=path.stat()
 assert st.st_size==record['size'] and st.st_mtime_ns==record['mtime_ns']
 actions,states,tasks=read_numeric(record)
 assert actions.shape==(record['rows'],58) and states.shape==actions.shape
 table=pq.read_table(path,columns=['episode_index'])
 assert np.all(np.asarray(table['episode_index'])==record['episode'])
 schema=pq.read_schema(path);source_present='source_timestamp_ns' in schema.names
 if not source_present:missing_source_time+=1
 else:
  t=np.asarray(pq.read_table(path,columns=['source_timestamp_ns'])['source_timestamp_ns'])
  assert np.all(np.diff(t)>0)
  assert np.allclose(np.diff(t)/1e9,1/30,atol=2e-4)
 for start in sorted(set([7,(record['rows']-52+7)//2,record['rows']-52])):
  target=actions[start-2:start+52];history=states[start-7:start+1]
  assert target.shape==(54,58) and history.shape==(8,58)
  inv=np.asarray(unnormalize_actions(normalize_actions(target,q01,q99),q01,q99))
  err=float(np.max(np.abs(inv-target)));max_inverse=max(max_inverse,err)
  np.testing.assert_allclose(inv,target,atol=2e-6,rtol=1e-6)
 records.append({'id':record['id'],'episode':record['episode'],'rows':record['rows'],
                 'source_recording_id':record['source_recording_id'],'root':record['root'],
                 'source_timestamp_ns_present':source_present})
print(json.dumps({'status':'passed_minimum_numeric_acceptance','source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'asset_seal':seal,'training_contract_sha256':sha(a.training_contract),'training_dataset':contract['dataset'],
 'actual_batch_size':contract['batch_size'],'frame_count':sum(r['rows'] for r in records),'episode_count':len(records),
 'action_semantics':'absolute','canonical_order':list(ORDER),'history_ticks':[-7,0],'target_ext_ticks':[-2,51],'output_ticks':[0,49],
 'normalization_inverse_max_rad':max_inverse,'missing_source_timestamps':missing_source_time,
 'checkpoint_read':False,'video_sync_verified':False,'records':records},ensure_ascii=False))
