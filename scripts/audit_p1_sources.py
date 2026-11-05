#!/usr/bin/env python3
"""Focused read-only lineage/copy/schema/quality checks in the source CPU runtime.

Inputs are explicit manifests; no discovery or copying of bags/videos/weights.
Exact frame fingerprints detect numeric copies and contiguous crops; they cannot
exclude lossy resampling, changed UUIDs or unprovided raw recording lineage.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from openpi.training.flash_native import read_numeric,CAMERAS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def fingerprints(a,s):
 return [hashlib.blake2b(row.tobytes(),digest_size=16).digest() for row in np.ascontiguousarray(np.concatenate([a,s],axis=1),dtype='<f4')]
def longest(mask):
 best=run=0
 for v in mask:
  run=run+1 if v else 0;best=max(best,run)
 return best
cfg=json.loads(Path(sys.argv[1]).read_text());manifest=json.loads(Path(cfg['manifest']).read_text());paired=json.loads(Path(cfg['paired_inputs']).read_text())
train=set(manifest['train_sources']);candidates=[r for r in paired['records'] if r['source_recording_id'] not in train]
# Candidate hashes are tiny; stream train rows against them to avoid a large index.
needed={h for r in candidates for h in fingerprints(*read_numeric(r)[:2])};matches={};records=[];initial=[];lineage={}
all_records=manifest['records']+candidates
for i,r in enumerate(all_records):
 a,s,t=read_numeric(r);root=Path(r['root']);features=r['info']['features'];ep=next(json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines() if json.loads(line)['episode_index']==r['episode'])
 camera={k:v for k,v in features.items() if k.startswith('observation.images.')}
 key=digest([features['action'],features['observation.state'],camera,r['tasks'],r['info']['fps']])
 delta=np.diff(a.astype(float),axis=0);sd=np.diff(s.astype(float),axis=0)
 row={'id':r['id'],'source':r['source_recording_id'],'training':r['source_recording_id'] in train,'rows':len(a),'episode':r['episode'],
      'resolved_root':str(root.resolve()),'schema_camera_prompt_sha256':key,'camera_features':camera,'tasks':r['tasks'],
      'action_state_numeric_sha256':hashlib.sha256(np.concatenate([a,s],axis=1).tobytes()).hexdigest(),
      'nan_inf':False,'dt_verified_30hz_by_original_reader':True,'episode_metadata':ep,'parquet_sha256':sha(r['path']),
      'max_action_increment_rad':float(np.max(np.abs(delta))),'p99_action_increment_rad':float(np.quantile(np.abs(delta),.99)),
      'max_state_increment_rad':float(np.max(np.abs(sd))),'p99_state_increment_rad':float(np.quantile(np.abs(sd),.99)),
      'state_command_rmse_rad':float(np.sqrt(np.mean((a.astype(float)-s)**2))),
      'zero_action_increment_fraction':float(np.mean(delta==0)),
      'missing_lineage_directories':[d for d in ('conversion','ingest') if not (root/d).exists()]}
 initial.append({'id':r['id'],'source':r['source_recording_id'],'training':row['training'],'state0':s[0].tolist(),'action0':a[0].tolist()})
 hashes=fingerprints(a,s)
 if row['training']:
  for f,h in enumerate(hashes):
   if h in needed:matches.setdefault(h,[]).append((r['id'],f))
 else:
  hit=[h in matches for h in hashes];best=0;previous={}
  for h in hashes:
   current={(ident,f):previous.get((ident,f-1),0)+1 for ident,f in matches.get(h,[])}
   best=max([best,*current.values()]);previous=current
  row['frames_exactly_matching_any_training_frame']=sum(hit);row['longest_candidate_run_of_any_matching_frames']=longest(hit)
  row['longest_ordered_same_training_record_copy_frames']=best
 records.append(row)
 if (i+1)%100==0:print(f'audited {i+1}/{len(all_records)}',file=sys.stderr,flush=True)
result={'manifest_sha256':sha(cfg['manifest']),'paired_inputs_sha256':sha(cfg['paired_inputs']),'camera_roles':CAMERAS,'records':records,'episode_initial_states':initial,
 'limitations':['UUID/source time and canonical numeric identity only; no raw bag content proof supplied','Exact fingerprints do not exclude lossy/resampled/modified duplicates','Camera calibration/serial numbers not in these info files','Finite jumps are distribution observations, not certified hardware limit checks']}
Path(cfg['output']).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'completed','training_records':len(manifest['records']),'candidate_records':len(candidates),'schema_camera_prompt_variants':len({r['schema_camera_prompt_sha256'] for r in records}),'candidate_copy_matches':[{k:r[k] for k in ('source','frames_exactly_matching_any_training_frame','longest_ordered_same_training_record_copy_frames')} for r in records if not r['training']]}))
