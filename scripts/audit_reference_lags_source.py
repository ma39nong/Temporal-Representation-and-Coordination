#!/usr/bin/env python3
"""Read-only offline lag profiles; future states here never enter model inference."""
import hashlib,json,sys,subprocess,site
from pathlib import Path
import numpy as np
from openpi.training.flash_native import read_numeric
cfg=json.loads(Path(sys.argv[1]).read_text());site.addsitedir(cfg['research_runtime'])
from trc.analysis.flash.reference import lag_profile
from trc.data.action_schema import GROUPS
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=cfg['source_commit']:raise ValueError('Source mismatch')
if sha(cfg['paired_inputs'])!=cfg['paired_inputs_sha256']:raise ValueError('Input mismatch')
out=Path(cfg['output']).resolve();inputs=json.loads(Path(cfg['paired_inputs']).read_text())
if out.exists() or out.is_relative_to(Path.cwd().resolve()) or any(out.is_relative_to(Path(r['root']).resolve()) for r in inputs['records']):raise ValueError('New research output required')
rows=[]
for r in inputs['records']:
 a,s,_=read_numeric(r)
 for group,idx in GROUPS.items():
  profile=lag_profile(a,s,idx,max_lag=10);zero=profile[10];best=min(profile,key=lambda x:x['position_rmse_rad']);dynamic=[v for v in profile if v['increment_correlation'] is not None];bestdynamic=max(dynamic,key=lambda x:x['increment_correlation']) if dynamic else None
  rows.append({'source':r['source_recording_id'],'record_id':r['id'],'group':group,'frames':len(a),'zero_lag_rmse_rad':zero['position_rmse_rad'],'best_position_lag':best['lag_frames'] if max(v['position_rmse_rad'] for v in profile)-min(v['position_rmse_rad'] for v in profile)>1e-12 else None,'best_position_rmse_rad':best['position_rmse_rad'],'best_increment_lag':bestdynamic['lag_frames'] if bestdynamic else None,'best_increment_correlation':bestdynamic['increment_correlation'] if bestdynamic else None,'profile':profile})
out.write_text(json.dumps({'rows':rows,'source_commit':cfg['source_commit'],'paired_inputs_sha256':sha(cfg['paired_inputs']),'script_sha256':sha(__file__),'limitation':'Grid alignment correlations do not identify hardware latency, clock offsets, engagement or controller dynamics.'},indent=2)+'\n');print(json.dumps({'status':'completed','records':len(inputs['records']),'group_profiles':len(rows)}))
