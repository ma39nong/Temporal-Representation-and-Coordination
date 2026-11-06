#!/usr/bin/env python3
"""Reuse sealed predictions for q0 semantics and training-only state strata."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from trc.io import file_sha256,write_report
from trc.data.action_schema import GROUPS,ORDER
from trc.analysis.flash.metrics import EVAL_GROUPS,summary,evaluate
from trc.analysis.flash.offset import decompose_offsets,grouped_offsets
from trc.analysis.flash.report import csv_rows
from trc.models.flash_coordinates import FlashCoordinates

p=argparse.ArgumentParser();p.add_argument('--exports',nargs='+',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
if out.exists():raise FileExistsError('New output required')
audit=json.loads(Path(args.audit).read_text());inputs=[]
for root in map(Path,args.exports):
 m=json.loads((root/'metadata.json').read_text());seal=json.loads((root/'seal.json').read_text())
 if file_sha256(root/'metadata.json')!=seal['metadata_sha256'] or file_sha256(root/'aligned.npz')!=seal['aligned_sha256']:raise ValueError('Export seal mismatch')
 with np.load(root/'aligned.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
 if not data['valid'].all():raise ValueError('Complete windows required')
 inputs.append((m,data,seal))
m,d,_=inputs[-1];a=d['truth'].astype(float);h=d['history'].astype(float);q=h[:,-1];np.testing.assert_array_equal(q,d['current_state'])
sources=np.array([w['source_recording_id'] for w in m['windows']]);seen=np.array([w['training_source'] for w in m['windows']],bool)
reference={x['source']:np.array(x['state0']) for x in audit['episode_initial_states']};initial=np.stack([reference[s] for s in sources]);train_initial=np.array([x['state0'] for x in audit['episode_initial_states'] if x['training']])
strata={};thresholds={};identity=[]
for kind,values,training in [('episode_first_retained_state',initial,train_initial),('window_reference_state',q,q[seen])]:
 center=training.mean(0);spread=np.maximum(training.std(0),1e-6);score=np.sqrt(np.mean(((values-center)/spread)**2,axis=1));tscore=np.sqrt(np.mean(((training-center)/spread)**2,axis=1));cuts=np.quantile(tscore,[.5,.9])
 label=np.where(score<=cuts[0],'low',np.where(score<=cuts[1],'middle','high'));strata[kind]=label
 thresholds[kind]={'training_only_cuts':cuts.tolist(),'training_reference_count':len(training),'std_floor':1e-6,'definition':'RMS of per-joint train-only standardized positions; diagnostic, not model normalization'}
 for i,w in enumerate(m['windows']):identity.append({'kind':kind,'window':i,'source':sources[i],'training_source':bool(seen[i]),'score':float(score[i]),'stratum':label[i]})
refrows=[];jointrows=[];relative=FlashCoordinates('state_relative',d['q01'],d['q99'],d['scale'],tuple(GROUPS));r=relative.actions(a,h);normalized=relative.normalize_coordinates(r)
masks={'training_sources':seen,'candidate_sources':~seen,**{'source:'+s:sources==s for s in sorted(set(sources))}}
for scope,mask in masks.items():
 for group,idx in EVAL_GROUPS.items():
  err=(a[:,0]-q)[mask][:,idx];rn=normalized[mask][...,idx]
  refrows.append({'scope':scope,'group':group,'windows':int(mask.sum()),'source_count':len(set(sources[mask])),**summary(err),
                  'signed_mean_rad':float(err.mean()),'normalized_residual_p99':float(np.quantile(np.abs(rn),.99)),'normalized_residual_max':float(np.abs(rn).max())})
 for j,name in enumerate(ORDER):
  err=(a[:,0]-q)[mask,j];jointrows.append({'scope':scope,'joint':name,'signed_mean_rad':float(err.mean()),**summary(err),
   'action_quantile_span':float(d['q99'][j]-d['q01'][j]+1e-6),'normalized_residual_p99':float(np.quantile(np.abs(normalized[mask,:,j]),.99)),'normalized_residual_max':float(np.abs(normalized[mask,:,j]).max())})
evalrows=[];temporal=[];offset=[];source_metrics=[]
for meta,z,seal in inputs:
 np.testing.assert_array_equal(z['truth'],d['truth']);np.testing.assert_array_equal(z['history'],d['history'])
 if [(w['sample_id'],w['current_row']) for w in meta['windows']]!=[(w['sample_id'],w['current_row']) for w in m['windows']]:raise ValueError('Window mismatch')
 for kind,labels in strata.items():
  for scope,scope_mask in [('training_sources',seen),('candidate_sources',~seen)]:
   for label in ('low','middle','high'):
    mask=scope_mask&(labels==label)
    if not mask.any():continue
    identity_row={'step':meta['checkpoint_step'],'coordinate_mode':meta.get('coordinate_mode','absolute'),'kind':kind,'scope':scope,'stratum':label,'windows':int(mask.sum()),'sources':len(set(sources[mask]))}
    mm=evaluate(z['truth'][mask],z['oracle'][mask],z['prediction'][mask],z['valid'][mask])
    evalrows.extend({**identity_row,**v} for v in mm['groups']);temporal.extend({**identity_row,**v} for v in mm['temporal'])
    offset.extend({**identity_row,**v} for v in grouped_offsets(decompose_offsets(z['truth'][mask],z['prediction'][mask])))
 for source in sorted(set(sources)):
  mask=sources==source;mm=evaluate(z['truth'][mask],z['oracle'][mask],z['prediction'][mask],z['valid'][mask])
  source_metrics.extend({'step':meta['checkpoint_step'],'coordinate_mode':meta.get('coordinate_mode','absolute'),'source':source,'training_source':bool(seen[mask][0]),**v} for v in mm['groups'])
out.mkdir(parents=True)
for name,rows in [('q0_group',refrows),('q0_joint',jointrows),('state_strata_windows',identity),('stratified_metrics',evalrows),('stratified_temporal',temporal),('stratified_offsets',offset),('per_source_metrics',source_metrics)]:csv_rows(out/(name+'.csv'),rows)
write_report(out/'reference_report.json',{'source_exports':[{'step':mm['checkpoint_step'],'coordinate_mode':mm.get('coordinate_mode','absolute'),'evaluation_precision':mm.get('evaluation_precision','legacy_bfloat16'),'seal':seal} for mm,_,seal in inputs],'q0_metrics':refrows,'strata':thresholds,'audit_sha256':file_sha256(args.audit),'research_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'code_sha256':file_sha256(__file__),
 'interpretation':'q0 is measured position reference, not GT command0. No hardware latency inferred. Strata are train-defined exploratory diagnostics, not new normalization.'},protected=list(map(Path,args.exports)))
print(json.dumps({'status':'completed','pooled_groups':[v for v in refrows if not v['scope'].startswith('source:')],'strata':thresholds},indent=2))
