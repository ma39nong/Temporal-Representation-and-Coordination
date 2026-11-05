#!/usr/bin/env python3
"""P1-0 analysis of existing sealed real predictions; never runs or trains a model."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from trc.io import file_sha256,write_report
from trc.data.action_schema import ORDER
from trc.data.normalization import load_stats
from trc.analysis.flash.metrics import EVAL_GROUPS,summary
from trc.analysis.flash.offset import decompose_offsets,grouped_offsets,anchor_correction,compare_constant_coefficient
from trc.analysis.flash.report import csv_rows

p=argparse.ArgumentParser();p.add_argument('--exports',nargs='+',required=True);p.add_argument('--audit',required=True);p.add_argument('--stats',required=True);p.add_argument('--output',required=True);args=p.parse_args()
out=Path(args.output)
if out.exists():raise FileExistsError('New output required')
audit=json.loads(Path(args.audit).read_text());stats=load_stats(args.stats);rows=[];compare=[];references=[];baselines=[];windows=[];joint_bias=[];scope_fig=[];provenance=[];window_distribution=[];component_arrays=[]
initial=audit['episode_initial_states'];train_initial=np.array([x['state0'] for x in initial if x['training']]);distribution=[]
# The reference distribution is estimated only from training initial states.
center=train_initial.mean(0);spread=train_initial.std(0);spread=np.maximum(spread,1e-6)
for source in sorted({x['source'] for x in initial if not x['training']}):
 q=np.array([x['state0'] for x in initial if x['source']==source])
 for group,idx in EVAL_GROUPS.items():
  normalized=(q[:,idx]-center[idx])/spread[idx]
  nearest=np.sqrt(np.mean(((q[:,None,idx]-train_initial[None,:,idx])/spread[idx])**2,axis=-1)).min(axis=1)
  distribution.append({'source':source,'group':group,'mean_abs_z_vs_training_episode_initials':float(np.abs(normalized).mean()),
                       'nearest_training_initial_standardized_rms':float(nearest.mean()),
                       'outside_training_initial_minmax_fraction':float(np.mean((q[:,idx]<train_initial[:,idx].min(0))|(q[:,idx]>train_initial[:,idx].max(0))))})
for folder in map(Path,args.exports):
 m=json.loads((folder/'metadata.json').read_text());seal=json.loads((folder/'seal.json').read_text())
 if m['data_kind']!='real' or file_sha256(folder/'aligned.npz')!=seal['aligned_sha256'] or file_sha256(folder/'metadata.json')!=seal['metadata_sha256']:raise ValueError('Sealed real inputs required')
 if file_sha256(args.stats)!=m['normalization_sha256']:raise ValueError('Actual model statistics mismatch')
 with np.load(folder/'aligned.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
 if not data['valid'].all():raise ValueError('This temporal-mean decomposition requires complete valid windows')
 a,pr,h,s=(np.asarray(data[k],np.float64) for k in ('truth','prediction','history','current_state'));np.testing.assert_array_equal(h[:,-1],s)
 step=m['checkpoint_step'];seen=np.array([w['training_source'] for w in m['windows']],bool);sources=np.array([w['source_recording_id'] for w in m['windows']])
 train_s=s[seen].astype(float);mean_s=train_s.mean(0);std_s=np.maximum(train_s.std(0),1e-6)
 for source in sorted(set(sources[~seen])):
  q=s[sources==source].astype(float)
  for group,idx in EVAL_GROUPS.items():
   normalized=(q[:,idx]-mean_s[idx])/std_s[idx]
   lo=stats['state']['q01'][idx];hi=stats['state']['q99'][idx]
   window_distribution.append({'step':step,'source':source,'group':group,'mean_abs_z_vs_training_window_q0':float(np.abs(normalized).mean()),
      'outside_model_state_quantile_fraction':float(np.mean((q[:,idx]<lo)|(q[:,idx]>hi))),
      'nearest_training_window_standardized_rms':float(np.sqrt(np.mean(((q[:,None,idx]-train_s[None,:,idx])/std_s[idx])**2,axis=-1)).min(axis=1).mean())})
 corrected=anchor_correction(pr,h);decomp=decompose_offsets(a,pr);component_arrays.append((step,decomp))
 masks={'training_sources':seen,'candidate_unseen_sources':~seen}
 masks.update({'source:'+source:sources==source for source in sorted(set(sources))})
 for scope,mask in masks.items():
  d={k:v[mask] for k,v in decomp.items()}
  rr=[{'step':step,'scope':scope,**r} for r in grouped_offsets(d)];rows.extend(rr)
  if not scope.startswith('source:'):scope_fig.extend(rr)
  compare.extend({'step':step,'scope':scope,**r} for r in compare_constant_coefficient(d['bias'],data['c_gt'][mask],data['c_pred'][mask],data['scale'],data['basis'],data['q01'],data['q99']))
  for group,idx in EVAL_GROUPS.items():
   for method,value in [('B0_absolute',pr),('B1_q0_translation',corrected),('constant_measured_state',np.broadcast_to(s[:,None],a.shape))]:
    baselines.append({'step':step,'scope':scope,'group':group,'method':method,**summary((value[mask]-a[mask])[...,idx])})
   for name,error in [('GT0_minus_q0',a[:,0]-s),('Prediction0_minus_q0',pr[:,0]-s),('Prediction0_minus_GT0',pr[:,0]-a[:,0])]:
    references.append({'step':step,'scope':scope,'group':group,'comparison':name,**summary(error[mask][:,idx])})
 for i,w in enumerate(m['windows']):
  identity={'step':step,'window_index':i,'sample_id':w['sample_id'],'episode':w['episode_index'],'frame':w['current_frame'],'source':w['source_recording_id'],'training_source':w['training_source']}
  for r in grouped_offsets({k:v[i:i+1] for k,v in decomp.items()}):windows.append({**identity,**r})
  for j,name in enumerate(ORDER):
   total=decomp['total_mse'][i,j];bias=decomp['offset_mse'][i,j]
   joint_bias.append({**identity,'joint':name,'bias_rad':float(decomp['bias'][i,j]),'offset_mse':float(bias),'shape_mse':float(decomp['shape_mse'][i,j]),'total_mse':float(total),'offset_fraction':float(bias/total) if total else None})
 provenance.append({'step':step,'seal':seal,'metadata_sha256':file_sha256(folder/'metadata.json'),'checkpoint_parameter_sha256':m['checkpoint_parameter_sha256'],'normalization_sha256':m['normalization_sha256']})
out.mkdir(parents=True)
for step,components in component_arrays:np.savez_compressed(out/f'offset_components_step{step}.npz',**components)
for name,value in [('offset_groups',rows),('coefficient0_vs_physical_bias',compare),('q0_gt0_pred0',references),('B0_B1_controls',baselines),('offset_windows',windows),('offset_per_joint_window',joint_bias),('episode_initial_distribution',distribution),('window_reference_distribution',window_distribution)]:csv_rows(out/(name+'.csv'),value)
report={'data_kind':'real','research_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'input_exports':provenance,'source_audit_sha256':file_sha256(args.audit),
 'component_files_sha256':{f.name:file_sha256(f) for f in out.glob('offset_components_step*.npz')},'analysis_code_sha256':{str(f):file_sha256(f) for f in [Path(__file__),Path('src/trc/analysis/flash/offset.py')]},'offset_metrics':rows,'coefficient0_vs_bias':compare,'initial_distribution':distribution,'window_reference_distribution':window_distribution,
 'B1_limit':'Only offline causal-input translation, not a trained B2 or deployment-safe controller. q0 is not GT command0.',
 'independence_limit':'Different source IDs/timestamps and no exact numeric copies do not prove raw bag identity/calibration independence.'}
write_report(out/'offset_report.json',report,protected=list(map(Path,args.exports)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
groups=['left_arm','left_hand','right_arm','right_hand'];x=np.arange(4)
for ax,scope,label in zip(axes,['training_sources','candidate_unseen_sources'],['485 training-source windows','20 candidate windows (4 sources)']):
 rr=[next(r for r in rows if r['step']==6000 and r['scope']==scope and r['group']==g) for g in groups]
 ax.bar(x,[r['offset_mse'] for r in rr],label='Mean offset MSE');ax.bar(x,[r['shape_mse'] for r in rr],bottom=[r['offset_mse'] for r in rr],label='Shape MSE')
 ax.set_xticks(x,[g.replace('_',' ') for g in groups],rotation=25);ax.set_title(label);ax.set_ylabel('Final physical MSE (rad²)');ax.legend(fontsize=8)
fig.suptitle('REAL Step6000: physical mean-offset decomposition (not coefficient allocation)')
fig.savefig(out/'figure1_physical_offset_decomposition.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,scope,title in zip(axes,['training_sources','candidate_unseen_sources'],['Training sources','Candidate sources']):
 rr=[r for r in baselines if r['step']==6000 and r['scope']==scope and r['group']=='full_action']
 ax.bar([r['method'].replace('_','\n') for r in rr],[r['rmse'] for r in rr]);ax.set_title(title);ax.set_ylabel('Final RMSE (rad)')
fig.suptitle('REAL offline controls: B1 forces q0; no deployment timing validation / no B2 training')
fig.savefig(out/'figure2_causal_anchor_control.png',dpi=180);plt.close(fig)
print(json.dumps({'status':'completed','scope_results':[r for r in scope_fig if r['group']=='full_action']},indent=2))
