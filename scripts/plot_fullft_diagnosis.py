#!/usr/bin/env python3
"""Scope separated figures from verified FullFT diagnostics; no synthetic plots."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trc.analysis.flash.metrics import evaluate

p=argparse.ArgumentParser();p.add_argument('--diagnosis',required=True);p.add_argument('--export',required=True);a=p.parse_args()
out=Path(a.diagnosis);r=json.loads((out/'diagnosis.json').read_text());meta=json.loads((Path(a.export)/'metadata.json').read_text())
assert meta['inference_mode']=='fresh_fullft_checkpoint_paired_raw_inputs'
groups=['left_arm','right_arm','left_hand','right_hand','full_action'];labels=[g.replace('_',' ').title() for g in groups]
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for ax,scope,title in zip(axes,['training_sources','candidate_unseen_sources'],['97 training sources / 485 windows','4 candidate unseen sources / 20 windows']):
 for offset,checkpoint,label,color in [(-.18,'historical_step250','Frozen Step250','#888888'),(.18,'fullft',f'FullFT Step{r["checkpoint_step"]}','#0074d9')]:
  rows=[next(v for v in r['metrics'] if v['scope']==scope and v['checkpoint']==checkpoint and v['method']=='total' and v['group']==g) for g in groups]
  ax.bar(np.arange(len(groups))+offset,[v['rmse'] for v in rows],width=.36,label=label,color=color)
 ax.set_xticks(np.arange(len(groups)),labels,rotation=25,ha='right');ax.set_ylabel('Total pooled RMSE (rad)');ax.set_title(title);ax.legend(fontsize=8)
fig.suptitle('REAL paired windows: descriptive comparison, training conditions differ',fontsize=11)
fig.savefig(out/'figure5_training_vs_candidate_unseen.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,scope,title in zip(axes,['training_sources','candidate_unseen_sources'],['Training sources','Candidate unseen sources']):
 rows=[v for v in r['coefficient_contribution'] if v['scope']==scope and v['group']=='full_action']
 ax.bar([v['degree'] for v in rows],[v['signed_allocated_prediction_mse'] for v in rows],color='#3d9970')
 ax.set_xlabel('Legendre degree');ax.set_ylabel('Signed allocated prediction MSE (rad²)');ax.set_title(title)
fig.suptitle('REAL coefficient-to-action attribution; allocations include cross-degree interactions',fontsize=10)
fig.savefig(out/'figure6_coefficient_action_contribution.png',dpi=180);plt.close(fig)
with np.load(Path(a.export)/'aligned.npz') as z:
 truth,oracle,pred=z['truth'],z['oracle'],z['prediction']
seen=np.asarray([v['training_source'] for v in meta['windows']],bool)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,mask,title in zip(axes,[seen,~seen],['Training sources','Candidate unseen sources']):
 m=evaluate(truth[mask],oracle[mask],pred[mask])
 for method,label,color in [('representation','GT–Oracle','#3d9970'),('prediction','Oracle–Prediction','#ff851b'),('total','GT–Prediction','#0074d9')]:
  rows=[v for v in m['temporal'] if v['group']=='full_action' and v['method']==method]
  ax.plot([v['tick'] for v in rows],[v['rmse'] for v in rows],label=label,color=color)
 ax.set_title(title);ax.set_xlabel('Prediction tick');ax.set_ylabel('Pooled RMSE (rad)');ax.legend(fontsize=8)
fig.suptitle(f'REAL FullFT Step{r["checkpoint_step"]}: temporal profile by source identity',fontsize=11)
fig.savefig(out/'figure7_temporal_by_source_scope.png',dpi=180);plt.close(fig)
