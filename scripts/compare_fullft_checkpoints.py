#!/usr/bin/env python3
"""Compare sealed, identical FullFT windows without running a model."""
import argparse,json
from pathlib import Path
import numpy as np
from trc.io import file_sha256,write_report
from trc.analysis.flash.metrics import evaluate
from trc.analysis.flash.report import csv_rows

p=argparse.ArgumentParser();p.add_argument('--exports',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output)
if out.exists():raise FileExistsError('New comparison output required')
reference=None;rows=[];identities=[]
for folder in map(Path,a.exports):
 meta=json.loads((folder/'metadata.json').read_text());seal=json.loads((folder/'seal.json').read_text())
 if file_sha256(folder/'aligned.npz')!=seal['aligned_sha256'] or file_sha256(folder/'metadata.json')!=seal['metadata_sha256']:raise ValueError('Export seal mismatch')
 if meta['inference_mode']!='fresh_fullft_checkpoint_paired_raw_inputs':raise ValueError('Fresh FullFT inference required')
 with np.load(folder/'aligned.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
 identities.append({'step':meta['checkpoint_step'],'seal':seal,'parameter_sha256':meta['checkpoint_parameter_sha256']})
 if reference:
  rm,rz=reference
  for key in ('source_commit','training_contract_sha256','normalization_sha256','historical_plan_sha256','rgb_sequence_sha256','windows'):
   if meta[key]!=rm[key]:raise ValueError('FullFT comparison contract/window mismatch: '+key)
  for key in ('truth','history','current_state','valid','q01','q99','basis','scale','c_gt','oracle'):np.testing.assert_array_equal(arrays[key],rz[key])
 else:reference=(meta,arrays)
 seen=np.asarray([w['training_source'] for w in meta['windows']],bool)
 for scope,mask in [('training_sources',seen),('candidate_unseen_sources',~seen),('all_paired',np.ones(len(seen),bool))]:
  m=evaluate(arrays['truth'][mask],arrays['oracle'][mask],arrays['prediction'][mask],arrays['valid'][mask])
  rows.extend({'step':meta['checkpoint_step'],'scope':scope,**r} for r in m['groups'])
out.mkdir(parents=True)
csv_rows(out/'paired_fullft_metrics.csv',rows)
write_report(out/'comparison.json',{'identical_inputs_and_oracle':True,'exports':identities,'metrics':rows,
 'code_sha256':file_sha256(__file__),'interpretation':'Same FullFT run, paired diagnostic windows; not independent generalization evidence'},protected=list(map(Path,a.exports)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,scope,title in zip(axes,['training_sources','candidate_unseen_sources'],['485 training-source windows','20 candidate unseen-source windows']):
 for method,label in [('representation','GT–Oracle'),('prediction','Oracle–Prediction'),('total','GT–Prediction')]:
  selected=sorted([r for r in rows if r['scope']==scope and r['method']==method and r['group']=='full_action'],key=lambda r:r['step'])
  ax.plot([r['step'] for r in selected],[r['rmse'] for r in selected],marker='o',label=label)
 ax.set_title(title);ax.set_xlabel('FullFT checkpoint step');ax.set_ylabel('Pooled RMSE (rad)');ax.legend(fontsize=8)
fig.suptitle('REAL paired FullFT checkpoints: same contract, source-separated diagnostics')
fig.savefig(out/'figure8_paired_fullft_progress.png',dpi=180);plt.close(fig)
print(json.dumps({'identical_inputs_and_oracle':True,'full_action_total':[r for r in rows if r['group']=='full_action' and r['method']=='total']},indent=2))
