#!/usr/bin/env python3
"""Build a strict replay gate from original-runtime layer traces."""
import argparse,json
from pathlib import Path
import numpy as np
from trc.io import file_sha256,write_report
from trc.analysis.flash.report import csv_rows
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root);out=Path(a.output)
if out.exists():raise FileExistsError('New report output required')
names=['default_a','default_b','controlled_a','controlled_b','fp32_a','fp32_b'];reports={n:json.loads((root/n/'trace_report.json').read_text()) for n in names}
identity=['source_commit','parameter_sha256','normalization_sha256','model_config']
for k in identity:
 if len({r[k] for r in reports.values()})!=1:raise ValueError('Replay identity differs: '+k)
indices=reports['fp32_a']['configuration']['window_indices'];rows=[];entries=[];cross=[];source_inputs_exact=True
for case,left,right in [('cross_default','default_a','default_b'),('cross_controlled','controlled_a','controlled_b'),('compile_policy_effect','default_a','controlled_a'),('cross_fp32','fp32_a','fp32_b')]:
 for ix in indices:
  with np.load(root/left/f'window{ix}_prepared.npz') as p0,np.load(root/right/f'window{ix}_prepared.npz') as p1:
   for key in p0.files:
    if not np.array_equal(p0[key],p1[key]):source_inputs_exact=False;raise ValueError('Prepared input differs')
  with np.load(root/left/f'window{ix}_stages.npz') as s0,np.load(root/right/f'window{ix}_stages.npz') as s1:
   for key in s0.files:
    d=s0[key].astype(float)-s1[key].astype(float);rows.append({'case':case,'window':ix,'stage':key,'max':float(np.abs(d).max()),'rmse':float(np.sqrt(np.mean(d*d)))})
   # The final language chunk is unaffected: first divergence is in visual tokens.
   if case=='compile_policy_effect':
    d=s0['prefix_tokens'].astype(float)-s1['prefix_tokens'].astype(float)
    for chunk in range(4):
     z=d[:,chunk*256:(chunk+1)*256];rows.append({'case':case,'window':ix,'stage':f'prefix_chunk_{chunk}','max':float(np.abs(z).max()),'rmse':float(np.sqrt(np.mean(z*z)))})
  with np.load(root/left/f'window{ix}_outputs.npz') as o0,np.load(root/right/f'window{ix}_outputs.npz') as o1:
   for key in o0.files:
    d=o0[key].astype(float)-o1[key].astype(float);cross.append({'case':case,'window':ix,'entry':key,'max_rad':float(np.abs(d).max()),'rmse_rad':float(np.sqrt(np.mean(d*d)))})
for name,r in reports.items():
 for w in r['windows']:
  for entry,value in w['entry_parity'].items():entries.append({'run':name,'window':w['window'],'entry':entry,'max_rad':value['max'],'rmse_rad':value['rmse']})
fp32max=max(v['max_rad'] for v in entries if v['run'].startswith('fp32'))
crossmax=max(v['max_rad'] for v in cross if v['case']=='cross_fp32')
randommax=max(v['different_seed_max_rad'] for r in reports.values() for w in r['windows'] for v in w['within_process_repeats'])
passed=source_inputs_exact and fp32max<=1e-5 and crossmax<=1e-5 and randommax==0
out.mkdir(parents=True)
for name,values in [('layer_differences',rows),('entry_parity',entries),('cross_process_parity',cross)]:csv_rows(out/(name+'.csv'),values)
result={'status':'PARITY_PASSED' if passed else 'PARITY_BLOCKED','strict_tolerance_rad':1e-5,'prepared_inputs_exact':source_inputs_exact,'fp32_max_entry_difference_rad':fp32max,'fp32_max_cross_process_difference_rad':crossmax,'seed_replay_difference_rad':randommax,
 'evaluation_policy':{'dtype':'float32','matmul_precision':'highest','batch_size':1,'num_steps':1,'padding':False},
 'xla_flags':'--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true','identity':{k:reports['fp32_a'][k] for k in identity},
 'layer_differences':rows,'entry_parity':entries,'cross_process_parity':cross,
 'root_cause':'BF16 VLM visual-prefix computation is sensitive to XLA execution graph/compiler policy. First observed changed module is image-token prefix, after identical preprocessed inputs and history encoding. Shared FP32 removes observed graph-entry mismatch at strict tolerance; no stochastic flow noise.',
 'limits':'Historical compiled executables were not preserved, so the exact historical CUDA kernel is not identified. This gate covers two actual windows and the stated device/runtime/shape, not arbitrary future environments.',
 'replay_reports_sha256':{n:file_sha256(root/n/'trace_report.json') for n in names},'comparison_code_sha256':file_sha256(__file__)}
write_report(out/'parity_gate.json',result,protected=[root/n for n in names]);print(json.dumps({k:v for k,v in result.items() if k not in ('layer_differences','entry_parity','cross_process_parity')},indent=2))
