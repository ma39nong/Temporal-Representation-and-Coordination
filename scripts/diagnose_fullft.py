#!/usr/bin/env python3
"""Paired source-scoped FullFT diagnosis, extending existing error attribution."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from trc.io import file_sha256,write_report
from trc.analysis.flash.metrics import evaluate,EVAL_GROUPS,summary
from trc.analysis.flash.diagnosis import coefficient_contributions,temporal_structure,tick_zero_alignment,conditioning_ranges
from trc.analysis.flash.report import csv_rows,build_report

p=argparse.ArgumentParser();p.add_argument('--export',required=True);p.add_argument('--historical-export',required=True);p.add_argument('--output',required=True);p.add_argument('--current-stats')
a=p.parse_args();root=Path(a.export);old=Path(a.historical_export);out=Path(a.output)
if out.exists():raise FileExistsError('New diagnostic output required')
meta=json.loads((root/'metadata.json').read_text())
for folder in (root,old):
 seal=json.loads((folder/'seal.json').read_text())
 assert file_sha256(folder/'aligned.npz')==seal['aligned_sha256'] and file_sha256(folder/'metadata.json')==seal['metadata_sha256']
with np.load(root/'aligned.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
with np.load(old/'aligned.npz',allow_pickle=False) as z:historical={k:z[k] for k in z.files}
oldmeta=json.loads((old/'metadata.json').read_text())
keys=lambda m:[(w['sample_id'],w['current_row'],w['source_recording_id']) for w in m['windows']]
assert keys(meta)==keys(oldmeta)
np.testing.assert_array_equal(arrays['truth'],historical['truth'])
np.testing.assert_array_equal(arrays['history'],historical['history'])
seen=np.asarray([w['training_source'] for w in meta['windows']],bool)
report=build_report(root,out,make_figures=True)
scopes={'all_paired':np.ones(len(seen),bool),'training_sources':seen,'candidate_unseen_sources':~seen}
scoped=[];contributions=[];pairs=[];temporal=[];overview=[];tick0=[];ranges=[];anchor_baseline=[];source_rows=[];decode_residuals={}
if a.current_stats:
 from trc.data.normalization import load_stats
 if file_sha256(a.current_stats)!=meta['normalization_sha256']:raise ValueError('Conditioning statistics identity mismatch')
 stats=load_stats(a.current_stats)
for scope,mask in scopes.items():
 if not mask.any():continue
 z={k:v[mask] if k in ('truth','oracle','prediction','c_gt','c_pred','history','current_state','valid') else v for k,v in arrays.items()}
 metric=evaluate(z['truth'],z['oracle'],z['prediction'],z['valid'])
 for group,idx in EVAL_GROUPS.items():
  anchor_baseline.append({'scope':scope,'group':group,**summary((z['current_state'][:,None,:]-z['truth'])[...,idx])})
 for row in metric['groups']:scoped.append({'scope':scope,'checkpoint':'fullft',**row})
 metric_old=evaluate(historical['truth'][mask],historical['oracle'][mask],historical['prediction'][mask],historical['valid'][mask])
 for row in metric_old['groups']:scoped.append({'scope':scope,'checkpoint':'historical_step250',**row})
 c=coefficient_contributions(z['truth'],z['oracle'],z['prediction'],z['c_gt'],z['c_pred'],z['scale'],z['basis'],z['q01'],z['q99'])
 decode_residuals[scope]=c['decode_residual_max_rad']
 contributions.extend({'scope':scope,**r} for r in c['rows']);pairs.extend({'scope':scope,**r} for r in c['pair_interactions'])
 t,o=temporal_structure(z['truth'],z['prediction']);temporal.extend({'scope':scope,**r} for r in t);overview.extend({'scope':scope,**r} for r in o)
 tick0.extend({'scope':scope,**r} for r in tick_zero_alignment(z['truth'],z['prediction'],z['history'],z['current_state']))
 if a.current_stats:
  ranges.extend({'scope':scope,**r} for r in conditioning_ranges(z['truth'],z['history'],z['current_state'],stats['state']['q01'],stats['state']['q99'],z['q01'],z['q99']))
for source in sorted({w['source_recording_id'] for w in meta['windows']}):
 mask=np.asarray([w['source_recording_id']==source for w in meta['windows']])
 for group,indices in EVAL_GROUPS.items():
  source_rows.append({'source_recording_id':source,'training_source':bool(seen[mask][0]),'windows':int(mask.sum()),
                      'group':group,**summary((arrays['prediction'][mask]-arrays['truth'][mask])[...,indices])})
for name,rows in [('scoped_group_metrics',scoped),('coefficient_action_contribution',contributions),('coefficient_interactions',pairs),
                  ('temporal_intervals',temporal),('temporal_structure',overview),('tick0_alignment',tick0),('conditioning_ranges',ranges),('causal_constant_state_baseline',anchor_baseline),('per_source_metrics',source_rows)]:csv_rows(out/(name+'.csv'),rows)
summary={'checkpoint_step':meta['checkpoint_step'],'counts':{'all':len(seen),'training_source_windows':int(seen.sum()),'candidate_unseen_windows':int((~seen).sum())},
 'paired_gt_history_exact':True,'oracle_difference_current_vs_historical_max_rad':float(np.max(np.abs(arrays['oracle']-historical['oracle']))),
 'metrics':scoped,'conditioning_ranges':ranges,'causal_anchor_baseline':anchor_baseline,'temporal_structure':overview,'coefficient_contribution':contributions,
 'historical_comparison_limit':'Descriptive pairing only: freeze/fullft, statistics, datasets and update budgets differ',
 'coefficient_intervention_limit':'Oracle repair uses true future coefficients only as attribution, never in model inference',
 'decode_residual_max_rad':decode_residuals,
 'export_seal':json.loads((root/'seal.json').read_text()),
 'historical_export_seal':json.loads((old/'seal.json').read_text()),
 'research_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'analysis_code_sha256':{str(f):file_sha256(f) for f in [Path(__file__),Path('src/trc/analysis/flash/diagnosis.py'),Path('scripts/plot_fullft_diagnosis.py')]},
 'current_stats_sha256':file_sha256(a.current_stats) if a.current_stats else None,
 'report_file_sha256':file_sha256(out/'report.json')}
write_report(out/'diagnosis.json',summary,protected=[root,old])
print(json.dumps({'step':meta['checkpoint_step'],'counts':summary['counts'],'full_action':[r for r in scoped if r['group']=='full_action' and r['method']=='total']},indent=2))
