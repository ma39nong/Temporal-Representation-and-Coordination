#!/usr/bin/env python3
"""Fixed four-model, recording-level report; never chooses a policy using GT."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from trc.io import file_sha256
from trc.analysis.flash.metrics import EVAL_GROUPS, summary
from trc.analysis.flash.offset import decompose_offsets
from trc.analysis.flash.report import csv_rows
from trc.data.action_schema import ORDER


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    protocol=json.loads((root/'frozen_protocol.json').read_text());audit=json.loads((root/'source_audit.json').read_text())
    if file_sha256(root/'source_audit.json')!=protocol['source_audit_sha256']:raise ValueError('Audit changed after freezing')
    arrays={};identities=[];reference=None;metadata=None
    for name in protocol['models']:
        folder=root/'exports'/name;m=json.loads((folder/'metadata.json').read_text());seal=json.loads((folder/'seal.json').read_text())
        for filename,key in [('aligned.npz','aligned_sha256'),('metadata.json','metadata_sha256')]:
            if file_sha256(folder/filename)!=seal[key]:raise ValueError('Export seal changed')
        if m['frozen_protocol_sha256']!=file_sha256(root/'frozen_protocol.json') or m['evaluation_precision']!='float32' or not m['future_perturbation_passed'] or not m['checkpoint_files_unchanged']:
            raise ValueError('Frozen evaluation/no-future gate failed')
        if m['checkpoint_parameter_sha256']!=protocol['models'][name]['parameter_sha256'] or m['prior_window_replay_max_rad']>1e-5:
            raise ValueError('Model or original-runtime replay mismatch')
        with np.load(folder/'aligned.npz',allow_pickle=False) as z:d={k:z[k].astype(float) for k in z.files}
        if d['truth'].shape!=(50,50,58) or not d['valid'].all():raise ValueError('Expected50 complete preregistered windows')
        if reference is None:reference=d;metadata=m
        else:
            for key in ('truth','oracle','c_gt','history','current_state','basis','scale','q01','q99','valid'):np.testing.assert_array_equal(d[key],reference[key])
            for key in ('windows','normalization_sha256','rgb_sequence_sha256'):
                if m[key]!=metadata[key]:raise ValueError('Paired observation identity differs')
        arrays[name]=d;identities.append({'model':name,'checkpoint':m['checkpoint_id'],'parameter_sha256':m['checkpoint_parameter_sha256'],
            'seal':seal,'prior_window_replay_max_rad':m['prior_window_replay_max_rad'],'runtime':m['runtime']})
    src=np.array([w['source_recording_id'] for w in metadata['windows']]);role=np.array([w['scope'] for w in metadata['windows']])
    if set(role)!={'development_success','locked_test_success'} or [int((role==r).sum()) for r in ('development_success','locked_test_success')]!=[20,30]:raise ValueError('Frozen split changed')
    scopes={r:role==r for r in ('development_success','locked_test_success')}
    scopes.update({s:src==s for s in sorted(set(src))})
    metrics=[];offsets=[];times=[];window_rows=[]
    for name,d in arrays.items():
        error=d['prediction']-d['truth'];dec=decompose_offsets(d['truth'],d['prediction'])
        for scope,mask in scopes.items():
            ident={'model':name,'scope':scope,'windows':int(mask.sum()),'sources':len(set(src[mask]))}
            for group,idx in EVAL_GROUPS.items():
                x=np.take(error[mask],idx,axis=-1)
                offsets.append({**ident,'group':group,'offset_mse':float(dec['offset_mse'][mask][:,idx].mean()),'shape_mse':float(dec['shape_mse'][mask][:,idx].mean())})
                for first,label in [(0,'all50'),(41,'last9')]:
                    y=x[:,first:];hit=(np.abs(y)>.5).any((1,2))
                    metrics.append({**ident,'group':group,'interval':label,'mse':float(np.mean(y*y)),**summary(y),
                        'increment_rmse_rad_per_step':float(np.sqrt(np.mean(np.diff(y,axis=1)**2))),
                        'over0p5_elements':int((np.abs(y)>.5).sum()),'over0p5_windows':int(hit.sum()),'over0p5_sources':len(set(src[mask][hit]))})
                if scope in ('development_success','locked_test_success'):
                    times.extend({**ident,'group':group,'tick':tick,**summary(x[:,tick])} for tick in range(50))
        for i,w in enumerate(metadata['windows']):
            window_rows.append({'model':name,'source':src[i],'role':role[i],'current_row':w['current_row'],**summary(error[i]),
                'offset_mse':float(dec['offset_mse'][i].mean()),'shape_mse':float(dec['shape_mse'][i].mean())})
    lookup={(r['model'],r['scope'],r['group'],r['interval']):r for r in metrics}
    paired=[]
    for source in sorted(set(src)):
        row={'source_recording_id':source,'role':role[src==source][0],'windows':int((src==source).sum())}
        for name in arrays:
            r=lookup[name,source,'full_action','all50'];b=lookup['D0',source,'full_action','all50']
            row.update({name+'_'+k:r[k] for k in ('rmse','mae','p95','p99','max')})
            row[name+'_minus_D0_rmse']=r['rmse']-b['rmse']
            row[name+'_minus_D0_mse']=r['mse']-b['mse']
        paired.append(row)
    directions=[]
    for cohort in ('development_success','locked_test_success'):
        for name in arrays:
            rows=[r for r in paired if r['role']==cohort];delta=np.array([r[name+'_minus_D0_rmse'] for r in rows])
            directions.append({'cohort':cohort,'model':name,'sources':len(rows),'improved':int((delta<0).sum()),'degraded':int((delta>0).sum()),'unchanged':int((delta==0).sum())})
    test=next(r for r in directions if r['cohort']=='locked_test_success' and r['model']=='D3')
    primary=lookup['D3','locked_test_success','full_action','all50']['rmse']<lookup['D0','locked_test_success','full_action','all50']['rmse'] and test['improved']>=5
    training=json.loads((root/'training_source_identity.json').read_text());initial_train=np.array([r['initial_state'] for r in training['records']])
    initial_stats=[]
    for r in audit['sources']:
        q=np.array(r['initial_state']);mask=src==r['source_recording_id']
        for group,idx in EVAL_GROUPS.items():
            baseline=initial_train[:,idx];lo,hi=np.quantile(baseline,[.01,.99],axis=0);current=q[idx]
            initial_stats.append({'source':r['source_recording_id'],'role':r['role'],'group':group,
                'initial_vs_training_mean_rms_rad':float(np.sqrt(np.mean((current-baseline.mean(0))**2))),
                'initial_joints_outside_training_initial_P01_P99':int(((current<lo)|(current>hi)).sum()),
                'joint_count':len(idx),'window_q0_mean_rad':float(reference['current_state'][mask][:,idx].mean()),
                'initial_state_mean_rad':float(current.mean())})
    out.mkdir(parents=True)
    for name,rows in [('metrics',metrics),('offsets',offsets),('temporal',times),('windows',window_rows),('source_pairs',paired),('initial_state_distribution',initial_stats)]:csv_rows(out/(name+'.csv'),rows)
    association=[]
    for cohort in ('development_success','locked_test_success'):
        rows=[r for r in paired if r['role']==cohort]
        distance=np.array([next(s['initial_vs_training_mean_rms_rad'] for s in initial_stats if s['source']==r['source_recording_id'] and s['group']=='full_action') for r in rows])
        gain=np.array([r['D0_rmse']-r['D3_rmse'] for r in rows])
        association.append({'role':cohort,'recordings':len(rows),'initial_distance_vs_D3_rmse_gain_descriptive_r':float(np.corrcoef(distance,gain)[0,1]),'no_causal_or_significance_claim':True})
    (out/'state_gain_association.json').write_text(json.dumps(association,indent=2)+'\n')
    decision={'verdict':'INDEPENDENT VALIDATION SUPPORTED' if primary else 'NOT SUPPORTED','primary_rule':protocol['decision_rule'],
        'directions':directions,'D3_locked_test_primary_passed':bool(primary),'frozen_protocol_sha256':file_sha256(root/'frozen_protocol.json'),
        'scope':'Independent from frozen training and prior505; user-reviewed successful demonstrations. Offline prediction only, not robot success rate.',
        'no_GT_model_switching':True,'no_hyperparameter_selection':True}
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    (out/'identity.json').write_text(json.dumps({'exports':identities,'research_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'analysis_code_sha256':file_sha256(__file__),'data_kind':'real','cohort_definition':'4 development and6 locked recordings, five windows each; no window-level independent CI'},indent=2)+'\n')
    plot(out,metrics,offsets,times,paired,metadata,arrays)
    oracle=[]
    for cohort in ('development_success','locked_test_success'):
        mask=role==cohort
        for group,idx in EVAL_GROUPS.items():
            oracle.append({'cohort':cohort,'group':group,**summary(np.take((reference['oracle']-reference['truth'])[mask],idx,axis=-1))})
    csv_rows(out/'oracle_metrics.csv',oracle)
    print(json.dumps({'decision':decision,'main':[r for r in metrics if r['scope'] in ('development_success','locked_test_success') and r['group']=='full_action' and r['interval']=='all50']},indent=2))


def plot(out,metrics,offsets,times,paired,metadata,arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax,cohort in zip(axes,('development_success','locked_test_success')):
        rows=[r for r in paired if r['role']==cohort]
        for name in arrays:ax.plot(range(len(rows)),[r[name+'_rmse'] for r in rows],'o-',label=name)
        ax.set_xticks(range(len(rows)),[r['source_recording_id'][:8] for r in rows],rotation=25);ax.set(title=cohort,ylabel='Absolute RMSE(rad)');ax.legend(fontsize=8)
    fig.suptitle('REAL human-reviewed successful tasks; frozen recording split, no tuning')
    fig.savefig(out/'source_rmse.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,group in zip(axes.flat,('left_arm','left_hand','right_arm','right_hand')):
        for name in arrays:
            rows=[r for r in times if r['scope']=='locked_test_success' and r['group']==group and r['model']==name]
            ax.plot([r['tick'] for r in rows],[r['rmse'] for r in rows],label=name)
        ax.axvspan(41,49,color='red',alpha=.1);ax.set(title=group,xlabel='Future tick',ylabel='RMSE(rad)');ax.legend(fontsize=8)
    fig.suptitle('REAL locked-test30 windows; final9 shaded, no operation-phase interpretation')
    fig.savefig(out/'group_temporal.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,cohort in zip(axes,('development_success','locked_test_success')):
        for i,name in enumerate(arrays):
            r=next(r for r in offsets if r['scope']==cohort and r['model']==name and r['group']=='full_action')
            ax.bar(np.arange(2)+(i-1.5)*.18,[r['offset_mse'],r['shape_mse']],width=.18,label=name)
        ax.set_xticks([0,1],['Offset','Shape']);ax.set(title=cohort,ylabel='MSE(rad squared)');ax.legend(fontsize=8)
    fig.suptitle('REAL paired frozen-model physical error decomposition')
    fig.savefig(out/'offset_shape.png',dpi=170);plt.close(fig)
    # Post-hoc descriptive cases only; no policy/metric selection is changed.
    test=np.array([w['scope']=='locked_test_success' for w in metadata['windows']]);cases=[]
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax,chosen in zip(axes,('D0','B2_FT250')):
        d=arrays[chosen];error=d['prediction']-d['truth'];score=np.where(test[:,None,None],np.abs(error),-1)
        i,t,j=np.unravel_index(score.argmax(),score.shape);w=metadata['windows'][i]
        ax.plot(d['truth'][i,:,j],label='GT',color='black',lw=2)
        ax.plot(d['oracle'][i,:,j],label='Oracle',ls='--')
        for name,value in arrays.items():ax.plot(value['prediction'][i,:,j],label=name)
        ax.set(title=chosen+' largest test error\n'+ORDER[j],xlabel='Future tick',ylabel='Absolute command(rad)');ax.legend(fontsize=8)
        cases.append({'selection':'largest locked-test element for '+chosen,'source':w['source_recording_id'],
            'row':w['current_row'],'tick':int(t),'joint':ORDER[j],'prediction_error_rad':float(error[i,t,j]),
            'oracle_error_at_same_element_rad':float(d['oracle'][i,t,j]-d['truth'][i,t,j])})
    fig.suptitle('REAL descriptive extreme cases; no future-label correction or robot outcome inference')
    fig.savefig(out/'extreme_cases.png',dpi=170);plt.close(fig)
    (out/'extreme_cases.json').write_text(json.dumps(cases,indent=2)+'\n')


if __name__=='__main__':main()
