#!/usr/bin/env python3
"""Sealed Step8000/B0-FT/B2-FT comparisons using existing FLASH diagnostics."""
import argparse,csv,json,statistics
from pathlib import Path
import numpy as np
from trc.io import file_sha256
from trc.analysis.flash.metrics import EVAL_GROUPS,summary,coefficient_metrics
from trc.analysis.flash.offset import decompose_offsets
from trc.analysis.flash.diagnosis import coefficient_contributions
from trc.analysis.flash.report import csv_rows
from trc.data.action_schema import ORDER


def load(folder):
    folder=Path(folder);m=json.loads((folder/'metadata.json').read_text());seal=json.loads((folder/'seal.json').read_text())
    for name,key in [('aligned.npz','aligned_sha256'),('metadata.json','metadata_sha256')]:
        if file_sha256(folder/name)!=seal[key]:raise ValueError('Export seal changed')
    if m['data_kind']!='real' or m['action_semantics']!='absolute' or m['evaluation_precision']!='float32':raise ValueError('Real physical FP32 required')
    with np.load(folder/'aligned.npz',allow_pickle=False) as z:d={k:z[k].astype(float) for k in z.files}
    if d['truth'].shape!=(505,50,58) or not d['valid'].all():raise ValueError('505 complete windows required')
    return m,d,seal


def main():
    p=argparse.ArgumentParser();p.add_argument('--exports-root',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--steps',nargs='+',type=int,default=[0,50,100,250]);p.add_argument('--review',required=True);p.add_argument('--output',required=True);p.add_argument('--training-root')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    bm,bd,bs=load(a.baseline);review=json.loads(Path(a.review).read_text());failed={r['source_recording_id'] for r in review['records'] if r['review_entries']}
    entries=[('Step8000',8000,bm,bd,bs)]
    for step in a.steps:
        for trial in ('B0','B2'):entries.append((trial,step,*load(Path(a.exports_root)/f'{trial}_step{step}')))
    metrics=[];offsets=[];times=[];windows=[];coeff=[];contributions=[];identities=[];arrays={};scope_sources={}
    for trial,step,m,d,seal in entries:
        for k in ('windows','normalization_sha256','rgb_sequence_sha256'):
            if m[k]!=bm[k]:raise ValueError('Paired observations differ: '+k)
        for k in ('truth','oracle','c_gt','history','current_state','basis','scale','q01','q99'):np.testing.assert_array_equal(d[k],bd[k])
        train=np.array([w['training_source'] for w in m['windows']]);src=np.array([w['source_recording_id'] for w in m['windows']]);bad=np.array([s in failed for s in src])
        if (train.sum(),(~train&~bad).sum(),(~train&bad).sum())!=(485,10,10):raise ValueError('Source scopes changed')
        scopes={'training_sources':train,'candidate_unrejected_quality_unverified':~train&~bad,'candidate_reviewed_failed':~train&bad,'all_candidates_exploratory':~train}
        scopes.update({w['sample_id']:src==w['source_recording_id'] for w in m['windows']})
        gt,oracle,pred=[d[k] for k in ('truth','oracle','prediction')];error=pred-gt;dec=decompose_offsets(gt,pred)
        ident={'trial':trial,'step':step};arrays[trial,step]=(m,d)
        identities.append({**ident,'checkpoint':m['checkpoint_id'],'parameter_sha256':m['checkpoint_parameter_sha256'],'training_contract_sha256':m['training_contract_sha256'],'seal':seal})
        for scope,mask in scopes.items():
            sid={**ident,'scope':scope,'windows':int(mask.sum()),'sources':len(set(src[mask]))}
            for group,idx in EVAL_GROUPS.items():
                x=np.take(error[mask],idx,axis=-1)
                offsets.append({**sid,'group':group,'offset_mse':float(dec['offset_mse'][mask][:,idx].mean()),'shape_mse':float(dec['shape_mse'][mask][:,idx].mean())})
                for start,end,label in [(0,50,'all50'),(41,50,'last9')]:
                    y=x[:,start:end];hit=np.any(np.abs(y)>.5,axis=(1,2))
                    metrics.append({**sid,'group':group,'interval':label,'mse':float(np.mean(y*y)),**summary(y),
                        'increment_error_rmse_rad_per_step':float(np.sqrt(np.mean(np.diff(y,axis=1)**2))),
                        'over_0p5_elements':int((np.abs(y)>.5).sum()),'over_0p5_windows':int(hit.sum()),'over_0p5_sources':len(set(src[mask][hit]))})
                if scope in list(scopes)[:4]:
                    for tick in range(50):times.append({**sid,'group':group,'tick':tick,**summary(x[:,tick])})
            # Original mathematical utilities map coefficients through sampled basis/scales.
            coeff.extend({**sid,**r} for r in coefficient_metrics(d['c_gt'][mask],d['c_pred'][mask],d['scale'],d['q01'],d['q99']))
            c=coefficient_contributions(gt[mask],oracle[mask],pred[mask],d['c_gt'][mask],d['c_pred'][mask],d['scale'],d['basis'],d['q01'],d['q99'])
            contributions.extend({**sid,**r} for r in c['rows'])
        for i,w in enumerate(m['windows']):
            hit=np.abs(error[i])>.5
            windows.append({**ident,'source':w['sample_id'],'window':i,'row':w['current_row'],'training':bool(train[i]),'review_failed':bool(bad[i]),
                'rmse':float(np.sqrt(np.mean(error[i]**2))),'offset_mse':float(dec['offset_mse'][i].mean()),'shape_mse':float(dec['shape_mse'][i].mean()),'over_0p5_elements':int(hit.sum())})
    out.mkdir(parents=True)
    for name,r in [('metrics',metrics),('offsets',offsets),('temporal',times),('windows',windows),('coefficients',coeff),('coefficient_contributions',contributions)]:csv_rows(out/(name+'.csv'),r)
    pairs=[];lookup={(r['trial'],r['step'],r['scope'],r['group'],r['interval']):r for r in metrics}
    for r in metrics:
        if r['trial']=='Step8000':continue
        baseline=lookup['Step8000',8000,r['scope'],r['group'],r['interval']]
        pairs.append({k:r[k] for k in ('trial','step','scope','group','interval','windows','sources')} |
                     {'rmse':r['rmse'],'minus_step8000_rmse':r['rmse']-baseline['rmse'],
                      'minus_own_step0_rmse':r['rmse']-lookup[r['trial'],0,r['scope'],r['group'],r['interval']]['rmse'],
                      'B2_minus_B0_rmse':lookup['B2',r['step'],r['scope'],r['group'],r['interval']]['rmse']-lookup['B0',r['step'],r['scope'],r['group'],r['interval']]['rmse']})
    csv_rows(out/'paired_comparisons.csv',pairs)
    (out/'identity.json').write_text(json.dumps({'data_kind':'real','exports':identities,'code_sha256':file_sha256(__file__),'same_inputs_oracle':True,
       'quality_classification_sha256':file_sha256(a.review),'inference':'FP32/highest NFE1; no future target in prediction; exploratory candidates only',
       'coefficient_interpretation':'Error in restored Absolute coefficient coordinates; actual basis/scale/quantile used; signed contribution to P-minus-Oracle MSE, not GT-total MSE.'},indent=2)+'\n')
    plot(out,metrics,offsets,times,contributions,arrays,max(a.steps))
    if a.training_root:training_report(Path(a.training_root),out)
    print(json.dumps({'output':str(out),'models':len(entries),'main_metrics':[r for r in metrics if r['scope'] in list(scopes)[:3] and r['group']=='full_action' and r['interval']=='all50']},indent=2))


def training_report(root,out):
    import matplotlib.pyplot as plt
    stats={};logs={}
    for trial in ('B0','B2'):
        path=next((root/'runs/checkpoints'/('trc_p15_flash_'+trial.lower())).glob('*/flash_metrics/training.jsonl'))
        rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if [r['step'] for r in rows]!=list(range(1,251)):raise ValueError('250 accepted updates required')
        if not all(np.isfinite(r[k]) for r in rows for k in ('loss','gradient_norm','vision_gradient_norm','language_gradient_norm','expert_gradient_norm')):raise ValueError('Invalid training metrics')
        if any(r['effective_samples']!=r['step']*256 for r in rows):raise ValueError('Batch budget changed')
        audit=root/'runs/execution'/trial;initial=json.loads((audit/'initial_state.json').read_text());saved=json.loads((audit/'saved_state_3.json').read_text());restored=json.loads((audit/'restored_state.json').read_text())
        if any(saved[k]!=restored[k] for k in ('params_sha256','opt_state_sha256','completed_step')):raise ValueError('Restore differs')
        steady=[r for r in rows if r['step'] not in (1,4)];stats[trial]={'accepted_updates':250,'sample_positions':64000,'initial':initial,
            'first20_loss':statistics.mean(r['loss'] for r in rows[:20]),
            'last20_loss':statistics.mean(r['loss'] for r in rows[-20:]),'median_step_plus_io_seconds':statistics.median(r['seconds']+r['data_wait_seconds'] for r in steady),'restored_step3_byte_identical':True,
            'first20_gradient_norm':statistics.mean(r['gradient_norm'] for r in rows[:20]),
            'last20_gradient_norm':statistics.mean(r['gradient_norm'] for r in rows[-20:]),
            'gradient_norm_above_common_clip1_steps':sum(r['gradient_norm']>1 for r in rows),
            'gpu0_jax_peak_bytes':max(r['peak_device_bytes'] for r in rows),'training_log_sha256':file_sha256(path)};logs[trial]=rows
    if any(stats['B0']['initial'][k]!=stats['B2']['initial'][k] for k in ('params_sha256','opt_state_sha256')):raise ValueError('Initial pair differs')
    if [(r['step'],r['learning_rate'],r['effective_samples']) for r in logs['B0']]!=[(r['step'],r['learning_rate'],r['effective_samples']) for r in logs['B2']]:raise ValueError('Actual LR or sample budget differs')
    (out/'training_summary.json').write_text(json.dumps(stats,indent=2)+'\n');csv_rows(out/'training_steps.csv',[{'trial':t,**r} for t,rows in logs.items() for r in rows])
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for trial,rows in logs.items():
        for ax,key in zip(axes,('loss','flow_matching_loss','polynomial_consistency_loss')):
            x=[r['step'] for r in rows];y=np.array([r[key] for r in rows]);ax.plot(x,y,alpha=.2);ax.plot(x[19:],np.convolve(y,np.ones(20)/20,mode='valid'),label=trial);ax.set(title=key,xlabel='Accepted updates');ax.legend()
    fig.suptitle('REAL transfer training; 3-step precheck included in250');fig.savefig(out/'training_loss.png',dpi=170);plt.close(fig)


def plot(out,metrics,offsets,times,contributions,arrays,final):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scopes=['training_sources','candidate_unrejected_quality_unverified','candidate_reviewed_failed'];titles=['Training485 windows','Unrejected10 (success unverified)','Reviewed failed10']
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,scope,title in zip(axes,scopes,titles):
        for trial in ('B0','B2'):
            r=sorted([r for r in metrics if r['trial']==trial and r['scope']==scope and r['group']=='full_action' and r['interval']=='all50'],key=lambda r:r['step']);ax.plot([x['step'] for x in r],[x['rmse'] for x in r],'o-',label=trial+'-FT')
        base=next(r for r in metrics if r['trial']=='Step8000' and r['scope']==scope and r['group']=='full_action' and r['interval']=='all50');ax.axhline(base['rmse'],color='gray',ls='--',label='Original Step8000');ax.set(title=title,xlabel='Transfer updates',ylabel='Physical RMSE(rad)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('REAL same-parameter warm start; candidate results exploratory');fig.savefig(out/'transfer_progress.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,group in zip(axes.flat,('left_arm','left_hand','right_arm','right_hand')):
        for trial,step in [('Step8000',8000),('B0',final),('B2',final)]:
            r=[r for r in times if r['trial']==trial and r['step']==step and r['scope']=='training_sources' and r['group']==group];ax.plot([x['tick'] for x in r],[x['rmse'] for x in r],label=trial)
        ax.axvspan(41,49,alpha=.1,color='red');ax.set(title=group,xlabel='Future tick',ylabel='RMSE(rad)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle(f'REAL training-source temporal errors; transfer Step{final}');fig.savefig(out/'group_temporal.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,scope,title in zip(axes,scopes,titles):
        for trial,step in [('Step8000',8000),('B0',final),('B2',final)]:
            r=[r for r in times if r['trial']==trial and r['step']==step and r['scope']==scope and r['group']=='full_action']
            ax.plot([x['tick'] for x in r],[x['rmse'] for x in r],label=trial)
        ax.axvspan(41,49,alpha=.1,color='red');ax.set(title=title,xlabel='Future tick',ylabel='RMSE(rad)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle(f'REAL transfer Step{final} versus original baseline; final9 shaded');fig.savefig(out/'cohort_temporal.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,scope,title in zip(axes,scopes,titles):
        for pos,(trial,step) in enumerate([('Step8000',8000),('B0',final),('B2',final)]):
            r=next(r for r in offsets if r['trial']==trial and r['step']==step and r['scope']==scope and r['group']=='full_action')
            ax.bar(np.arange(2)+(pos-1)*.24,[r['offset_mse'],r['shape_mse']],width=.24,label=trial)
        ax.set_xticks([0,1],['Offset','Centered shape']);ax.set(title=title,ylabel='Physical MSE(rad²)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'REAL transfer Step{final}: exact physical error decomposition');fig.savefig(out/'offset_shape.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,scope,title in zip(axes,scopes,titles):
        for trial,step in [('Step8000',8000),('B0',final),('B2',final)]:
            r=sorted([r for r in contributions if r['trial']==trial and r['step']==step and r['scope']==scope and r['group']=='full_action'],key=lambda x:x['degree'])
            ax.plot([x['degree'] for x in r],[x['component_rmse_rad'] for x in r],'o-',label=trial)
        ax.set(title=title,xlabel='Legendre degree',ylabel='Decoded component RMSE(rad)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('REAL weighted per-degree residuals; squared components are NOT additive');fig.savefig(out/'coefficient_components.png',dpi=170);plt.close(fig)
    m,d2=arrays['B2',final];_,d0=arrays['B0',final];_,base=arrays['Step8000',8000];gt=base['truth'];e2=d2['prediction']-gt;e0=d0['prediction']-gt;train=np.array([w['training_source'] for w in m['windows']]);delta=np.mean(e2**2-e0**2,(1,2));zero=arrays['B2',0][1];ez=zero['prediction']-gt;vs_zero=np.mean(e2**2-ez**2,(1,2));chosen=[int(np.flatnonzero(train)[np.argmax(delta[train])]),int(np.flatnonzero(~train)[np.argmax(vs_zero[~train])]),int(np.argmax(np.max(np.abs(e2),(1,2))))]
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained');cases=[]
    labels=['Worst training regression vs B0-FT' if delta[chosen[0]]>0 else 'Smallest training gain vs B0-FT', 'Worst candidate regression vs B2 Step0' if vs_zero[chosen[1]]>0 else 'Candidate unchanged or improved vs B2 Step0', 'Largest B2 error']
    for ax,i,label in zip(axes,chosen,labels):
        tick,joint=np.unravel_index(np.abs(e2[i]).argmax(),e2[i].shape)
        for val,name in [(gt,'GT'),(base['oracle'],'Oracle'),(base['prediction'],'Step8000'),(d0['prediction'],'B0-FT'),(zero['prediction'],'B2 Step0'),(d2['prediction'],'B2-FT')]:ax.plot(np.arange(50),val[i,:,joint],label=name)
        w=m['windows'][i];ax.set(title=label+'\n'+ORDER[joint],xlabel='Prediction tick',ylabel='Absolute command(rad)');ax.legend(fontsize=8);cases.append({'selection':label,'window':i,'source':w['sample_id'],'row':w['current_row'],'joint':int(joint),'joint_name':ORDER[joint],'tick':int(tick),'B2_rmse':float(np.sqrt(np.mean(e2[i]**2))),'B0_rmse':float(np.sqrt(np.mean(e0[i]**2))),'B2_step0_rmse':float(np.sqrt(np.mean(ez[i]**2))),'mse_change_vs_B0':float(delta[i]),'mse_change_vs_B2_step0':float(vs_zero[i])})
    fig.suptitle(f'REAL transfer Step{final} post-hoc cases; no phase/FK/task-success inference');fig.savefig(out/'tail_cases.png',dpi=170);plt.close(fig);(out/'tail_cases.json').write_text(json.dumps(cases,indent=2)+'\n')


if __name__=='__main__':main()
