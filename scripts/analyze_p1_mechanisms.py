#!/usr/bin/env python3
"""Read-only post-hoc analysis of the eight sealed P1-2 prediction exports."""
import argparse
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
from trc.analysis.flash.metrics import EVAL_GROUPS, summary
from trc.analysis.flash.offset import decompose_offsets
from trc.analysis.flash.diagnosis import paired_component_gain
from trc.analysis.flash.report import csv_rows
from trc.data.action_schema import ORDER
from trc.io import file_sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--exports-root',required=True)
    p.add_argument('--source-audit',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    root=Path(args.exports_root);data={};metas={};seals={};reference=None
    for step in (0,100,250,500):
        for trial,mode in [('B0','absolute'),('B2','state_relative')]:
            folder=root/f'{trial}_step{step}'
            seal=json.loads((folder/'seal.json').read_text())
            for f,k in [('aligned.npz','aligned_sha256'),('metadata.json','metadata_sha256')]:
                if file_sha256(folder/f)!=seal[k]:raise ValueError('Export checksum differs')
            m=json.loads((folder/'metadata.json').read_text())
            if (m['coordinate_mode'],m['checkpoint_step'],m['data_kind'],m['action_semantics'],m['action_unit'],m['K'],m['evaluation_precision'])!=(mode,step,'real','absolute','rad',6,'float32'):raise ValueError('Experiment identity differs')
            with np.load(folder/'aligned.npz',allow_pickle=False) as z:d={k:z[k].astype(np.float64) for k in z.files}
            if not d['valid'].all() or d['truth'].shape!=(505,50,58):raise ValueError('Expected 505 complete windows')
            windows=[(w['sample_id'],w['current_row'],w['source_recording_id'],w['training_source']) for w in m['windows']]
            if reference is None:reference=(windows,d,m)
            else:
                if windows!=reference[0]:raise ValueError('Unpaired windows')
                for k in ('truth','oracle','history','current_state','basis','scale','q01','q99','c_gt'):
                    np.testing.assert_array_equal(d[k],reference[1][k])
                for k in ('normalization_sha256','rgb_sequence_sha256','training_protocol','dataset','source_commit','upstream_runtime_commit','history_ticks','target_ext_ticks','horizon_ticks'):
                    if m[k]!=reference[2][k]:raise ValueError('Input/contract differs')
            if (trial,0) in metas and m['training_contract_sha256']!=metas[trial,0]['training_contract_sha256']:raise ValueError('Trial contract changed')
            data[trial,step]=d;metas[trial,step]=m;seals[f'{trial}_step{step}']=seal
    m=reference[2];d=reference[1];a=d['truth'];q=d['current_state']
    sources=np.array([w['source_recording_id'] for w in m['windows']]);seen=np.array([w['training_source'] for w in m['windows']])
    if (seen.sum(),(~seen).sum())!=(485,20):raise ValueError('Scope changed')
    aliases={w['source_recording_id']:w['sample_id'] for w in m['windows']}
    audit=json.loads(Path(args.source_audit).read_text())
    audit_records={r['source']:r for r in audit['records']}
    # Training-only episode-initial state moments; diagnostic z distances, not a data mask.
    train0=np.array([r['state0'] for r in audit['episode_initial_states'] if r['training']])
    mu=train0.mean(0);sigma=np.maximum(train0.std(0),1e-3)
    src0={r['source']:np.array(r['state0']) for r in audit['episode_initial_states']}
    rows=[];window_rows=[];groups={k:EVAL_GROUPS[k] for k in ('left_arm','left_hand','right_arm','right_hand','full_action')}
    masks={'training_sources':seen,'candidate_sources':~seen}
    masks.update({aliases[s]:sources==s for s in sorted(set(sources))})
    for step in (0,100,250,500):
        for trial in ('B0','B2'):
            pred=data[trial,step]['prediction'];e=pred-a;dec=decompose_offsets(a,pred)
            for scope,mask in masks.items():
                for group,idx in groups.items():
                    x=e[mask][...,idx];gt=a[mask][...,idx];state=q[mask][:,idx]
                    entry={'trial':trial,'step':step,'source':scope,'training':bool(seen[mask].all()),'windows':int(mask.sum()),'group':group,
                           'mse':float(np.mean(x*x)),**summary(x),
                           'offset_mse':float(np.mean(dec['offset_mse'][mask][:,idx])),
                           'shape_mse':float(np.mean(dec['shape_mse'][mask][:,idx])),
                           'increment_error_rmse_rad_per_step':float(np.sqrt(np.mean(np.diff(x,axis=1)**2))),
                           'gt_centered_rms_rad':float(np.sqrt(np.mean((gt-gt.mean(1,keepdims=True))**2))),
                           'q0_mean_rad':float(state.mean()),'q0_std_rad':float(state.std()),
                           'q0_train_z_rms':float(np.sqrt(np.mean(((state-mu[idx])/sigma[idx])**2))),
                           'gt0_minus_q0_rmse_rad':float(np.sqrt(np.mean((gt[:,0]-state)**2))),
                           'gt0_minus_q0_p99_rad':float(np.quantile(np.abs(gt[:,0]-state),.99)),
                           'abs_error_over_0p1_fraction':float(np.mean(np.abs(x)>.1)),
                           'abs_error_over_0p2_fraction':float(np.mean(np.abs(x)>.2)),
                           'abs_error_over_0p5_fraction':float(np.mean(np.abs(x)>.5))}
                    rows.append(entry)
            for i,w in enumerate(m['windows']):
                window_rows.append({'trial':trial,'step':step,'source':aliases[sources[i]],'training':bool(seen[i]),'window':i,'row':w['current_row'],
                                    'rmse':float(np.sqrt(np.mean(e[i]**2))),
                                    'offset_mse':float(dec['offset_mse'][i].mean()),'shape_mse':float(dec['shape_mse'][i].mean())})
    lookup={(r['trial'],r['step'],r['source'],r['group']):r for r in rows};paired=[]
    for r in rows:
        if r['trial']!='B0':continue
        z=lookup['B2',r['step'],r['source'],r['group']]
        paired.append({k:r[k] for k in ('step','source','training','windows','group')} |
                      {f'{trial}_{k}':v[k] for trial,v in [('B0',r),('B2',z)] for k in ('rmse','mae','p95','p99','max','offset_mse','shape_mse','increment_error_rmse_rad_per_step')} |
                      {'mse_gain':r['mse']-z['mse'],'rmse_change':z['rmse']-r['rmse'],
                       'offset_mse_gain':r['offset_mse']-z['offset_mse'],'shape_mse_gain':r['shape_mse']-z['shape_mse']})
    regress=[s for s in aliases.values() if lookup['B0',500,s,'full_action']['training'] and
             lookup['B2',500,s,'full_action']['rmse']>lookup['B0',500,s,'full_action']['rmse']]
    # Source (not window) summaries: descriptive only, no pseudo-independent CI.
    source_stats=[]
    for step in (0,100,250,500):
        for training in (True,False):
            ids=[s for s in aliases.values() if lookup['B0',step,s,'full_action']['training']==training]
            changes=np.array([lookup['B2',step,s,'full_action']['rmse']-lookup['B0',step,s,'full_action']['rmse'] for s in ids])
            source_stats.append({'step':step,'training':training,'sources':len(ids),'improved':int((changes<0).sum()),'regressed':int((changes>0).sum()),
                'median_rmse_change':float(np.median(changes)),'q25_rmse_change':float(np.quantile(changes,.25)),
                'q75_rmse_change':float(np.quantile(changes,.75)),
                'macro_mean_rmse_B0':float(np.mean([lookup['B0',step,s,'full_action']['rmse'] for s in ids])),
                'macro_mean_rmse_B2':float(np.mean([lookup['B2',step,s,'full_action']['rmse'] for s in ids]))})
    coeff=[];temporal=[];joint=[];checks=[]
    for step in (0,100,250,500):
        p0,p2=[data[t,step]['prediction'] for t in ('B0','B2')]
        c=[]
        for trial in ('B0','B2'):
            z=data[trial,step]
            c.append(np.einsum('tk,nkj,k,j->nktj',z['basis'],z['c_pred']-z['c_gt'],z['scale'],(z['q99']-z['q01']+1e-6)/2))
        for scope,mask in masks.items():
            # All 101 sources get coefficient attribution and temporal profiles.
            for group,idx in groups.items():
                selected=[np.take(v[mask],idx,axis=-1) for v in (a,d['oracle'],p0,p2,*c)]
                gain=paired_component_gain(*selected)
                checks.append(gain['decode_remainder_max_rad'])
                for k,val in enumerate(gain['signed_final_mse_gain']):
                    coeff.append({'step':step,'source':scope,'group':group,'degree':k,'signed_final_mse_gain_rad2':float(val),
                                  'decode_roundoff_gain_rad2':gain['roundoff_gain'],'total_mse_gain_rad2':gain['total_mse_gain']})
                e0,e2=selected[2]-selected[0],selected[3]-selected[0]
                for tick in range(50):
                    temporal.append({'step':step,'source':scope,'group':group,'tick':tick,'B0_rmse':float(np.sqrt(np.mean(e0[:,tick]**2))),
                                     'B2_rmse':float(np.sqrt(np.mean(e2[:,tick]**2)))})
        if step==500:
            for scope,mask in masks.items():
                for j in range(58):
                    e0,e2=(p0-a)[mask,:,j],(p2-a)[mask,:,j]
                    joint.append({'source':scope,'joint':j,'name':ORDER[j],'B0_rmse':float(np.sqrt(np.mean((p0-a)[mask,:,j]**2))),
                                  'B2_rmse':float(np.sqrt(np.mean((p2-a)[mask,:,j]**2))),
                                  'mse_change':float(np.mean(e2**2)-np.mean(e0**2)),
                                  'B0_bias_rad':float(e0.mean()),'B2_bias_rad':float(e2.mean()),
                                  'B0_increment_rmse':float(np.sqrt(np.mean(np.diff((p0-a)[mask,:,j],axis=1)**2))),
                                  'B2_increment_rmse':float(np.sqrt(np.mean(np.diff((p2-a)[mask,:,j],axis=1)**2)))})
    state_profiles=[]
    for s,alias in aliases.items():
        mask=sources==s
        for j in range(58):
            state_profiles.append({'source':alias,'training':bool(seen[mask].all()),'joint':j,'name':ORDER[j],
                'episode_initial_state_rad':float(src0[s][j]),'q0_mean_rad':float(q[mask,j].mean()),'q0_std_rad':float(q[mask,j].std()),
                'q0_min_rad':float(q[mask,j].min()),'q0_max_rad':float(q[mask,j].max()),
                'gt0_minus_q0_mean_rad':float((a[mask,0,j]-q[mask,j]).mean()),
                'gt0_minus_q0_rmse_rad':float(np.sqrt(np.mean((a[mask,0,j]-q[mask,j])**2)))})
    out.mkdir(parents=True)
    for name,table in [('metrics',rows),('paired_sources',paired),('windows',window_rows),('source_summary',source_stats),
                       ('coefficient_gain',coeff),('temporal',temporal),('joints',joint),('state_profiles',state_profiles)]:csv_rows(out/(name+'.csv'),table)
    cohort=[]
    for name,ids in [('regressing8',regress),('improving89',[s for s in aliases.values() if lookup['B0',500,s,'full_action']['training'] and s not in regress])]:
        for group in groups:
            r=[lookup['B0',500,s,group] for s in ids];z=[lookup['B2',500,s,group] for s in ids]
            cohort.append({'cohort':name,'group':group,'sources':len(ids),**{k+'_median':float(np.median([x[k] for x in r])) for k in
                ('q0_train_z_rms','gt0_minus_q0_rmse_rad','gt_centered_rms_rad')},
                'offset_worse_sources':sum(y['offset_mse']>x['offset_mse'] for x,y in zip(r,z)),
                'shape_worse_sources':sum(y['shape_mse']>x['shape_mse'] for x,y in zip(r,z)),
                'increment_worse_sources':sum(y['increment_error_rmse_rad_per_step']>x['increment_error_rmse_rad_per_step'] for x,y in zip(r,z))})
    csv_rows(out/'cohorts.csv',cohort)
    manifest={'data_kind':'real','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'script_sha256':file_sha256(__file__),'source_audit_sha256':file_sha256(args.source_audit),'export_seals':seals,
        'analysis_module_sha256':file_sha256(Path(__file__).resolve().parents[1]/'src/trc/analysis/flash/diagnosis.py'),
        'regressing_training_source_aliases':regress,'source_mapping':aliases,'max_decode_remainder_rad':max(checks),
        'normalization_sha256':m['normalization_sha256'],'training_contract_sha256':m['training_contract_sha256'],
        'statistics':'Sources are descriptive clusters. No window-level CI or independent-test claims.',
        'coefficient_allocation':'Physical sampled basis, quantile span and original scaling; signed interactions plus oracle residual; float remainder separate.',
        'source_quality':{aliases[s]:audit_records[s] for s in aliases},
        'checkpoint_ids':{f'{t}_{st}':{'id':meta['checkpoint_id'],'parameter_sha256':meta['checkpoint_parameter_sha256']} for (t,st),meta in metas.items()}}
    (out/'identity.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    plot(out,paired,regress,aliases,sources,m,data,cohort)
    print(json.dumps({'output':str(out),'regressing_sources':regress,'max_decode_remainder_rad':max(checks)}))


def plot(out,paired,regress,aliases,sources,m,data,cohort):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'savefig.dpi':170})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,training in zip(axes,(True,False)):
        for step in (100,250,500):
            r=[r for r in paired if r['step']==step and r['group']=='full_action' and r['training']==training and r['source'] not in ('training_sources','candidate_sources')]
            ax.scatter([x['offset_mse_gain'] for x in r],[x['shape_mse_gain'] for x in r],label=f'Step{step}',alpha=.7,s=18)
        ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
        ax.set(xlabel='B0 - B2 offset MSE (rad²)',ylabel='B0 - B2 shape MSE (rad²)',title='97 training sources' if training else '4 candidate sources (exploratory)');ax.legend()
    fig.suptitle('REAL paired source gains: positive values favor B2');fig.savefig(out/'source_gain.png');plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,12),layout='constrained')
    a=data['B0',500]['truth'];o=data['B0',500]['oracle'];p0=data['B0',500]['prediction'];p2=data['B2',500]['prediction']
    cases=[]
    for ax,s in zip(axes.flat,regress):
        original=next(k for k,v in aliases.items() if v==s);ids=np.flatnonzero(sources==original)
        delta=(p2-a)**2-(p0-a)**2;ij=np.unravel_index(np.argmax(delta[ids].mean(1)),(len(ids),58));i,j=int(ids[ij[0]]),int(ij[1])
        for value,label in [(a,'GT'),(o,'Oracle'),(p0,'B0'),(p2,'B2')]:ax.plot(np.arange(50)/30,value[i,:,j],label=label,lw=1.4)
        ax.set(title=f'{s} | row {m["windows"][i]["current_row"]} | {ORDER[j]}',xlabel='Prediction seconds',ylabel='Absolute command (rad)');ax.grid(alpha=.2)
        cases.append({'source':s,'window':i,'row':m['windows'][i]['current_row'],'joint':j,'joint_name':ORDER[j],
                      'selection':'Largest B2-minus-B0 joint trajectory MSE within this regressing source'})
    axes.flat[0].legend();fig.suptitle('REAL Step500: all 8 regressing training sources; post-hoc cases, no phase labels')
    fig.savefig(out/'regression_cases.png');plt.close(fig)
    (out/'regression_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');tails=[]
    for ax,training in zip(axes,(True,False)):
        mask=np.array([w['training_source']==training for w in m['windows']])
        for name,pred in [('B0',p0),('B2',p2)]:
            errors=np.abs((pred-a)[mask]).ravel();q=np.linspace(.95,.9999,180)
            ax.plot(q,np.quantile(errors,q),label=name)
            for quantile in (.95,.99,.999,.9995,.9999,1.):
                tails.append({'trial':name,'training':training,'quantile':quantile,
                              'absolute_error_rad':float(np.quantile(errors,quantile)),
                              'count_over_0p5_rad':int((errors>.5).sum()),'element_count':len(errors)})
        ax.set(title='Training 485 windows' if training else 'Candidate 20 windows (exploratory)',xlabel='Absolute-error quantile',ylabel='Error (rad)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('REAL Step500: tail distribution, not just the maximum');fig.savefig(out/'tail_quantiles.png');plt.close(fig)
    csv_rows(out/'tail_quantiles.csv',tails)
    temporal=list(csv.DictReader((out/'temporal.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,group in zip(axes.flat,('left_arm','left_hand','right_arm','right_hand')):
        rows=[r for r in temporal if r['source']=='training_sources' and r['step']=='500' and r['group']==group]
        for trial in ('B0','B2'):ax.plot([int(r['tick']) for r in rows],[float(r[trial+'_rmse']) for r in rows],label=trial)
        ax.set(title=group.replace('_',' '),xlabel='Future tick (30 Hz)',ylabel='Physical RMSE (rad)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('REAL Step500 training sources: group-specific late-horizon regression')
    fig.savefig(out/'group_temporal.png');plt.close(fig)


if __name__=='__main__':main()
