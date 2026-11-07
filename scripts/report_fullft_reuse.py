#!/usr/bin/env python3
"""Reuse sealed FullFT evaluations; separate rejected quality from independence."""
import argparse
import json
from pathlib import Path
import numpy as np
from trc.io import file_sha256
from trc.analysis.flash.metrics import EVAL_GROUPS,summary
from trc.analysis.flash.offset import decompose_offsets
from trc.analysis.flash.report import csv_rows
from trc.data.action_schema import ORDER


def main():
    p=argparse.ArgumentParser();p.add_argument('--exports',nargs='+',required=True)
    p.add_argument('--review',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    review=json.loads(Path(a.review).read_text())
    failed={r['source_recording_id'] for r in review['records'] if r['review_entries']}
    rows=[];offsets=[];times=[];windows=[];cases=[];identities=[];reference=None;last=None
    for folder in map(Path,a.exports):
        meta=json.loads((folder/'metadata.json').read_text());seal=json.loads((folder/'seal.json').read_text())
        for f,k in [('aligned.npz','aligned_sha256'),('metadata.json','metadata_sha256')]:
            if file_sha256(folder/f)!=seal[k]:raise ValueError('Export seal differs')
        if meta['data_kind']!='real' or meta['evaluation_precision']!='float32' or meta['action_semantics']!='absolute':raise ValueError('Physical paired FP32 real exports required')
        with np.load(folder/'aligned.npz',allow_pickle=False) as z:d={k:z[k].astype(float) for k in z.files}
        if d['truth'].shape!=(505,50,58) or not d['valid'].all():raise ValueError('Complete fixed 505 windows required')
        if reference:
            rm,rd=reference
            for k in ('windows','normalization_sha256','training_contract_sha256','rgb_sequence_sha256','source_commit'):
                if meta[k]!=rm[k]:raise ValueError('Checkpoint comparison differs: '+k)
            for k in ('truth','oracle','c_gt','history','current_state','q01','q99','basis','scale'):np.testing.assert_array_equal(d[k],rd[k])
        else:reference=(meta,d)
        step=meta['checkpoint_step'];identities.append({'step':step,'checkpoint':meta['checkpoint_id'],'parameter_sha256':meta['checkpoint_parameter_sha256'],'seal':seal})
        train=np.array([w['training_source'] for w in meta['windows']]);sources=np.array([w['source_recording_id'] for w in meta['windows']]);bad=np.array([s in failed for s in sources])
        if (train.sum(),(~train&~bad).sum(),(~train&bad).sum())!=(485,10,10):raise ValueError('Source quality scope changed')
        scopes={'training_sources':train,'candidate_unrejected_quality_unverified':~train&~bad,'candidate_reviewed_failed':~train&bad,'all_candidates_exploratory':~train}
        scopes.update({w['sample_id']:sources==w['source_recording_id'] for w in meta['windows']})
        gt,oracle,pred=[d[k] for k in ('truth','oracle','prediction')];dec=decompose_offsets(gt,pred)
        for scope,mask in scopes.items():
            for group,idx in EVAL_GROUPS.items():
                offsets.append({'step':step,'scope':scope,'group':group,'windows':int(mask.sum()),
                    'offset_mse':float(dec['offset_mse'][mask][:,idx].mean()),'shape_mse':float(dec['shape_mse'][mask][:,idx].mean())})
                for method,err in [('total',pred-gt),('representation',oracle-gt),('prediction',pred-oracle)]:
                    x=np.take(err[mask],idx,axis=-1)
                    for start,end,label in [(0,50,'all50'),(41,50,'last9')]:
                        val=x[:,start:end]
                        rows.append({'step':step,'scope':scope,'group':group,'method':method,'interval':label,'windows':int(mask.sum()),
                            'mse':float(np.mean(val**2)),**summary(val),'over_0p5_count':int((np.abs(val)>.5).sum())})
                    if scope in list(scopes)[:4] and method=='total':
                        for t in range(50):times.append({'step':step,'scope':scope,'group':group,'tick':t,**summary(x[:,t])})
        for i,w in enumerate(meta['windows']):
            windows.append({'step':step,'window':i,'source':w['sample_id'],'row':w['current_row'],'training':bool(train[i]),'review_failed':bool(bad[i]),
                'rmse':float(np.sqrt(np.mean((pred[i]-gt[i])**2))),'offset_mse':float(dec['offset_mse'][i].mean()),'shape_mse':float(dec['shape_mse'][i].mean())})
        if last is None or step>last[0]:last=(step,meta,d,scopes)
    if len(set(x['step'] for x in identities))!=len(identities):raise ValueError('Duplicate checkpoints')
    out.mkdir(parents=True)
    for name,r in [('metrics',rows),('offsets',offsets),('temporal',times),('windows',windows)]:csv_rows(out/(name+'.csv'),r)
    (out/'identity.json').write_text(json.dumps({'exports':identities,'same_inputs_and_oracle':True,'review_sha256':file_sha256(a.review),
        'code_sha256':file_sha256(__file__),'quality_caveat':'Unrejected is not verified successful. No candidate source is a proven independent success test.',
        'inference_precision':'float32/highest; original FLASH NFE1; zero training updates'},indent=2)+'\n')
    plot(out,rows,offsets,times,last)
    print(json.dumps({'output':str(out),'full_action_total':[r for r in rows if r['group']=='full_action' and r['interval']=='all50' and r['method']=='total' and r['scope'] in list(last[3])[:4]]},indent=2))


def plot(out,rows,offsets,times,last):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scopes=['training_sources','candidate_unrejected_quality_unverified','candidate_reviewed_failed']
    names=['Training sources (485 windows)','Unrejected candidates (10; success unknown)','Reviewed failed demonstrations (10)']
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,scope,title in zip(axes,scopes,names):
        for method in ('representation','prediction','total'):
            r=sorted([r for r in rows if r['scope']==scope and r['group']=='full_action' and r['interval']=='all50' and r['method']==method],key=lambda x:x['step'])
            ax.plot([x['step'] for x in r],[x['rmse'] for x in r],'o-',label=method)
        ax.set(title=title,xlabel='FullFT optimizer step',ylabel='Physical RMSE (rad)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('REAL FullFT paired progress; candidate independence incomplete');fig.savefig(out/'fullft_progress.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,scope,title in zip(axes,scopes,names):
        for kind in ('offset_mse','shape_mse'):
            r=sorted([r for r in offsets if r['scope']==scope and r['group']=='full_action'],key=lambda x:x['step'])
            ax.plot([x['step'] for x in r],[x[kind] for x in r],'o-',label=kind)
        ax.set(title=title,xlabel='FullFT optimizer step',ylabel='Physical MSE (rad²)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('REAL exact per-window offset / centered-shape decomposition');fig.savefig(out/'fullft_offsets.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,scope,title in zip(axes,scopes,names):
        for step in sorted({r['step'] for r in times}):
            r=[r for r in times if r['step']==step and r['scope']==scope and r['group']=='full_action']
            ax.plot([x['tick'] for x in r],[x['rmse'] for x in r],label=f'Step{step}')
        ax.axvspan(41,49,alpha=.1,color='red');ax.set(title=title,xlabel='Prediction tick (30 Hz)',ylabel='Physical RMSE (rad)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('REAL temporal error; shaded final 9 ticks');fig.savefig(out/'fullft_temporal.png',dpi=170);plt.close(fig)
    step,meta,d,scopes=last;fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');cases=[]
    for ax,group in zip(axes.flat,('left_arm','left_hand','right_arm','right_hand')):
        idx=EVAL_GROUPS[group];error=np.abs(d['prediction']-d['truth'])[...,idx];i,t,j=np.unravel_index(error.argmax(),error.shape);j=idx[j]
        for key in ('truth','oracle','prediction'):ax.plot(np.arange(50)/30,d[key][i,:,j],label=key)
        w=meta['windows'][i];ax.set(title=f'{group} | {w["sample_id"]} row{w["current_row"]}\n{ORDER[j]}',xlabel='Seconds',ylabel='Absolute command (rad)');ax.legend(fontsize=8)
        cases.append({'source':w['sample_id'],'window':int(i),'row':w['current_row'],'joint':int(j),'joint_name':ORDER[j],'tick':int(t),
            'training':w['training_source'],'absolute_error_rad':float(error.max())})
    fig.suptitle(f'REAL FullFT Step{step}: worst trajectory per group; post-hoc, no phase labels');fig.savefig(out/'fullft_extreme_cases.png',dpi=170);plt.close(fig)
    (out/'extreme_cases.json').write_text(json.dumps(cases,indent=2)+'\n')


if __name__=='__main__':main()
