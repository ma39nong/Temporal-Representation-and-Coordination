#!/usr/bin/env python3
"""P2-0 scoped feasibility. Never materializes prior locked-test numeric rows."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from trc.io import file_sha256,object_sha256
from trc.analysis.flash.crs import selected_npz_rows,mix_actions,oracle_weights,StateReference,source_balanced_weights,fit_threshold
from trc.analysis.flash.metrics import summary,EVAL_GROUPS
from trc.analysis.flash.offset import decompose_offsets
from trc.analysis.flash.report import csv_rows


def load_pair(a,b,role):
    ma=json.loads((a/'metadata.json').read_text());mb=json.loads((b/'metadata.json').read_text())
    required={'action_semantics':'absolute','action_unit':'rad','K':6,'coefficient_tokens':7,
              'history_ticks':[-7,0],'target_ext_ticks':[-2,51],'horizon_ticks':[0,49],
              'normalization_sha256':'a5d7bd2dd422148736c69f1bf52a721d5770ba810f3bace9a2dd413395544d52',
              'upstream_runtime_commit':'d208d414efbb78483f44bfef4c2a9314115dd632'}
    for folder,m in [(a,ma),(b,mb)]:
        if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Frozen action/statistics/runtime contract mismatch')
        seal=json.loads((folder/'seal.json').read_text())
        for file,key in [('aligned.npz','aligned_sha256'),('metadata.json','metadata_sha256')]:
            if file_sha256(folder/file)!=seal[key]:raise ValueError('Input seal changed')
        if m['checkpoint_parameter_sha256']!='40b2c2c4dcdd6fb88dd847d5762d2071fba5028af0942c6dfe457409fe5f5919' or m['evaluation_precision']!='float32':raise ValueError('Frozen Step8000 FP32 required')
    for key in ('windows','normalization_sha256','rgb_sequence_sha256'):
        if ma[key]!=mb[key]:raise ValueError('Unpaired exports')
    if (ma['coordinate_mode'],mb['coordinate_mode'])!=('absolute','state_relative'):raise ValueError('D0/D3 endpoints required')
    idx=[i for i,w in enumerate(ma['windows']) if (w['training_source'] if role=='training' else w['scope']=='development_success')]
    values={k:selected_npz_rows(a/'aligned.npz',k,idx) for k in ('truth','prediction','history','current_state','valid')}
    for key in ('truth','history','current_state','valid'):
        np.testing.assert_array_equal(values[key],selected_npz_rows(b/'aligned.npz',key,idx))
    if not values['valid'].all():raise ValueError('Complete valid windows required')
    values['d0']=values.pop('prediction');values['d3']=selected_npz_rows(b/'aligned.npz','prediction',idx)
    np.testing.assert_array_equal(values['history'][:,-1],values['current_state'])
    rows=[{**ma['windows'][i],'analysis_role':role,'original_export_row':i} for i in idx]
    return values,rows,{'absolute_seal':json.loads((a/'seal.json').read_text()),'relative_seal':json.loads((b/'seal.json').read_text()),'selected_rows':idx,
        'excluded_sources':sorted({w['source_recording_id'] for i,w in enumerate(ma['windows']) if i not in idx}),
        'numeric_policy':'Only selected NPY rows deserialized; full export bytes hashed solely for integrity.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    plan={'fixed_weights':[0,.25,.5,.75,1],'features':['state_z','state_nn','history_motion'],
        'feature_floor_rad':.05,'unsupervised_threshold_quantile':.99,'supervised_threshold':'high-score ->D3 only; fit using training folds, nominal RMSE constraint1.1xD0',
        'outer_cv':'4 folds; each holds1 of4 development recordings and24/25 of97 training recordings; all rows of a recording together',
        'seed':20261110,'fit_weights':'equal cohorts; equal recordings inside cohort; equal windows inside recording',
        'calibration':'nominal training scores exclude entire own recording, including reference mean/std',
        'acceptance':'Old-source OOF RMSE and P99 <=1.1xD0, >0.5rad elements not increased; development OOF RMSE at least10% belowD0 and >=3/4 source gains; report local dynamics and identity confounding regardless.',
        'gate_condition':'No neural training if evidence remains collection identity or simple rules fail. Prior6 test sources never loaded numerically.',
        'fusion':'Output-space chunk-constant convex mixture, NOT intermediate input-coordinate interpolation.'}
    (out/'analysis_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    old,old_rows,old_id=load_pair(Path(cfg['training_absolute']),Path(cfg['training_relative']),'training')
    dev,dev_rows,dev_id=load_pair(Path(cfg['development_absolute']),Path(cfg['development_relative']),'development')
    rows=old_rows+dev_rows;data={k:np.concatenate([old[k],dev[k]]) for k in old}
    sources=np.array([r['source_recording_id'] for r in rows]);roles=np.array([r['analysis_role'] for r in rows])
    forbidden=set(dev_id['excluded_sources'])
    if set(sources)&forbidden or [int((roles==r).sum()) for r in ('training','development')]!=[485,20]:raise ValueError('Allowlist violated')
    if [len(set(sources[roles==r])) for r in ('training','development')]!=[97,4]:raise ValueError('Recording identities changed')
    h=data['history'];a=data['truth'].astype(float);d0=data['d0'].astype(float);d3=data['d3'].astype(float)
    loss0=np.mean((d0-a)**2,(1,2));loss3=np.mean((d3-a)**2,(1,2));binary,continuous=oracle_weights(a,d0,d3)
    selected={f'fixed_{w:g}':np.full(len(rows),w) for w in plan['fixed_weights']}
    selected.update(oracle_window_binary=binary,oracle_window_convex=continuous)
    record_oracle=np.zeros(len(rows))
    for source in set(sources):
        m=sources==source;record_oracle[m]=float(loss3[m].mean()<loss0[m].mean())
    selected['oracle_recording_binary']=record_oracle
    selected['cohort_identity_control_OFFLINE_ONLY']=(roles=='development').astype(float)
    fold=np.full(len(rows),-1)
    for role in ('training','development'):
        ids=sorted(set(sources[roles==role]),key=lambda s:object_sha256({'seed':plan['seed'],'recording':s}))
        for i,source in enumerate(ids):fold[sources==source]=i%4
    for name in plan['features']:
        selected['q99_'+name]=np.zeros(len(rows));selected['supervised_'+name]=np.zeros(len(rows))
    selected['OOF_selected_fixed']=np.zeros(len(rows));fits=[];score_rows=[]
    timings=[]
    for k in range(4):
        fit=fold!=k;val=~fit;nominal=fit&(roles=='training')
        if set(sources[fit])&set(sources[val]):raise ValueError('Recording leakage')
        reference=StateReference(h[nominal]);all_score=reference.score(h)
        # Reference-dependent nominal distances are calibrated outside own source.
        calibrated={name:values.copy() for name,values in all_score.items()}
        for source in set(sources[nominal]):
            mask=nominal&(sources==source);r=StateReference(h[nominal&(sources!=source)])
            for name,v in r.score(h[mask]).items():calibrated[name][mask]=v
        weights=source_balanced_weights(sources[fit],roles[fit])
        choices=[(float(np.dot(weights,np.mean((mix_actions(d0[fit],d3[fit],w)-a[fit])**2,(1,2)))),w) for w in plan['fixed_weights']]
        fixed=min(choices)[1];selected['OOF_selected_fixed'][val]=fixed
        for name in plan['features']:
            threshold=float(np.quantile(calibrated[name][nominal],.99))
            sup=fit_threshold(calibrated[name][fit],loss0[fit],loss3[fit],weights,roles[fit]=='training')
            selected['q99_'+name][val]=(all_score[name][val]>threshold).astype(float)
            selected['supervised_'+name][val]=(all_score[name][val]>sup).astype(float)
            fits.append({'fold':k,'feature':name,'nominal_sources':len(set(sources[nominal])),'fit_development_sources':sorted(map(str,set(sources[fit&(roles=='development')]))),'validation_development_sources':sorted(map(str,set(sources[val&(roles=='development')]))),'q99_threshold':threshold,'supervised_threshold':None if not np.isfinite(sup) else sup,'supervised_infinite_threshold':'D0_always' if sup==np.inf else ('D3_always' if sup==-np.inf else None),'selected_fixed_weight':fixed})
            for i in np.flatnonzero(val):score_rows.append({'row':int(i),'fold':k,'source':sources[i],'role':roles[i],'feature':name,'score':float(all_score[name][i]),'D3_advantage_mse':float(loss0[i]-loss3[i]),'D3_better':bool(binary[i])})
        # Small CPU-only causal feature + scalar-threshold latency, not GPU inference.
        sample=h[np.flatnonzero(val)[:1]]
        for _ in range(30):
            start=time.perf_counter();reference.score(sample);timings.append(time.perf_counter()-start)
    metrics=[];offsets=[];windows=[];scope_masks={'training':roles=='training','development':roles=='development'}
    scope_masks.update({s:sources==s for s in sorted(set(sources))})
    for name,weight in selected.items():
        pred=mix_actions(d0,d3,weight);err=pred-a;dec=decompose_offsets(a,pred)
        for scope,mask in scope_masks.items():
            ident={'method':name,'scope':scope,'windows':int(mask.sum()),'sources':len(set(sources[mask]))}
            for group,idx in EVAL_GROUPS.items():
                x=np.take(err[mask],idx,axis=-1)
                offsets.append({**ident,'group':group,'offset_mse':float(dec['offset_mse'][mask][:,idx].mean()),'shape_mse':float(dec['shape_mse'][mask][:,idx].mean())})
                for first,label in [(0,'all50'),(41,'last9')]:
                    y=x[:,first:];hit=(np.abs(y)>.5).any((1,2))
                    metrics.append({**ident,'group':group,'interval':label,**summary(y),'mse':float(np.mean(y*y)),
                        'increment_rmse':float(np.sqrt(np.mean(np.diff(y,axis=1)**2))),'over0p5_elements':int((np.abs(y)>.5).sum()),'over0p5_windows':int(hit.sum()),'over0p5_sources':len(set(sources[mask][hit]))})
        for i,r in enumerate(rows):windows.append({'method':name,'row':i,'source':sources[i],'role':roles[i],'fold':int(fold[i]),'lambda':float(weight[i]),'mse':float(np.mean(err[i]**2))})
    out_files=[('metrics',metrics),('offsets',offsets),('windows',windows),('fold_fits',fits),('out_of_fold_scores',score_rows)]
    for name,values in out_files:csv_rows(out/(name+'.csv'),values)
    counts=[]
    for role in ('training','development'):
        mask=roles==role;mixed=sum(0<binary[sources==s].sum()<int((sources==s).sum()) for s in set(sources[mask]))
        counts.append({'role':role,'windows':int(mask.sum()),'D3_better_windows':int(binary[mask].sum()),'recordings_with_both_endpoint_labels':int(mixed),
                       'window_oracle_mse':float(np.mean(np.minimum(loss0,loss3)[mask])),'recording_oracle_mse':float(np.mean(np.where(record_oracle>0,loss3,loss0)[mask]))})
    identity={'training_input':old_id,'development_input':dev_id,'forbidden_prior_test_sources':sorted(forbidden),'allowed_sources':sorted(set(sources)),
              'excluded_prior_test_numeric_rows_loaded':0,'script_sha256':file_sha256(__file__),'core_sha256':file_sha256(Path(__file__).parents[1]/'src/trc/analysis/flash/crs.py'),
              'gate_training':'No neural gate trained. Thresholds are fold-local scalar baseline fits only.','cpu_feature_score_median_ms':float(np.median(timings)*1000),'CPU_feature_score_p95_ms':float(np.quantile(timings,.95)*1000),'oracle_label_structure':counts}
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    np.savez(out/'allowed_inputs.npz',truth=a,d0=d0,d3=d3,history=h)
    (out/'allowed_windows.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'oracle_label_structure':counts,'summary':[r for r in metrics if r['scope'] in ('training','development') and r['group']=='full_action' and r['interval']=='all50']},indent=2))


if __name__=='__main__':main()
