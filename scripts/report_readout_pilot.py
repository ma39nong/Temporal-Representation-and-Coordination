"""Physical-space readout pilot reports, using existing error decomposition code."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from trc.analysis.flash.crs import selected_npz_rows
from trc.analysis.flash.metrics import evaluate,summary,coefficient_metrics,EVAL_GROUPS
from trc.analysis.flash.diagnosis import coefficient_contributions
from trc.analysis.flash.offset import decompose_offsets,grouped_offsets


def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--exports',required=True,type=Path)
    p.add_argument('--baseline',required=True,type=Path)
    p.add_argument('--continuous-indices',required=True,type=Path)
    p.add_argument('--cases',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--training-records',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    continuous=set(json.loads(args.continuous_indices.read_text())['indices'])
    cases=json.loads(args.cases.read_text())['entries']
    focus={cases[i-1]['high']['export_window'] for i in (7,8,9)}
    controls={c['control']['export_window'] for c in cases if c['control'] is not None}
    variants=[('B0',8000,args.baseline)]+[(r.name.split('_step')[0],int(r.name.split('_step')[1]),r)
              for r in sorted(args.exports.glob('B*_step*')) if (r/'seal.json').exists()]
    totals=[];groups=[];temporal=[];degrees=[];sources=[];window_rows=[];diagnostics={}
    reference=None;case_plot={};case_metrics=[];case_degrees=[]
    for trial,step,root in variants:
        meta=json.loads((root/'metadata.json').read_text())
        if trial=='B0':
            indices=[i for i,w in enumerate(meta['windows']) if w['training_source']]
            windows=[dict(meta['windows'][i],original_export_index=i,continuous_support=i in continuous) for i in indices]
        else:
            indices=list(range(len(meta['windows'])));windows=meta['windows']
        if len(indices)!=485:raise ValueError('Require frozen legacy 485 windows')
        data={k:selected_npz_rows(root/'aligned.npz',k,indices) for k in ('truth','oracle','prediction','c_gt','c_pred')}
        with np.load(root/'aligned.npz') as z:
            basis={k:z[k] for k in ('scale','basis','q01','q99')}
        keys=[(w['source_recording_id'],w['current_frame']) for w in windows]
        if reference is None:reference=(keys,data['truth'],data['oracle'])
        if keys!=reference[0]:raise ValueError('Paired source/window order differs')
        np.testing.assert_array_equal(data['truth'],reference[1])
        np.testing.assert_allclose(data['oracle'],reference[2],atol=2e-6,rtol=1e-5)
        scopes={'legacy485':list(range(485)),
                'continuous423':[i for i,w in enumerate(windows) if w['original_export_index'] in continuous],
                'focus3':[i for i,w in enumerate(windows) if w['original_export_index'] in focus],
                'control9':[i for i,w in enumerate(windows) if w['original_export_index'] in controls]}
        if [len(v) for v in scopes.values()]!=[485,423,3,9]:raise ValueError('Frozen diagnostic subsets changed')
        if trial=='B0' or step==250:
            for i in scopes['focus3']:
                case_plot[(trial,windows[i]['original_export_index'])]=dict(
                    truth=data['truth'][i],oracle=data['oracle'][i],prediction=data['prediction'][i],window=windows[i])
            for role in ('focus3','control9'):
                for i in scopes[role]:
                    a,o,pred,cg,cp=(data[k][i:i+1] for k in ('truth','oracle','prediction','c_gt','c_pred'))
                    tag=dict(trial=trial,step=step,role=role,original_export_index=windows[i]['original_export_index'],
                             source=windows[i]['source_recording_id'],frame=windows[i]['current_frame'])
                    offsets=decompose_offsets(a,pred)
                    for name in ('left_arm','left_hand','right_arm','right_hand'):
                        idx=EVAL_GROUPS[name];d=np.take(pred.astype(float)-a,idx,axis=-1)
                        case_metrics.append({**tag,'group':name,**summary(d),
                            'last9_rmse':summary(d[:,-9:])['rmse'],
                            'adjacent_change_rmse':summary(np.diff(d,axis=1))['rmse'],
                            'oracle_rmse':summary(np.take(o-a,idx,axis=-1))['rmse'],
                            'prediction_oracle_rmse':summary(np.take(pred-o,idx,axis=-1))['rmse'],
                            'offset_mse':float(offsets['offset_mse'][:,idx].mean()),
                            'shape_mse':float(offsets['shape_mse'][:,idx].mean())})
                    contribution=coefficient_contributions(a,o,pred,cg,cp,**basis)
                    case_degrees.extend({**tag,**r} for r in contribution['rows'] if r['group'] in ('left_arm','left_hand','right_arm','right_hand'))
        for scope,selected in scopes.items():
            a,o,pred,cg,cp=(data[k][selected] for k in ('truth','oracle','prediction','c_gt','c_pred'))
            errors=evaluate(a,o,pred);offset=grouped_offsets(decompose_offsets(a,pred))
            contribution=coefficient_contributions(a,o,pred,cg,cp,**basis)
            tag=dict(trial=trial,step=step,scope=scope)
            groups.extend({**tag,**r} for r in errors['groups'])
            temporal.extend({**tag,**r} for r in errors['temporal'])
            degrees.extend({**tag,**r} for r in contribution['rows'])
            delta=pred.astype(float)-a
            extreme=np.abs(delta)>.5
            affected=np.any(extreme,axis=(1,2))
            metric=next(r for r in errors['groups'] if r['method']=='total' and r['group']=='full_action')
            off=next(r for r in offset if r['group']=='full_action')
            totals.append({**tag,**metric,'windows':len(selected),'offset_mse':off['offset_mse'],'shape_mse':off['shape_mse'],
                'left_arm_last9_rmse':summary(delta[:,-9:,:7])['rmse'],
                'adjacent_change_rmse':summary(np.diff(delta,axis=1))['rmse'],
                'over_half_rad_elements':int(extreme.sum()),'over_half_rad_windows':int(affected.sum()),
                'over_half_rad_sources':len({windows[selected[i]]['source_recording_id'] for i in np.flatnonzero(affected)})})
            diagnostics[f'{trial}_step{step}_{scope}']=dict(offsets=offset,
                coefficients=coefficient_metrics(cg,cp,basis['scale'],basis['q01'],basis['q99']),
                interactions=contribution['pair_interactions'],decode_residual_max_rad=contribution['decode_residual_max_rad'],
                mse_decomposition=errors['mse_decomposition'])
            window_rows.extend({**windows[selected[r['window_index']]],**r,**tag} for r in errors['windows'] if r['method']=='total')
            if scope=='continuous423':
                ids=np.asarray([windows[i]['source_recording_id'] for i in selected])
                for source in sorted(set(ids)):
                    mask=ids==source
                    decomposition=decompose_offsets(a[mask],pred[mask])
                    for name,idx in EVAL_GROUPS.items():
                        d=np.take(delta[mask],idx,axis=-1)
                        sources.append({**tag,'source':source,'group':name,'windows':int(mask.sum()),**summary(d),
                            'offset_mse':float(decomposition['offset_mse'][:,idx].mean()),
                            'shape_mse':float(decomposition['shape_mse'][:,idx].mean()),
                            'last9_rmse':summary(d[:,-9:])['rmse'],
                            'oracle_rmse':summary(np.take(o[mask]-a[mask],idx,axis=-1))['rmse'],
                            'prediction_oracle_rmse':summary(np.take(pred[mask]-o[mask],idx,axis=-1))['rmse'],
                            'adjacent_change_rmse':summary(np.diff(d,axis=1))['rmse'],
                            'over_half_rad_elements':int((np.abs(d)>.5).sum())})
    for name,rows in [('summary',totals),('groups',groups),('temporal',temporal),('degrees',degrees),('sources',sources),('windows',window_rows)]:
        csv_write(args.output/(name+'.csv'),rows)
    csv_write(args.output/'case_metrics.csv',case_metrics)
    csv_write(args.output/'case_degrees.csv',case_degrees)
    (args.output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    index={(r['trial'],r['step'],r['source'],r['group']):r for r in sources}
    pairs=[]
    for r in sources:
        if r['trial']=='B0':continue
        for reference_trial in ('B0','B1','B2'):
            if reference_trial==r['trial']:continue
            reference_step=8000 if reference_trial=='B0' else r['step']
            peer=index.get((reference_trial,reference_step,r['source'],r['group']))
            if peer is None:continue
            pairs.append(dict(candidate=r['trial'],reference=reference_trial,step=r['step'],
                source=r['source'],group=r['group'],windows=r['windows'],
                rmse_improvement=peer['rmse']-r['rmse'],mse_improvement=peer['rmse']**2-r['rmse']**2,
                p99_improvement=peer['p99']-r['p99'],max_improvement=peer['max']-r['max'],
                last9_rmse_improvement=peer['last9_rmse']-r['last9_rmse'],
                adjacent_change_rmse_improvement=peer['adjacent_change_rmse']-r['adjacent_change_rmse']))
    csv_write(args.output/'source_pairs.csv',pairs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for trial in ('B1','B2','B3'):
        rows=sorted((r for r in totals if r['trial']==trial and r['scope']=='continuous423'),key=lambda r:r['step'])
        if not rows:continue
        for ax,key in zip(axes,('rmse','left_arm_last9_rmse','p99')):
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],marker='o',label=trial)
            baseline=next(r[key] for r in totals if r['trial']=='B0' and r['scope']=='continuous423')
            ax.axhline(baseline,color='gray',ls='--');ax.set(xlabel='Optimizer updates',ylabel=key+' (rad)')
    for ax in axes:ax.legend()
    fig.suptitle('REAL single-pick training-source diagnostics, common 423 continuous windows')
    fig.tight_layout();fig.savefig(args.output/'paired_learning.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for trial,step in [('B0',8000),('B1',250),('B2',250),('B3',250)]:
        for ax,group in zip(axes,('full_action','left_arm')):
            rows=sorted((r for r in temporal if r['trial']==trial and r['step']==step and r['scope']=='continuous423' and r['method']=='total' and r['group']==group),key=lambda r:r['tick'])
            if rows:ax.plot([r['tick'] for r in rows],[r['rmse'] for r in rows],label=trial)
            ax.set(xlabel='Prediction tick (30Hz)',ylabel='RMSE (rad)',title=group)
    for ax in axes:ax.legend()
    fig.suptitle('REAL matched continuous training-source windows')
    fig.tight_layout();fig.savefig(args.output/'temporal_profile.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    ticks=np.arange(50)/30
    for ax,i in zip(axes,sorted(focus)):
        row=case_plot[('B0',i)]
        ax.plot(ticks,row['truth'][:,6],color='black',label='GT')
        ax.plot(ticks,row['oracle'][:,6],color='gray',ls='--',label='Oracle K6')
        for trial in ('B0','B1','B2','B3'):
            if (trial,i) in case_plot:ax.plot(ticks,case_plot[(trial,i)]['prediction'][:,6],label=trial)
        ax.axvspan(41/30,49/30,color='gray',alpha=.12)
        w=row['window'];ax.set(title=f"{w['source_recording_id'][:8]}, frame {w['current_frame']}",xlabel='Future seconds',ylabel='Left FR3 joint7 (rad)')
    axes[-1].legend(fontsize=8)
    fig.suptitle('REAL predeclared three training-source cases; shaded last nine ticks')
    fig.tight_layout();fig.savefig(args.output/'focus_left_joint7.png',dpi=160);plt.close(fig)
    if args.training_records:
        records=json.loads(args.training_records.read_text());stats={}
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        base=np.asarray([r['loss'] for r in records['B1']])
        for trial,rows in records.items():
            if [r['step'] for r in rows]!=list(range(1,251)):
                raise ValueError('Training records must contain exactly accepted updates 1..250')
            if any(r['vision_gradient_norm'] or r['language_gradient_norm'] for r in rows):
                raise ValueError('Unexpected frozen VLM gradient')
            steady=[r for r in rows if r['step'] not in (1,6)]
            stats[trial]=dict(updates=250,effective_samples=rows[-1]['effective_samples'],
                median_update_seconds=float(np.median([r['seconds'] for r in steady])),
                median_samples_per_second_with_data=float(np.median([256/(r['seconds']+r['data_wait_seconds']) for r in steady])),
                resume_peak_device_bytes=max(max(r['peak_device_bytes']) for r in rows[5:]),
                initialization_and_smoke_peak_device_bytes=max(max(r['peak_device_bytes']) for r in rows[:5]),
                first20_mean_loss=float(np.mean([r['loss'] for r in rows[:20]])),
                last20_mean_loss=float(np.mean([r['loss'] for r in rows[-20:]])),
                throughput_scope='steady compute + data wait; excludes initialization, compile, checkpoint IO')
            values=np.asarray([r['loss'] for r in rows]);kernel=np.ones(20)/20
            axes[0].plot(np.arange(20,251),np.convolve(values,kernel,mode='valid'),label=trial)
            if trial!='B1':axes[1].plot(np.arange(20,251),np.convolve(values-base,kernel,mode='valid'),label=trial+' minus B1')
        axes[0].set(xlabel='Optimizer update',ylabel='FM + consistency loss, 20-step mean')
        axes[1].axhline(0,color='gray',ls='--');axes[1].set(xlabel='Optimizer update',ylabel='Paired loss difference, 20-step mean')
        for ax in axes:ax.legend()
        fig.suptitle('REAL same-window/same-key training; absolute loss curves nearly overlap')
        fig.tight_layout();fig.savefig(args.output/'training_loss.png',dpi=160);plt.close(fig)
        (args.output/'training_summary.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps([r for r in totals if r['scope']=='continuous423'],indent=2))


if __name__=='__main__':main()
