"""Reproducible JSON/CSV/figures for one authenticated FLASH export."""
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from trc.io import file_sha256, write_report
from trc.data.action_schema import ORDER
from .metrics import evaluate, coefficient_metrics, joint_time_diagnostics, EVAL_GROUPS


def csv_rows(path, rows):
    rows=list(rows)
    if not rows: return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)


def figures(output, arrays, metrics, metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Full server paths remain in the report; they overflow scientific figures.
    display_id=(f"Checkpoint Step{metadata['checkpoint_step']}" if 'checkpoint_step' in metadata
                else Path(metadata['checkpoint_id']).name[-50:])
    data_label=f"REAL | {display_id} | {metadata['evaluation_scope']}"
    kinds=['representation','prediction','total'];colors=['#3d9970','#ff851b','#0074d9']
    lookup={(r['method'],r['group']):r for r in metrics['groups']}
    fig,ax=plt.subplots(figsize=(7.6,4.3),layout='constrained')
    vals=[lookup[(k,'full_action')]['rmse'] for k in kinds]
    ax.bar(['GT vs Oracle','Oracle vs Expert','GT vs Expert'],vals,color=colors)
    ax.set_ylabel('Pooled joint RMSE (rad)');ax.set_title('Figure 1. Representation vs prediction error\n'+data_label,fontsize=10)
    ax.bar_label(ax.containers[0],fmt='%.5f',padding=3);ax.set_ylim(0,max(vals)*1.22)
    fig.savefig(output/'figure1_representation_prediction.png',dpi=180);plt.close(fig)
    groups=['left_arm','left_hand','right_arm','right_hand']
    # Per-window total MAE distribution; repr/pred RMSE remains in CSV.
    a,o,p=(arrays[k] for k in ('truth','oracle','prediction'))
    fig,ax=plt.subplots(figsize=(7.6,4.3),layout='constrained')
    values=[np.mean(np.abs(p-a)[...,EVAL_GROUPS[g]],axis=(1,2)) for g in groups]
    ax.boxplot(values,tick_labels=[g.replace('_',' ').title() for g in groups],showfliers=False)
    ax.set_ylabel('Per-window total MAE (rad)');ax.set_title('Figure 2. Arm–hand error distribution\n'+data_label,fontsize=10)
    fig.savefig(output/'figure2_group_error.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained',sharey=True)
    for ax,group in zip(axes,['all_arms','all_hands','full_action']):
        for method,color in zip(kinds,colors):
            rows=[r for r in metrics['temporal'] if r['method']==method and r['group']==group]
            ax.plot([r['tick'] for r in rows],[r['rmse'] for r in rows],label=method,color=color)
        ax.set_title(group.replace('_',' ').title());ax.set_xlabel('Prediction tick (0–49)')
    axes[0].set_ylabel('Pooled RMSE (rad)');axes[-1].legend(fontsize=8)
    fig.suptitle('Figure 3. Temporal error profile\n'+data_label,fontsize=10)
    fig.savefig(output/'figure3_temporal_error.png',dpi=180);plt.close(fig)
    cases=[]
    for group in groups:
        idx=EVAL_GROUPS[group]
        errors=np.abs(p-a)[...,idx]
        i,t,j=np.unravel_index(np.argmax(errors),errors.shape)
        cases.append((int(i),idx[int(j)],group))
    fig,axes=plt.subplots(2,2,figsize=(11,6.7),layout='constrained')
    for ax,(i,j,group) in zip(axes.flat,cases):
        w=metadata['windows'][i]
        for val,label,color in [(a,'GT','#222222'),(o,'Oracle','#3d9970'),(p,'Prediction','#0074d9')]:
            ax.plot(np.arange(50)/30,val[i,:,j],label=label,color=color)
        ax.set_title(f"{group}: {ORDER[j]} | window {i}\nepisode {w['episode_index']}, frame {w.get('current_frame',w['current_row'])}",fontsize=8)
        ax.set_xlabel('Seconds from current frame');ax.set_ylabel('Absolute command (rad)')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Figure 4. High-error real trajectory cases\n'+data_label,fontsize=10)
    fig.savefig(output/'figure4_high_error_cases.png',dpi=180);plt.close(fig)


def build_report(export_dir, output, *, make_figures=True):
    export_dir,output=Path(export_dir).resolve(),Path(output).resolve()
    if output.is_relative_to(export_dir): raise ValueError('Do not overwrite export inputs')
    if output.exists(): raise FileExistsError('Use a new result directory')
    metadata=json.loads((export_dir/'metadata.json').read_text())
    if metadata.get('data_kind')!='real' or metadata.get('action_semantics')!='absolute' or metadata.get('action_unit')!='rad':
        raise ValueError('Scientific report requires authenticated real Absolute rad exports')
    if metadata.get('horizon_ticks')!=[0,49] or metadata.get('K')!=6: raise ValueError('FLASH alignment mismatch')
    with np.load(export_dir/'aligned.npz',allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    if len(metadata['windows'])!=len(arrays['truth']): raise ValueError('Window provenance mismatch')
    if make_figures and not arrays['valid'].all():
        raise ValueError('Current figure renderer requires all-valid windows; metrics support masks')
    # Authenticate export integrity seal, written only after original-source validation.
    seal=json.loads((export_dir/'seal.json').read_text())
    if seal['aligned_sha256']!=file_sha256(export_dir/'aligned.npz') or seal['metadata_sha256']!=file_sha256(export_dir/'metadata.json'):
        raise ValueError('Export integrity seal mismatch')
    metrics=evaluate(arrays['truth'],arrays['oracle'],arrays['prediction'],arrays['valid'])
    coeff=coefficient_metrics(arrays['c_gt'],arrays['c_pred'],arrays['scale'],arrays['q01'],arrays['q99'])
    time=joint_time_diagnostics(arrays['truth'],arrays['prediction'],arrays['valid'])
    for row in metrics['windows']:
        w=metadata['windows'][row['window_index']]
        row.update(sample_id=w['sample_id'],episode_index=w['episode_index'],current_frame=w.get('current_frame',w['current_row']),
                   source_recording_id=w['source_recording_id'])
    output.mkdir(parents=True)
    for name,rows in [('group_metrics',metrics['groups']),('temporal_metrics',metrics['temporal']),
                      ('per_joint',metrics['joints']),('per_window',metrics['windows']),
                      ('coefficient_by_degree',coeff),('joint_increment_profile',time)]:
        csv_rows(output/(name+'.csv'),rows)
    group_cases=[]
    for group,indices in EVAL_GROUPS.items():
        err=np.where(arrays['valid'][...,indices],np.abs(arrays['prediction'].astype(np.float64)-arrays['truth'].astype(np.float64))[...,indices],-np.inf)
        i,t,j=np.unravel_index(np.argmax(err),err.shape)
        w=metadata['windows'][int(i)]
        group_cases.append({'group':group,'window_index':int(i),'tick':int(t),'joint':ORDER[indices[int(j)]],
                            'absolute_error_rad':float(err[i,t,j]),'sample_id':w['sample_id'],
                            'episode_index':w['episode_index'],'current_frame':w.get('current_frame',w['current_row']),
                            'source_recording_id':w['source_recording_id']})
    csv_rows(output/'worst_group_cases.csv',group_cases)
    worst=sorted([r for r in metrics['windows'] if r['method']=='total'],key=lambda r:r['max'],reverse=True)[:20]
    summary={'status':('completed_real_saved_prediction_attribution' if metadata.get('inference_mode','').startswith('authenticated_saved') else 'completed_real_fresh_inference_attribution'),'metadata':metadata,
             'evaluation_unit':'rad','aggregation':'pooled_valid_joint_values; overlapping windows are not independent trials',
             'errors':metrics['groups'],'mse_decomposition':metrics['mse_decomposition'],'worst_20_windows':worst,
             'export_seal':seal,
             'analysis_code_sha256':{name:file_sha256(Path(__file__).with_name(name)) for name in ('report.py','metrics.py')},'research_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'research_tracked_changes':subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(),
             'limitations':[('Saved historical predictions; no fresh checkpoint restore' if metadata.get('inference_mode','').startswith('authenticated_saved') else 'Fresh original checkpoint inference; no optimizer updates'),
                            'No comparison to current FullFT as a training-step ablation',
                            'Coefficient errors separated by degree/scale; recovered C_pred is labelled',
                            'Joint increment errors are not physical coordination failure metrics',
                            'No verified FK/phase labels or executed robot outcomes',
                            'Inspect per-window training_source/scope; historical holdout does not imply current FullFT holdout']}
    write_report(output/'report.json',summary,protected=[export_dir])
    if make_figures:
        figures(output,arrays,metrics,metadata)
    return summary
