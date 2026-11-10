#!/usr/bin/env python3
"""Reporting only on P2-0's allowlisted pack and recording-out-of-fold outputs."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from trc.analysis.flash.crs import mix_actions
from trc.analysis.flash.report import csv_rows
from trc.analysis.flash.metrics import EVAL_GROUPS


def read(path):return list(csv.DictReader(path.open()))

def correlation(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.input);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    rows=json.loads((root/'allowed_windows.json').read_text());sources=np.array([r['source_recording_id'] for r in rows]);roles=np.array([r['analysis_role'] for r in rows])
    data=np.load(root/'allowed_inputs.npz',allow_pickle=False);a,d0,d3=(data[k] for k in ('truth','d0','d3'))
    metrics=read(root/'metrics.csv');windows=read(root/'windows.csv');scores=read(root/'out_of_fold_scores.csv')
    def metric(method,scope,group='full_action',interval='all50'):
        return next(r for r in metrics if (r['method'],r['scope'],r['group'],r['interval'])==(method,scope,group,interval))
    methods=list(dict.fromkeys(r['method'] for r in windows));aggregate=[];source_table=[];temporal=[];decision=[]
    for method in methods:
        lam=np.array([float(r['lambda']) for r in windows if r['method']==method]);pred=mix_actions(d0,d3,lam);err=pred-a
        for role in ('training','development'):
            m=roles==role;r=metric(method,role);base=metric('fixed_0',role)
            gained=sum(float(metric(method,s)['rmse'])<float(metric('fixed_0',s)['rmse'])-1e-12 for s in set(sources[m]))
            aggregate.append({**r,'sources_improved':gained,'rmse_change_vs_D0':float(r['rmse'])-float(base['rmse']),'D3_selected_windows':int((lam[m]==1).sum()),'soft_mixed_windows':int(((lam[m]>0)&(lam[m]<1)).sum())})
            for group,idx in EVAL_GROUPS.items():
                delta=err[m][...,idx]
                for t in range(50):temporal.append({'method':method,'role':role,'group':group,'tick':t,'rmse':float(np.sqrt(np.mean(delta[:,t]**2))),'mae':float(np.abs(delta[:,t]).mean())})
        for s in sorted(set(sources)):
            role=roles[sources==s][0];r=metric(method,s);base=metric('fixed_0',s)
            source_table.append({**r,'role':role,'rmse_change_vs_D0':float(r['rmse'])-float(base['rmse']),'D3_selected_windows':int((lam[sources==s]==1).sum())})
        old,new=metric(method,'training'),metric(method,'development');b=metric('fixed_0','training')
        nwin=sum(float(metric(method,s)['rmse'])<float(metric('fixed_0',s)['rmse']) for s in set(sources[roles=='development']))
        decision.append({'method':method,'nominal_rmse_pass':float(old['rmse'])<=1.1*float(b['rmse']),
            'nominal_p99_pass':float(old['p99'])<=1.1*float(b['p99']),'nominal_extreme_pass':int(old['over0p5_elements'])<=int(b['over0p5_elements']),
            'development_rmse_pass':float(new['rmse'])<=.9*float(metric('fixed_0','development')['rmse']),
            'development_3of4_pass':nwin>=3})
    association=[]
    for name in ('state_z','state_nn','history_motion'):
        subset=sorted([r for r in scores if r['feature']==name],key=lambda r:int(r['row']))
        x=np.array([float(r['score']) for r in subset]);y=np.array([float(r['D3_advantage_mse']) for r in subset]);label=y>0
        auc=float(np.mean((x[label,None]>x[~label][None,:])+.5*(x[label,None]==x[~label][None,:])))
        dx=x.copy();dy=y.copy()
        for s in set(sources):
            mask=sources==s;dx[mask]-=x[mask].mean();dy[mask]-=y[mask].mean()
        association.append({'feature':name,'pooled_advantage_correlation':correlation(x,y),'pooled_endpoint_AUC':auc,
                            'recording_demeaned_correlation':correlation(dx,dy),'warning':'Exploratory descriptive association, correlated windows; no IID confidence interval or generalization claim.',
                            'cohort_ranges':{role:{'min':float(x[roles==role].min()),'median':float(np.median(x[roles==role])),'max':float(x[roles==role].max())} for role in ('training','development')}})
    for name,values in [('aggregate',aggregate),('source_comparison',source_table),('temporal',temporal)]:csv_rows(out/(name+'.csv'),values)
    passing=[r['method'] for r in decision if r['method'].startswith(('q99_', 'supervised_')) and all(v for k,v in r.items() if k.endswith('_pass'))]
    result={'acceptance':decision,'associations':association,'decision':'REQUIRES MECHANISM REVIEW' if passing else 'INSUFFICIENT EVIDENCE','passing_causal_rules':passing,
            'gate_implemented':False,'reason':'Joint criteria are necessary, not sufficient for mechanism evidence. Inspect source confounding before any gate training; no neural selector trained.'}
    (out/'feasibility.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for role,c in [('training','tab:blue'),('development','tab:orange')]:
        xx=[0,.25,.5,.75,1];yy=[float(metric(f'fixed_{w:g}',role)['rmse']) for w in xx]
        ax[0].plot(xx,yy,'o-',label=role,color=c)
    ax[0].set(xlabel='Fixed output fusion lambda',ylabel='Physical RMSE (rad)',title='Fixed fusion trade-off');ax[0].legend()
    chosen=['fixed_0','fixed_1','q99_state_nn','supervised_state_nn','oracle_window_convex']
    for method in chosen:
        ax[1].scatter(float(metric(method,'training')['rmse']),float(metric(method,'development')['rmse']),label=method)
    ax[1].axvline(1.1*float(metric('fixed_0','training')['rmse']),color='gray',linestyle='--');ax[1].set(xlabel='Training-source RMSE (rad)',ylabel='Development RMSE (rad)',title='Recording-out-of-fold rules');ax[1].legend(fontsize=7)
    fig.suptitle('REAL data: 97 training + 4 development recordings; prior 6 tests excluded');fig.tight_layout();fig.savefig(out/'tradeoff.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,name in zip(axes,('state_z','state_nn','history_motion')):
        for role,color in [('training','tab:blue'),('development','tab:orange')]:
            rr=[r for r in scores if r['feature']==name and r['role']==role]
            ax.scatter([float(r['score']) for r in rr],[float(r['D3_advantage_mse']) for r in rr],s=9,alpha=.6,label=role,color=color)
        ax.axhline(0,color='gray');ax.set(xlabel=name,ylabel='D0 MSE - D3 MSE (rad squared)')
    axes[0].legend();fig.suptitle('REAL data: causal scores vs endpoint advantage (OOF)');fig.tight_layout();fig.savefig(out/'state_association.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,role in zip(axes,('training','development')):
        for method in ('fixed_0','fixed_1','q99_state_nn','supervised_state_nn'):
            r=[r for r in temporal if r['method']==method and r['role']==role and r['group']=='left_arm']
            ax.plot([v['tick'] for v in r],[v['rmse'] for v in r],label=method)
        ax.axvspan(41,49,alpha=.1,color='gray');ax.set(title=role,xlabel='Prediction tick (30 Hz)',ylabel='Left arm RMSE (rad)');ax.legend(fontsize=7)
    fig.suptitle('REAL data: left-arm temporal error; shaded last 9 ticks');fig.tight_layout();fig.savefig(out/'left_arm_temporal.png',dpi=180);plt.close(fig)
    print(json.dumps({'decision':result['decision'],'associations':association,'acceptance':decision},indent=2))

if __name__=='__main__':main()
