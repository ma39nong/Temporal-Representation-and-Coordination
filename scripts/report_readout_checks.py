"""Report real CPU functional smoke, explicitly not a paired pilot evaluation."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from trc.analysis.flash.metrics import evaluate,coefficient_metrics,summary
from trc.analysis.flash.offset import decompose_offsets,grouped_offsets


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--basis-export',required=True);a=p.parse_args();out=Path(a.output)
    accepted=json.loads((out/'full_smoke_acceptance.json').read_text());native=np.load(out/'baseline_real_batch.npz')
    first=np.load(out/'B1_expert_smoke_predictions.npz');truth,oracle=first['truth'],first['oracle']
    with np.load(Path(a.basis_export)/'aligned.npz') as basis:
        scale,lo,hi=(basis[k] for k in ('scale','q01','q99'))
    rows=[];allmetrics={}
    for name in ('B0','B1','B2','B3'):
        z=first if name=='B1' else (np.load(out/(name+'_expert_smoke_predictions.npz')) if name!='B0' else None)
        pred=native['prediction'] if name=='B0' else z['prediction']
        cp=native['source']+native['velocity'] if name=='B0' else z['c_pred']
        cg=native['target_coefficients'] if name=='B0' else z['c_gt']
        if z is not None:np.testing.assert_array_equal(truth,z['truth']);np.testing.assert_array_equal(oracle,z['oracle'])
        metric=evaluate(truth,oracle,pred)
        coeff=coefficient_metrics(cg,cp,scale,lo,hi)
        offset=grouped_offsets(decompose_offsets(truth,pred))
        allmetrics[name]={'errors':metric,'coefficients':coeff,'offsets':offset,
            'left_arm_last9':summary((pred-truth)[:,-9:,:7]),
            'adjacent_increment_rad':summary(np.diff(pred.astype(float)-truth,axis=1)),
            'over_half_rad_elements':int(np.sum(np.abs(pred-truth)>.5)),
            'windows':1,'recording_id':'be79493d531944aba5bc9e85eeaa7af2','frame':848,
            'scope':'CPU FP32 cached-prefix functional smoke on one reused training window; NOT 250-step or generalization evidence'}
        for r in metric['groups']:rows.append({'trial':name,**r})
    with (out/'cpu_smoke_groups.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'cpu_smoke_metrics.json').write_text(json.dumps(allmetrics,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for name,trial in accepted['trials'].items():
        values=trial['full_expert_cpu_smoke']['steps'];ax.plot([v['step'] for v in values],[v['loss'] for v in values],label=name)
    ax.set(xlabel='CPU functional update (Batch1)',ylabel='FM + consistency loss',title='REAL single-window functional smoke; flow/noise key changes each step')
    ax.legend();fig.tight_layout();fig.savefig(out/'cpu_smoke_loss.png',dpi=160);plt.close(fig)
    orders=json.loads((out/'order_summary.json').read_text())['overview']
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,group in zip(axes.flat,('left_arm','left_hand','right_arm','right_hand')):
        for role in ('focus','control'):
            row=next(v for v in orders if v['role']==role and v['group']==group)
            ax.plot(range(7),np.sqrt(row['mean_self_energy_by_degree']),marker='o',label=f"{role} n={row['windows']}")
        ax.set(xlabel='Legendre degree (not frequency)',ylabel='Standalone physical component RMSE (rad)',title=group,yscale='log');ax.legend()
    fig.suptitle('REAL selected training-source windows; standalone energies are not additive')
    fig.tight_layout();fig.savefig(out/'order_residual_structure.png',dpi=160);plt.close(fig)
    print(json.dumps({k:next(r for r in v['errors']['groups'] if r['method']=='total' and r['group']=='full_action') for k,v in allmetrics.items()},indent=2))


if __name__=='__main__':main()
