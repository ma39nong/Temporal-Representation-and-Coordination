"""Frozen clean-case coefficient residuals; NFE=1 flow-time-zero interpretation only."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from trc.analysis.flash.crs import selected_npz_rows
from trc.analysis.flash.diagnosis import coefficient_contributions
from trc.data.action_schema import GROUPS


def main():
    p=argparse.ArgumentParser();p.add_argument('--export',required=True);p.add_argument('--cases',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.export);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cases=json.loads(Path(a.cases).read_text())['entries'];high=[cases[n-1]['high'] for n in (7,8,9)]
    controls={e['control']['export_window']:e['control'] for e in cases if e['control'] is not None}
    samples=[('focus',r) for r in high]+[('control',r) for r in controls.values()]
    if len(controls)!=9 or any(r['crossing_cuts'] for _,r in samples):raise ValueError('Expected three clean cases plus nine clean controls')
    indices=[r['export_window'] for _,r in samples]
    data={k:selected_npz_rows(root/'aligned.npz',k,indices) for k in ('truth','oracle','prediction','c_gt','c_pred')}
    rows=[]
    with np.load(root/'aligned.npz') as z:
        for i,(role,record) in enumerate(samples):
            report=coefficient_contributions(*[data[k][i:i+1] for k in ('truth','oracle','prediction','c_gt','c_pred')],z['scale'],z['basis'],z['q01'],z['q99'])
            for r in report['rows']:
                if r['group'] not in GROUPS:continue
                idx=GROUPS[r['group']];delta=(data['c_pred'][i]-data['c_gt'][i]).astype(float)[r['degree'],idx]
                re=(data['oracle'][i]-data['truth'][i]).astype(float)[:,idx]
                pe=(data['prediction'][i]-data['oracle'][i]).astype(float)[:,idx]
                rows.append({'role':role,'window':record['export_window'],'source':record['source_recording_id'],
                             'frame':record['frame'],**r,'scaled_coefficient_mse':float(np.mean(delta**2)),
                             'scaled_coefficient_bias':float(np.mean(delta)),
                             'target_scaled_coefficient_mean':float(np.mean(data['c_gt'][i,r['degree'],idx])),
                             'target_scaled_coefficient_std':float(np.std(data['c_gt'][i,r['degree'],idx])),
                             'oracle_rmse':float(np.sqrt(np.mean(re**2))), 'prediction_oracle_rmse':float(np.sqrt(np.mean(pe**2)))})
    with (out/'order_residuals.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    overview=[]
    for role in ('focus','control'):
        for group in GROUPS:
            rr=[r for r in rows if r['role']==role and r['group']==group];nr=len(rr)//7
            energy=np.array([[r['self_mse'] for r in rr if r['window']==win] for win in dict.fromkeys(r['window'] for r in rr)])
            allocation=np.array([[r['signed_allocated_prediction_mse'] for r in rr if r['window']==win] for win in dict.fromkeys(r['window'] for r in rr)])
            winners=np.argmax(energy,axis=1)
            overview.append({'role':role,'group':group,'windows':nr,'largest_self_energy_degree_counts':np.bincount(winners,minlength=7).tolist(),
                'mean_self_energy_by_degree':energy.mean(0).tolist(),
                'mean_signed_allocation_by_degree':allocation.mean(0).tolist(),
                'caution':'Orders are polynomial basis indices, not frequency bands. Scaled coefficients cannot be mixed as physical importance.'})
    result={'data_kind':'real','scope':'selected training-source cases only','windows':indices,'overview':overview,
       'velocity_interpretation':'NFE=1 only: c_pred-c_gt equals velocity(tau=0,source)-(target-source); not arbitrary-time training FM error.',
       'case_sha256':hashlib.sha256(Path(a.cases).read_bytes()).hexdigest(),
       'export_metadata_sha256':hashlib.sha256((root/'metadata.json').read_bytes()).hexdigest()}
    (out/'order_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(overview,indent=2))


if __name__=='__main__':main()
