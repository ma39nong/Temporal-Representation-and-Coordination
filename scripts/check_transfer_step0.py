#!/usr/bin/env python3
"""Strict zero-update gate; a representation switch may hurt, never hidden."""
import argparse,json
from pathlib import Path
import numpy as np
from report_p1_transfer import load
from trc.analysis.flash.metrics import summary
from trc.models.flash_coordinates import FlashCoordinates
from trc.data.action_schema import GROUPS
from trc.io import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--exports-root',required=True)
    p.add_argument('--execution',required=True);p.add_argument('--config',required=True);p.add_argument('--review',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    cfg=json.loads(Path(args.config).read_text());m,ref,_=load(args.baseline);entries={t:load(Path(args.exports_root)/(t+'_step0')) for t in ('B0','B2')}
    states={t:json.loads((Path(args.execution)/t/'initial_state.json').read_text()) for t in ('B0','B2')}
    if any(states['B0'][k]!=states['B2'][k] for k in ('params_sha256','opt_state_sha256','completed_step')) or states['B0']['completed_step']!=0:raise ValueError('Initial parameter/Adam states differ')
    for trial,(meta,d,_) in entries.items():
        if meta['checkpoint_parameter_sha256']!=cfg['initial_parameter_sha256'] or not meta['future_perturbation_passed']:raise ValueError('Initializer or no-future gate differs')
        if meta['windows']!=m['windows'] or meta['rgb_sequence_sha256']!=m['rgb_sequence_sha256']:raise ValueError('Window/RGB identity differs')
        for k in ('truth','oracle','history','current_state','basis','scale','q01','q99','c_gt'):np.testing.assert_array_equal(d[k],ref[k])
        np.testing.assert_array_equal(d['history'][:,-1],d['current_state'])
    difference=entries['B0'][1]['prediction']-ref['prediction']
    np.testing.assert_allclose(difference,0,atol=1e-5,rtol=0)
    z=entries['B2'][1];adapter=FlashCoordinates('state_relative',z['q01'],z['q99'],z['scale'],tuple(GROUPS))
    relative=adapter.coefficients(z['c_gt'],z['history']);np.testing.assert_allclose(adapter.coefficients(relative,z['history'],inverse=True),z['c_gt'],atol=1e-12,rtol=1e-12)
    actions=adapter.actions(z['truth'],z['history']);np.testing.assert_allclose(adapter.actions(actions,z['history'],inverse=True),z['truth'],atol=1e-12,rtol=1e-12)
    normalized=2*(z['truth']-z['q01'])/(z['q99']-z['q01']+1e-6)-1
    qnorm=2*(z['current_state']-z['q01'])/(z['q99']-z['q01']+1e-6)-1
    np.testing.assert_allclose(adapter.normalize_coordinates(actions),normalized-qnorm[:,None],atol=1e-12,rtol=1e-12)
    train=np.array([w['training_source'] for w in m['windows']]);review=json.loads(Path(args.review).read_text());failed={r['source_recording_id'] for r in review['records'] if r['review_entries']};bad=np.array([w['source_recording_id'] in failed for w in m['windows']]);rows=[]
    for name,mask in [('training_sources',train),('candidate_unrejected_quality_unverified',~train&~bad),('candidate_reviewed_failed',~train&bad)]:
        for trial,(_,d,_) in entries.items():rows.append({'scope':name,'trial':trial,**summary((d['prediction']-d['truth'])[mask])})
    result={'status':'STEP0_PAIR_VERIFIED','config_sha256':file_sha256(args.config),'code_sha256':file_sha256(__file__),
        'B0_vs_step8000_max_rad':float(np.max(np.abs(difference))),'B0_vs_step8000_rmse_rad':float(np.sqrt(np.mean(difference**2))),
        'initial_states':states,'same_full_parameters':True,'same_zero_adam':True,'same_input_identities':True,'future_perturbation_passed_both':True,
        'all505_coordinate_roundtrips_and_normalization_order':'passed','physical_metrics':rows,
        'interpretation':'B2 coordinate switch may change output substantially. No compensation or parameter change; report its own Step0 and original baseline.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
