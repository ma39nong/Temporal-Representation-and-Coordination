#!/usr/bin/env python3
"""P1-7 preregistration on human-reviewed recordings, before model inference.

Run only in the explicitly pinned source dependency runtime. Does not import
model weights. Dataset paths and the human outcome evidence come from config.
"""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import av
from openpi.training.flash_native import NativeDataset, read_numeric, CAMERAS
from trc.data.lerobot_adapter import LeRobotAdapter
from trc.data.action_schema import canonicalize, verify_features
from trc.data.validation import audit_episode
from trc.io import file_sha256, object_sha256


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def video_identity(path, expected_frames):
    with av.open(str(path)) as c:
        stream=c.streams.video[0]
        times=np.array([float(f.time) for f in c.decode(stream)])
        if len(times)!=expected_frames or not np.allclose(times,np.arange(expected_frames)/30,rtol=0,atol=1e-5):
            raise ValueError('Video frame count or PTS mismatch: '+str(path))
        return {'path':str(path),'sha256':file_sha256(path),'frames':len(times),
                'fps':float(stream.average_rate),'max_pts_error_s':float(np.max(np.abs(times-np.arange(expected_frames)/30)))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(cfg['output'])
    from openpi.training import flash_native
    if not Path(flash_native.__file__).resolve().is_relative_to(Path(cfg['source_snapshot']).resolve()):
        raise ValueError('Explicit pinned source runtime required')
    if (out/'frozen_protocol.json').exists():raise FileExistsError('Protocol already frozen')
    review=json.loads(Path(cfg['review']).read_text())
    if review['task_outcome']!='success' or review['applies_to_all_10_selected_recordings'] is not True:
        raise ValueError('Explicit human success review required')
    train=json.loads(Path(cfg['training_manifest']).read_text())
    identities=json.loads(Path(cfg['training_identity']).read_text())
    if len(identities['records'])!=480 or identities['manifest_sha256']!=train['sha256']:raise ValueError('Full training registry required')
    expected_norm='a5d7bd2dd422148736c69f1bf52a721d5770ba810f3bace9a2dd413395544d52'
    if file_sha256(cfg['norm_stats'])!=expected_norm:raise ValueError('Matching original statistics required')
    baseline=json.loads((Path(cfg['baseline'])/'metadata.json').read_text())
    forbidden=set(train['train_sources'])|{w['source_recording_id'] for w in baseline['windows']}
    forbidden_numeric={r['numeric_sha256'] for r in identities['records']}
    forbidden_parquet={r['parquet_sha256'] for r in identities['records']}
    audits=[];records=[];remote_records=[];windows=[];numeric={};roles={};annotations={}
    for spec in cfg['datasets']:
        root=Path(spec['local_root']);adapter=LeRobotAdapter(root);verify_features(adapter.info['features'],'converter58')
        meta={e['episode_index']:e for e in adapter.episodes}
        candidates=sorted(spec['episodes'],key=lambda ep:object_sha256({'seed':cfg['split_seed'],'source':meta[ep]['source_recording_id']}))
        development=set(candidates[:spec['development_recordings']])
        tasks={str(t['task_index']):t['task'] for t in map(json.loads,(root/'meta/tasks.jsonl').read_text().splitlines())}
        for ep in sorted(spec['episodes']):
            e=meta[ep];source=e['source_recording_id'];ident=object_sha256(source)[:16]
            if source in forbidden:raise ValueError('Training/old candidate recording overlap')
            role='development_success' if ep in development else 'locked_test_success';roles[source]=role
            d=adapter.read_episode(ep);audit=audit_episode(d,fps=30,layout='converter58')
            if adapter.info['fps']!=30 or audit['errors'] or len(d['action'])!=e['length']:raise ValueError('Numeric audit failed')
            if not np.all(d['episode_index']==ep):raise ValueError('Episode index mismatch')
            a=canonicalize(d['action'],'converter58');s=canonicalize(d['observation.state'],'converter58',state=True)
            numeric_sha=hashlib.sha256(a.tobytes()+s.tobytes()).hexdigest();parquet=adapter.episode_path(ep)
            if numeric_sha in forbidden_numeric or file_sha256(parquet) in forbidden_parquet:raise ValueError('Duplicate training content')
            native={'id':ident,'root':str(root),'episode':ep,'rows':len(a),'source_recording_id':source,'path':str(parquet),'info':adapter.info,'tasks':tasks}
            na,ns,nt=read_numeric(native);np.testing.assert_array_equal(na,a);np.testing.assert_array_equal(ns,s)
            records.append(native);numeric[ident]=(a,s,nt)
            remote={**native,'root':spec['remote_root'],'path':str(Path(spec['remote_root'])/parquet.relative_to(root))};remote_records.append(remote)
            ann_path=root/f'annotations/subtasks/coarse/episode_{ep:06d}.json';ann=json.loads(ann_path.read_text())
            if ann['source']['source_recording_id']!=source or not all(seg['reviewed'] for seg in ann['segments']):raise ValueError('Phase evidence differs')
            for rel,digest in ann['source']['sha256'].items():
                if file_sha256(root/rel)!=digest:raise ValueError('Annotated data identity changed')
            annotations[source]={'path':str(ann_path),'sha256':file_sha256(ann_path)}
            gap=abs((e['source_end_ns']-e['source_start_ns'])-(len(a)-1)*1e9/30)
            if gap>1000:raise ValueError('Original interval endpoint mapping mismatch')
            audits.append({'dataset':spec['id'],'episode':ep,'source_recording_id':source,'role':role,'task_outcome':'human_reviewed_success',
                'source_start_ns':e['source_start_ns'],'source_end_ns':e['source_end_ns'],'source_interval_endpoint_error_ns':gap,
                'numeric_sha256':numeric_sha,'parquet_sha256':file_sha256(parquet),'audit':audit,'initial_state':s[0].tolist(),
                'frames':len(a),'legal_windows':len(a)-58,'annotation':annotations[source]})
            for row in np.linspace(7,len(a)-52,cfg['windows_per_recording'],dtype=int):
                windows.append({'id':ident,'start':int(row),'evaluation_role':role})
    if len(audits)!=10 or len({r['source_recording_id'] for r in audits})!=10 or len(windows)!=50:raise ValueError('Expected10 disjoint recordings /50 predeclared windows')
    if len({r['numeric_sha256'] for r in audits})!=10:raise ValueError('Duplicate new numeric recordings')
    # Exact-copy and recording-time checks complement immutable source IDs.
    train_times=[r['source_end_ns'] for r in identities['records'] if r.get('source_end_ns') is not None]
    if train_times and min(r['source_start_ns'] for r in audits)<=max(train_times):raise ValueError('New recording chronology requires additional lineage review')
    video_jobs=[]
    for r in records:
        for key in CAMERAS.values():
            rel=r['info']['video_path'].format(episode_chunk=r['episode']//r['info']['chunks_size'],episode_index=r['episode'],video_key=key)
            video_jobs.append((Path(r['root'])/rel,r['rows']))
    with ThreadPoolExecutor(max_workers=3) as pool:
        videos=list(pool.map(lambda pair:video_identity(*pair),video_jobs))
    native=NativeDataset.__new__(NativeDataset);native.records={r['id']:r for r in records};native.numeric=numeric
    prior=cfg['old_parity'];old_m=baseline;old_i=prior['old_index'];ow=old_m['windows'][old_i]
    old_root=Path(prior['local_root']);old_info=json.loads((old_root/'meta/info.json').read_text())
    old_ep=prior['episode'];old_path=old_root/old_info['data_path'].format(episode_chunk=old_ep//old_info['chunks_size'],episode_index=old_ep)
    old_rows=next(e['length'] for e in map(json.loads,(old_root/'meta/episodes.jsonl').read_text().splitlines()) if e['episode_index']==old_ep)
    old_record={'id':ow['sample_id'],'root':str(old_root),'episode':old_ep,'rows':old_rows,
                'source_recording_id':ow['source_recording_id'],'path':str(old_path),'info':old_info,'tasks':{str(t['task_index']):t['task'] for t in map(json.loads,(old_root/'meta/tasks.jsonl').read_text().splitlines())}}
    native.records[old_record['id']]=old_record
    # Server-side parity source root is recorded in the original export.
    old_remote={**old_record,'root':str(Path(ow['source_path']).parents[2]),'path':ow['source_path'],
                'episode':ow['episode_index']};remote_records.append(old_remote)
    parity_window={'id':old_record['id'],'start':ow['current_row']}
    parity={}
    for name,folder in [('D0',Path(cfg['baseline'])),('D3',Path(cfg['old_exports'])/'B2_step0'),('B0_FT250',Path(cfg['old_exports'])/'B0_step250'),('B2_FT250',Path(cfg['old_exports'])/'B2_step250')]:
        with np.load(folder/'aligned.npz',allow_pickle=False) as z:
            parity[name]=z['prediction'][old_i];parity['truth']=z['truth'][old_i];parity['history']=z['history'][old_i]
    ph,pt,_,_=native.window(parity_window);np.testing.assert_array_equal(ph,parity['history']);np.testing.assert_array_equal(pt[2:52],parity['truth'])
    rgb={}
    for w in [*windows,parity_window]:
        for role in CAMERAS:rgb[f"{w['id']}:{w['start']}:{role}"]=hashlib.sha256(native.image(w,role).tobytes()).hexdigest()
    values=[native.window(w) for w in windows]
    np.savez(out/'reference_arrays.npz',truth=np.stack([t[1][2:52] for t in values]),history=np.stack([t[0] for t in values]))
    np.savez(out/'parity_reference.npz',**parity)
    inputs={'records':remote_records,'windows':windows,'rgb_sha256':rgb,'parity_window':parity_window,'historical_plan_sha256':'not_old505_new_preregistered50'}
    json_write(out/'paired_inputs.json',inputs)
    json_write(out/'source_audit.json',{'sources':audits,'videos':videos,'source_ids_disjoint_training480_and_old505':True,
        'canonical_numeric_and_parquet_duplicate_check':'none_against_all480','training_latest_known_source_end_ns':max(train_times) if train_times else None,
        'missing_local_episode0':'Intentionally not selected; already imported/excluded in upstream selection. Metadata retained unchanged.',
        'raw_timing_limit':'Only original recording endpoints preserved; per-packet capture/receive timestamps were dropped upstream.'})
    models={}
    for name,folder,mode in [('D0',Path(cfg['baseline']),'absolute'),('D3',Path(cfg['baseline']),'state_relative'),('B0_FT250',Path(cfg['old_exports'])/'B0_step250','absolute'),('B2_FT250',Path(cfg['old_exports'])/'B2_step250','state_relative')]:
        meta=json.loads((folder/'metadata.json').read_text());models[name]={'parameter_sha256':meta['checkpoint_parameter_sha256'],'coordinate_mode':mode,'checkpoint':meta['checkpoint_id'],'normalization_sha256':meta['normalization_sha256']}
    protocol={'schema_version':1,'client_date':'2026-11-10','status':'LOCKED_BEFORE_NEW_MODEL_PREDICTIONS','models':models,
        'success_review_sha256':file_sha256(cfg['review']),'source_audit_sha256':file_sha256(out/'source_audit.json'),
        'paired_inputs_sha256':file_sha256(out/'paired_inputs.json'),'reference_arrays_sha256':file_sha256(out/'reference_arrays.npz'),
        'parity_reference_sha256':file_sha256(out/'parity_reference.npz'),'split_seed':cfg['split_seed'],'split':'within A/B by hash(seed,original recording);3A+1B development,4A+2B test',
        'windows_per_recording':5,'window_selection':'uniform linspace7..length-52; full[-7,51] support; no prediction-dependent selection',
        'evaluation':{'dtype':'float32','matmul_precision':'highest','batch_size':1,'NFE':1,'padding':False,'PRNG':'original deterministic inference ignores rng; no noise; model seed42 lineage'},
        'metrics':['MAE','RMSE','offset_MSE','shape_MSE','four_joint_groups','50_tick_profile','left_arm_last9','increment_RMSE','P95','P99','Max','over0.5rad events','per_source_vs_D0'],
        'decision_rule':{'primary':'D3 compared with D0 on locked_test_success only; descriptive confirmation, no significance claim with6 recordings',
            'supported':'D3 pooled test RMSE lower than D0 and at least5 of6 test recordings improve; report all local/tail costs separately',
            'not_supported':'Primary criterion not met; no post-hoc model replacement or threshold change',
            'secondary':'B0_FT250/B2_FT250 and development sources fully reported, not used to rescue primary; original training reliability remains separate',
            'no_GT_switching':True,'robot_success_rate_claim':False},
        'test_reuse_rule':'No coordinate/hyperparameter/model tuning on locked test; any future reuse for tuning retires it to development.',
        'preparation_code_sha256':file_sha256(__file__),'training_identity_sha256':file_sha256(cfg['training_identity'])}
    json_write(out/'frozen_protocol.json',protocol)
    print(json.dumps({'sources':len(audits),'frames':sum(r['frames'] for r in audits),'windows':len(windows),'roles':roles,'protocol_sha256':file_sha256(out/'frozen_protocol.json')},indent=2))


if __name__=='__main__':main()
