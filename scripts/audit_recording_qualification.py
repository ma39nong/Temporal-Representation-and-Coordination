#!/usr/bin/env python3
"""P2-1A actual recording audit. Existing reader/schema/audit; no model imports."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from trc.io import file_sha256
from trc.data.lerobot_adapter import LeRobotAdapter
from trc.data.action_schema import verify_features,canonicalize
from trc.data.validation import audit_episode
from trc.data.qualification import state_stratum,inverse_compact_time


def load(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def video_audit(path,n,thumbs):
    import av
    before=(path.stat().st_size,path.stat().st_mtime_ns);sha=file_sha256(path)
    with av.open(str(path)) as c:
        s=c.streams.video[0];s.thread_type='NONE';s.codec_context.thread_count=1
        times=[float(p.pts*s.time_base) for p in c.demux(s) if p.pts is not None and p.size]
        times=np.sort(times);expected=np.arange(n)/30
        good=len(times)==n and np.allclose(times,expected,rtol=0,atol=1e-5)
        result={'path':str(path),'sha256':sha,'bytes':before[0],'packet_count':len(times),'expected_frames':n,'fps':float(s.average_rate),'width':s.width,'height':s.height,'pts_status':'verified' if good else 'failed','pts_max_error_s':float(np.max(np.abs(times-expected))) if len(times)==n else None,'full_decode':'not_run','samples':[]}
    for frame_id in (0,n//2,n-1):
        with av.open(str(path)) as c:
            s=c.streams.video[0];s.codec_context.thread_count=1;c.seek(int((frame_id/30)/s.time_base),stream=s,backward=True)
            found=None
            for f in c.decode(s):
                if float(f.time)>=frame_id/30-1e-5:found=f;break
            if found is None or abs(float(found.time)-frame_id/30)>1e-5:raise ValueError('Sample frame unavailable or wrong PTS')
            image=found.to_image();image.thumbnail((320,240));thumbs.append((frame_id,image))
            result['samples'].append({'frame':frame_id,'pts_s':float(found.time)})
    if before!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Video changed during audit')
    return result


def lineage(source,meta,raw):
    bags=[b for b in raw['bags'] if b['recording_id']==source]
    result={'status':'unknown','raw_bag_matches':bags,'export_matches':[],'mapping_status':'unknown','issues':[]}
    if len(bags)!=1:result['issues'].append('missing_or_ambiguous_raw_bag');return result
    b=bags[0]
    if not b['files'] or not all(f['exists'] and f['bytes']>0 for f in b['files']):result['issues'].append('raw_volume_missing');return result
    exact=[]
    for ex in raw['exports']:
        for ep in ex['episodes']:
            if ep.get('source_recording_id')==source and all(ep.get(k)==meta.get(k) for k in ('length','source_start_ns','source_end_ns')):
                for proof in ex['source_proofs']:
                    if proof['source_episode_name']==Path(b['bag_path']).name:exact.append((ex,ep,proof))
    if not exact:result['issues'].append('no_exact_source_interval_length_export_link');return result
    for ex,ep,proof in exact:
        summary=proof['proof_files'].get('summary.json',{}).get('document',{});jobs=summary.get('jobs',[])
        job=next((v for v in jobs if v.get('source')==b['state_evidence']['source_bag']),None)
        if not job:continue
        volumes=job.get('readiness',{}).get('volumes',[]);expected={v.get('name'):v.get('bytes') for v in volumes}
        size_ok=bool(expected) and all(expected.get(v['name'])==v['bytes'] for v in b['files'])
        original_ok=job.get('original_sha256_verified') is True and job.get('verification',{}).get('status')=='passed' and summary.get('dataset_verification_passed') is True and size_ok
        tr=proof['proof_files'].get('trim_report.json',{});trim=tr.get('document',{})
        mapping_ok=trim.get('timestamps')=='compact_shared_time_mapping' and isinstance(trim.get('mapping'),list)
        result['export_matches'].append({'export_root':ex['root'],'export_episode':ep,'proof_files':{k:{'path':v['path'],'sha256':v['sha256']} for k,v in proof['proof_files'].items()},'original_source_hash_verification_reused':original_ok,'current_raw_volume_sizes_match':size_ok,'trim_report':trim,'job_verification':job.get('verification'),'readiness_volumes':volumes,'source_profile':job.get('input_profile'),'source_verification_time_ns':job.get('updated_ns')})
        if original_ok and mapping_ok:
            result['status']='verified';result['mapping_status']='verified';result['mapping']=trim['mapping'];break
    if result['status']!='verified':result['issues'].append('missing_sha_alignment_or_compact_mapping_proof')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load(a.config);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);(out/'evidence').mkdir(exist_ok=True);(out/'contact_sheets').mkdir(exist_ok=True)
    for name,digest in cfg['pins'].items():
        if file_sha256(cfg[name])!=digest:raise ValueError('Frozen input changed: '+name)
    raw=load(cfg['raw_provenance']);training=load(cfg['training_identity']);old=load(cfg['prior_registry']);stats=load(cfg['state_statistics'])['statistics'];ref=np.load(cfg['state_reference'],allow_pickle=False)
    prior=set(old['old_candidates'])|set(old['p17_used']);train_ids={r['source_recording_id'] for r in training['records']};train_nums={r['numeric_sha256'] for r in training['records']};train_parquet={r['parquet_sha256'] for r in training['records']}
    inventory=load(cfg['candidate_inventory']);allowed=set(cfg['recording_ids']);locations={s:[] for s in allowed}
    for d in inventory['datasets']:
        for m in d['records']:
            if m.get('source_recording_id') in allowed:locations[m['source_recording_id']].append((d,m))
    adapters={};rows=[];seen_numeric={}
    from PIL import Image,ImageDraw
    for index,source in enumerate(sorted(allowed)):
        path=out/'evidence'/f'{source}.json'
        if path.exists():raise FileExistsError(path)
        choices=sorted(locations[source],key=lambda x:('1109' in x[0]['root'],x[0]['root']))
        if not choices:raise ValueError('Candidate not in frozen inventory')
        d,m=choices[0];root=Path(d['root']);ep=m['episode_index']
        r={'recording_id':source,'dataset_root':str(root),'episode_index':ep,'batch':root.name,'source_aliases':[{'root':x['root'],'episode':y['episode_index']} for x,y in choices],
           'success_status':'uncertain','reviewer_status':'pending_human','task_type':'unknown','scene_class':'unknown','scene_review_status':'unknown','review_evidence':[],
           'data_contract_status':'unknown','training_isolation':'unknown','prior_exclusion_status':'verified' if source not in prior else 'failed','exclusion_reason':[]}
        try:

            if str(root) not in adapters:adapters[str(root)]=LeRobotAdapter(root)
            adapter=adapters[str(root)];actual=next(e for e in adapter.episodes if e['episode_index']==ep)
            if actual!=m:raise ValueError('Candidate metadata changed')
            verify_features(adapter.info['features'],'converter58')
            if adapter.info['fps']!=30:raise ValueError('Expected30Hz')
            parquet=adapter.episode_path(ep);before=(parquet.stat().st_size,parquet.stat().st_mtime_ns);data=adapter.read_episode(ep);audit=audit_episode(data,fps=30,layout='converter58');r['numeric_audit']=audit;r['parquet_path']=str(parquet);r['parquet_sha256']=file_sha256(parquet)
            if before!=(parquet.stat().st_size,parquet.stat().st_mtime_ns):raise ValueError('Parquet changed')
            if audit['errors'] or len(data['action'])!=m['length'] or not np.all(data['episode_index']==ep):
                r['data_contract_status']='failed';raise ValueError('Numeric audit failed')
            act=canonicalize(data['action'],'converter58');state=canonicalize(data['observation.state'],'converter58',state=True);numeric_sha=hashlib.sha256(act.tobytes()+state.tobytes()).hexdigest();r['numeric_sha256']=numeric_sha;r['initial_state']=state[0].tolist()
            duplicate=numeric_sha in seen_numeric;r['duplicate_candidate_id']=seen_numeric.get(numeric_sha);seen_numeric[numeric_sha]=source
            isolated=source not in train_ids and numeric_sha not in train_nums and r['parquet_sha256'] not in train_parquet;r['training_isolation']='verified' if isolated else 'failed'
            r['training_isolation_scope']='Frozen480 source IDs and full numeric/Parquet hashes plus raw lineage; no arbitrary subtrajectory similarity test.'
            r['state']=state_stratum(state[0],ref['initial_state'],ref['mean'],ref['scale'],stats)
            r['source_lineage']=lineage(source,m,raw);r['source_range']={k:m.get(k) for k in ('source_start_ns','source_end_ns','segment_id')}
            cuts=[]
            if r['source_lineage']['mapping_status']=='verified':
                compact=int(m['source_start_ns'])+(np.arange(len(state),dtype=np.int64)*1000000000//30)
                original,cuts=inverse_compact_time(compact,r['source_lineage']['mapping']);r['original_time_reconstruction']={'status':'verified','first_ns':int(original[0]),'last_ns':int(original[-1]),'cut_rows':cuts,'inverse_policy':'right side of deletion seam; no history/target may cross any seam'}
            r['valid_segments']=[[int(a),int(b)] for a,b in zip([0,*cuts],[*cuts,len(state)])];r['legal_full_support_windows']=sum(max(0,b-a-58) for a,b in r['valid_segments'])
            annotation=root/f'annotations/subtasks/coarse/episode_{ep:06d}.json';r['annotation_status']='unknown'
            if annotation.exists():
                ann=load(annotation);same=ann.get('source',{}).get('source_recording_id')==source;r['annotation_status']='verified' if same else 'failed';r['annotation_path']=str(annotation);r['annotation_sha256']=file_sha256(annotation)
                r['annotated_harvest_segments']=sum(s.get('label')=='harvest' for s in ann.get('segments',[]));r['annotated_task_prompt']=ann.get('task_prompt');r['all_stage_segments_reviewed']=all(s.get('reviewed') for s in ann.get('segments',[]))
                r['task_type_evidence']='Stage count/prompt are intent only, not actual fruit count or completed task; human review required.'
            thumbs=[];r['videos']=[]
            for key in ('observation.images.cam0','observation.images.cam1','observation.images.cam2'):
                video=root/adapter.info['video_path'].format(episode_index=ep,episode_chunk=ep//adapter.info['chunks_size'],video_key=key)
                samples=[];v=video_audit(video,len(state),samples);v['camera_key']=key;r['videos'].append(v);thumbs.extend((key,f,img) for f,img in samples)
            sheet=Image.new('RGB',(960,720),'#ffffff');draw=ImageDraw.Draw(sheet)
            for i,(key,frame,img) in enumerate(thumbs):
                x=(i%3)*320;y=(i//3)*240;sheet.paste(img,(x,y+20));draw.text((x+2,y+2),f'{key.split(".")[-1]} frame {frame}',fill='black')
            sheet.save(out/'contact_sheets'/f'{source}.jpg',quality=80)
            video_ok=all(v['pts_status']=='verified' and v['fps']==30 for v in r['videos'])
            r['camera_action_alignment']={'status':'verified' if video_ok else 'failed','scope':'All compressed video packet PTS compared with numeric frame/30; original source alignment proof reused. No sensor latency ground truth or full bitstream decode claim.'}
            r['data_contract_status']=('failed' if not video_ok else ('verified' if r['legal_full_support_windows']>0 and r['source_lineage']['mapping_status']=='verified' else 'unknown'))
            if duplicate:r['exclusion_reason'].append('duplicate_candidate_numeric_content')
            if not isolated:r['exclusion_reason'].append('training_overlap')
        except Exception as e:r['exclusion_reason'].append(type(e).__name__+': '+str(e));r['technical_error']=str(e)
        r['exclusion_reason']+=['human_success_unverified','actual_task_type_unverified','scene_reference_comparison_unverified']
        write(path,r);rows.append({'recording_id':source,'evidence':str(path),'data_status':r['data_contract_status'],'state_class':r.get('state',{}).get('class'),'lineage':r.get('source_lineage',{}).get('status')});print(json.dumps({'checked':index+1,'total':len(allowed),**rows[-1]},ensure_ascii=False),flush=True)
        time.sleep(.03)
    write(out/'audit_index.json',rows)

if __name__=='__main__':main()
