#!/usr/bin/env python3
"""Materialize all90 source reviews/CSV and a local-video human review board."""
import argparse,csv,json,html
from pathlib import Path
from collections import Counter,defaultdict
from trc.io import file_sha256
from trc.data.qualification import coverage_cell


def load(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();cfg=load(args.config);out=Path(cfg['audit_output']);evidence=sorted((out/'evidence').glob('*.json'));rows=[load(p) for p in evidence]
    if len(rows)!=90 or len({r['recording_id'] for r in rows})!=90:raise ValueError('Exactly90 unique sources required')
    local=defaultdict(list)
    for ident,entries in cfg.get('video_overrides',{}).items():
        for key,path in entries.items():local[ident,key].append(Path(path))
    limits={x['group']:x for x in load(cfg['state_statistics'])['statistics']}
    for root in cfg['local_video_search_roots']:
        for meta in Path(root).rglob('meta/episodes.jsonl'):
            folder=meta.parent.parent;info=load(folder/'meta/info.json')
            for ep in map(json.loads,meta.read_text().splitlines()):
                ident=ep.get('source_recording_id')
                for key in ('observation.images.cam0','observation.images.cam1','observation.images.cam2'):
                    video=folder/info['video_path'].format(episode_index=ep['episode_index'],episode_chunk=ep['episode_index']//info['chunks_size'],video_key=key)
                    if video.is_file():local[ident,key].append(video)
    hashes={};reviews=[];flat=[];videos=[]
    for r in rows:
        source=r['recording_id'];vp=[]
        for v in r['videos']:
            match=None
            for candidate in local[source,v['camera_key']]:
                k=str(candidate)
                if k not in hashes:hashes[k]=file_sha256(candidate)
                if hashes[k]==v['sha256']:match=str(candidate.resolve());break
            vp.append({'camera_key':v['camera_key'],'remote_path':v['path'],'local_path':match,'sha256':v['sha256'],'duration_s':r['numeric_audit']['frames']/30})
            videos.append({'recording_id':source,**vp[-1]})
        bag=r['source_lineage']['raw_bag_matches'][0];proof_path=Path('outputs/p2_1a/evidence')/f'{source}.json';sha=file_sha256(out/'evidence'/f'{source}.json')
        review={'recording_id':source,'review_entry':'outputs/p2_1a/human_review.html#'+source,'dataset_label':Path(r['dataset_root']).name,'episode_index':r['episode_index'],
            'task_type':'unknown','success_status':'uncertain','review_evidence':[],'reviewer_status':'pending_human','reviewer':None,'reviewed_at':None,'full_task_video_checked':False,'video_scope':'cleaned_export_with_original_gap_sidecar',
            'source_lineage':{'status':r['source_lineage']['status'],'evidence':str(proof_path),'sha256':sha},'data_contract_status':r['data_contract_status'],
            'state':r['state'],'scene':{'class':'unknown','reviewer_status':'pending_human','reference_condition_id':None,'reason':None,'fruit_count':None,'truss_geometry':None,'occlusion':None,'initial_spatial_relation':None},
            'exclusion_reason':r['exclusion_reason'],'annotated_harvest_segments':r.get('annotated_harvest_segments'),'annotated_task_prompt':r.get('annotated_task_prompt'),
            'valid_segments':r['valid_segments'],'legal_full_support_windows':r['legal_full_support_windows'],'requires_original_seam_sidecar':True,
            'training_isolation':r['training_isolation'],'prior_exclusion_status':r['prior_exclusion_status'],'cell':None}
        review['state']['group_classes']={g:('near_marginal' if d<=limits[g]['training_LOO_nn_q95'] else 'far' if d>limits[g]['training_LOO_nn_q99'] else 'ambiguous') for g,d in r['state']['nearest_distances'].items()}
        review['cell']=coverage_cell(review['success_status'],review['data_contract_status'],review['source_lineage']['status'],review['training_isolation'],review['prior_exclusion_status'],review['state']['class'],review['scene']['class'],review['task_type'])
        reviews.append(review)
        record={'recording_id':source,'dataset_root':r['dataset_root'],'episode_index':r['episode_index'],'raw_bag_path':bag['bag_path'],'raw_metadata_path':bag['metadata_path'],'raw_collection_state':bag['collection_state_path'],'lerobot_parquet':r['parquet_path'],
            'success_status':'uncertain','task_type':'unknown','reviewer_status':'pending_human','source_lineage_status':r['source_lineage']['status'],'data_contract_status':r['data_contract_status'],'training_isolation_status':r['training_isolation'],'prior_exclusion_status':r['prior_exclusion_status'],
            'state_class':r['state']['class'],'state_far_groups':'|'.join(r['state']['far_groups']),'near_common_reference_index':r['state']['common_near_reference_index'],**{f'{g}_distance':d for g,d in r['state']['nearest_distances'].items()},**{f'{g}_class':c for g,c in review['state']['group_classes'].items()},'scene_class':'unknown','A_D_cell':'','batch':r['batch'],'raw_source_interval_status':r['source_lineage']['mapping_status'],
            'frames':r['numeric_audit']['frames'],'compact_cut_boundaries':len(r['original_time_reconstruction']['cut_rows']),'legal_full_support_windows':r['legal_full_support_windows'],
            'annotated_harvest_segments':r.get('annotated_harvest_segments'),'video_PTS_status':r['camera_action_alignment']['status'],'contact_sheet':str(out/'contact_sheets'/f'{source}.jpg'),'evidence':str(proof_path),'evidence_sha256':sha,'exclusion_reason':'|'.join(r['exclusion_reason'])}
        for i,v in enumerate(vp):record[f'cam{i}_remote_path']=v['remote_path'];record[f'cam{i}_local_path']=v['local_path'] or ''
        flat.append(record);r['review_videos']=vp
    def csv_write(path,values):
        with Path(path).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    csv_write(out/'recording_qualification.csv',flat);csv_write(out/'video_review_index.csv',videos)
    manifest={'schema_version':1,'purpose':'Human review queue; technical verification is not task success. No source is enrolled until explicit review and all coverage gates pass.','base_commit':'17f544a','contract_scope':{'numeric':'Existing float32 canonical58/state116 schema, finite values and30Hz audit','videos':'All packet PTS/counts; first/middle/last decode; not full bitstream decode','lineage':'Raw ID+published exact interval/length+existing SHA/alignment proof, current raw volume size; no fresh multi-TB bag rehash','human_task_success':'Not verified by technical checks'},'records':reviews}
    write(cfg['review_output'],manifest)
    summary={'status':'BLOCKED BY REVIEW/DATA','checked':90,'success':0,'failure':0,'uncertain':90,'lineage_verified':sum(r['source_lineage']['status']=='verified' for r in rows),'data_contract_verified':sum(r['data_contract_status']=='verified' for r in rows),
        'state_distances_computed':90,'state_classification':dict(Counter(r['state']['class'] for r in rows)),'scene_classified':0,'local_full_video_files_hash_verified':sum(v['local_path'] is not None for v in videos),'remote_video_files':len(videos),
        'frames':sum(r['numeric_audit']['frames'] for r in rows),'cut_boundaries':sum(len(r['original_time_reconstruction']['cut_rows']) for r in rows),'legal_windows':sum(r['legal_full_support_windows'] for r in rows),'cells':{c:{'qualified':0,'target':4,'deficit':4} for c in 'ABCD'},
        'state_only_upper_bounds':{'near':sum(r['state']['class']=='near' for r in rows),'far':sum(r['state']['class']=='far' for r in rows)},'independence_note':'No cell qualifies; batch concentration cannot yet be assessed for enrolled cells.'}
    write(out/'qualification_summary.json',summary)
    cards=[]
    for r in sorted(rows,key=lambda x:(x['state']['class']=='ambiguous',x['state']['class'],x['recording_id'])):
        source=r['recording_id'];rv=next(x for x in reviews if x['recording_id']==source);vid=[]
        for v in r['review_videos']:
            uri=Path(v['local_path']).as_uri() if v['local_path'] else None
            vid.append('<div><b>'+html.escape(v['camera_key'])+'</b>'+('<video controls preload="none" src="'+html.escape(uri)+'"></video>' if uri else '<p>本地无SHA匹配副本；请打开远端视频</p>')+'<p class="path">'+html.escape(v['remote_path'])+'</p></div>')
        cards.append(f'''<details class="card" id="{source}" data-id="{source}"><summary>{source} · {html.escape(r['batch'])} / ep{r['episode_index']} · state={r['state']['class']}</summary>
        <p>完整导出视频约{r['numeric_audit']['frames']/30:.1f}秒，原时间轴有{len(r['original_time_reconstruction']['cut_rows'])}处拼接；删除段可能影响任务完整性，必要时结合原录制或采集者证据。阶段标注harvest={r.get('annotated_harvest_segments')}，不能据此判成功/实际采摘数量。未与训练场景作人工对照。</p>
        <img loading="lazy" src="contact_sheets/{source}.jpg" alt="首中尾抽样，仅供定位"/>
        <p><button onclick="playTogether(this)">同步播放三路</button> <button onclick="pauseTogether(this)">暂停三路</button></p><div class="videos">{''.join(vid)}</div><p><a href="evidence/{source}.json">源谱系、删除段映射与SHA证据</a></p>
        <label>任务结果 <select class="success"><option>uncertain</option><option>success</option><option>failure</option></select></label>
        <label>实际任务 <select class="task"><option>unknown</option><option>single_pick</option><option>continuous_multi</option></select></label>
        <label>场景相对训练 <select class="scene"><option>unknown</option><option>near</option><option>different</option></select></label>
        <label><input type="checkbox" class="watched"/>已完整核对任务视频及实际摘取数量</label>
        <label>训练场景参考卡/证据 <input class="reference" placeholder="无证据则保持unknown"/></label>
        <label>结果/场景理由与果串、遮挡、初始相对位置 <textarea class="reason"></textarea></label></details>''')
    js='''
function playTogether(button){const videos=[...button.closest('.card').querySelectorAll('video')];const t=videos[0]?.currentTime||0;videos.forEach(v=>{v.currentTime=t;v.play().catch(()=>{});});}
function pauseTogether(button){button.closest('.card').querySelectorAll('video').forEach(v=>v.pause());}
const original=JSON.parse(document.getElementById('records').textContent);
document.getElementById('save').onclick=()=>{
 const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('请填写实际人工审核人');return;}
 const result=structuredClone(original);
 document.querySelectorAll('.card').forEach(card=>{
  const r=result.records.find(r=>r.recording_id===card.dataset.id),watched=card.querySelector('.watched').checked;
  const success=card.querySelector('.success').value,task=card.querySelector('.task').value,scene=card.querySelector('.scene').value,ref=card.querySelector('.reference').value.trim(),reason=card.querySelector('.reason').value.trim();
  if(!watched)return;
  r.success_status=success;r.task_type=task;r.reviewer=reviewer;r.reviewed_at=new Date().toISOString();r.full_task_video_checked=true;r.reviewer_status='human_reviewed';
  r.review_evidence=[{type:'human_full_video_review',source_evidence:r.source_lineage.evidence,video_scope:r.video_scope,reason:reason}];
  if(scene!=='unknown'&&ref&&reason){r.scene.class=scene;r.scene.reference_condition_id=ref;r.scene.reason=reason;r.scene.reviewer_status='human_reviewed';}
  r.cell=null;r.exclusion_reason=['requires_reviewed_file_validation_and_coverage_rebuild'];
 });
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.download='p2_1a_recording_review.reviewed.json';a.click();URL.revokeObjectURL(a.href);
};
'''
    reference_cards=''
    if cfg.get('training_reference_index'):
        references=load(cfg['training_reference_index'])
        reference_cards='<details><summary>7个训练场景参考候选：先人工确认代表性，再作场景对照</summary>'+''.join('<p>'+html.escape(v['reference_id'])+' · '+html.escape(v['dataset'])+'</p><img loading="lazy" src="training_reference_images/'+html.escape(v['image'])+'"/>' for v in references)+'</details>'
    page='''<!doctype html><meta charset="utf-8"><title>P2-1A 90来源人工审核</title><style>body{font:15px sans-serif;max-width:1150px;margin:25px auto;color:#223}header{position:sticky;top:0;background:white;padding:12px;border-bottom:2px solid #ccc}details{margin:15px 0;padding:12px;border:1px solid #bbb}summary{cursor:pointer;font-weight:bold}img{max-width:700px;width:100%}.videos{display:flex;gap:8px}.videos>div{width:32%}video{width:100%}.path{overflow-wrap:anywhere;font-size:11px}label{display:block;margin:8px 0}textarea{width:95%;height:60px}input.reference{width:65%}</style><header><h1>P2-1A 人工审核队列：90个真实录制</h1><p>先审核明确near/far的5条，其余85条ambiguous不能改阈值凑格。缩略图只作定位；没有AI成功判定。展开后可播放SHA匹配的本地完整三路视频。</p><label>实际审核人 <input id="reviewer"/></label><button id="save">导出人工审核记录（不修改原始数据）</button><p>success仅在完整任务核对后填写；场景分类须附训练参考证据。导出后仍需核验并重建覆盖统计，不会自动入组。</p></header>'''+reference_cards+''.join(cards)+'<script type="application/json" id="records">'+json.dumps(manifest,ensure_ascii=False).replace('</','<\\/')+'</script><script>'+js+'</script>'
    (out/'human_review.html').write_text(page)
    (out/'human_review_instructions.md').write_text('打开human_review.html：优先完成5条near/far来源，观看完整三路视频，确认实际摘取数量、完整任务成功/失败，并用训练场景参考卡标近/异。未知保持uncertain/unknown。未完整审核不得勾选确认。导出reviewed JSON后交回研究仓库进行证据核验和覆盖重建。远端视频逐条路径见video_review_index.csv；不能用缩略图替代完整审核。\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
