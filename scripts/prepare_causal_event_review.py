"""Build the two-recording evidence queue; never promote visual candidates to GT."""
import csv,html,json,sys
from pathlib import Path
from trc.data.causal_supervision import reviewed_progress,validate_window


def main(config_path,output):
    cfg=json.loads(Path(config_path).read_text());out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    source=json.loads(Path(cfg['source_review']).read_text())['records']
    clocks={r['recording_id']:r for r in json.loads(Path(cfg['clocks']).read_text())['clocks']}
    rows=[];reviews=[];pages=[];alignment=[]
    for source_row in source:
        ident=source_row['recording_id'];clock=clocks[ident]
        annotations=json.loads(Path(source_row['annotation_path']).read_text())['segments']
        events=[]
        for j,s in enumerate(annotations):
            events.append({'recording_id':ident,'event_id':f'offline_phase_{j}','event_type':'phase',
                'event_timestamp':s['start_frame']/30,'timestamp_domain':'export_seconds','phase':s['label'],
                'evidence':{'path':source_row['annotation_path'],'sha256':source_row['annotation_sha256'],'segment_index':j},
                'confidence':'reviewed_offline_phase_only','review_status':'verified','verified_by':'existing_human_phase_review',
                'confirmation_available_at':None,'confirmation_timestamp_domain':'export_seconds',
                'causal_availability_verified':False,'object_id':None})
        for j,candidate in enumerate(cfg['candidates'][ident]):
            kind,start,end,description=candidate
            event={'recording_id':ident,'event_id':f'visual_candidate_{j}','event_type':kind,
                'event_timestamp':[start,end],'timestamp_domain':'export_seconds',
                'evidence':{'video_paths':[v['path'] for v in source_row['videos']],
                    'observation':description,'review_method':'assistant inspection of phase context contact sheets',
                    'interval_is':'navigation bracket, NOT a verified unique event or exhaustive event census'},
                'confidence':'tentative','review_status':'pending_human','verified_by':None,
                'confirmation_available_at':None,'confirmation_timestamp_domain':'export_seconds',
                'causal_availability_verified':False,'object_id':None,
                'crosses_cut_rows':[c for c in clock['cut_rows'] if start<c/30<=end]}
            events.append(event)
        review={'recording_id':ident,'goal':cfg['goal_confirmation'],
            'coverage':{'reset_at':None,'complete_until':None,'review_status':'pending_human','deleted_intervals_reviewed':False,'evidence':None},
            'events':events,'recording_stop_reason':'unknown; goal completion does not establish recorder stop cause',
            'whole_task_success_evidence':source_row['whole_task_success'],
            'remaining_count':None,'target_count':None,'completed_count':None,'progress_valid':False}
        reviews.append(review)
        legal=blocked=0
        for frame in range(7,clock['frames']-51):
            try:validate_window(frame,clock)
            except ValueError:blocked+=1;continue
            legal+=1
            condition=reviewed_progress(review,frame,clock)['condition']
            if condition.progress_valid or condition.completed_count is not None:raise ValueError('Unreviewed events leaked into progress')
        alignment.append({'recording_id':ident,'legal_continuous_windows':legal,'cross_cut_windows_rejected':blocked,
            'target_count':None,'task_goal':'harvest_all','completed_count':None,'remaining_count':None,
            'progress_valid':False,'causal_input_events':0})
        for e in events:rows.append({**e,'evidence':json.dumps(e['evidence'],ensure_ascii=False)})
        videos=''.join(f'<video width="320" controls preload="metadata" src="{html.escape(Path(v["path"]).as_uri())}"></video>' for v in source_row['videos'])
        table=''.join('<tr><td>'+html.escape(str(e[k]))+'</td><td>'+html.escape(str(e['event_timestamp']))+'</td><td>'+html.escape(e['review_status'])+'</td></tr>' for e in events for k in ['event_id'])
        pages.append(f'<h2>{ident}</h2>{videos}<p>All times: cleaned export seconds. Cuts: {clock["cut_rows"]}. All visual event candidates need human confirmation.</p><table border="1">{table}</table>')
    (out/'event_reviews.json').write_text(json.dumps({'records':reviews},indent=2,ensure_ascii=False))
    (out/'alignment.json').write_text(json.dumps(alignment,indent=2))
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'event_review.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><h1>Two-recording causal event review</h1><p>Inspect physical event, object identity, confirmation time, placement destination, retries and deletion gaps. CSV/JSON remain pending until reviewed. No automatic count is generated.</p>'+''.join(pages))
    print(json.dumps({'sources':len(reviews),'candidate_navigation_intervals':sum(len(cfg['candidates'][r['recording_id']]) for r in reviews),'alignment':alignment}))

if __name__=='__main__':main(*sys.argv[1:])
