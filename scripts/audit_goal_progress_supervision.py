"""Inspect a configured SMALL subset; do not infer fruit events from phases.

Configuration names existing annotation/video files and frozen source evidence.
No videos are transcoded and original datasets remain read-only.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import subprocess


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(config_path, output):
    cfg=json.loads(Path(config_path).read_text());out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    if not cfg['recordings']:raise ValueError('At least one explicitly selected recording required')
    rows=[];queue=[]
    for row in cfg['recordings']:
        annotation=Path(row['annotation_path'])
        if sha(annotation)!=row['annotation_sha256']:raise ValueError('Annotation identity changed')
        doc=json.loads(annotation.read_text());segments=doc['segments']
        previous=0
        for s in segments:
            if s['start_frame']!=previous or s['end_frame']<=previous or s.get('reviewed') is not True:
                raise ValueError('Phase review/coverage changed')
            previous=s['end_frame']
        if previous!=row['frames']:raise ValueError('Incomplete phase coverage')
        videos=[]
        for video in row['videos']:
            path=Path(video)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=width,height,avg_frame_rate','-of','json',str(path)]))['streams'][0]
            pixels=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(path),
                '-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'])
            if len(pixels)!=probe['width']*probe['height']*3:raise ValueError('First-frame decode failed')
            videos.append({'path':str(path),'bytes':path.stat().st_size,
                'first_frame_shape':[probe['height'],probe['width'],3],
                'average_rate':probe['avg_frame_rate'],'full_decode':False})
        harvests=[s for s in segments if s['label']=='harvest']
        # Phase intervals are navigation hints, never confirmed pick events.
        for s in harvests:
            queue.append({'recording_id':row['recording_id'],'navigation_start_frame':s['start_frame'],
                'navigation_end_frame_exclusive':s['end_frame'], 'hint':'review whole task; harvest phase is NOT one fruit',
                'target_count':'unknown','object_id':'','pick_success_at':'','placement_at':'',
                'confirmation_available_at':'','retry_at':'','reviewer':'','evidence':'',
                'clock_mapping_and_cut_review':'pending',
                'videos':';'.join(row['videos']), 'annotation_path':row['annotation_path']})
        rows.append({**row,'videos':videos,'phase_annotation_verified':True,
            'phase_labels':[s['label'] for s in segments], 'harvest_segments':len(harvests),
            'target_count':'unknown','per_fruit_success_events':'unknown','placement_events':'unknown',
            'retry_events':'unknown','online_phase_available_at':'unknown','causal_progress_eligible':False,
            'reason':'Whole-task human success and offline phase review do not provide count/causal-event supervision'})
    (out/'supervision_audit.json').write_text(json.dumps({'config_sha256':sha(config_path),
        'recordings_inspected':len(rows),'records':rows,'eligible_goal_progress_recordings':0},indent=2))
    with (out/'event_review_queue.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(queue[0]));writer.writeheader();writer.writerows(queue)
    print(json.dumps({'recordings_inspected':len(rows),'review_navigation_intervals':len(queue),
                     'eligible_goal_progress_recordings':0}))


if __name__=='__main__':main(*sys.argv[1:])
