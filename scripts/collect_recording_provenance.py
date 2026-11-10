#!/usr/bin/env python3
"""Read small bag/conversion proofs only. Never opens rosbag message databases."""
import argparse,json,hashlib
from pathlib import Path


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();cfg=load(Path(args.config));allowed=set(cfg['recording_ids']);bags=[];exports=[]
    for root in cfg['bag_roots']:
        for folder in sorted(Path(root).glob('episode*')):
            state=folder/'collection_state.json';meta=folder/'metadata.yaml'
            if not state.exists():continue
            j=load(state);ident=j.get('source_recording_id')
            if ident not in allowed:continue
            import yaml
            y=yaml.safe_load(meta.read_text()) if meta.exists() else {};info=y.get('rosbag2_bagfile_information',{})
            files=[{'name':name,'exists':(folder/name).is_file(),'bytes':(folder/name).stat().st_size if (folder/name).is_file() else None} for name in info.get('relative_file_paths',[])]
            bags.append({'recording_id':ident,'bag_path':str(folder),'collection_state_path':str(state),'collection_state_sha256':sha(state),'metadata_path':str(meta),'metadata_sha256':sha(meta) if meta.exists() else None,'files':files,
                'bag_start_ns':info.get('starting_time',{}).get('nanoseconds_since_epoch'),'duration_ns':info.get('duration',{}).get('nanoseconds'),'message_count':info.get('message_count'),
                'state_evidence':{k:j.get(k) for k in ('source_bag','state','finalized','quality','failures','source_warnings','boundary_warnings','alignment_quality','device_identities','calibration_ids','workcell_id','workcell_config_hash','validation_report','updated_at_ns','milestones')},
                'explicit_task_outcome':{k:v for k,v in j.items() if k in ('task_success','task_outcome','success_review','complete_task')}})
    for root in cfg['lerobot_roots']:
        for folder in sorted(Path(root).iterdir()):
            if folder.name.startswith('.') or not folder.is_dir():continue
            ep=folder/'meta/episodes.jsonl'
            if not ep.exists():continue
            rows=[json.loads(x) for x in ep.read_text().splitlines()];chosen=[r for r in rows if r.get('source_recording_id') in allowed]
            if not chosen:continue
            proofs=[]
            for source in sorted((folder/'ingest/sources').glob('episode*')):
                # Published per-source datasets may have compacted away episodes.jsonl.
                # Bind by raw bag route/name, then retain the parent export's actual ID row.
                matched=[b for b in bags if Path(b['bag_path']).name==source.name and Path(b['bag_path']).parent.name==folder.parent.name]
                matched_ids={b['recording_id'] for b in matched}
                ss=[r for r in chosen if r.get('source_recording_id') in matched_ids]
                if not ss:continue
                proof={'source_episode_name':source.name,'episodes':ss,'binding':'parent export recording ID + raw bag route/name; summary original source checked by consumer','proof_files':{}}
                for name in ('summary.json','trim_report.json','manifest.json'):
                    path=source/name
                    if path.exists():proof['proof_files'][name]={'path':str(path),'sha256':sha(path),'document':load(path)}
                proofs.append(proof)
            exports.append({'root':str(folder),'episodes_path':str(ep),'episodes_sha256':sha(ep),'episodes':chosen,'source_proofs':proofs})
    print(json.dumps({'bag_metadata_only':True,'bags':bags,'exports':exports,'allowed_count':len(allowed)},ensure_ascii=False))

if __name__=='__main__':main()
