"""Collect only existing conversion JSON deletion maps, never bag/Parquet/video."""
import argparse,bisect,hashlib,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text());rows=[]
    for r in cfg['records']:
        root=Path(cfg['source_roots'][r['dataset']]);ep=r['episode']
        meta=next(json.loads(x) for x in (root/'meta/episodes.jsonl').read_text().splitlines() if json.loads(x)['episode_index']==ep)
        for k,v in [('source_recording_id',r['source_recording_id']),('length',r['rows']),('source_start_ns',r['source_start_ns']),('source_end_ns',r['source_end_ns'])]:
            if meta[k]!=v:raise ValueError('Frozen training source metadata differs: '+r['id']+' '+k)
        path=root/f'conversion/records/{ep:06d}.json';proof=json.loads(path.read_text())
        found=next(v for v in proof['episodes'] if v['episode_index']==ep)
        if any(found[k]!=meta[k] for k in ('length','source_start_ns','source_end_ns')):raise ValueError('Conversion proof interval differs')
        n=r['rows'];grid=[r['source_start_ns']+(i*1000000000)//30 for i in range(n)]
        if abs(grid[-1]-r['source_end_ns'])>1:raise ValueError('Uniform source grid inconsistent')
        total=0;last=None;cuts=[]
        for m in proof['cleaning']['mapping']:
            start,end,size=(int(m[k]) for k in ('source_start_ns','source_end_ns','removed_ns'))
            if end<=start or end-start!=size or (last is not None and start<last):raise ValueError('Bad trim mapping')
            cut=bisect.bisect_left(grid,start-total)
            if 0<cut<n:cuts.append(cut)
            total+=size;last=end
        cuts=sorted(set(cuts));bounds=[0,*cuts,n];count=sum(max(0,b-a-58) for a,b in zip(bounds[:-1],bounds[1:]))
        rows.append({'id':r['id'],'source_recording_id':r['source_recording_id'],'rows':n,'cuts':cuts,
            'legal_windows':count,'proof_path':str(path),'proof_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_start_ns':r['source_start_ns'],'source_end_ns':r['source_end_ns']})
    out={'scope':'exact frozen all480 conversion metadata only; no raw message audit',
         'training_manifest_sha256':cfg['training_manifest_sha256'],'records':rows,'source_count':len(rows),
         'cut_boundaries':sum(len(r['cuts']) for r in rows),'legal_windows':sum(r['legal_windows'] for r in rows),
         'compact_clock_windows':sum(max(0,r['rows']-58) for r in rows)}
    print(json.dumps(out))


if __name__=='__main__':main()
