"""Validate edited event-review JSON against fixed clocks; never train a model."""
from dataclasses import asdict
import hashlib,json,sys
from pathlib import Path
from trc.data.causal_supervision import reviewed_progress,validate_window


def main(reviews_path,clocks_path,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    reviews=json.loads(Path(reviews_path).read_text())['records']
    clocks={r['recording_id']:r for r in json.loads(Path(clocks_path).read_text())['clocks']}
    results=[]
    for r in reviews:
        clock=clocks[r['recording_id']];valid=unknown=blocked=0;snapshots=[]
        for frame in range(7,clock['frames']-51):
            try:validate_window(frame,clock)
            except ValueError:blocked+=1;continue
            c=reviewed_progress(r,frame,clock)
            valid+=int(c['condition'].progress_valid);unknown+=int(not c['condition'].progress_valid)
            # Export only confirmed snapshots; unknown input must not silently
            # become a zero-count training label.
            if c['condition'].progress_valid:snapshots.append({'frame':frame,
                'condition':asdict(c['condition']),'remaining_count':c['remaining_count']})
        results.append({'recording_id':r['recording_id'],'verified_progress_windows':valid,
            'unknown_progress_windows':unknown,'cross_cut_windows_rejected':blocked,'confirmed_snapshots':snapshots})
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    out.write_text(json.dumps({'reviews_sha256':sha(reviews_path),'clocks_sha256':sha(clocks_path),
        'results':results,'restriction':'development-only supervision check; not a training launcher or independent evaluation'},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='confirmed_snapshots'} for r in results]))

if __name__=='__main__':main(*sys.argv[1:])
