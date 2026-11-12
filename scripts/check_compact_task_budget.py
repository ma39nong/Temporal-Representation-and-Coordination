"""Pinned real SentencePiece check; bounds, legacy behavior and mixed-bin probes."""
import json,sys,hashlib,csv
from pathlib import Path
import numpy as np
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize
from trc.models.task_condition import TaskCondition,task_prompt,checked_tokens,parse_compact_prompt


def main(config,output):
    cfg=json.loads(Path(config).read_text());tok=PaligemmaTokenizer(256);sp=tok._tokenizer
    def text(prompt,bins):return f"Task: {prompt}, State: {' '.join(map(str,bins))};\nAction: "
    def n(prompt,bins):return len(sp.encode(text(prompt,bins),add_bos=True))
    prompts=[]
    for goal in ('pick_tomatoes','harvest_all'):
        for target in ([None,*range(1,10)] if goal=='pick_tomatoes' else [None]):
            for count in (None,*range(10)):
                for phase in ([None] if count is None else [None,'pickup','harvest','return','adjust']):
                    for stop in (False,True):
                        c=TaskCondition(goal,target,count,count is not None,phase,stop,0,('SYNTHETIC audit',))
                        p=task_prompt(c,'compact_v1')
                        restored=parse_compact_prompt(p,observation_time=0,evidence=c.evidence)
                        if c!=restored:raise ValueError('Lossy task condition encoding')
                        prompts.append(p)
    reference=prompts[-1];base=n(reference,[0]*58);deltas=[]
    for v in range(-1,256):
        increase=n(reference,[v]+[0]*57)-base
        if n(reference,[v]*58)!=base+58*increase:raise ValueError('Nonseparable state grammar')
        deltas.append(increase)
    worst=int(np.argmax(deltas))-1
    for position in range(58):
        b=[0]*58;b[position]=worst
        if n(reference,b)!=base+max(deltas):raise ValueError('Position-dependent state pieces')
    rng=np.random.default_rng(42)
    bound=n(reference,[worst]*58)
    for _ in range(2000):
        b=rng.integers(-1,256,58).tolist()
        if n(reference,b)!=base+sum(deltas[v+1] for v in b):raise ValueError('Mixed-bin factorization differs')
    lengths=[n(p,[worst]*58) for p in prompts]
    if max(lengths)>256:raise ValueError('Compact grammar exceeds model budget')
    original=next(r for r in json.loads(Path(cfg['training_manifest']).read_text())['records']
                  if r['source_recording_id']==cfg['recording_id'])['tasks']['0']
    result={'template_version':'truss-progress-v1','prompt_variants_checked':len(prompts),
            'all_metadata_roundtrip_exact':True,'supported_target_counts':'1..9 or unknown/all',
            'supported_completed_counts':'0..9 or unknown','compact_max_tokens':max(lengths),
            'compact_min_worst_state_tokens':min(lengths),'mixed_bin_factorization_checks':2000,
            'legacy_original_worst_tokens':n(original,[worst]*58),
            'legacy_p24_worst_tokens':n(task_prompt(TaskCondition('pick_tomatoes',3,0,True,None,False,0,('test',)),'goal_progress'),[worst]*58),
            'state_bin_values_checked':257,'state_positions_checked':58,
            'always_runtime_length_guard':True,'scope':'Verified numeric-token factorization probes, not exhaustive 257**58 enumeration',
            'module_sha256':hashlib.sha256(Path(cfg['condition_module']).read_bytes()).hexdigest()}
    # Original tokenizer equality on nontruncated inputs, and rejection rather
    # than silent truncation on both original and new oversized text.
    stat=normalize.load(Path(cfg['norm_stats']).parent)['state']
    q=np.load(cfg['reference_arrays'])['history'][0,-1]
    state=(q-stat.q01)/(stat.q99-stat.q01+1e-6)*2-1
    for p in (original,prompts[-1]):
        a=tok.tokenize(p,state);b=checked_tokens(p,state,sp.encode)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    result['original_template_equality_on_real_state']=True
    result['original_real_tokens']=int(tok.tokenize(original,state)[1].sum())
    result['compact_real_tokens']=int(checked_tokens(prompts[-1],state,sp.encode)[1].sum())
    # Runtime rejection is retained for unsupported or maliciously long text.
    try:checked_tokens(original+' verylong'*100,state,sp.encode)
    except ValueError:result['oversize_rejected']=True
    else:raise ValueError('Oversized text not rejected')
    Path(output).write_text(json.dumps(result,indent=2))
    with Path(output).with_suffix('.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['compact_prompt','worst_state_tokens'])
        writer.writerows(zip(prompts,lengths))
    print(json.dumps(result))

if __name__=='__main__':main(*sys.argv[1:])
