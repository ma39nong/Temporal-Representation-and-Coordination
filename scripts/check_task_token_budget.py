"""Explicit original-tokenizer probe; no weights, training or robot commands.

Usage: python scripts/check_task_token_budget.py private_runtime.json result.json
Tests the one configured REAL state and synthetic bins; the bound is specific to
the recorded prompt, not all possible goal counts/phases.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize
from trc.models.task_condition import checked_tokens,causal_condition,task_prompt
c=json.load(open(sys.argv[1]));tokenizer=PaligemmaTokenizer(max_len=256)
stat=normalize.load(Path(c['norm_stats']).parent)['state']
q=np.load(c['reference_arrays'])['history'][0,-1]
s=(q-stat.q01)/(stat.q99-stat.q01+1e-6)*2-1
rec=next(r for r in json.load(open(c['training_manifest']))['records'] if r['source_recording_id']==c['recording_id'])
prompt=rec['tasks']['0']
a=tokenizer.tokenize(prompt,s);b=checked_tokens(prompt,s,tokenizer._tokenizer.encode)
assert all(np.array_equal(x,y) for x,y in zip(a,b))
condition=causal_condition(target_count=3,observation_time=0,reset_at=0,reset_evidence='SYNTHETIC',coverage_until=0)
text=task_prompt(condition,'goal_progress')
# Verify independent numeric tokenization before deriving a worst-case bound.
def count(bins):
 t=f"Task: {text}, State: {' '.join(map(str,bins))};\nAction: "
 return len(tokenizer._tokenizer.encode(t,add_bos=True))
base=count([0]*58);deltas=[]
for v in range(-1,256):
 inc=count([v]+[0]*57)-base
 assert count([v]*58)==base+58*inc
 deltas.append(inc)
start=time.perf_counter()
for i in range(100):checked_tokens(text,s,tokenizer._tokenizer.encode)
result={'original_tokenization_exact':True,'original_token_count':int(a[1].sum()),
 'real_condition_tokens':int(checked_tokens(text,s,tokenizer._tokenizer.encode)[1].sum()),
 'verified_uniform_bin_worst_bound_for_goal3_progress0_phase_unknown':base+58*max(deltas),
 'per_call_token_encoding_ms_mean_100':(time.perf_counter()-start)*10,
 'scope':'Tokenization only on CPU; not GPU or deployment latency; over-budget inputs fail closed'}
Path(sys.argv[2]).write_text(json.dumps(result,indent=2));print(json.dumps(result))
