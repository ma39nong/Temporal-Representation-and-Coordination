"""Validate the sole B0/B2 difference before touching a source runtime."""
import hashlib,json


def validate_protocol(config):
    shared=config['shared'];trials=config['trials'];evaluation=config['evaluation']
    if set(trials)!={'B0','B2'} or trials['B0']!={'coordinate_mode':'absolute'} or trials['B2']!={'coordinate_mode':'state_relative'}:
        raise ValueError('Only coordinate_mode may differ; per-trial overrides forbidden')
    if not 500<=shared['num_train_steps']<=1000 or shared['batch_size']<=0 or shared['fsdp_devices']<=0 or shared['batch_size']%shared['fsdp_devices']:
        raise ValueError('Matched 500-1000 step budget and divisible global batch required')
    if shared['train_dtype']!='bfloat16' or evaluation!={'dtype':'float32','matmul_precision':'highest','batch_size':1,'num_steps':1,'padding':False}:
        raise ValueError('Pinned common training/evaluation precision protocol required')
    for key in ('model_seed','data_seed','augmentation_seed','flow_seed'):
        if not isinstance(shared[key],int):raise ValueError('Explicit integer seeds required')
    if set(shared)!={'num_train_steps','batch_size','fsdp_devices','train_dtype','model_seed','data_seed','augmentation_seed','flow_seed'}:
        raise ValueError('Loss/model/optimizer overrides are not part of this experiment')
    payload=json.dumps({'shared':shared,'evaluation':evaluation},sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(payload).hexdigest()
