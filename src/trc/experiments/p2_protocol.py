"""Shared frozen-VLM protocol; only the final residual readout may differ."""
import hashlib,json
from trc.models.velocity_readout import adapter_count


def validate_order_protocol(config):
    expected={'B0':{'mode':'none','updates':0},'B1':{'mode':'none','updates':250},
              'B2':{'mode':'unified','updates':250},'B3':{'mode':'order_specific','updates':250}}
    if config['trials']!=expected:raise ValueError('Only readout mode may differ between training groups')
    shared=config['shared']
    if shared['batch_size']!=256 or shared['devices']!=8 or shared['optimizer_state_policy']!='reset_all_trainable_groups':
        raise ValueError('Matched global256/eight-device/fresh Adam protocol required')
    if shared['freeze_vlm'] is not True or shared['preserve_frozen_parameter_dtype'] is not True:
        raise ValueError('Freeze VLM without silently casting original FP32 parameters')
    if shared['train_dtype']!='bfloat16' or config['evaluation']['dtype']!='float32':raise ValueError('Precision protocol differs')
    if config['representation']!={'coordinate':'absolute','horizon':50,'action_dim':58,'degree':6,'tokens':7,'num_flow_steps':1}:
        raise ValueError('Representation/loss not part of this experiment')
    if config['data']['legal_windows']!=362918 or config['data']['source_count']!=480 or not config['data']['cut_manifest_sha256']:
        raise ValueError('Frozen all480 continuous-window proof required')
    for name,count in config['added_parameters'].items():
        if count!=adapter_count(expected[name]['mode']):raise ValueError('Parameter count differs')
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
