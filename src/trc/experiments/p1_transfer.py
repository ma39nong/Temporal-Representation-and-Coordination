"""Specific Step8000 transfer protocol; does not weaken the P1-2 base gate."""
import hashlib
import json


def validate_transfer_protocol(config):
    s=config['shared']
    expected={'num_train_steps':250,'batch_size':256,'fsdp_devices':8,'train_dtype':'bfloat16',
              'model_seed':42,'data_seed':42001,'augmentation_seed':42002,'flow_seed':42}
    if s!=expected:raise ValueError('Sealed 250-update / 256 batch transfer protocol required')
    if config['trials']!={'B0':{'coordinate_mode':'absolute'},'B2':{'coordinate_mode':'state_relative'}}:
        raise ValueError('Only coordinate_mode may differ')
    if config['experiment_kind']!='step8000_transfer' or config['optimizer_state_policy']!='reset_both':
        raise ValueError('Explicit warm-start and symmetric fresh Adam required')
    if config['expected_schedule']!={'warmup_steps':25,'peak_lr':2.5e-6,'decay_steps':250,'decay_lr':2.5e-6}:
        raise ValueError('Shared predeclared warmup/constant schedule required')
    if config['expected_optimizer']!={'b1':.9,'b2':.95,'eps':1e-8,'weight_decay':1e-10,'clip_gradient_norm':1.0}:
        raise ValueError('Original shared AdamW required')
    if config['evaluation']!={'dtype':'float32','matmul_precision':'highest','batch_size':1,'num_steps':1,'padding':False}:
        raise ValueError('Shared FP32 evaluation required')
    if config['initial_parameter_sha256']!='40b2c2c4dcdd6fb88dd847d5762d2071fba5028af0942c6dfe457409fe5f5919':
        raise ValueError('Verified Step8000 initializer required')
    return hashlib.sha256(json.dumps({k:config[k] for k in ('shared','trials','expected_optimizer',
        'expected_schedule','optimizer_state_policy','initial_parameter_sha256','evaluation')},sort_keys=True).encode()).hexdigest()
