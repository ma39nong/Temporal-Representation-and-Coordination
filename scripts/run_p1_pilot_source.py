#!/usr/bin/env python3
"""Execute the sealed P1 pair through the original trainer, with audited restart.

No optimizer/model/data implementation is replaced. Hooks only select saves,
record exact state hashes and split the 500 accepted updates at completed step 3.
Run from the pinned source launcher. Spawn children have no entrypoint effects.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import runpy
import site
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--prepare-entry', required=True)
    parser.add_argument('--trial', choices=['B0', 'B2'], required=True)
    parser.add_argument('--phase', choices=['initialize', 'precheck', 'resume'], required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    site.addsitedir(cfg['research_runtime'])
    transfer = cfg.get('experiment_kind') == 'step8000_transfer'
    if transfer:
        from trc.experiments.p1_transfer import validate_transfer_protocol as validate_protocol
    else:
        from trc.experiments.p1_protocol import validate_protocol
    validate_protocol(cfg)
    import jax
    import numpy as np
    import scripts.train_flash as trainer
    from openpi.training import checkpoints

    if jax.default_backend() != 'gpu' or jax.device_count() != 8:
        raise ValueError('The sealed eight-GPU mesh is required')
    budget = 250 if transfer else 500
    if args.phase == 'initialize' and not transfer:raise ValueError('Initialize-only phase is for transfer')
    if cfg['shared']['num_train_steps'] != budget or cfg['shared']['batch_size'] != 256:
        raise ValueError('Execution differs from its sealed budget')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != cfg['source_commit']:
        raise ValueError('Source identity changed')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Source must remain clean')
    root = Path(cfg['output_root'])
    audit = root / 'execution' / args.trial
    audit.mkdir(parents=True, exist_ok=True)
    identity = {'config_sha256': sha(args.config), 'runner_sha256': sha(__file__),
                'prepare_entry_sha256': sha(args.prepare_entry), 'source_commit': cfg['source_commit']}
    originals = {name: getattr(checkpoints, name) for name in
                 ('initialize_checkpoint_dir', 'save_state', 'restore_state')}
    original_init, original_data, original_main = trainer.init_train_state, trainer.NativeDataset, trainer.main
    captured = {}

    def fingerprint(state):
        """Hash every parameter/moment byte, preserving each device shard identity."""
        result = {}
        for category, tree in [('params', state.params), ('opt_state', state.opt_state)]:
            h = hashlib.sha256()
            for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
                h.update(str(path).encode())
                h.update(str(getattr(value, 'shape', None)).encode())
                h.update(str(getattr(value, 'dtype', None)).encode())
                if isinstance(value, jax.Array):
                    for shard in value.addressable_shards:
                        h.update(str(shard.index).encode())
                        h.update(np.asarray(jax.device_get(shard.data)).tobytes())
                else:
                    h.update(np.asarray(value).tobytes())
            result[category + '_sha256'] = h.hexdigest()
        return {**result, 'completed_step': int(state.step)}

    def record_state(state, label):
        tick = time.perf_counter()
        record = fingerprint(state)
        record['hash_seconds'] = time.perf_counter() - tick
        (audit / (label + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'state_identity': label, **record}), flush=True)
        return record

    def dataset(*a, **kw):
        captured['dataset'] = original_data(*a, **kw)
        return captured['dataset']

    def initialize(*a, **kw):
        manager, resuming = originals['initialize_checkpoint_dir'](*a, **kw)
        captured['manager'] = manager
        return manager, resuming

    def init_state(*a, **kw):
        state, layout = original_init(*a, **kw)
        if not kw.get('resume', False):
            if transfer:
                from flax import nnx
                from openpi.training.flash_checkpoint import parameter_digest
                pure = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), state.params.filter(nnx.Param).to_pure_dict())
                if parameter_digest(pure) != cfg['initial_parameter_sha256']:raise ValueError('Actual initialized parameters differ from Step8000')
                del pure
                for value in jax.tree.leaves(state.opt_state):
                    if isinstance(value, jax.Array) and any(np.any(np.asarray(jax.device_get(shard.data)) != 0) for shard in value.addressable_shards):
                        raise ValueError('Fresh Adam moments/counters are not zero')
            own = record_state(state, 'initial_state')
            peer = root / 'execution' / ('B2' if args.trial == 'B0' else 'B0') / 'initial_state.json'
            if peer.exists():
                previous = json.loads(peer.read_text())
                if any(own[k] != previous[k] for k in ('params_sha256', 'opt_state_sha256')):
                    raise ValueError('Actual B0/B2 initial parameter or optimizer bytes differ')
            originals['save_state'](captured['manager'], state, captured['dataset'], 0)
            captured['manager'].wait_until_finished()
        return state, layout

    def save(manager, state, data, step):
        if step not in ((3, 50, 100, 250) if transfer else (3, 100, 250, 500)):
            return
        originals['save_state'](manager, state, data, step)
        manager.wait_until_finished()
        if step in (3, budget):
            record_state(state, 'saved_state_' + str(step))

    def restore(*a, **kw):
        state = originals['restore_state'](*a, **kw)
        actual = record_state(state, 'restored_state')
        expected = json.loads((audit / ('initial_state.json' if transfer and args.phase == 'precheck' else 'saved_state_3.json')).read_text())
        if any(actual[k] != expected[k] for k in ('completed_step', 'params_sha256', 'opt_state_sha256')):
            raise ValueError('Restored model/Adam state is not byte-identical to accepted step 3')
        return state

    # Canonical paired evaluation runs separately in FP32 after training.
    # Avoid invoking an extra full-batch monitor every step solely to select saves.
    def monitor(*a, **kw):
        return {}

    def execute(config):
        if args.phase == ('initialize' if transfer else 'precheck'):
            if (audit / 'execution_identity.json').exists():
                raise FileExistsError('Precheck cannot overwrite a previous attempt')
            config = replace(config, save_interval=1, keep_period=1, log_interval=1)
            with (audit / 'sealed_config.pkl').open('wb') as stream:
                pickle.dump(config, stream)
            (audit / 'execution_identity.json').write_text(json.dumps(
                {**identity, 'sealed_config_sha256': sha(audit / 'sealed_config.pkl')}, indent=2) + '\n')
            if not transfer:config = replace(config, num_train_steps=3)
        elif transfer and args.phase == 'precheck':
            gate = json.loads(Path(cfg['step0_gate']).read_text())
            if gate['status'] != 'STEP0_PAIR_VERIFIED' or gate['config_sha256'] != sha(args.config):raise ValueError('Step0 pair not verified')
            config = replace(config, resume=True, num_train_steps=3)
        else:
            config = replace(config, resume=True)
        trainer.NativeDataset = dataset
        trainer.init_train_state = init_state
        trainer.validation = monitor
        checkpoints.initialize_checkpoint_dir = initialize
        checkpoints.save_state = save
        checkpoints.restore_state = restore
        os.environ['FLASH_INPUT_PIPELINE'] = 'multiprocess'
        os.environ['WANDB_MODE'] = 'offline'
        for name in ('tmp', 'wandb'):
            (root / name).mkdir(parents=True, exist_ok=True)
        os.environ['TMPDIR'] = str(root / 'tmp')
        os.environ['WANDB_DIR'] = str(root / 'wandb')
        if transfer and args.phase == 'initialize':
            # The source TrainConfig requires positive num_train_steps. Stop at
            # the tracking context immediately before the first update, AFTER
            # its original initialization, Step0 save and contract writing.
            from openpi.training import tracking
            class InitializationComplete(Exception):pass
            original_session = tracking.session
            def stop_before_updates(*a, **kw):raise InitializationComplete()
            tracking.session = stop_before_updates
            try:
                original_main(config)
            except InitializationComplete:
                pass
            finally:
                tracking.session = original_session
        else:
            original_main(config)
        expected_step = 0 if args.phase == 'initialize' else (3 if args.phase == 'precheck' else budget)
        marker = Path(config.checkpoint_dir) / str(expected_step) / '_CHECKPOINT_METADATA'
        if not json.loads(marker.read_text()).get('commit_timestamp_nsecs'):
            raise ValueError('Final checkpoint is incomplete')
        result = {'trial': args.trial, 'phase': args.phase, 'accepted_updates': expected_step,
                  'checkpoint': str(marker.parent), 'training_config_identity': identity}
        (audit / (args.phase + '_complete.json')).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)

    if args.phase == ('initialize' if transfer else 'precheck'):
        trainer.main = execute
        sys.argv = [args.prepare_entry, '--config', args.config, '--start-training', args.trial]
        runpy.run_path(args.prepare_entry, run_name='__main__')
    else:
        saved = json.loads((audit / 'execution_identity.json').read_text())
        if any(saved[k] != v for k, v in identity.items()) or saved['sealed_config_sha256'] != sha(audit / 'sealed_config.pkl'):
            raise ValueError('Execution code/config changed across restart')
        prior = 'initialize_complete.json' if transfer and args.phase == 'precheck' else 'precheck_complete.json'
        if not (audit / prior).exists() or (audit / (args.phase + '_complete.json')).exists():
            raise ValueError('Resume requires exactly one successful three-update precheck')
        with (audit / 'sealed_config.pkl').open('rb') as stream:
            config = pickle.load(stream)
        execute(config)


if __name__ == '__main__':
    main()
