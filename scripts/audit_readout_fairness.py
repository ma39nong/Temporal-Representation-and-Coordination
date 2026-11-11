"""Read-only P2-3 accounting, with optional CPU shape/optimizer checkpoint audit.

Runtime mode requires the explicit pinned OpenPI environment used by P2-2.
It reads NPZ headers and the Adam counter, not the multi-GB array payloads.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def window_audit(manifest, cuts):
    from trc.data.flash_sampling import ContinuousFlashIndex
    index = ContinuousFlashIndex(manifest['records'], cuts, 42001)
    proof = {r['id']: r for r in cuts['records']}
    rows = []
    for r in manifest['records']:
        p = proof[r['id']]
        bounds = [0, *p['cuts'], r['rows']]
        legal = sum(max(0, b-a-58) for a, b in zip(bounds, bounds[1:]))
        compact = max(0, r['rows']-58)
        rows.append(dict(recording_id=r['source_recording_id'], id=r['id'],
                         root=r['root'], frames=r['rows'], cuts=len(p['cuts']),
                         compact_windows=compact, continuous_windows=legal,
                         excluded=compact-legal))
    if len({r['recording_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated original recording')
    if any('/no-block-one/' not in r['root'] for r in rows):
        raise ValueError('Non-single-pick root in frozen training data')
    tasks = sorted({text for r in manifest['records'] for text in r['tasks'].values()})
    totals = {k: sum(r[k] for r in rows) for k in
              ('frames', 'cuts', 'compact_windows', 'continuous_windows', 'excluded')}
    if totals['continuous_windows'] != index.total:
        raise ValueError('Sampler/accounting mismatch')
    return dict(sources=len(rows), sources_without_windows=sum(r['continuous_windows']==0 for r in rows),
                tasks=tasks, **totals), rows


def runtime_audit(config_path):
    import os
    import numpy as np
    import jax
    import optax
    from flax import nnx, traverse_util
    from openpi.models.pi0_flash import Pi0FlashConfig
    from openpi.training.flash_checkpoint import is_vlm_parameter
    from trc.models.flash_order_readout import ReadoutFlashConfig
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or jax.default_backend() != 'cpu':
        raise ValueError('Shape audit must leave GPUs untouched')
    cfg = json.loads(Path(config_path).read_text())
    for relative, digest in json.loads(Path(cfg['source_manifest']).read_text())['files'].items():
        if sha(Path(cfg['source_root']) / relative) != digest:
            raise ValueError('Pinned source mismatch: '+relative)
    if sha(cfg['training_contract']) != cfg['training_contract_sha256']:
        raise ValueError('Contract mismatch')
    contract = json.loads(Path(cfg['training_contract']).read_text())
    node = ast.parse(contract['model'], mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in ('Pi0FlashConfig', 'CoordinateFlashConfig'):
        raise ValueError('Unknown model configuration')
    kwargs = {v.arg: ast.literal_eval(v.value) for v in node.keywords}
    if kwargs.pop('coordinate_mode', 'absolute') != 'absolute':
        raise ValueError('Absolute required')
    base = Pi0FlashConfig(**{**kwargs, 'freeze_vlm': True, 'dtype': 'float32'})
    result = {}
    original = None
    for trial, mode in [('B1', 'none'), ('B2', 'unified'), ('B3', 'order_specific')]:
        modelcfg = ReadoutFlashConfig.from_base(base, mode)
        m = nnx.eval_shape(modelcfg.create, jax.random.key(42))
        train = nnx.state(m, modelcfg.get_trainable_filter())
        flat = traverse_util.flatten_dict(train.to_pure_dict())
        frozen = traverse_util.flatten_dict(nnx.state(m, nnx.All(nnx.Param, modelcfg.get_freeze_filter())).to_pure_dict())
        assert not any(is_vlm_parameter(k) for k in flat)
        assert all(is_vlm_parameter(k) for k in frozen)
        if original is None:
            original = flat
        assert all(k in flat and flat[k].shape == v.shape and flat[k].dtype == v.dtype for k, v in original.items())
        extra = set(flat)-set(original)
        assert extra == (set() if mode=='none' else {('action_out_proj','adapter_down'), ('action_out_proj','adapter_up')})
        tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(2.5e-6, b1=.9, b2=.95, eps=1e-8, weight_decay=1e-10))
        adam = jax.eval_shape(tx.init, train)
        # Same flatten order and optimizer construction as the executed CPU smoke.
        expected = jax.tree.leaves((train, adam))
        checkpoint = Path(cfg['output']) / (trial+'_expert_adam_step5.npz')
        headers = []
        with zipfile.ZipFile(checkpoint) as archive:
            assert len(archive.namelist()) == len(expected)
            for i, leaf in enumerate(expected):
                with archive.open(f'leaf{i}.npy') as stream:
                    version = np.lib.format.read_magic(stream)
                    shape, _, dtype = np.lib.format._read_array_header(stream, version)
                    assert shape == leaf.shape and dtype == leaf.dtype, (trial, i, shape, leaf.shape)
                    if shape == ():
                        assert int(np.frombuffer(stream.read(), dtype=dtype)[0]) == 5
                    headers.append(dict(leaf=i, shape=list(shape), dtype=str(dtype)))
        result[trial] = dict(mode=mode, trainable_parameters=sum(v.size for v in flat.values()),
            frozen_parameters=sum(v.size for v in frozen.values()),
            trainable_leaves=[dict(path='/'.join(k), shape=list(v.shape), dtype=str(v.dtype), elements=v.size) for k,v in sorted(flat.items())],
            frozen_leaves=[dict(path='/'.join(k), shape=list(v.shape), dtype=str(v.dtype)) for k,v in sorted(frozen.items())],
            optimizer='one clip-global-norm + AdamW over ALL trainable Expert and adapter leaves; no separate adapter group',
            checkpoint_headers=headers, checkpoint_bytes=checkpoint.stat().st_size,
            checkpoint_header_identity_only=True, adam_step=5,
            original_expert_paths_shapes_dtypes_equal=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path)
    p.add_argument('--cuts', type=Path)
    p.add_argument('--runtime-config', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.runtime_config:
        result = runtime_audit(args.runtime_config)
        (args.output/'parameter_optimizer_audit.json').write_text(json.dumps(result, indent=2))
    else:
        summary, rows = window_audit(json.loads(args.manifest.read_text()), json.loads(args.cuts.read_text()))
        summary.update(manifest_file_sha256=sha(args.manifest), cuts_file_sha256=sha(args.cuts))
        (args.output/'window_accounting.json').write_text(json.dumps(summary, indent=2))
        with (args.output/'window_accounting.csv').open('w') as stream:
            w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
