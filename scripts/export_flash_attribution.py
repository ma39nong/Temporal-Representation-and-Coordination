#!/usr/bin/env python3
"""Explicit fixed-source process adapter. No external imports in the core package."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from trc.io import file_sha256,write_report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    path=Path(args.config).resolve();cfg=json.loads(path.read_text());out=Path(args.output).resolve()
    source=Path(cfg['source_root']).expanduser().resolve();runtime=Path(cfg['runtime']).expanduser().absolute()  # preserve venv executable symlink
    if out.exists(): raise FileExistsError('Never overwrite earlier experiment output')
    for key in ('data_dir','checkpoint','predictions','audit','plan','training_contract','checkpoint_reference'):
        if cfg.get(key):
            protected=Path(cfg[key]).expanduser().resolve()
            if out==protected or out.is_relative_to(protected):
                raise ValueError('Output must be outside read-only experiment inputs')
    if cfg.get('mode') not in ('cached','native'):
        raise ValueError('mode must be cached or native')
    if out.is_relative_to(source): raise ValueError('Export must not modify original repository')
    # Opt-in source/runtime paths are configuration, not implicit package dependencies.
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS=cfg.get('jax_platforms','cpu'))
    env['PYTHONPATH']=os.pathsep.join(str(source/v) for v in ('src','.','packages/openpi-client/src'))
    env['MPLCONFIGDIR']=str(out.parent/'matplotlib_cache')
    bridge=Path(__file__).resolve().with_name('flash_source_bridge.py')
    try:
        subprocess.run([str(runtime),str(bridge),'--config',str(path),'--output',str(out)],cwd=source,env=env,check=True)
    except subprocess.CalledProcessError:
        if out.exists():
            write_report(out/'blocked.json',{'status':'blocked','reason':'Original-source adapter failed; no completed export',
                         'config_sha256':file_sha256(path)},protected=[source,path])
        raise
    write_report(out/'experiment_config.json',cfg,protected=[source,path])
    seal={'aligned_sha256':file_sha256(out/'aligned.npz'),'metadata_sha256':file_sha256(out/'metadata.json'),
          'bridge_sha256':file_sha256(bridge),'config_sha256':file_sha256(path)}
    write_report(out/'seal.json',seal,protected=[source,path])
