#!/usr/bin/env python3
"""Create original offline tracking metadata for initialized zero-update runs."""
import argparse,hashlib,json,os,pickle,site
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());site.addsitedir(cfg['research_runtime'])
    from openpi.training import tracking
    root=Path(cfg['output_root']);os.environ['WANDB_MODE']='offline';os.environ['WANDB_DIR']=str(root/'wandb')
    for trial in ('B0','B2'):
        audit=root/'execution'/trial;sealed=audit/'sealed_config.pkl';identity=json.loads((audit/'execution_identity.json').read_text())
        if hashlib.sha256(sealed.read_bytes()).hexdigest()!=identity['sealed_config_sha256']:raise ValueError('Sealed config differs')
        with sealed.open('rb') as stream:c=pickle.load(stream)
        ckpt=Path(c.checkpoint_dir);log=ckpt/'flash_metrics/training.jsonl'
        if (ckpt/'wandb_id.txt').exists() or (log.exists() and log.read_text().strip()):raise ValueError('Tracking exists or updates already executed')
        if not json.loads((ckpt/'0/_CHECKPOINT_METADATA').read_text()).get('commit_timestamp_nsecs'):raise ValueError('No committed Step0')
        with tracking.session(c,resuming=False,start_step=0):pass
        (audit/'tracking_initialized_before_updates.json').write_text(json.dumps({'source_api':'tracking.session','mode':'offline','accepted_updates':0,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
