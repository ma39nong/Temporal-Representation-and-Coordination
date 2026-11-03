#!/usr/bin/env python3
"""Config is JSON (also valid YAML 1.2), avoiding an extra YAML dependency."""
import argparse
import json
from pathlib import Path
from trc.data.validation import audit_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('configs/p0_data_audit.yaml'))
    p.add_argument('--output', type=Path, default=Path('outputs/audit.json'))
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    for key in ('dataset_root', 'norm_stats_path'):
        if config.get(key):
            config[key] = str((args.config.resolve().parent / config[key]).resolve())
    report = audit_dataset(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'real_data_read', 'observed_frames', 'observed_episodes')}, ensure_ascii=False))
    return 0 if report['status'] == 'passed_with_warnings' else 2


if __name__ == '__main__':
    raise SystemExit(main())
