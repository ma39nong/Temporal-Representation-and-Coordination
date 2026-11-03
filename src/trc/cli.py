"""One portable entry point for offline P0 audit/identity/normalization checks."""
import argparse
import json
import sys
from trc.config import read_config, protected_roots
from trc.io import write_report
from trc.data.manifest import build_manifest, recording_split
from trc.project import audit_project, project_assets, asset_report, check_roundtrip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'manifest', 'split', 'assets-check', 'roundtrip'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-fraction', type=float, default=.8)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    parser.add_argument('--test-fraction', type=float, default=0.)
    parser.add_argument('--max-windows', type=int, default=3)
    args = parser.parse_args(argv)
    cfg = None
    try:
        cfg = read_config(args.config)
        if args.command == 'audit':
            report = audit_project(cfg)
        elif args.command == 'manifest':
            report = {'status': 'passed', **build_manifest(cfg)}
        elif args.command == 'split':
            manifest = build_manifest(cfg)
            report = {'status': 'passed', **recording_split(manifest, seed=args.seed, fractions={
                'train': args.train_fraction, 'validation': args.validation_fraction, 'test': args.test_fraction})}
        elif args.command == 'assets-check':
            # With no dataset mount, integrity can still be checked but cannot be called verified.
            from trc.config import dataset_specs
            manifest = build_manifest(cfg) if dataset_specs(cfg) else None
            report = asset_report(project_assets(cfg, manifest))
        else:
            report = check_roundtrip(cfg, max_windows=args.max_windows)
        import platform
        import numpy as np
        from importlib.metadata import version
        report['runtime'] = {'python': platform.python_version(), 'numpy': np.__version__, 'trc': version('trc-research')}
        write_report(args.output, report, protected=[*protected_roots(cfg), args.config])
        print(json.dumps({k: report[k] for k in ('status', 'real_data_read', 'observed_frames', 'observed_episodes', 'vector_contract_ready', 'formal_data_ready') if k in report}, ensure_ascii=False))
        return 0 if report['status'] in ('passed', 'passed_with_warnings', 'verified') else 2
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, AssertionError) as exc:
        report = {'status': 'blocked', 'command': args.command, 'errors': [str(exc)]}
        # Never overwrite a configured input even on failure.
        if cfg is not None:
            try:
                write_report(args.output, report, protected=[*protected_roots(cfg), args.config])
            except (OSError, ValueError):
                pass
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        return 2
