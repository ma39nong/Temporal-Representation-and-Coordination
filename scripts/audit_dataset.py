#!/usr/bin/env python3
"""Compatibility wrapper for the repository-local trc audit command."""
import sys
from trc.cli import main

if __name__ == '__main__':
    args = sys.argv[1:]
    if '--config' not in args:
        args.extend(['--config', 'configs/p0_data_audit.yaml'])
    if '--output' not in args:
        args.extend(['--output', 'outputs/audit.json'])
    raise SystemExit(main(['audit', *args]))
