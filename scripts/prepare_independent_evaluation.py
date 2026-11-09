#!/usr/bin/env python3
"""Prepare numeric window plan only when new reviewed recordings are supplied."""
import argparse
import json
from pathlib import Path
from trc.config import read_config, protected_roots
from trc.data.manifest import build_manifest
from trc.data.independent_evaluation import prepare_plan
from trc.io import write_report


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); cfg = json.loads(Path(a.config).read_text())
    if Path(a.output).exists(): raise FileExistsError(a.output)
    def read(key): return json.loads(Path(cfg[key]).read_text())
    protected = [a.config]
    if not all(cfg.get(k) for k in ('collection_config', 'reviews_path', 'exclusion_registry_path')):
        report = {'status': 'BLOCKED_NEW_REVIEWED_DATA_NOT_CONFIGURED', 'real_data_read': False,
                  'formal_independent_test_locked': False,
                  'reason': 'New original-recording-disjoint successful tasks, review evidence and full exclusion registry not supplied. Old20 candidates are ineligible.'}
    else:
        collection = read_config(cfg['collection_config']); manifest = build_manifest(collection)
        protected += [*protected_roots(collection), cfg['reviews_path'], cfg['exclusion_registry_path'], cfg['collection_config']]
        report = prepare_plan(collection, manifest, read('reviews_path'), read('exclusion_registry_path'), cfg['protocol'])
    write_report(a.output, report, protected=protected)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
