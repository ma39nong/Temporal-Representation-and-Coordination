#!/usr/bin/env python3
"""Repeatable evaluation of authenticated real exports, without OpenPI dependencies."""
import argparse
import json
from trc.analysis.flash.report import build_report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',required=True);p.add_argument('--output',required=True)
    p.add_argument('--no-figures',action='store_true');args=p.parse_args()
    report=build_report(args.export,args.output,make_figures=not args.no_figures)
    print(json.dumps({'status':report['status'],'full_action': [r for r in report['errors'] if r['group']=='full_action']},indent=2))
