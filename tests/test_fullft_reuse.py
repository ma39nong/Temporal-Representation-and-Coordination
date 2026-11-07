import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class FullftReuse(unittest.TestCase):
    def test_help_needs_no_original_runtime_or_gpu(self):
        for name in ('audit_fullft_checkpoint_source.py','report_fullft_reuse.py','report_p1_transfer.py','check_transfer_step0.py'):
            r=subprocess.run([sys.executable,str(ROOT/'scripts'/name),'--help'],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)

    def test_synthetic_export_cannot_be_reported_as_real(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);export=root/'export';export.mkdir()
            meta=export/'metadata.json';meta.write_text(json.dumps({'data_kind':'synthetic'}))
            arrays=export/'aligned.npz';arrays.write_bytes(b'not a real trajectory')
            (export/'seal.json').write_text(json.dumps({'metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest(),
                                                      'aligned_sha256':hashlib.sha256(arrays.read_bytes()).hexdigest()}))
            review=root/'review.json';review.write_text('{"records":[]}')
            out=root/'report'
            r=subprocess.run([sys.executable,str(ROOT/'scripts/report_fullft_reuse.py'),'--exports',str(export),
                              '--review',str(review),'--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('Physical paired FP32 real exports required',r.stderr)
            self.assertFalse(out.exists())
