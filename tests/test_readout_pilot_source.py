import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('readout_pilot',Path(__file__).parents[1]/'scripts/run_readout_pilot.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReadoutPilotSourceTests(unittest.TestCase):
    def test_only_sealed_step_is_loaded_without_main_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.py'
            path.write_text("raise RuntimeError('main must not run')\ndef step(x):\n    return x+1\n")
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(runner.load_step(path,digest,{})(3),4)
            path.write_text('def step(x):\n    return x-1\n')
            with self.assertRaises(ValueError):runner.load_step(path,digest,{})

    def test_missing_step_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.py';path.write_text('def other(): pass\n')
            with self.assertRaises(ValueError):runner.load_step(path,runner.sha(path),{})
