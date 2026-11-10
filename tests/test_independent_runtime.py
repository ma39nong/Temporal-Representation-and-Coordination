"""No OpenPI import or GPU required for saved-config parser checks."""
import unittest
from scripts.evaluate_fullft_paired_source import model_kwargs_from_contract


class IndependentRuntime(unittest.TestCase):
    def test_literal_model_only(self):
        result=model_kwargs_from_contract("CoordinateFlashConfig(action_dim=58, dtype='bfloat16', action_q01=(0.,1.), coordinate_mode='state_relative')")
        self.assertEqual(result,{'action_dim':58,'dtype':'bfloat16','action_q01':(0.,1.)})

    def test_never_execute_untrusted_config_repr(self):
        for text in ["__import__('os').system('false')", "Pi0FlashConfig(dtype=str(1))",
                     "Pi0FlashConfig(**{})", "Pi0FlashConfig(58)", "Pi0FlashConfig(a=1,a=2)"]:
            with self.subTest(text=text), self.assertRaises((ValueError,SyntaxError)):
                model_kwargs_from_contract(text)
