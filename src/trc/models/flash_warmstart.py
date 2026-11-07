"""Optional strict full-parameter loader for the explicitly selected source runtime."""
from dataclasses import dataclass
import numpy as np


@dataclass
class StrictFlashWarmStart:
    params_path: str
    expected_sha256: str

    def load(self,reference):
        from flax import traverse_util
        from openpi.models import model
        from openpi.training.flash_checkpoint import parameter_digest
        loaded=model.restore_params(self.params_path,restore_type=np.ndarray)
        if parameter_digest(loaded)!=self.expected_sha256:raise ValueError('Warm-start parameter identity differs')
        ref,src=map(traverse_util.flatten_dict,(reference,loaded))
        if set(ref)!=set(src):raise ValueError('Full parameter paths differ; no random projection retention allowed')
        for key,v in src.items():
            if v.shape!=ref[key].shape or v.dtype!=ref[key].dtype:raise ValueError('Warm-start shape/dtype differs: '+str(key))
        return loaded
