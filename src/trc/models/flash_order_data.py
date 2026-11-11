"""Explicit source-runtime sampler adapter; no OpenPI dependency in core indexing."""
import hashlib,json
from pathlib import Path
from openpi.training.flash_native import NativeDataset
from trc.data.flash_sampling import ContinuousFlashIndex


class CutAwareNativeDataset(NativeDataset):
    def __init__(self,config,*,tokenizer=None,cut_manifest,expected_sha256):
        path=Path(cut_manifest)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('Training cut identity differs')
        super().__init__(config,tokenizer=tokenizer)
        self.cut_manifest_path=path
        self.safe_index=ContinuousFlashIndex(self.train,json.loads(path.read_text()),config.data_seed)

    def index(self,position):
        return self.safe_index.index(position)

    def data_config(self):
        result=super().data_config();original=result.save_extra_assets
        def extra(destination):
            import shutil
            original(destination)
            shutil.copyfile(self.cut_manifest_path,Path(destination)/'flash_training_cuts.json')
        result.save_extra_assets=extra
        return result
