"""Explicit storage → physical canonical → training target → physical decoding."""
from dataclasses import dataclass
import numpy as np
from .action_schema import canonicalize
from .assets import NormalizationAssets
from .normalization import normalize, arm_bridge
from .semantics import transform_actions


@dataclass(frozen=True)
class ActionCodec:
    assets: NormalizationAssets
    layout: str
    arm_semantics: str = 'delta'
    require_verified: bool = True

    def __post_init__(self):
        if self.layout not in ('canonical58', 'converter58') or self.arm_semantics not in ('delta', 'absolute'):
            raise ValueError('Explicit supported layout and arm semantics required')
        if self.require_verified and self.assets.provenance_status != 'verified':
            raise ValueError('Codec requires dataset-bound, explicitly pinned transform contract')
        # The audited T3 source contract always uses delta arms.
        if self.arm_semantics != 'delta':
            raise ValueError('T3 codec supports the audited delta-arms/absolute-hands contract only')

    def _bridge(self, x, inverse=False):
        if self.assets.metadata['mode'] == 'global_quantile':
            return x.copy()
        stats = self.assets.stats['actions']
        return arm_bridge(x, q01=stats['q01'], q99=stats['q99'],
                          arm_mean=self.assets.arrays['arm_mean'], arm_std=self.assets.arrays['arm_std'],
                          std_floor=self.assets.metadata['std_floor'], inverse=inverse)

    def encode(self, storage_actions, storage_anchor):
        physical = canonicalize(storage_actions, self.layout)
        anchor = canonicalize(storage_anchor, self.layout, state=True)
        if physical.shape[-2:] != tuple(self.assets.metadata['shape']):
            raise ValueError('Full configured action horizon required')
        delta = transform_actions(physical, anchor)
        target = self._bridge(normalize(delta, self.assets.stats['actions']))
        return {'state': normalize(anchor, self.assets.stats['state']), 'actions': target,
                'anchor_state': anchor, 'physical_actions': physical}

    def decode(self, model_actions, canonical_anchor):
        x = np.asarray(model_actions)
        if x.shape[-2:] != tuple(self.assets.metadata['shape']):
            raise ValueError('Full configured action horizon required')
        delta = normalize(self._bridge(x, inverse=True), self.assets.stats['actions'], inverse=True)
        return transform_actions(delta, canonical_anchor, inverse=True)

    def roundtrip(self, storage_actions, storage_anchor):
        encoded = self.encode(storage_actions, storage_anchor)
        decoded = self.decode(encoded['actions'], encoded['anchor_state'])
        return float(np.max(np.abs(decoded - encoded['physical_actions'])))
