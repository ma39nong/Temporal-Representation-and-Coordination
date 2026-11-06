# OpenPI / tomato-openpi provenance

Apache-2.0 license text is preserved in LICENSE in this directory.

The local P0 APIs adapt numerical algorithms and contracts from tomato-openpi:

- Commit 2ab68c8b8090284fca6d258f86bf5098d2f1302b, src/openpi/shared/delta_paper.py: bundle validation and arm normalization bridge.
- Same commit, src/openpi/training/norm_provenance.py: dataset content/transform binding validation.
- Same commit, src/openpi/training/dataset_collection.py: ordered collection metadata and feature consistency.
- main commit 4d0cb07e02181ed3bcdd876cad69446b22cfb075, src/openpi/policies/fr3_sharpa_multimodal_policy.py and src/openpi/transforms.py: action/state layouts, anchor deltas, normalization formulas.

Adaptations: repository-local imports, strict configuration, independent NumPy and
optional PyArrow APIs, dataset fingerprint checks without OpenPI model/config
imports, explicit offline provenance and JSON reports. No driver/model/training
modules or data assets are copied. Source names and commits are historical
provenance, not runtime paths or dependencies. No new copyright holder is inferred.

FLASH attribution integration (2026-11-04): an explicit, optional process adapter
reuses the original paper-flash-test runtime at e64c9a41ecb24494ea6c81e2df3dd4bbbfdcc45c:
shared/flash_polynomial.py, training/flash_data.py, flash_native.py,
models/pi0_flash.py, model.py and scripts/flash/evaluate_p25.py. No original
FLASH model or polynomial implementation is copied/reimplemented in trc. The
standalone research code computes metrics, authenticates exported predictions
and calls original APIs in their configured environment. Original math lineage
from 534ed023542a20dad55e1827b0a5adad9b760d4d differs only in dataclass eq flag
for flash_polynomial.py; target/history/reconstruction equations are unchanged.

P1-1 extension: `src/trc/models/flash_openpi.py` subclasses the original
`Pi0Flash`/`Pi0FlashConfig` and adapts the small coefficient-loss/integration
routines from tomato-openpi `paper-flash-test@d208d414efbb78483f44bfef4c2a9314115dd632`
(`src/openpi/models/pi0_flash.py`). The unchanged backbone, polynomial fitting,
optimizer and FSDP trainer remain explicit source-runtime dependencies. Only
coordinate transforms and experiment guards are added. OpenPI's Apache-2.0
license remains in this directory; no original copyright notices were removed.
