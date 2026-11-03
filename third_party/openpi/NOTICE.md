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
