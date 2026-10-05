# Dataset Validator Self-test

- Validator engineering status: **PASS**
- Data-governance test suite: **6/6 PASS**
- Covered: valid manifest freeze, actual dimensions/SHA, UNKNOWN rejection, duplicate content, path traversal, deterministic perturbation naming and label inheritance.
- Current Core manifest status: **DATA_PENDING_EMPTY_DATASET**

`DATASET_VALIDATION.json` correctly rejects the empty ground-truth template with `EMPTY_DATASET`. This is an expected data-gap result, not a validator implementation failure. A formal Core manifest must never pass with zero samples.
