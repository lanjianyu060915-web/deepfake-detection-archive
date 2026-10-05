# Automated Test Summary

| Suite | Result | Tests | Notes |
| --- | --- | ---: | --- |
| Data governance | PASS | 6/6 | manifest freeze, SHA, validator, duplicate ID/content/path rejection, UNKNOWN rejection, deterministic perturbations, naming and label inheritance |
| Evaluator/stub | PASS | 7/7 | known formulas, strict labels, zero denominator, conditional AUC, grouping/parent aggregation, all success/failure stub branches |
| Frozen baseline verification | PASS | 50/50 | read-only SHA/size/config verification; no model import or inference |

Evaluator fixture: TP=3, FP=1, TN=4, FN=2, N=10, Accuracy=0.7, Precision=0.75, Recall=0.6, Specificity=0.8, F1=0.6666666666666666, confirmed-semantics AUC=0.76.

Stub modes exercised: `all_correct`, `partial_wrong`, `all_real`, `all_fake`, `crash`, `invalid_json`, `status_error`, `timeout`, plus `no_output` and `missing_label`. Every system failure retained the fixed denominator.

No real deepfake model was run.
