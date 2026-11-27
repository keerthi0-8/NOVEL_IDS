# NOVEL-IDS saved source-test evaluation report

Built only from frozen Phase C saved prediction CSVs and results.json. No model inference was run; source test partitions remain closed.

Confusion matrices and classification metrics use forced predicted_family classes, including rows selectively marked Uncertain or Unfamiliar. Selective decisions are shown separately.

## nsl_kdd_test_independent

Accuracy 78.9671%; macro precision 84.0600%; macro recall 59.4120%; macro F1 62.7273%.
Weighted precision 82.6533%; weighted recall 78.9671%; weighted F1 76.5124%; rows 21,880.
Selective decisions: {"Known_attack": 7453, "Normal": 10965, "Unfamiliar": 3462}
OOD: {"aupr": 0.5778624957725649, "auroc": 0.8799609827515754, "fpr_at_95_tpr": 0.3583815028901734, "positive_rows": 3715, "test_only_attack_rows": 3715}

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 72.34% | 96.74% | 82.78% | 9,606 |
| DoS | 93.64% | 79.19% | 85.81% | 7,060 |
| Probe_Recon | 73.31% | 78.04% | 75.60% | 2,291 |
| UNSW_Exploits | — | — | — | 0 |
| UNSW_Fuzzers | — | — | — | 0 |
| UNSW_Generic | — | — | — | 0 |
| UNSW_Analysis | — | — | — | 0 |
| UNSW_Backdoor | — | — | — | 0 |
| UNSW_Shellcode | — | — | — | 0 |
| UNSW_Worms | — | — | — | 0 |
| NSL_R2L | 97.69% | 20.69% | 34.15% | 2,856 |
| NSL_U2R | 83.33% | 22.39% | 35.29% | 67 |


## nsl_kdd_test_full

Accuracy 79.3781%; macro precision 84.3618%; macro recall 59.7993%; macro F1 63.0586%.
Weighted precision 82.9654%; weighted recall 79.3781%; weighted F1 76.9644%; rows 22,544.
Selective decisions: {"Known_attack": 7971, "Normal": 11077, "Unfamiliar": 3496}
OOD: {"aupr": 0.5711614079229518, "auroc": 0.8772566209073817, "fpr_at_95_tpr": 0.3676705331488773, "positive_rows": 3750, "test_only_attack_rows": 3750}

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 72.36% | 96.59% | 82.74% | 9,711 |
| DoS | 93.77% | 80.31% | 86.52% | 7,460 |
| Probe_Recon | 74.66% | 79.22% | 76.87% | 2,421 |
| UNSW_Exploits | — | — | — | 0 |
| UNSW_Fuzzers | — | — | — | 0 |
| UNSW_Generic | — | — | — | 0 |
| UNSW_Analysis | — | — | — | 0 |
| UNSW_Backdoor | — | — | — | 0 |
| UNSW_Shellcode | — | — | — | 0 |
| UNSW_Worms | — | — | — | 0 |
| NSL_R2L | 97.69% | 20.49% | 33.87% | 2,885 |
| NSL_U2R | 83.33% | 22.39% | 35.29% | 67 |


## unsw_nb15_test_independent

Accuracy 76.1692%; macro precision 47.4386%; macro recall 51.4332%; macro F1 46.0816%.
Weighted precision 83.3666%; weighted recall 76.1692%; weighted F1 78.6651%; rows 73,791.
Selective decisions: {"Known_attack": 35776, "Normal": 25971, "Uncertain": 6418, "Unfamiliar": 5626}
OOD: {"aupr": null, "auroc": null, "fpr_at_95_tpr": null, "positive_rows": 0, "test_only_attack_rows": 0}

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 94.36% | 74.02% | 82.96% | 35,528 |
| DoS | 21.15% | 39.62% | 27.58% | 2,085 |
| Probe_Recon | 63.10% | 82.61% | 71.55% | 2,358 |
| UNSW_Exploits | 72.73% | 76.47% | 74.56% | 9,330 |
| UNSW_Fuzzers | 32.33% | 56.18% | 41.05% | 5,801 |
| UNSW_Generic | 99.99% | 95.96% | 97.93% | 17,149 |
| UNSW_Analysis | 0.00% | 0.00% | 0.00% | 552 |
| UNSW_Backdoor | 25.84% | 3.97% | 6.89% | 579 |
| UNSW_Shellcode | 23.69% | 69.59% | 35.35% | 365 |
| UNSW_Worms | 41.18% | 15.91% | 22.95% | 44 |
| NSL_R2L | — | — | — | 0 |
| NSL_U2R | — | — | — | 0 |


## unsw_nb15_test_full

Accuracy 75.0170%; macro precision 48.7564%; macro recall 51.8085%; macro F1 46.9846%.
Weighted precision 81.8966%; weighted recall 75.0170%; weighted F1 77.3766%; rows 82,332.
Selective decisions: {"Known_attack": 38874, "Normal": 26959, "Uncertain": 10743, "Unfamiliar": 5756}
OOD: {"aupr": null, "auroc": null, "fpr_at_95_tpr": null, "positive_rows": 0, "test_only_attack_rows": 0}

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 94.52% | 73.75% | 82.85% | 37,000 |
| DoS | 31.80% | 53.88% | 40.00% | 4,089 |
| Probe_Recon | 71.09% | 80.89% | 75.68% | 3,496 |
| UNSW_Exploits | 68.79% | 69.37% | 69.08% | 11,132 |
| UNSW_Fuzzers | 31.13% | 54.06% | 39.51% | 6,062 |
| UNSW_Generic | 99.99% | 96.18% | 98.05% | 18,871 |
| UNSW_Analysis | 0.00% | 0.00% | 0.00% | 677 |
| UNSW_Backdoor | 25.84% | 3.95% | 6.85% | 583 |
| UNSW_Shellcode | 23.23% | 70.11% | 34.89% | 378 |
| UNSW_Worms | 41.18% | 15.91% | 22.95% | 44 |
| NSL_R2L | — | — | — | 0 |
| NSL_U2R | — | — | — | 0 |


## pooled_test_independent

Accuracy 76.8091%; macro precision 58.0652%; macro recall 49.2126%; macro F1 47.6658%.
Weighted precision 80.9509%; weighted recall 76.8091%; weighted F1 77.6792%; rows 95,671.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 87.41% | 78.86% | 82.91% | 45,134 |
| DoS | 64.98% | 70.17% | 67.47% | 9,145 |
| Probe_Recon | 67.61% | 80.36% | 73.43% | 4,649 |
| UNSW_Exploits | 72.73% | 76.47% | 74.56% | 9,330 |
| UNSW_Fuzzers | 32.33% | 56.18% | 41.05% | 5,801 |
| UNSW_Generic | 99.99% | 95.96% | 97.93% | 17,149 |
| UNSW_Analysis | 0.00% | 0.00% | 0.00% | 552 |
| UNSW_Backdoor | 25.84% | 3.97% | 6.89% | 579 |
| UNSW_Shellcode | 23.69% | 69.59% | 35.35% | 365 |
| UNSW_Worms | 41.18% | 15.91% | 22.95% | 44 |
| NSL_R2L | 97.69% | 20.69% | 34.15% | 2,856 |
| NSL_U2R | 83.33% | 22.39% | 35.29% | 67 |


## pooled_test_full

Accuracy 75.9545%; macro precision 57.7374%; macro recall 48.5081%; macro F1 47.1142%.
Weighted precision 79.9829%; weighted recall 75.9545%; weighted F1 76.8291%; rows 104,876.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 87.65% | 78.50% | 82.82% | 46,711 |
| DoS | 61.53% | 70.95% | 65.91% | 11,549 |
| Probe_Recon | 72.49% | 80.21% | 76.16% | 5,917 |
| UNSW_Exploits | 68.79% | 69.37% | 69.08% | 11,132 |
| UNSW_Fuzzers | 31.13% | 54.06% | 39.51% | 6,062 |
| UNSW_Generic | 99.99% | 96.18% | 98.05% | 18,871 |
| UNSW_Analysis | 0.00% | 0.00% | 0.00% | 677 |
| UNSW_Backdoor | 25.84% | 3.95% | 6.85% | 583 |
| UNSW_Shellcode | 23.23% | 70.11% | 34.89% | 378 |
| UNSW_Worms | 41.18% | 15.91% | 22.95% | 44 |
| NSL_R2L | 97.69% | 20.49% | 33.87% | 2,885 |
| NSL_U2R | 83.33% | 22.39% | 35.29% | 67 |


## Cross-dataset confusion

- nsl_kdd test_independent: 0/21,880 (0.000%) predicted into the other dataset's exclusive label space.
- nsl_kdd test_full: 0/22,544 (0.000%) predicted into the other dataset's exclusive label space.
- unsw_nb15 test_independent: 34,861/73,791 (47.243%) predicted into the other dataset's exclusive label space.
- unsw_nb15 test_full: 39,776/82,332 (48.312%) predicted into the other dataset's exclusive label space.

## Weak-class predicted labels

### nsl_kdd test_independent
- NSL_R2L (n=2856): Normal=2244, NSL_R2L=591, Probe_Recon=20
- NSL_U2R (n=67): Normal=40, NSL_U2R=15, NSL_R2L=12
### nsl_kdd test_full
- NSL_R2L (n=2885): Normal=2273, NSL_R2L=591, Probe_Recon=20
- NSL_U2R (n=67): Normal=40, NSL_U2R=15, NSL_R2L=12
### unsw_nb15 test_independent
- UNSW_Analysis (n=552): DoS=463, UNSW_Exploits=44, Normal=42
- UNSW_Backdoor (n=579): DoS=469, UNSW_Exploits=71, UNSW_Backdoor=23
- UNSW_Worms (n=44): UNSW_Exploits=32, UNSW_Worms=7, UNSW_Shellcode=2
- UNSW_Fuzzers (n=5801): UNSW_Fuzzers=3259, Normal=1169, DoS=931
- UNSW_Shellcode (n=365): UNSW_Shellcode=254, Probe_Recon=50, UNSW_Fuzzers=34
### unsw_nb15 test_full
- UNSW_Analysis (n=677): DoS=546, UNSW_Exploits=86, Normal=42
- UNSW_Backdoor (n=583): DoS=469, UNSW_Exploits=75, UNSW_Backdoor=23
- UNSW_Worms (n=44): UNSW_Exploits=32, UNSW_Worms=7, UNSW_Shellcode=2
- UNSW_Fuzzers (n=6062): UNSW_Fuzzers=3277, Normal=1180, DoS=1102
- UNSW_Shellcode (n=378): UNSW_Shellcode=265, Probe_Recon=51, UNSW_Fuzzers=35

## Decision statistics and coverage

### nsl_kdd test_independent
Decision counts: {"Known_attack": 7453, "Normal": 10965, "Unfamiliar": 3462}
| Coverage | Accuracy among highest-confidence rows | Rows |
|---:|---:|---:|
| 10% | 100.000% | 2,188 |
| 20% | 98.880% | 4,376 |
| 30% | 98.537% | 6,564 |
| 40% | 98.572% | 8,752 |
| 50% | 98.528% | 10,940 |
| 60% | 98.317% | 13,128 |
| 70% | 93.980% | 15,315 |
| 80% | 87.649% | 17,504 |
| 90% | 82.749% | 19,692 |
| 100% | 78.967% | 21,880 |

### nsl_kdd test_full
Decision counts: {"Known_attack": 7971, "Normal": 11077, "Unfamiliar": 3496}
| Coverage | Accuracy among highest-confidence rows | Rows |
|---:|---:|---:|
| 10% | 100.000% | 2,254 |
| 20% | 98.935% | 4,508 |
| 30% | 98.581% | 6,763 |
| 40% | 98.580% | 9,017 |
| 50% | 98.501% | 11,272 |
| 60% | 98.248% | 13,526 |
| 70% | 94.392% | 15,780 |
| 80% | 88.179% | 18,035 |
| 90% | 83.149% | 20,289 |
| 100% | 79.378% | 22,544 |

### unsw_nb15 test_independent
Decision counts: {"Known_attack": 35776, "Normal": 25971, "Uncertain": 6418, "Unfamiliar": 5626}
| Coverage | Accuracy among highest-confidence rows | Rows |
|---:|---:|---:|
| 10% | 100.000% | 7,379 |
| 20% | 100.000% | 14,758 |
| 30% | 100.000% | 22,137 |
| 40% | 100.000% | 29,516 |
| 50% | 99.967% | 36,895 |
| 60% | 97.213% | 44,274 |
| 70% | 91.247% | 51,653 |
| 80% | 86.340% | 59,032 |
| 90% | 82.065% | 66,411 |
| 100% | 76.169% | 73,791 |

### unsw_nb15 test_full
Decision counts: {"Known_attack": 38874, "Normal": 26959, "Uncertain": 10743, "Unfamiliar": 5756}
| Coverage | Accuracy among highest-confidence rows | Rows |
|---:|---:|---:|
| 10% | 100.000% | 8,233 |
| 20% | 100.000% | 16,466 |
| 30% | 100.000% | 24,699 |
| 40% | 100.000% | 32,932 |
| 50% | 99.888% | 41,166 |
| 60% | 95.937% | 49,399 |
| 70% | 89.908% | 57,632 |
| 80% | 84.983% | 65,865 |
| 90% | 79.788% | 74,098 |
| 100% | 75.017% | 82,332 |

## How to read these numbers

Accuracy is dominated by large classes such as Normal and UNSW_Generic. Macro metrics weigh supported classes equally and are pulled down by weak rare classes. Support is the true sample count, not an error measure. The two datasets have different label spaces; pooled values are aggregate summaries only. NSL-KDD test includes attack types absent from KDDTrain+ training. The saved prediction files do not retain fine-grained attack subtype seen/test-only metadata, so the requested fine-type recall split cannot be computed. Baseline per-row probability/OOD/decision files are absent; HGB/MLP aggregate results exist. Plain Transformer test results and a sixth ablation are unavailable because test is closed.


## S2c — post-hoc validation-tuned class-prior decision layer

This decision layer / v2 head was designed after the primary model's test results were seen; its test numbers are not an unbiased estimate and are shown beside the original, pre-registered result.

Bias SHA-256: 95c395b55296e7a58b947d985629bebd22a9173243c2c353efb30a7d428119b7

| Dataset / split | View | Accuracy | Macro-F1 | Weighted-F1 | Selective coverage | Accepted accuracy |
|---|---|---:|---:|---:|---:|---:|
| nsl_kdd_test_independent | Original frozen decision layer | 78.9671% | 62.7273% | 76.5124% | 84.1773% | 85.7694% |
| nsl_kdd_test_independent | Post-hoc adjusted | 77.4086% | 56.9686% | 73.5766% | 84.1590% | 85.5110% |
| nsl_kdd_test_full | Original frozen decision layer | 79.3781% | 63.0586% | 76.9644% | 84.4925% | 86.0195% |
| nsl_kdd_test_full | Post-hoc adjusted | 77.8655% | 57.3264% | 74.1106% | 84.4748% | 85.7698% |
| unsw_nb15_test_independent | Original frozen decision layer | 76.1692% | 46.0816% | 78.6651% | 83.6782% | 82.3133% |
| unsw_nb15_test_independent | Post-hoc adjusted | 73.1729% | 46.3751% | 76.7160% | 85.0686% | 77.9284% |
| unsw_nb15_test_full | Original frozen decision layer | 75.0170% | 46.9846% | 77.3766% | 79.9604% | 82.6531% |
| unsw_nb15_test_full | Post-hoc adjusted | 72.3267% | 47.4809% | 75.6831% | 81.1750% | 78.5226% |


## Baseline comparisons and unavailable outputs

The saved aggregate independent-test HGB comparisons are:
- NSL-KDD unified label mode: HGB macro-F1 0.640996 versus NOVEL-IDS primary 0.627273. This comparison uses the same unified family label mode.
- UNSW-NB15 native label mode: HGB macro-F1 0.548124 versus NOVEL-IDS primary 0.460816.

The baseline validation results recorded in the final report are:

| Dataset | Label mode | HGB validation macro-F1 | MLP selected seed | MLP validation macro-F1 |
|---|---|---:|---:|---:|
| NSL-KDD | native | 0.8122 | 42 | 0.8170 |
| NSL-KDD | unified | 0.9719 | 44 | 0.9118 |
| UNSW-NB15 | native | 0.6183 | 44 | 0.5773 |
| UNSW-NB15 | unified | 0.6183 | 43 | 0.5732 |

Do not compare NSL native and unified label modes as if they were the same task. Baseline per-row test prediction/probability/OOD/decision files were not saved, so baseline selective coverage, accepted accuracy, risk-coverage curves, and per-row OOD metrics are unavailable. MLP test per-row outputs are unavailable in the saved Phase C artifact set. No plain Transformer test baseline artifact/checkpoint or sixth ablation exists in the closed test run. These items are unavailable, not zero.


## Additional saved Phase C metrics and baseline comparisons

The following macro one-vs-rest AUC values were present in the frozen `results.json` and are reported here from saved metrics only:

| Dataset / view | Macro ROC-AUC (OVR) | Macro PR-AUC (OVR) |
|---|---:|---:|
| NSL-KDD independent | 0.945973 | 0.728371 |
| NSL-KDD full | 0.947383 | 0.730809 |
| UNSW-NB15 independent | 0.964292 | 0.507606 |
| UNSW-NB15 full | 0.958511 | 0.504558 |

Saved aggregate baseline results (same frozen test artifacts; no baseline model was re-run):

| Dataset / view | Label mode | Model | Accuracy | Macro-F1 | Weighted-F1 | Notes |
|---|---|---|---:|---:|---:|---|
| NSL-KDD independent | Unified | HGB | 78.4872% | 64.0996% | 75.5750% | Direct unified-label comparison |
| NSL-KDD independent | Unified | MLP | 74.9269% | 56.2857% | 70.8302% | Direct unified-label comparison |
| NSL-KDD full | Unified | HGB | 78.9123% | 64.4076% | 76.0579% | Direct unified-label comparison |
| NSL-KDD full | Unified | MLP | 75.4524% | 56.7479% | 71.4672% | Direct unified-label comparison |
| UNSW-NB15 independent | Native/unified | HGB | 71.7066% | 54.8124% | 76.1917% | Native and unified mappings coincide in saved output |
| UNSW-NB15 independent | Native | MLP | 72.7839% | 45.3642% | 76.3824% | |
| UNSW-NB15 independent | Unified | MLP | 71.3664% | 45.1895% | 74.8422% | |
| UNSW-NB15 full | Native/unified | HGB | 68.6963% | 52.0531% | 73.9981% | Native and unified mappings coincide in saved output |
| UNSW-NB15 full | Native | MLP | 71.8894% | 46.1630% | 75.3607% | |
| UNSW-NB15 full | Unified | MLP | 70.6530% | 46.3622% | 73.9893% | |

NSL-KDD native-label HGB/MLP aggregate metrics are also saved, but the independent view excludes 3,715 unsupported raw test-label rows and the full view excludes 3,750. They are not directly comparable with the unified-label metrics or the primary unified classifier.

### Independent-view risk-coverage (forced predictions)

Risk is (1-\text{accuracy}) among the highest-confidence rows at each coverage level. These are ranking/coverage summaries from saved primary predictions, not selective decision-policy coverage.

| Coverage | NSL accuracy | NSL risk | UNSW accuracy | UNSW risk |
|---:|---:|---:|---:|---:|
| 10% | 100.000% | 0.000% | 100.000% | 0.000% |
| 20% | 98.880% | 1.120% | 100.000% | 0.000% |
| 30% | 98.537% | 1.463% | 100.000% | 0.000% |
| 40% | 98.572% | 1.428% | 100.000% | 0.000% |
| 50% | 98.528% | 1.472% | 99.967% | 0.033% |
| 60% | 98.317% | 1.683% | 97.213% | 2.787% |
| 70% | 93.980% | 6.020% | 91.247% | 8.753% |
| 80% | 87.649% | 12.351% | 86.340% | 13.660% |
| 90% | 82.749% | 17.251% | 82.065% | 17.935% |
| 100% | 78.967% | 21.033% | 76.169% | 23.831% |

OOD evaluation is available for the NSL-KDD test-only-attack proxy (independent AUROC 0.879961, AUPR 0.577862, FPR@95TPR 0.358382; full AUROC 0.877257, AUPR 0.571161, FPR@95TPR 0.367671). UNSW saved OOD positive support is zero, so AUROC/AUPR/FPR@95TPR are unavailable, not zero.


## 2026-11-27 owner-authorized continuation — source-format validation and CIC second run

### Phase C integrity reconciliation

See docs/PHASEC_INTEGRITY_NOTE.md. The recovery manifest SHA matches the value recorded in results.json; 25/26 recovery entries match, with the sole mismatch being the recovery evaluator file. All 24 saved prediction CSV hashes and the frozen model, Stage 5/6, taxonomy, label map, calibration and threshold artifacts match their recorded hashes. The original evaluator version was not recovered. Saved results predate the current evaluator file and remain hash-verified; exact replay from the current evaluator is not guaranteed. No source-test inference was rerun and no test features were reopened.

### Source-format demo diagnosis and correction

The previous six demo summaries had used the generic unseen-schema route, so the NSL-KDD and UNSW-NB15 source columns were not recognized as source schemas. The current source fingerprint normalizes BOM/whitespace/case and documented aliases, excludes label and dataset metadata columns, and requires an exact feature-set match to the stored source preprocessor. The source path reuses the training-fitted DatasetPreprocessor; it does not fit medians, scalers, or vocabularies on the 500-row upload. Unknown schemas retain semantic matching and target-only normalization.

| Raw source sample | Matched features | Compatibility | Trust | Decisions | Demo accuracy | Demo macro-F1 |
|---|---:|---:|---|---|---:|---:|
| NSL-KDD labeled | 41/41 | 1.000 | High | Normal 261; Known attack 227; Unfamiliar 12 | 1.000 | 1.000 |
| NSL-KDD unlabeled | 41/41 | 1.000 | High | Normal 261; Known attack 227; Unfamiliar 12 | — | — |
| UNSW-NB15 labeled | 42/42 | 1.000 | High | Known attack 244; Normal 142; Uncertain 79; Unfamiliar 35 | 0.792 | 0.5907 |
| UNSW-NB15 unlabeled | 42/42 | 1.000 | High | Known attack 244; Normal 142; Uncertain 79; Unfamiliar 35 | — | — |

These accuracy and macro-F1 values are 500-row validation demo-sample metrics, not official test results. Labeled and unlabeled prediction outputs are identical for each pair. The sample generator reads raw validation rows with original names/values; it did not use processed/unified artifacts, so the source sample files and sample manifest were not regenerated.

Raw-CSV API probability parity against saved Phase-E source validation probabilities, applying the already-shipped additive class-prior bias in log-probability space:
- NSL-KDD: maximum absolute probability difference 2.47e-7; maximum OOD difference 5.71e-6.
- UNSW-NB15: maximum absolute probability difference 5.73e-7; maximum OOD difference 4.72e-6.
Both probability comparisons pass the 1e-5 limit. The first comparison attempt used the wrong arithmetic for the bias; the corrected log-probability-space comparison is the reported result.

A CIC metadata numeric-column regression was found during the first full Option A attempt: an object-dtype calibration column reached the numeric normalizer. The unseen-schema path now uses external metadata's declared numeric type and coerces both calibration and prediction chunks consistently; malformed tokens become missing/masked values. No frozen model or source preprocessing artifact changed. Focused suite: 18 passed before this correction; 21 passed in 9.73s after the new regression test.

### CIC-DDoS2019 second run — Option A headline

The prepared sample contains 137,312 rows, 86 raw feature columns, seed 42, and a 15,000-row per-label cap. Free disk was 16.88 GiB before starting, so the full prepared sample was used. The feature file, separate evaluation-label file, and mapping hashes match the pre-run manifest and the mapping hash recorded in PROGRESS.md. The first 162,276-row CIC run remains preserved. Labels were not opened until after Option A inference completed.

| Measure | Option A result |
|---|---:|
| Rows | 137,312 |
| Feature compatibility | 0.465116 |
| Match counts | 4 exact; 0 semantic; 72 group-only; 10 dataset-specific |
| Trust | Medium |
| Selective decisions | 137,312 Unfamiliar; 0 accepted |
| Forced best-guess attack-vs-normal accuracy | 0.568377 |
| Forced attack precision / recall / F1 | 0.974329 / 0.529392 / 0.686034 |
| 1-P(Normal) AUROC / AUPR | 0.771223 / 0.966248 |
| OOD score AUROC / AUPR | 0.589223 / 0.870222 |

The forced-decision results use best_guess_class != Normal and intentionally ignore the selective Uncertain/Unfamiliar decision. They are ranking/class-score diagnostics, not accepted operational coverage. Under the requested AUROC wording, 1-P(Normal) has a “useful signal”; the OOD score has “no usable signal”. All rows remain Unfamiliar under the frozen decision rule.

Normal/DoS confusion matrix (all non-BENIGN labels collapsed to DoS; forced best-guess prediction):
| Actual \ Predicted | Normal | DoS |
|---|---:|---:|
| Normal | 13,294 | 1,706 |
| DoS | 57,561 | 64,751 |

Per-label forced attack detection rates (BENIGN is the false-positive rate):

| Raw label | Rows | Detection rate |
|---|---:|---:|
| BENIGN | 15,000 | 0.113733 |
| DrDoS_DNS | 7,500 | 0.437200 |
| DrDoS_LDAP | 7,500 | 0.544667 |
| DrDoS_MSSQL | 7,500 | 0.856400 |
| DrDoS_NTP | 7,500 | 0.432800 |
| DrDoS_NetBIOS | 7,500 | 0.755600 |
| DrDoS_SSDP | 7,500 | 0.458400 |
| DrDoS_UDP | 7,500 | 0.463600 |
| LDAP | 15,000 | 0.661533 |
| MSSQL | 7,500 | 0.848267 |
| NetBIOS | 7,500 | 0.816933 |
| Portmap | 7,500 | 0.971067 |
| Syn | 15,000 | 0.000400 |
| UDP | 7,500 | 0.487733 |
| UDP-lag | 7,500 | 0.008400 |
| UDPLag | 1,873 | 0.911372 |
| WebDDoS | 439 | 0.015945 |

Option B strict-source-Level-1 normalization was started on the same prepared sample but the API stopped at its hard resource guard when free RAM fell below 2 GiB. No Option B predictions or metrics were produced and no retry was attempted. This is a documented incomplete item, not a measured Option B result. The aggregate file has no original per-source-file identity, so per-file detection rates are unavailable.

Artifacts: C:\NOVEL_IDS_runs\cic\second_run_20261127\CIC_DDoS2019_SECOND_RUN_REPORT.md, cic_second_run_final_report.json, option_a_predictions.csv, option_a_summary.json, and option_a_evaluation_metrics.json. The first CIC run remains untouched.

### S0 validation-only runtime audit

A 64-row raw NSL-KDD validation batch was forwarded through Stage 5–11. All tensors were finite. Shapes: Z_s/Z_d/R/R_proj/G/Z_f/A/U/Z_u/Z_t are 64x128; tokens are 64x78x128; C is 64x1; classifier logits are 64x12. Gate G mean/std were 0.362738/0.365859; alignment C mean/std were 0.407726/0.025548. Full parameter counts, trainability flags, tensor ranks and min/max statistics are in docs/pipeline_truth_check.json. No test partition was loaded.


## CIC-DDoS2019 second-run comparison — 2026-11-27

Both normalization options completed on the same 137,312-row seed-42 sample. Option A uses target-only unlabeled normalization. Option B uses source-training statistics for Level-1 features and target-unlabeled statistics for non-Level-1 features. Both produced 137,312 Unfamiliar decisions (0% accepted coverage). Option A forced accuracy/F1 = 0.568377/0.686034; Option B = 0.544432/0.665465. 1-P(Normal) AUROC: 0.771223 (A), 0.794376 (B). OOD AUROC: 0.589223 (A), 0.547361 (B). Forced scores ignore the selective Unfamiliar decision and are diagnostic only. Accepted CIC transfer is not demonstrated. Full per-label comparison and confusion matrices are in C:\NOVEL_IDS_runs\cic\second_run_20261127\CIC_DDoS2019_SECOND_RUN_REPORT.md.
