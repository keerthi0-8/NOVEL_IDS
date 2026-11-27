# NOVEL-IDS demo sample pack — verified 2026-11-27

## Contents and provenance

Six CSVs are preserved: labeled/unlabeled pairs for NSL-KDD, UNSW-NB15, and CIC-DDoS2019. Each has 500 rows. The original sample manifest hashes and row counts remain valid; no sample CSV was regenerated or replaced.

| Pair | Source | Raw feature columns | Extra columns |
|---|---|---:|---|
| NSL-KDD validation | Active v2 validation derived from KDDTrain+.txt; seed 42 | 41 | difficulty_level; optional label |
| UNSW-NB15 validation | Project-designated validation source rows from UNSW_NB15_testing-set.csv; seed 42 | 42 | id; optional label |
| CIC-DDoS2019 | Seeded capped draw from the existing 162,276-row / 13-file partial sample; seed 42 | 86 | optional label |

The source sample builder reads raw source rows using the stored validation row indices and preserves original source column names and raw values. These are not unified/processed feature arrays. Labeled files add a lowercase label column; paired unlabeled files omit it. Pass --labels-column label for labeled samples so the API excludes labels before onboarding and inference.

## Source-schema recognition and preprocessing

The API normalizes column names by removing a UTF-8 BOM, trimming/collapsing whitespace, and case-folding. Documented aliases include protocol type → protocol_type, src bytes → src_bytes, dst bytes → dst_bytes, source bytes → sbytes, destination bytes → dbytes, source packets → spkts, and destination packets → dpkts. Label/outcome columns and the dataset metadata columns difficulty_level (NSL-KDD) and id (UNSW-NB15) are excluded from the feature fingerprint.

A source schema is recognized only when the normalized feature-name set exactly matches the stored source preprocessor's feature-name set, with no duplicate/unknown feature. NSL-KDD matches 41/41; UNSW-NB15 matches 42/42. Both return compatibility 1.0 and High trust, and use the stored training-fitted DatasetPreprocessor.transform path (training medians, scalers and vocabularies). Upload rows do not refit source normalization.

Unknown schemas continue through the existing semantic feature matching path and unlabeled target-only median/IQR normalization. CIC metadata-driven numeric columns are explicitly coerced to numeric in both inference chunks and the unlabeled calibration frame; malformed tokens become missing values and are masked. No fitted source preprocessor or model artifact was changed.

## Refreshed demo outputs

Outputs and summaries are in demo_samples/outputs/source_schema_fix_20261127/.

| Sample | Recognized dataset | Compatibility | Trust | Decision breakdown | Labeled demo metrics |
|---|---|---:|---|---|---|
| NSL-KDD labeled | nsl_kdd | 1.000 | High | Normal 261; Known attack 227; Unfamiliar 12 | Accuracy 1.000; macro-F1 1.000 |
| NSL-KDD unlabeled | nsl_kdd | 1.000 | High | Normal 261; Known attack 227; Unfamiliar 12 | — |
| UNSW-NB15 labeled | unsw_nb15 | 1.000 | High | Known attack 244; Normal 142; Uncertain 79; Unfamiliar 35 | Accuracy 0.792; macro-F1 0.5907 |
| UNSW-NB15 unlabeled | unsw_nb15 | 1.000 | High | Known attack 244; Normal 142; Uncertain 79; Unfamiliar 35 | — |
| CIC-DDoS2019 labeled | unknown schema (CIC metadata supplied) | 0.465116 | Medium | Unfamiliar 500 | Not a source-dataset metric |
| CIC-DDoS2019 unlabeled | unknown schema (CIC metadata supplied) | 0.465116 | Medium | Unfamiliar 500 | — |

Labeled and unlabeled prediction columns are identical for each pair. The source sample accuracy/macro-F1 values are explicitly demo-sample metrics, not official test results or research headline results.

Raw source validation parity was checked on 500 rows per source dataset against saved Phase-E validation probabilities after applying the already-shipped additive class-prior bias in log-probability space:
- NSL-KDD maximum absolute probability difference: 2.47e-7 (limit 1e-5); OOD difference 5.71e-6.
- UNSW-NB15 maximum absolute probability difference: 5.73e-7 (limit 1e-5); OOD difference 4.72e-6.

## CIC full second-run result

The separate 137,312-row CIC Option A run and evaluation are documented in C:\NOVEL_IDS_runs\cic\second_run_20261127\CIC_DDoS2019_SECOND_RUN_REPORT.md. All 137,312 rows were Unfamiliar; compatibility was 0.465116 / Medium. Forced best-guess ranking metrics are not accepted-decision coverage. Option B was stopped at the explicit 2-GiB free-RAM guard and has no metrics. The first 162,276-row CIC run is preserved.

## KDD Cup 99 optional sample

NOT SUPPLIED. data/demo_sources/kddcup.data_10_percent and kddcup.names were not present. No substitute was created. If supplied later, preserve the 41 feature names in kddcup.names order and strip trailing label dots. Because NSL-KDD derives from KDD Cup 99, it is a related/same-schema comparison, not an unseen-dataset evaluation.

## Restrictions and interpretation

NSL-KDD and UNSW-NB15 samples use validation rows only, never official test partitions. CIC samples are from the existing partial aggregate; missing files were not synthesized. Do not present any 500-row demo metrics as official test or generalization performance.
