# NOVEL-IDS Decision Record

## User-approved continuation decisions
Source for every entry below: user message, 2026-11-25. Status: approved.

1. Diagnostic lifecycle: Continue only PID 35816; check approximately every 15 minutes. Terminate only after 30 minutes without CPU progress or after four elapsed hours without final JSON. If rerun is required, make it resumable with progress logging and per-candidate partial results; after three hours, use a seeded duplicate-grouped training subsample up to 50,000 rows per dataset and label the deviation.
2. Phase 2 candidate selection: Run candidates 09 and 07, identified as the two highest P3 mean-gain unique residual configurations and labelled as having failed the original rank gate.
3. Phase 2 arms and objective: Four seed-42 arms: candidates 09 and 07 crossed with lambda_decorr 0.1 and 1.0. Keep budgets, Stage 5 snapshot and all other losses/coefficients unchanged. Compute per-dataset float32 L_decorr with sub-batch 256, the same variance-floor mask, 10% linear warm-up, and no gradient into Z_s or Z_d. L_info and L_proj remain defined on R, not R_proj.
4. Phase 2 selection and seeds: Select the highest validation probe-gain arm among finite arms with positive reconstruction skill on both datasets and no more than 25% dimensions at the variance floor. Rank is reported but does not block the three-seed run. Run seeds 42, 43, 44 for the selected arm. If none qualifies, record that and continue with the nine existing candidates.
5. Resource controls: Before training, verify GPU processes with nvidia-smi, at least 4 GB free RAM, and CPU thread cap of 2 (or 4 on at least 8 logical cores). Desktop GPU use of a few hundred MiB is acceptable. Peak GPU memory must remain below 6 GB; record before and after.
6. Rank-gate interpretation: Retain and report original participation-rank thresholds (64-D >=16; 128-D >=32), but they do not block acceptance because intrinsic E_d participation rank is approximately 10–14. Document this as a deviation from the original rank gate.
7. Stage 6 acceptance criterion: Require finite values; positive reconstruction skill on both datasets; <=25% dimensions at variance floor; Var(E_d)/Var(Z_d) >=0.001; and probe gain >=0.01 with paired-bootstrap 95% CI excluding zero, above both null controls under Rule B, and above seed-to-seed SD.
8. Final residual selection: Choose the highest mean validation probe-gain configuration among nine candidates and Phase 2 arms/seeds satisfying the acceptance criterion. Within one seed SD, prefer smaller width, then smaller lambda_red/lambda_decorr. Use its R_proj (128-D) downstream. If none qualifies, record the approved negative-result wording and carry the highest-gain candidate downstream so later gates/ablations can quantify its effect.
9. No residual rescaling: Do not rescale or normalize E_d. If L_info is negligible versus L_mag, report magnitudes and gradient norms without changing the objective.
10. Unified taxonomy mappings: Normal <- UNSW Normal and NSL-KDD normal; DoS <- UNSW DoS and NSL-KDD DoS-family labels; Probe_Recon <- UNSW Reconnaissance and NSL-KDD Probe-family labels. Keep all other categories dataset-prefixed: UNSW_Exploits, UNSW_Fuzzers, UNSW_Generic, UNSW_Analysis, UNSW_Backdoor, UNSW_Shellcode, UNSW_Worms, NSL_R2L, NSL_U2R. If verified counts or semantics contradict a mapping, revert that mapping to dataset-specific.
11. Taxonomy granularity and support: Group NSL-KDD fine labels into its own families for Stage 11 only; Stage 5 remains native. Report native fine-grained results secondarily. Record per-class train/validation/test supports; flag validation support below about 20 and do not tune on those classes. Create config/taxonomy_unified.yaml with status approved, evidence per mapping and approval source user message, 2026-11-25. Native separate heads are ablation only.
12. Trainable/frozen stages: Train Stage 7 gate, Stage 8 W_a/W_u projections (C stop-gradient), Stage 10 Transformer including W_t and CLS projection, and Stage 11 classifier. Freeze Stage 5, Stage 6 including R_proj, and Stage 8 reference statistics. No joint fine-tuning.
13. Unified training loss: L = L_cls + lambda_coral*L_coral + lambda_gate*L_gate_reg. L_cls is class-weighted cross-entropy on unified taxonomy, inverse-square-root class weights from training labels only, normalized and clipped to max ratio 10–20 and logged; label smoothing 0. L_coral is normalized CORAL on A=W_a Z_f weighted by C, per-dataset centered, sub-batch >=256, float32, 10% linear warm-up. lambda_gate defaults to 0 and is switchable.
14. Training budget and stopping: Equal-size 256-per-dataset sub-batches, dataset weighting 0.5/0.5, AdamW lr 5e-4, weight decay 1e-4, gradient clipping 1.0, max 20,000 steps, validation every 500. Monitor validation mean macro-F1 over present unified classes averaged across datasets; improvement must beat best by 0.5% relative. Stop after six consecutive non-improving validations after at least eight evaluations. Save best/last atomically in local non-synced run directory. Fixed seeds, mixed precision, Stage 5/6 feature-set dropout and value masking in training only; eval/no masking in validation.
15. CORAL selection: Screen lambda_coral in {0, 0.05, 0.2, 1.0} at seed 42. Then run seeds 42–44 for lambda 0 and best nonzero candidate. Reject if either dataset validation macro-F1 drops by more than max(5 percentage points, 2x seed SD) against lambda 0. Among remaining candidates choose strongest alignment; if none pass, choose 0. Gate regularizer stays 0 unless gates saturate (<0.01 or >0.99 mean for all dimensions in both datasets), then test lambda_gate=0.01 and retain only if it respects the same rejection rule.
16. Temperature calibration: Fit one global temperature using pooled source validation logits only.
17. OOD and uncertainty thresholds: Use class-conditional Mahalanobis on Z_t with shared covariance and ridge shrinkage, excluding/reporting classes with too few training samples; average its training/validation-IN z-score with energy-score z-score. tau_ood is validation 95th percentile; tau_conf is validation 10th percentile of temperature-calibrated max probability. Report coverage-accuracy and OOD curves. No test/unseen threshold tuning. A held-out-class development experiment (one attack class per dataset, separate development runs) is approved only to validate OOD scoring.
18. Decision precedence: Unfamiliar if OOD score > tau_ood; else Uncertain if calibrated max probability < tau_conf; else Normal or Known attack with category/confidence. Never emit a category outside taxonomy.
19. Baselines: Train per-dataset HistGradientBoosting with native and unified labels, plain MLP, and common-features-only Transformer. Same splits; three seeds where stochastic; train-fit/validation-tune only.
20. Ablations: Three seeds each, same data/budgets/selection: shared-only/no residual; residual with fixed G=1; residual with adaptive gate; full selective alignment primary; plus common-only, alignment off/on, native heads/unified head. Report accuracy, macro/weighted F1, per-class recall/support, residual norm share and per-dataset gate statistics.
21. One-shot source-test evaluation: Before test access, write full configuration hash and checkpoint hashes to PROGRESS.md and confirm no selection used test data. Evaluate independently: UNSW 73,791 independent rows and full file separately; NSL KDDTest+ 21,880 independent rows and full 22,544 separately. Report confusion matrices (raw/normalized), accuracy, precision, recall, F1, macro/weighted F1, per-class P/R/F1/support, FPR/FNR/detection rate, OVR ROC-AUC/PR-AUC where defined, OOD AUROC/AUPR/FPR@95%TPR, Uncertain rate and accepted-prediction accuracy. NSL test-only attack types are natural unfamiliar traffic. Never rerun after seeing results; if a bug is found, report and ask.
22. Required diagnostics: Dataset-identity probe on Z_s and common-features-only reference are required.
23. Unseen-data protocol and exclusions: CIC-DDoS2019 is not part of this task until supplied under data/unseen/. Then onboard without source-code change; headline unsupervised per-dataset normalization Option A and report strict Option B; no retraining/fine-tuning; labels evaluation only; unsupported classes reported. Run leave-one-dataset-out diagnostics before unseen evaluation. Joint fine-tuning Stage 5/6, architecture changes, E_d rescaling/loss changes, Stage 5 snapshot changes and test/unseen use for decisions are not approved.

## Operating constraints adopted with this record
- Stage 5 files/configs/checkpoints/logs/artifacts and data/splits/preprocessing artifacts are immutable.
- New integration/training code is written only to new files.
- No package installation, Git commands or file deletion.
- Only append to PROGRESS.md; do not edit NOVEL_IDS_MASTER_SPEC.md.
- Minimal fixes to Stage 6–12-created code are allowed only when a failing test proves a defect and the fix changes no equation, default, threshold, label, data, split or design decision; max two attempts per defect, with diff and rerun reported.
- Invocation issues may be corrected for up to three attempts per issue.
- Hard stops are the finite-value, provenance/hash, data-leak/forbidden-file, GPU/process/memory/disk, architecture/loss/forbidden-change, residual-variance, OFF-mode-equivalence and repeated-invocation conditions in the user continuation prompt.

## Approval ledger
| Decision | Source | Status |
|---:|---|---|
| 1 | user message, 2026-11-25 | approved |
| 2 | user message, 2026-11-25 | approved |
| 3 | user message, 2026-11-25 | approved |
| 4 | user message, 2026-11-25 | approved |
| 5 | user message, 2026-11-25 | approved |
| 6 | user message, 2026-11-25 | approved |
| 7 | user message, 2026-11-25 | approved |
| 8 | user message, 2026-11-25 | approved |
| 9 | user message, 2026-11-25 | approved |
| 10 | user message, 2026-11-25 | approved |
| 11 | user message, 2026-11-25 | approved |
| 12 | user message, 2026-11-25 | approved |
| 13 | user message, 2026-11-25 | approved |
| 14 | user message, 2026-11-25 | approved |
| 15 | user message, 2026-11-25 | approved |
| 16 | user message, 2026-11-25 | approved |
| 17 | user message, 2026-11-25 | approved |
| 18 | user message, 2026-11-25 | approved |
| 19 | user message, 2026-11-25 | approved |
| 20 | user message, 2026-11-25 | approved |
| 21 | user message, 2026-11-25 | approved |
| 22 | user message, 2026-11-25 | approved |
| 23 | user message, 2026-11-25 | approved |

## 2026-11-25 - User-approved NOVEL-IDS fast-path decisions (replaces slow Phase 2 runner)

Source for all decisions below: user message, 2026-11-25. Status: approved. These decisions supplement the existing continuation prompt and supersede it only where explicitly stated here.

1. Stop redundant old evaluation: after arm-1 checkpoint and metadata completeness verification, terminate PID 11932 only and re-evaluate from saved checkpoint using the fast evaluation. Leave PID 35816 alone unless prior continuation stall rules apply; lower its priority only if it exists. S0 verified best.pt, last.pt, both metadata files, global_step/best_step=10000, and SHA-256 values; PID 11932 was then terminated. PID 35816 was absent and was not modified.
2. Stage 6 training must use GPU when the prior run used CPU. Keep covariance calculations float32 and require fixed-batch CPU/GPU loss equivalence within 1e-5. S0 found the old trainer already used CUDA; equivalence check is still required for the new cache pipeline.
3. Cache frozen Stage 5 outputs in new scripts/files/run directories, float32. Training bank K=8 per training row (draw 0 clean, draws 1-7 seeded approved feature-set dropout and 5% value masking); validation clean plus fixed seeded masks only where protocol requires. Stage6 samples one row and one bank draw each step. Verify cached vs on-the-fly Z_s/Z_d within 1e-6 and Stage6 losses within 1e-5. Label finite-bank augmentation as a deviation and report RAM/file size. Use K=4 only if K=8 exceeds the stated 6 GB RAM or 10 GB disk limit.
4. Reuse P0 predictions if saved; otherwise fit P0 once per dataset and cache predictions. Cache P1/P2 features for reuse across candidate arms. S0 targeted search found no existing Phase1 P0 prediction/feature cache; new reusable caches must be generated.
5. Replace repeated sklearn F1 bootstrap calls with vectorized paired bootstrap, retaining 2,000 repetitions, seed 42, paired indices, original-validation-support classes, zero-division=0 and 2.5/97.5 percentile CI. Verify against sklearn on 50 identical resamples with maximum absolute difference <=1e-12.
6. Preserve multinomial logistic-regression probe class and hyperparameters. Standardize feature blocks with training statistics. If one dataset fit exceeds 10 minutes, n_jobs or tol=1e-3 may be used and the deviation reported.
7. Up to three Stage6 arms may run concurrently only after measuring one arm below 2 GB GPU memory, with total GPU memory below 5.5 GB, free RAM above 3 GB, and two CPU threads per process. This overrides the earlier no-concurrent-training rule for these Phase2 arms only.
8. Rerun all six seed-42 configurations under one identical cached pipeline: candidate09 and candidate07 with lambda_decorr=0, plus each candidate with lambda_decorr in {0.1,1.0}. Compare old arm-1 result against its rerun and report difference / bootstrap-CI relation. Then run seeds43 and44 for the best configuration chosen under the continuation decisions. The old rank-gate stop is superseded; rank remains reported but non-blocking.
9. Every long job requires a 100-step timed smoke test, progress log at least every 60 seconds (step, losses, elapsed, ETA), and a time projection. If one run projects over two hours, before launch use cache/GPU, reduce validation frequency to 1,000 steps, and reduce max steps to fit within two hours while retaining early stopping; record reductions. Investigate any log silence exceeding 10 minutes.
10. Stage7-11 training must use GPU, with Stage5 and Stage6 frozen and computed on the fly in eval/no_grad mode. Do not cache 78-token embeddings. Apply the two-hour run cap and progress/time-projection rules from item 9; screening and final runs use identical settings.
11. Apply the continuation prompt's Stage6 acceptance criterion, configuration selection, and outcome wording; continue automatically to later phases without asking for approvals covered here. If seeds43/44 cannot be run in reasonable time, use paired bootstrap CI and null controls, mark seed-SD condition not assessed, continue, and run seeds later in background if GPU is free.

No architecture, loss definition, threshold, or Section H decision is changed by this prompt. All hard stops in Section 3 remain binding.

## 2026-11-25 — User clarification: exhaustive hard-stop list for equivalence recovery and fast-path continuation

Source: user message, 2026-11-25. Status: approved.

The user's latest message explicitly states "Hard stops (only these)" and supplies the exhaustive current list: non-finite values after one seeded retry; protected-hash mismatch; any need to touch Stage 5 files, datasets, splits, artifacts, or test/unseen data; a needed change to an approved loss or architecture; GPU memory above 6 GB; or the same invocation failing three times. This latest explicit list supersedes any earlier standalone hard-stop condition not repeated in that list. In particular, free RAM below 2 GB is not itself a hard stop under this latest instruction. Keep arms sequential; do not run concurrent arms unless the previously approved >3 GB free-RAM and GPU conditions are met. Continue recording RAM and resource observations; do not modify source data or protected files.
 
## 2026-11-25 - User decision: revised Stage6 acceptance and automatic continuation
Source: explicit user decision, 2026-11-25. Status: approved; this decision overrides the earlier Stage6 decision-gate stop.

1. Revised NullN interpretation: NullN is an untrained h applied to E_d at the same width. It preserves E_d information and tests whether training h matters, not whether R adds information. The revised criterion is finite values; positive reconstruction skill on both datasets; at most 25% variance-floor dimensions; Var(E_d)/Var(Z_d)>=0.001; P3 gain>=0.01 with paired-bootstrap 95% CI excluding zero and above NullG; and above seed-to-seed SD when assessed. Disclose explicitly in every report: "revised after results were seen, on logical grounds". Preserve the original criterion and its result alongside the revised one.
2. Final Stage6 configuration is Candidate09, lambda_decorr=1.0. Freeze its checkpoint and hashes; use its 128-D R_proj downstream. Optional narrower R experiment is noted only and is not authorized.
3. Run seeds43/44 for Candidate09 lambda1 and Candidate07 lambda0.1 (training plus P3 probes only); reuse cached P0 and seed42 null controls; cap logistic-regression max_iter and report non-convergence. Run in background during Stage7-11 only if GPU/RAM limits allow; otherwise later. If seed SD exceeds gain, record limitation without changing selected configuration.
4. Required wording: "Stage 6 accepted with qualifications: the residual carries measurable detection information beyond random features; the learned encoder does not exceed a random projection of E_d; the original rank gate fails; the effect of L_decorr is not distinguishable from noise on a single seed."
5. Do not change architecture, losses, or Stage5 snapshot; do not redesign residual representation.
6. Continue phases P3-P9 automatically: taxonomy, Stage7-11 training, baselines/ablations, diagnostics, one-shot source test, final report. The ablation tests whether the residual helps the end-to-end detector; Stage6 evidence suggests its effect may be small.
7. Hard stops remain only those listed in the continuation and speed-up prompts.

### Stage6 post-hoc evidence reconciliation
- The original Phase2 phase2_selection.json remains unchanged and records NO_QUALIFYING_STAGE6_ARM under the old NullN criterion.
- Revised Stage6 criterion passes on seed42 for selected Candidate09 lambda1 except seed-SD is pending seeds43/44. P3 vs P0 mean gain 0.047090, paired bootstrap CI [0.023268,0.067280]. P3 vs NullG mean gain difference 0.054451, CI [0.027827,0.074053]. P3 vs NullN difference 0.008975, CI [-0.008846,0.026874]. P3 vs P2 difference 0.007672, CI [-0.008531,0.024498]. Therefore random Gaussian NullG is exceeded; learned h does not significantly exceed untrained h on this seed; P3 does not significantly exceed P2.
- Report rank as non-blocking and preserve discrepancy: required user wording says original rank gate fails, but stored selected-arm participation values are 34.74498 (NSL-KDD) and 35.82167 (UNSW-NB15), above the 128-D threshold 32 in the earlier record. Do not silently alter either the user-approved wording or numeric evidence.
- Frozen Stage6 checkpoint: C:\NOVEL_IDS_runs\stage6\accepted_candidate09_lambda1_seed42\best.pt; SHA256 1c66f6f8d1a029266e5a637b5db4d6f49881988bf16462d2c3fbc09cd704ecbd. Last checkpoint SHA256 01e41bc0849ebd772da2028f8cf95b4a955bcf92e9e2a79da2073ef114687018; Stage5 snapshot SHA256 9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b. Freeze manifest is freeze_manifest.json.
- Per-arm paired comparisons (2,000 replicates, seed42) saved to C:\NOVEL_IDS_runs\stage6\phase2_fastpath_20261125\revised_stage6_arm_comparisons.json.
- Approved taxonomy config created at config/taxonomy_unified.yaml; validated against all source train/validation labels. NSL_U2R validation support=10 (<20), explicitly flagged.
 
### Clarification: Stage6 rank metrics refer to different representations
The 21:20 addendum initially compared the selected arm's R effective-rank participation values (NSL 34.74498, UNSW 35.82167) to the original rank threshold. That comparison was incorrect: the original rank gate concerns the intrinsic participation rank of E_d, approximately 10-14, against the 128-D threshold 32. The R effective-rank values are a separate diagnostic and do not imply that the E_d rank gate passed. The required user wording "the original rank gate fails" is retained. Final reporting will show both metrics separately.
 
## 2026-11-25 - Stage8 CORAL lambda screening (seed42)
Four candidate lambda_coral values {0,0.05,0.2,1.0} were trained under the approved Stage7-11 pipeline, using seed42, 20k max steps, validation every500, six-validation early stopping after minimum8 validations. Best results:
- lambda0: mean macro-F1 .73723568 (NSL .927274, UNSW .547197), best step6000, stop9000.
- lambda0.05: mean .72995713 (NSL .914416, UNSW .545498), raw validation CORAL .0941437, best step6000, stop9000.
- lambda0.2: mean .73689327 (NSL .933731, UNSW .540056), raw validation CORAL .1429870, best step7000, stop10000.
- lambda1: mean .73492020 (NSL .931850, UNSW .537991), raw validation CORAL .1711599, best step6000, stop9000.
All nonzero values remain within5 percentage points of lambda0 on both datasets. Among nonzero candidates, lambda0.05 has the lowest raw validation CORAL and is the provisional strongest-alignment choice. Seed43/44 lambda0 and lambda0.05 are required to assess seed SD and confirm the per-dataset drop rule. Gate saturation must be evaluated per dimension, not only aggregate means.
 
### Stage7 gate saturation and seed-repeat follow-up
Per-dimension gate means were evaluated on the full validation split for every seed42 CORAL screen. Dimensions with mean gate<0.01 or>0.99: lambda0 NSL5/128, UNSW6/128; lambda0.05 NSL5/128, UNSW8/128; lambda0.2 NSL5/128, UNSW8/128; lambda1 NSL3/128, UNSW6/128. No model has all dimensions saturated in both datasets; lambda_gate remains0, as approved.
Provisional lambda_coral=0.05 is selected for its lowest raw validation CORAL among nonzero candidates and <5pp per-dataset macro-F1 drops versus lambda0. Seed43/44 PhaseB runs are in progress via a resource-gated sequence. The sequence runs lambda0 seeds43/44 and lambda0.05 seeds43/44, and waits for free RAM>=3GiB and GPU use<5.5GiB before each job. Do not start duplicates. Stage6 seed repeats are deferred until these jobs finish.

### 2026-11-25 - Stage8 CORAL Phase B three-seed selection (auto-generated)

lambda        mean_macro          nsl_mean         unsw_mean             nsl_sd            unsw_sd         coral_mean
------        ----------          --------         ---------             ------            -------         ----------
     0  0.73994749592194  0.93311261047127  0.54678238137261 0.0208217596685659 0.0129970106547491                  0
  0.05 0.733635809177152 0.920402150706078 0.546869467648225  0.013681131961975 0.0180656376690428 0.0854402830203374



- Per-dataset allowed drops: NSL=0.05; UNSW=0.05. Observed lambda0.05 drops: NSL=0.0127104597651913; UNSW=-8.70862756154889E-05. Pass=True. Chosen lambda=0.05.
- Selection operationalization: lowest mean validation raw CORAL discrepancy among nonzero candidates passing the macro-F1 safeguard.
- Full per-seed results and hashes: C:\NOVEL_IDS_runs\stage7_11\coral_phaseB_selection.json


## 2026-11-26 — Primary checkpoint selection and Phase A hard-stop
Source: user unblock-and-finish instruction, 2026-11-26. Status: approved selection rule applied; Phase A halted by explicit hard stop.

- Selected seed 43 from the existing lambda_coral=0.05 runs because its recorded best validation mean macro-F1 (0.7381580062) exceeds seed 42 (0.7299571297) and seed 44 (0.7327922916). Best checkpoint: `C:\\NOVEL_IDS_runs\\stage7_11\\coral_screen_lambda0p05_seed43\\best.pt`; SHA256 `349505b6ebae528246d72b66785b28e296ecdf4ccc6d775244005ae3fd5147c0`. This is validation-only selection; no test/unseen data was accessed.
- Phase A calibration extraction was launched against source train/validation only. Free RAM fell to 1.79 GiB, below the user's 2 GiB hard-stop threshold. The calibration process was terminated immediately. No temperature, OOD thresholds, OOD statistics, dataset-identity probe result, or held-out-class OOD result was produced by that attempt. Do not resume Phase A or launch further work while the hard-stop resource condition persists.


## 2026-11-26 — Phase A primary freeze and calibration
Source: user unblock-and-finish instruction, 2026-11-26, Phase A. Status: primary checkpoint selected and frozen from existing three-seed validation runs; calibration completed on source train/validation only.

- Seed 43 selected by highest recorded validation mean macro-F1 among seeds 42/43/44 (0.7381580062). Primary checkpoint SHA256 `349505b6ebae528246d72b66785b28e296ecdf4ccc6d775244005ae3fd5147c0`; Stage5 snapshot and Stage6 Candidate09 hashes are recorded in `PROGRESS.md` and `C:\\NOVEL_IDS_runs\\phaseA_primary_seed43_final_retry2\\phaseA_calibration.json`.
- Global temperature T=0.94386619; pooled validation tau_ood=1.36760807 (95th percentile of combined score); tau_conf=0.42745653 (10th percentile calibrated maximum probability). OOD combination uses equal weights for z-scored class-conditional Mahalanobis and energy. To remain above the 2 GiB RAM hard stop, source-training OOD normalizers were fit using seeded uniform samples of 20,000 rows per dataset; all validation rows were retained. The Z_s identity probe used 10,000 sampled training rows per dataset.
- Dataset-identity probe validation accuracy 0.997055, balanced accuracy 0.996782, ROC-AUC 0.999985. This is strong evidence that Z_s encodes dataset identity; it is a diagnostic finding, not a test-set result.
- Held-out-class OOD proxy used NSL_R2L and UNSW_Shellcode, excluding each class from its OOD reference and masking its logit. Detection rates at 5% known-validation FPR: NSL_R2L 0.9701 (AUROC 0.9806); UNSW_Shellcode 0.0182 (AUROC 0.7408). This is explicitly exploratory only: the frozen Stage7–11 checkpoint was trained with both classes, so this is not a strict leave-class-out retraining experiment. Do not describe it as strict held-out-class generalization.
- Validation macro-F1 gap remains: NSL-KDD 0.910316 vs UNSW-NB15 0.565572. No test/unseen tuning or access occurred. Common-features-only reference is still pending.
- Hash back-reference convention: `primary_frozen_config.json` stores SHA256 of the calibration JSON payload excluding its `primary_config_sha256` back-reference; `phaseA_calibration.json` stores the actual primary config SHA256. Both were independently verified.
- A first calibration attempt hit the RAM hard stop and was terminated; later runs used 256-row inference batches and seeded training subsamples. A 100-batch smoke test completed in 6.7 seconds. All retry/deviation details are preserved in run logs.


## 2026-11-26 — Phase B implementation choices and reductions
Source: user unblock-and-finish instruction, 2026-11-26. Status: applied within approved scope.

- Frozen inputs: Stage 5 snapshot SHA256 `9c4655286d6866e245320adc770dac7662d83b04fa2aedb3c9712e406dcf02b`; Stage 6 Candidate09/lambda_decorr=1.0 SHA256 `1c66f6f8d1a029266e5a637b5db4d6f49881988bf16462d2c3fbc09cd704ecbd`; approved taxonomy unchanged.
- Train/validation-only input cache verified with manifest SHA256 `8902ccc94b7b488e33067b6f46515d4db69608300a87554c74dd910f697a41bf`. It contains full and common-feature-only Stage5 embeddings, token arrays and availability masks for both datasets and both train/validation splits. Chunk continuity, row counts, shapes, dtypes, finite values and binary masks were checked. No test/unseen input was cached.
- Finite augmentation deviation: one deterministic augmented Stage5 input draw per training row is cached (rather than K=8); validation remains clean. This is explicitly a finite augmentation bank and will be reported as such. Stage5 weights and files remain frozen. Token arrays are opened per batch to avoid holding the complete token cache in RAM.
- Phase B 100-step timed smoke runs are performed before each corresponding 1,000-step pilot. Pilot settings use batch 256 per dataset, validation interval 1,000 for full pilot, and progress logging at <=60-second intervals. The first shared-only attempt crossed the 2 GiB free-RAM hard-stop boundary at 1.99 GiB and was stopped; token-map access was changed to per-batch mapping before retry. Subsequent observed free RAM remained above 4.4 GiB and GPU peak allocation was about 468 MiB.
- Planned full ablation runs use 5,000 max steps, validation interval 1,000, with a 40-minute per-run cap. This is within the measured per-step projection; any run exceeding its cap is reduced before launch. Two seeds (42,43) are used where budget permits. The primary full selective-alignment model is reused from the validation-selected seed43 checkpoint; residual-adaptive is the alignment-off comparison. A paired native-head mode is added to the same runner for the native-vs-unified Transformer head comparison; its smoke test is required before the full native-head runs.
- Ablation variants: shared-only, residual fixed G=1, residual adaptive (alignment off), common-features-only; full alignment on reuses primary. Native-head versus unified-head comparison uses paired residual-adaptive architecture and the same seeds. No test/unseen data is used for selection.
- No change to approved equations, Stage5/6 snapshots, data, labels, splits, preprocessing artifacts, or thresholds is authorized or performed by these implementation choices.


### Phase B time-budget clarification (2026-11-26)
The paired native-head Transformer ablation is reduced to 2,000 max steps per seed (42 and 43), validation every 500 steps, to keep the entire Phase B matrix within its four-hour wall-clock budget after the 5,000-step primary ablation variants. This is a documented budget-driven step reduction; it is not a change to the native-head architecture or loss. The earlier general note that native-head runs use the same 5,000-step budget is superseded by this clarification.


### Phase B aggregate-budget reduction (2026-11-26)
The first full ablation seed42 run's 60-second timed projection was approximately 24 minutes for 5,000 steps. Extrapolating the remaining seven 5,000-step runs plus the paired native-head checks would exceed the four-hour Phase B wall-clock budget. The current shared-only seed42 run was allowed to finish at 5,000 steps; all subsequent full ablation runs are reduced to 4,000 max steps with validation every 500 steps, retaining both seeds 42 and 43. This documented aggregate-budget reduction keeps the matrix within the phase budget and does not skip any variant. Native-head runs remain at 2,000 steps per seed.

### 2026-11-26 Phase C one-shot outcome: INCOMPLETE / CONSUMED (19:30 local)

The actual Phase C one-shot evaluator verified the supplemental freeze manifest and opened the NSL-KDD preprocessed source-test artifact. It halted before the first model inference while mapping test labels: the official test artifact contains raw label `saint`, which is not present in the approved unified taxonomy. No predictions, classification metrics, OOD metrics, or baseline metrics were produced. No `results.json` or `results.partial.json` was written. The one-shot access has occurred; under the approved no-rerun rule, do not reopen either source-test artifact or rerun Phase C. This is an incomplete one-shot attempt, not a successful evaluation. The evaluator script and frozen manifest are preserved for audit; the taxonomy is not modified post-test.

- 2026-11-26 Phase E complete: validation-only checkpoint reproducibility rerun exactly reproduced Phase A temperature/thresholds and source-validation metrics. Saved per-row validation probabilities, OOD scores and gate means; generated Phase B curves, per-class F1, OOD histograms, reliability diagrams, gate statistics and baseline confusion matrices. Created root README.md and docs/FINAL_REPORT_2026-11-26.md. Phase C remains incomplete/consumed; Phase D LODO complete; CIC-DDoS2019 unseen evaluation pending.


## 2026-11-26 evening — User-authorized Phase C recovery and finish-today scope

Source: user message, 2026-11-26 evening. Status: pre-approved; execution authorized within the exact scope of the user's finish-today prompt.

- The owner amends the consumed Phase C one-shot rule for one recovery only. The original failed attempt and logs remain unchanged. Recovery must be a single frozen inference run; after it completes, test partitions close permanently. Only a crash-only metrics-stage rerun from saved prediction files is allowed, with no value-computing code changes.
- Approved NSL-KDD label mapping: DoS attack types -> DoS; Probe attack types including saint -> Probe_Recon; R2L -> NSL_R2L; U2R -> NSL_U2R; normal -> Normal. This is an evaluation label map only; the approved model taxonomy and Stage 5 labels are not changed.
- Approved UNSW-NB15 label normalization: strip whitespace, case-insensitive; Backdoors aliases Backdoor. Existing approved taxonomy mappings remain unchanged.
- Any undocumented/unmapped label is represented as UNSUPPORTED_<label>, excluded from known-class metrics and counted as unfamiliar-positive in OOD evaluation.
- Evaluator hardening, validation-only and synthetic dry runs, label-vocabulary audit, frozen hashes, two-stage saved predictions/metrics, specified model/baseline scope and test closure are authorized.
- Backend inference API, tests and documentation are authorized. CIC-DDoS2019 evaluation is authorized only if data exists in data/unseen; mapping is evaluation-only and no target tuning/retraining is allowed. Packaging and final documentation are authorized. No frontend work and no Git commands.
- The source for the NSL-KDD family list is the project-local documentation search (no official list found locally), followed by published NSL-KDD attack-type tables: MDPI Sensors 2025 Table 3 (https://www.mdpi.com/1424-8220/25/8/2614), corroborated by SAGE 2021 Appendix Table A1 (https://journals.sagepub.com/doi/10.1089/big.2020.0263). The explicit saint -> Probe mapping is supported by both.


## 2026-11-26 evening — Phase C recovery preflight outcome and stop

Source: user message, 2026-11-26 evening. Status: preflight completed; recovery inference NOT run.

- Created evaluation-only config/phaseC_label_map.yaml (SHA256 cedd0569671f4f52dcbbc037a00490d863a6c3efbc08e12346dd0b11779f3837). It includes all owner-specified NSL-KDD families and unified mapping, including saint -> Probe_Recon; UNSW normalization and Backdoors -> Backdoor alias; unsupported fallback. config/taxonomy_unified.yaml was not modified.
- Created scripts/phaseC_recovery_preflight.py. Synthetic fixture hand check passed: known-only accuracy=0.75; DoS precision=1.0, recall=0.5, F1=0.6666666667; one unsupported sample counted; zero-support NSL_R2L excluded from the fixture's supported-class calculation.
- Original Phase C validation-only preflight completed successfully for primary plus five configured ablation entries (shared-only, fixed gate, adaptive/alignment-off, common-only reduced, native-head adaptive) and selected HGB/MLP baseline paths. Log: PREFLIGHT_COMPLETE; no test prediction or inference was run during this validation preflight.
- Label vocabulary report was generated at results/phaseC_label_vocabulary_report.json. NSL-KDD test vocabulary: 38 raw labels, all mapped; saint -> Probe_Recon. UNSW-NB15 test vocabulary: 10 raw labels, all mapped. Both independent and full test artifacts have the same unique label vocabularies.
- **Deviation / hard stop:** both test artifacts are compressed Joblib files. joblib.load(..., mmap_mode='r') emitted a warning that mmap_mode is ignored for compressed files; consequently, loading the artifact for the label-vocabulary audit materialized its full object, including feature arrays, even though the script accessed only the labels field and never inspected/used feature values. This violates the strict labels-only/no-features condition in Section 2A step 4. No predictions, probabilities, metrics, or inference were produced. Per Section 6, stop Phase C here; do not run stage 1, do not reopen test artifacts, and do not claim recovery evaluation completed. The original failed attempt remains preserved unchanged.
- Existing Phase C original evaluator SHA256 remains d34ff19f47f4e5b84a5b963f08d5ebfc1d47f06b449345efef13f0a5cd888c71; original freeze manifest and original phaseC_progress.log remain unchanged. Label-map SHA256 is above; primary frozen config SHA256 remains 55f8373490743e0eaf4776818aa826f3b26af51a27d4a4041fbfb40f66495759. No recovery inference freeze/run was initiated.
- No Phase C predictions or Phase C results.json/results.partial.json exist in the recovery directory. Test partitions are NOT marked closed because the authorized recovery inference did not complete.
- CIC-DDoS2019 folder data/unseen is absent. No unseen evaluation performed.
- No backend inference API, requirements.txt, or final API documentation has yet been created in this execution. No Git commands run. No Stage 5 source files, dataset files, labels, splits, preprocessing artifacts, approved taxonomy, Stage 6 checkpoint, or primary checkpoint were modified.


## 2026-11-26 — Phase C recovery attempt stopped before prediction

Source: user-authorized single Phase C recovery, 2026-11-26. Status: INCOMPLETE; no predictions or metrics.

- The recovery evaluator verified the frozen manifest and opened the first source-test artifact. It then stopped on a code error before any model forward pass: the test split was not registered in the source helper before `forward_rows` attempted to access it (KeyError: `test_independent`).
- No per-row predictions, probabilities, OOD scores, metrics, or results.json were produced. No model inference batch completed. The original Phase C failed attempt and this recovery log are preserved.
- The recovery one-shot authorization is consumed. Do not reopen either source-test artifact or rerun inference. A metrics-only rerun is not applicable because no saved predictions exist. Further source-test inference requires a new explicit user authorization.
- Recovery evaluator and frozen manifest remain in `C:\\NOVEL_IDS_runs\\phaseC_recovery_20261126`; label-only audit is `raw_label_vocabulary_audit.json`. Stage 5/6, taxonomy, data, splits, and checkpoints remain unchanged.


## 2026-11-26 ? Second Phase C recovery authorization and corrected loader freeze

Source: user message, 2026-11-26 (?continue?), explicitly authorizing continuation after the documented pre-forward loader failure. The recovery wrapper now registers each already-loaded test split in helper.datasets before source_batch access. No architecture, checkpoint, labels, preprocessing or thresholds changed.

- Corrected recovery evaluator SHA256: `a4cd369b3466f9b49732b08de9c2728de9858f5be9946c34cd27cc1dd8cbe0c9`.
- Updated supplemental freeze manifest SHA256: `fe890f1124328419e393a44359cc2c51a51756368ab34b4e2b67fa366a920dfc`.
- This is a separately authorized recovery attempt; the previous failed run and all logs are retained. This attempt remains one-shot. No existing predictions or partial results were found before launch.
- The same label-only audit and frozen source/model/configuration hashes remain prerequisites.


## 2026-11-26 — Second Phase C recovery attempt stopped by RAM hard stop

Source: user message, 2026-11-26 (“continue”), authorizing another attempt after the prior loader-registration failure. The corrected loader registration worked: the first source-test split was registered and passed into the row evaluator. The run then stopped before the first model forward pass because free system RAM fell below the explicit 2 GiB hard-stop threshold after loading the source arrays. Runtime error: `HARD STOP free RAM below 2 GiB during Phase C`. No prediction batch completed; no saved per-row predictions, probabilities, OOD scores, metrics, results.partial.json or results.json were produced. GPU remained idle. Do not retry under this authorization: the one-shot attempt is consumed and the explicit RAM hard stop applies. A further attempt requires resource remediation and fresh explicit user authorization. The first loader failure, this RAM-stop log, all frozen manifests and source artifacts are retained.


## 2026-11-26 fresh status check and safe continuation
- Fresh remote check: free RAM 5.08 GiB at inspection time. The inference API package directory is absent. Primary seed-43 checkpoint SHA256 matches the frozen hash.
- Phase C remains stopped: the second recovery passed loader registration, then hit free RAM <2 GiB after loading source-test arrays, before model forward. No outputs/metrics exist. The one-shot authorization is consumed; no retry performed.
- CIC Drive `G:` mount is currently inaccessible. Earlier inventory established 13 CSV files (~13.50 GB); evaluation remains pending.
- Documentation corrected to reflect the RAM-stop outcome, actual Drive access state, and pending API. Existing regression result remains 108 passed / 1 failed.

## Phase C source-test evaluation ? completed 2026-11-26

- Status: COMPLETE; one frozen evaluation run; 24 per-row prediction CSVs; `results.json` is the authoritative metrics artifact.
- Result directory: `C:\NOVEL_IDS_runs\phaseC_recovery_20261126`.
- Evaluated: primary seed 43, five validation-selected ablation variants, HGB and MLP baselines; UNSW independent (73,791) and full (82,332), NSL independent (21,880) and full (22,544).
- NSL-KDD primary: independent accuracy 0.7897, macro-F1 0.6273; full accuracy 0.7938, macro-F1 0.6306. Test-only attack OOD AUROC 0.8800 / AUPR 0.5779 (independent); FPR@95TPR 0.3584.
- UNSW-NB15 primary: independent accuracy 0.7617, macro-F1 0.4608; full accuracy 0.7502, macro-F1 0.4698. OOD AUROC/AUPR not defined because the approved UNSW evaluation partition contains no positive OOD rows for this proxy.
- Important comparison: UNSW native HGB macro-F1 0.5481 (independent) exceeds primary model 0.4608; NSL unified HGB macro-F1 0.6410 is slightly above primary 0.6273. Do not claim the transformer is best overall. Native-vs-unified NSL metrics are different label spaces and not directly comparable.
- All predictions are saved before metrics calculation; Stage 2 metrics were reloaded from saved CSVs. Test partitions are now closed; do not rerun or tune against them.
- Frozen hashes verified at run start and rechecked after completion: Stage5 snapshot `9c465528...dcf02b`, Stage6 Candidate09 `1c66f6f8...ecbd`, primary checkpoint `349505b6...5147c0`.


## Final execution update ? 2026-11-26

### Phase C source-test evaluation

- COMPLETE. `C:\NOVEL_IDS_runs\phaseC_recovery_20261126\results.json` is authoritative; 24 per-row prediction CSVs are saved in the adjacent `predictions` folder.
- NSL-KDD primary: independent partition accuracy 0.789671, macro-F1 0.627273; full KDDTest+ accuracy 0.793781, macro-F1 0.630586. Test-only attack OOD: independent AUROC 0.879961, AUPR 0.577862, FPR@95TPR 0.358382; full AUROC 0.877257, AUPR 0.571161, FPR@95TPR 0.367671.
- UNSW-NB15 primary: independent accuracy 0.761692, macro-F1 0.460816; full accuracy 0.750170, macro-F1 0.469846. OOD AUROC/AUPR are undefined because the approved UNSW evaluation partitions have zero positive rows for this OOD proxy.
- Baselines: NSL independent unified HGB macro-F1 0.640996 vs primary 0.627273; UNSW independent native HGB macro-F1 0.548124 vs primary 0.460816. Native and unified NSL label spaces are not directly comparable. Do not claim the primary transformer is best overall.
- Test partitions are closed. No further tuning, selection, or rerun against them.

### Stage 12 and inference API

- Added `src/novel_ids/decision/stage12.py` with the approved precedence: Unfamiliar if OOD exceeds validation tau; otherwise Uncertain if calibrated confidence is below validation tau; otherwise Normal or Known attack.
- Added `src/novel_ids/inference/` with `load_model`, `onboard_csv`, `predict_csv`, `evaluate_predictions`, and CLI `python -m novel_ids.inference.predict`. It loads frozen Stage 5/6 and primary Stage 7-11, reconstructs OOD references from source-training samples only, uses Option A target-only unlabeled median/IQR normalization, supports CPU fallback, and batches model inference.
- Synthetic inference smoke: 32 rows, all outputs finite; unknown/unmatched synthetic traffic was conservatively classified Unfamiliar. Targeted Stage 12 + onboarding tests: 5 passed.
- Unseen residual-branch routing is an engineering inference policy: evaluate both frozen source branches and select the branch with higher Stage 8 compatibility C, tie-break by lower Stage 8 unseen distance. It adds no trained parameters and uses no labels, but still requires empirical validation on CIC-DDoS2019.
- Remaining API limitations: full streaming preprocessing for very large CSVs is not implemented; strict Option B is not exposed; validation-parity against a saved source-validation prediction artifact (max abs diff 1e-5) remains outstanding.

### Remaining blocked work

- CIC-DDoS2019 was previously inventoried on Google Drive (`G:\\My Drive\\CiC-DDos`, 13 CSV files, ~13.50 GB), but the G: mount is currently disconnected. No CIC evaluation was run. Restore Drive access before Phase D Option A/Option B evaluation.
- Leave-one-dataset-out validation diagnostics remain weak: NSL-KDD?UNSW-NB15 macro-F1 0.1312; UNSW-NB15?NSL-KDD macro-F1 0.2968 on shared classes. These are validation diagnostics, not CIC test results.
- Full test suite: 180 passed, 3 failed. Failure 1: `test_trainer_finite_losses_and_gradients` expects a gradient for `unknown_category_embedding`; Stage 5 remains protected and unchanged. Failures 2?3: raw-data leakage tests cannot open the NSL-KDD and UNSW-NB15 files because the Google Drive G: mount is disconnected.
- Stage 6 qualification wording (retain exactly): ?Stage 6 accepted with qualifications: the residual carries measurable detection information beyond random features; the learned encoder does not exceed a random projection of E_d; the original rank gate fails; the effect of L_decorr is not distinguishable from noise on a single seed.? Record `NO_QUALIFYING_STAGE6_ARM` under the original rule and Candidate 09 / lambda_decorr=1.0 under the revised criterion, with the note ?revised after results were seen, on logical grounds?.
- No Stage 5/6 training, checkpoint, threshold, or architecture changes were made during this continuation. No Git commands or package installations were performed.


## Final machine checkpoint ? 2026-11-26 21:xx local

- No active NOVEL-IDS Python process remains; Phase C and the API smoke process exited successfully.
- Final machine resources: 6.40 GiB free physical RAM; NVIDIA GPU 0 MiB used / 6001 MiB free / 0% utilization; C: disk 20.87 GB free.
- Google Drive G: is not mounted (`G:\My Drive\CiC-DDos` inaccessible). CIC-DDoS2019 evaluation is blocked by data access, not RAM.
- Final API smoke: PASS on 8 synthetic rows; finite confidence and OOD scores; both frozen residual branches selected by Stage-8 compatibility/distance. All 8 were Unfamiliar on this deliberately unmatched synthetic input; this is a smoke test, not a performance claim.
- Targeted Stage 12 and onboarding tests: 5 passed. Full suite: 180 passed, 3 failed (one pre-existing Stage 5 gradient expectation; two raw-data leakage tests blocked by unavailable G: source files).
- Protected Stage 5/6 source and checkpoint verification: PASS. Phase C results JSON exists with 24 prediction CSVs.
- Remaining to close project: restore G: access and run approved CIC Option A/Option B evaluation; complete validation-parity check (<=1e-5); implement full streaming preprocessing and strict Option B in the inference API; resolve the existing Stage 5 synthetic-gradient test without modifying protected code.
- Project is idle now because all currently accessible authorized work is complete; no process runs between turns.


## 2026-11-26 ? CIC-DDoS2019 missing-file constraint

- User reports five files missing from the CIC-DDoS2019 dataset. This is now an explicit constraint for Phase D. The previous inventory found 13 CSV files (~13.50 GB) under `G:\My Drive\CiC-DDos`; the current machine has no G: drive and no local `data/unseen` folder, so the missing filenames and expected total cannot yet be verified. Do not infer an 18-file expected set or invent filenames.
- Added `docs/CIC_DDOS2019_PARTIAL_DATASET_PLAN.md`. It requires authoritative manifest reconciliation, per-file SHA256/row/label/schema inventory, chunked processing, per-file/per-class sampling under the approved 300k cap, per-file and aggregate metrics, and explicit partial-coverage reporting. No missing file may be synthesized, duplicated, or silently ignored. CIC labels remain evaluation-only; Option A uses only unlabeled target features, Option B only exact feature matches and frozen source stats.
- No CIC evaluation was started because the G: mount is disconnected. No model, source data, labels, splits, or thresholds were changed.


## 2026-11-26 — CIC Drive inventory and streaming sample preparation

- Corrected earlier access diagnosis: G: is mounted and `G:\\My Drive\\CiC-DDos` is accessible in the remote Windows session.
- Recursive inventory: 13 CSV files, 14,492,315,542 bytes, in `01-12` and `03-11`. Metadata inventory artifacts: `C:\\NOVEL_IDS_runs\\phaseD_cic_partial_20261126\\cic_file_inventory.csv` and `cic_inventory_summary.json`.
- Exact reported filenames present: `MSSQL.csv` (03-11) and `Syn.csv` (01-12). Exact names absent: `NetBIOS.csv`, `TFTP.csv`, `UDP.csv`. Related prefixed variants include `DrDoS_NetBIOS.csv`, `DrDoS_UDP.csv`, `DrDoS_MSSQL.csv`; do not treat them as exact replacements absent an authoritative manifest.
- All 13 files share the same 88-column header in header inspection; label column is ` Label`. Preliminary first-10,000-row checks show a filename/content mismatch: `03-11/LDAP.csv` contains raw label `NetBIOS`, and `03-11/MSSQL.csv` contains raw label `LDAP`. Preserve raw row labels; never infer labels from filenames.
- Full-file SHA-256 inventory was attempted but was too slow on the streamed Google Drive mount and produced no output; that process was stopped. Metadata-only inventory is complete; no hashes are claimed.
- Added `scripts/prepare_cic_partial_sample.py`: bounded chunk scan (default 50,000 rows), full per-file row/label/missing/Inf profile, fixed-seed per-file/raw-label reservoir sample, original row index + source filename sidecar, 300,000-row hard cap, no raw-file modification. Bytecode compile and small synthetic code-path smoke test passed; synthetic smoke is not a research result.
- Actual streaming sample job started as PID 12524. Last progress: completed `01-12/DrDoS_DNS.csv` (5,074,413 rows; labels DrDoS_DNS 5,071,011, BENIGN 3,402), completed `01-12/DrDoS_LDAP.csv` (2,181,542 rows; DrDoS_LDAP 2,179,930, BENIGN 1,612), and was scanning `01-12/DrDoS_MSSQL.csv` (2,000,000 rows read at last progress). Sample/profile outputs are written only after the full scan completes. No inference/predictions/metrics yet.
- Next: monitor PID 12524; verify completed sample/profile and resource limits; reconcile full raw-label counts; implement frozen Option A sample-only unlabeled normalization through the existing API; assess Option B exact-match frozen-stat availability; run frozen zero-shot prediction once; calculate per-file and aggregate evaluation-only metrics and coverage limitations. Do not modify weights, thresholds, taxonomy, source partitions, or raw CSVs.


## 2026-11-26 — Phase D sample scan retry (no inference consumed)

- First full sample scan (50,000-row pandas C-engine chunks) failed with `OSError: [Errno 22] Invalid argument` while reading `01-12/DrDoS_NetBIOS.csv` after 2,000,000 rows from that file. It produced no sample CSV, sample manifest, profile JSON, or predictions. This was a data-read failure, not a model result.
- A test of the same file using 10,000-row chunks and the pandas C engine successfully read 2,000,000 rows in 9.4 seconds. A Python-engine label-only check also read 2,000,000 rows successfully, but was slower. No raw file was modified.
- Restarted full sample preparation with the same seed (20261126), same 7,500-per-file/raw-label reservoir cap and 300,000-row hard cap, now using 10,000-row chunks. Active sample job PID 9304. Latest observed progress: `01-12/DrDoS_DNS.csv` at 900,000 rows.
- The failed first inference-wait attempt is archived as `phaseD_auto_inference_attempt1.log`; it stopped before model inference because no sample existed. No CIC predictions or evaluation metrics have been generated. A fresh inference waiter (PID 10376) and metrics waiter (PID 3568) are queued; the latter waits for saved predictions and exits on a recorded stop/failure.
- Do not rerun inference if its output artifacts appear. The inference waiter is configured for one frozen CPU Option A pass, labels excluded, then the metrics waiter calculates evaluation-only metrics from saved predictions and the immutable sample manifest.


## 2026-11-27 ? Finish-today prompt final reconciliation

Source: user message, 2026-11-26 evening (finish-today prompt), plus execution verification on 2026-11-27. This entry supersedes earlier provisional status entries above where they say Phase C had no metrics, CIC evaluation was pending, or the API was incomplete. Historical failed attempts and logs remain preserved unchanged.

### Section 2A ? Phase C recovery ruling
- Used the approved NSL-KDD fine-label family map, including `saint` ? Probe ? `Probe_Recon`; UNSW label normalization and `Backdoors`/`Backdoor` alias; unknown labels remain `UNSUPPORTED_<label>` and are excluded from known-class metrics while counted for OOD.
- Evaluator was hardened and validation/synthetic dry-run checks passed. The final frozen recovery produced per-row predictions before calculating metrics; `results.json` is authoritative and 24 prediction CSVs are preserved under `C:\NOVEL_IDS_runs\phaseC_recovery_20261126`.
- Source-test metrics are reported in the final report and `PROGRESS.md`. `TEST PARTITIONS CLOSED ? 2026-11-26`. No source-test partition was accessed during API completion on 2026-11-27.

### Section 2B ? Backend inference API
- Implemented `load_model`, `onboard_csv`, `predict_csv`, `evaluate_predictions`, and the CLI. CSV profile/onboarding and prediction use bounded chunks; calibration is a deterministic unlabeled sample capped at 50,000 rows; prediction uses bounded batches; CPU fallback is supported.
- `OnboardingReport.config` now contains ordered schema, confidence-bearing feature matches, compatibility and target-only Option A normalization scope/statistics. Prediction summaries include decision counts, taxonomy class counts, thresholds, onboarding/config, model provenance and streaming settings.
- Validation-data input/output example pair is saved under `docs/examples/`; rows are from the NSL-KDD training-derived validation split only, with no labels. GPU source-validation parity passed the 1e-5 probability tolerance.
- Final synthetic end-to-end API stress run passed on 37 rows with renamed/shuffled/extra columns, all-NaN, infinity and unknown categorical values. Predictions with and without a `Label` column were bit-identical; all 37 deliberately unmatched synthetic rows were Unfamiliar (engineering evidence only). Focused API/Stage12 suite: 9 passed.
- The API loader uses source training/validation inputs and source-training OOD references; it does not load test partitions.

### Section 2C ? CIC-DDoS2019
- Partial data evaluation completed on 162,276 sampled rows across 13 available CSVs. Option A used unlabeled target-only calibration; 87/87 features were dataset-specific; strict Option B had zero exact matches and was inapplicable. All decisions were Unfamiliar and accepted coverage was 0%. Missing exact files: `NetBIOS.csv`, `TFTP.csv`, `UDP.csv`. No retraining, fine-tuning or threshold change.

### Section 2D ? Packaging
- README, `requirements.txt`, `docs/INFERENCE_API.md`, `PROGRESS.md`, and final report updated. Requirements were captured with `pip freeze`; no installs. No Git commands.
- Stage 5/6 protected files, weights, losses, architecture, thresholds, splits and preprocessing artifacts were not modified. The last full suite run before the final API contract test addition was 182 passed / 1 failed (protected Stage 5 synthetic gradient expectation); the final focused suite after all edits passed 9/9.


## 2026-11-27 ? Corrective audit of Phase C completeness

The exact finish-today prompt was checked against the frozen artifacts. All 26 freeze-manifest hashes and all 24 recorded Phase C model prediction CSV hashes match. The saved Phase C recovery includes the primary and five ablation variants, plus HGB/MLP metrics; no plain Transformer baseline artifact exists.

The saved Phase C baseline records contain confusion matrices/aggregate metrics but no per-row baseline prediction files with probability/OOD/decision fields. Primary metric objects omit explicit `uncertain_rate` and `accuracy_on_accepted`. The prompt requests six ablation variants in addition to the primary, while only five ablations are present in the frozen manifest/results.

No repair is authorized after the completed one-shot: the only prompt exception allows a metrics-only replay after a metrics-stage crash, and no crash occurred. Do not reopen test data, regenerate baseline predictions, add a test-time ablation, or recompute the missing metrics. These omissions are recorded as final deviations.

Additional API regression tests now cover malformed CSV, missing requested label, mixed non-numeric values, 100k-row chunked onboarding with bounded calibration, and the free-RAM hard stop. Focused API/onboarding/streaming/Stage12 tests: 13 passed. The source-validation parity harness uses `load_model()` and the internal source `batch_forward` helper, not `predict_csv()` on raw source CSV; the separate raw validation example pair is generated through `predict_csv()` but uses the unseen schema path. Preserve this distinction in future reporting.

## 2026-11-27 — Few-hours plan resource hard stop
- Source: user message, 2026-11-27.
- Decision: apply the explicit hard stop because initial preflight measured 1.11 GiB free RAM, below the 2 GiB minimum. Stop all further project work until memory is available.
- Free disk at preflight: 17.46 GiB. No GPU measurement was completed before stopping.
- No project computation, model inference, training, demo sample generation, test-prediction analysis, source/config edits, or protected-file edits were performed in this resumed session. The only intended writes are this decision entry and the progress entry documenting the stop.


## 2026-11-27 — resumed few-hours plan: resource-aware continuation

- Source: user message, 2026-11-27.
- User clarified that if RAM is constrained, continue other feasible steps rather than waiting for RAM recovery. This does not authorize crossing the project's explicit minimum 2 GiB free-RAM guardrail or starting memory-intensive jobs below it.
- Latest live preflight: 2.12 GiB free RAM; 17.22 GiB free disk. RAM is marginally above the hard floor. Proceeded with read-only inspection and documentation reconciliation only; no checkpoint load, training, bulk CSV processing or inference.
- Found docs/STATUS_VERIFIED.md was stale: it reported 25 collection errors / zero tests executed, while later progress and API documentation record a focused suite of 13 passing tests and GPU validation parity. Status file replaced with a dated evidence reconciliation and explicit remaining unknowns.
- Found docs/EVALUATION_REPORT.txt stale relative to the final Phase C counts: it reports 18,418 NSL independent and 61,747 UNSW independent evaluated rows, while final recovery records identify 21,880 and 73,791 source rows. It must not be treated as the final report without an authorized rebuild from saved prediction artifacts.
- Demo pack was absent (only demo_samples/outputs directory existed). No raw source validation CSVs were found in data/raw; only processed validation split joblib artifacts exist. KDD Cup 99 files were not located. No demo samples were fabricated.
- First CIC run remains the only documented CIC inference: 162,276 rows, 87 dataset-specific features, all Unfamiliar. No second run or semantic mapping is claimed.
- No protected Stage 5/6 file, dataset, split, preprocessing artifact, frozen checkpoint, or test partition was changed or reopened.


## 2026-11-27 execution decisions and audit trail

Source for all entries below: user message, 2026-11-27.

- S1: Use only saved Phase C prediction files/results for evaluation reporting. Do not rerun model inference on source test features. Preserve the original source-test result as the headline.
- S2a: Add best_guess_class and best_guess_probability to every row, including Uncertain/Unfamiliar rows; add dataset-level compatibility counts/score and configuration-backed High/Medium/Low trust; keep existing field meanings.
- S2c: Fit one additive per-class bias on calibrated source-validation log-probabilities only, using coordinate ascent over [-2.0, 2.0] in 0.25 increments, three passes, seed 42, and 0.002 squared-bias penalty. Validation gate passed: mean macro-F1 improvement 2.23 percentage points; dataset accuracy changes +0.09 pp NSL-KDD and -0.91 pp UNSW-NB15. Ship the bias in the inference path; retain original test metrics as headline and disclose the post-hoc nature of adjusted test numbers.
- Part D: Use only active validation partitions for NSL-KDD/UNSW-NB15, user-supplied KDD Cup 99 10% data if present, and CIC files already on the laptop. KDD Cup 99 source file was not supplied; do not replace it. Keep raw column names/values and separate labels from inference inputs.
- S3 mapping: Create CIC feature metadata and mapping before prediction. Mapping is documentation/descriptor-based and uses no CIC labels or performance. Level 1 only for flow duration, protocol, forward bytes and backward bytes; Level 2 is group_only; Level 3 remains dataset-specific. Mapping SHA-256 is recorded in PROGRESS.md before the second-run inference.
- S3 Option A: headline uses unlabeled target per-dataset median/IQR normalization (previously approved Option A). S3 Option B: strict comparison uses equal-dataset-mean source-training medians/IQRs for Level-1 numeric features only; other numeric features continue using unlabeled target statistics. No retraining, fine-tuning, threshold changes, or label use for mapping.
- S3 sample: use at most 150,000 rows from the locally available CIC aggregate, per-class cap 15,000, seed 42; keep labels in a separate evaluation-only file. Original per-row source-file identity was not preserved in the aggregate, so per-attack-file rates must be marked unavailable.
- S4: skip today. Also defer the unseen-dataset selection prompt and few-shot adaptation. Record as future work, not done.
- Protected scope: no Stage 5 source file, Stage 5/6 checkpoint, dataset, split, preprocessing artifact, or raw data was modified. No training or source-test inference was performed.


## 2026-11-27 - mandatory frozen-artifact integrity stop
- Source: user message, 2026-11-27; NOVEL-IDS few-hours plan Section 4.
- During the live S0 validation audit, `phasec.verify_manifest()` computed SHA-256 `7bd0200411bfe1d680e2b2e558163a4dc7dd520315062a6c91db4985fc3c8a4c` for the frozen `phaseC_evaluator_script` entry and it did not match the hash recorded in `freeze_manifest.json`.
- This is an explicit hard stop. No further model loading, evaluation, inference, training, source/config changes, or CIC retries are permitted until the project owner reconciles the frozen evaluator file and manifest outside this run. No attempt was made to overwrite the manifest, recalculate an accepted baseline, or bypass the check.
- The S0 script exited before loading the primary checkpoint or executing a validation batch. The report must not claim a live S0 runtime audit.


## 2026-11-27 — continuation scope clarification after Phase C integrity stop

- The frozen `phaseC_evaluator_script` mismatch blocks executing the Phase C evaluator and workflows that load the model through the current inference loader (which imports `run_phaseC_one_shot.py`). The manifest must not be rewritten or bypassed.
- The mismatch does not prohibit the user's explicitly ordered saved-prediction-only S1 report, read-only demo artifact verification, synthetic API tests that do not load a checkpoint, or documentation/packaging. Those independent tasks were continued.
- S1 report and figures were rebuilt from saved Phase C prediction CSVs and `results.json` only. No test feature rows were reopened and no model inference was run. Original pre-registered metrics remain the headline.
- The class-prior post-hoc comparison was computed from existing original/adjusted prediction CSVs only. It is disclosed as post-hoc; it does not replace the original headline.
- The six demo CSV hashes and row counts match the sample manifest. All six existing generic unseen-route summaries report 500 Unfamiliar, compatibility 0.0, Low trust. Documentation now states this rather than implying source-schema compatibility.
- Focused API/onboarding/streaming/Stage-12 synthetic tests passed 13/13 in 7.21 seconds without calling `load_model()`.
- CIC mapping files and hashes were verified read-only. No second CIC inference was run because the current API loader would import the mismatched Phase C evaluator module.


## 2026-11-27 — authoritative recovery-manifest reconciliation and final audit

- The original one-shot manifest predates the approved Phase C recovery amendment. Its `phaseC_evaluator_script` mismatch against the amended current script is expected and is not the active integrity failure.
- The completed recovery `results.json` references `C:\NOVEL_IDS_runs\phaseC_recovery_20261126\freeze_manifest.json`; its manifest SHA-256 `aab8674579ee39cbc54e9816ace0c76429413260102b4f4d33cfc6d3dd23c201` matches the value recorded in `results.json`.
- A read-only check of all 26 recovery manifest entries found 25 matches and one genuine mismatch: `phaseC_recovery_evaluator` (`scripts/phaseC_recovery_execute.py`), recorded SHA-256 `a63b65debd48127e94825f5d51e205f877de33219dd6338f907be7ba231c8cb2`, current SHA-256 `5b20ab90ee76f3872e65fc0fc9e76402f8abc55719da9d0eae6f2f363d2ecd3d`.
- The earlier S0 verifier calls `run_phaseC_one_shot.verify_manifest()` with the default original-manifest path; its first failure was therefore against the superseded manifest. The recovery-manifest audit then found the genuine recovery evaluator mismatch. No model was loaded after this audit and no manifest was modified.
- Final focused synthetic API/onboarding/streaming/Stage-12 tests rerun: 13 passed in 41.93s. Python compilation of inference API, CLI and `app.py` passed. These checks did not load the checkpoint.
- S1 report was supplemented with saved macro ROC-AUC/PR-AUC, HGB and MLP aggregate comparisons, and explicit independent-view risk-coverage values. All were extracted from frozen saved result/prediction artifacts; no inference or test feature reopening.
- Plan remains incomplete only where explicitly blocked/skipped: S0 runtime checkpoint audit and S3 mapping-conditioned Option A/Option B second-run inference/evaluation. S4 remains intentionally skipped. Do not bypass the active frozen recovery-evaluator mismatch.


## 2026-11-27 evening — owner-authorized integrity ruling, source-format correction and CIC run

Source: user message, 2026-11-27 evening.

### Section 1 — Phase C evaluator integrity ruling

- Do not edit either freeze manifest and do not rewrite the evaluator to match its recorded hash.
- The recovery manifest is authoritative. Its recorded SHA matches the value in results.json. The 25/26 audit matches every saved result/model entry except phaseC_recovery_evaluator.
- Recorded evaluator SHA: a63b65debd48127e94825f5d51e205f877de33219dd6338f907be7ba231c8cb2. Current SHA: 5b20ab90ee76f3872e65fc0fc9e76402f8abc55719da9d0eae6f2f363d2ecd3d.
- No exact-hash original copy was found in the recovery directory/project scripts. The original version is not recovered; no diff or hunk classification is possible. Current script mtime is later than results.json. Saved results and 24 prediction CSVs remain hash-verified; exact current-script replay is not guaranteed.
- Apply the owner ruling: the evaluator is complete and no longer a protected file for execution. S0 validation-only audit and CIC inference do not execute the recovery evaluator. No source-test inference was rerun.

### Section 2 — source-format demos

- The raw NSL-KDD and UNSW-NB15 samples already contain original source names and raw values; the generator selected raw validation rows. No regeneration, superseded folder, or sample-manifest replacement was necessary.
- Exact normalized schema fingerprints identify NSL-KDD 41/41 and UNSW-NB15 42/42. The route excludes label and dataset metadata columns, applies documented aliases, and uses the stored source-training-fitted preprocessors. Both schemas return compatibility 1.0 / High trust.
- The previous zero-compatibility outputs were produced by the generic unseen-schema route before source recognition was reflected in demo output generation.
- Source raw API predictions match saved Phase-E validation probabilities after applying the existing additive class-prior bias in log-probability space: max abs difference 2.47e-7 NSL and 5.73e-7 UNSW (limit 1e-5). Labeled/unlabeled prediction outputs are identical.
- Labeled 500-row demo metrics: NSL accuracy/macro-F1 1.000/1.000; UNSW 0.792/0.5907. Explicitly demo-only, not test results.
- CIC demo pair remains unknown schema, compatibility 0.465116 / Medium, all 500 Unfamiliar.
- An additive unseen-schema numeric conversion was necessary for CIC metadata-driven normalization: declared numeric/continuous/integer/binary columns are coerced in both calibration and prediction frames. Invalid tokens become masked missing values. This does not alter frozen model/preprocessing artifacts. Added a regression test.

### Section 3 — CIC second run

- Earlier 137,312-row attempt stopped at 73,728 rows with 3.06 GiB free RAM and no recorded inference error; PROGRESS.md records the earlier conservative integrity gate as the stop. Mapping SHA matched the pre-inference record.
- Rechecked mapping SHA 6cccab77a7892d469731955a519405a6eb1d4b370a3cacfbc7aa6a31f957eb59, prepared feature/label hashes, seed 42, 137,312 rows, 86 columns and 15,000 per-label cap. Disk free was 16.88 GiB; full sample retained.
- Option A completed. Compatibility 0.465116 (4 exact, 72 group-only, 10 dataset-specific), Medium trust, 137,312 Unfamiliar decisions, zero accepted coverage. Forced best-guess attack-vs-normal accuracy 0.568377, precision 0.974329, recall 0.529392, F1 0.686034. 1-P(Normal) AUROC/AUPR 0.771223/0.966248 (“useful signal”); OOD AUROC/AUPR 0.589223/0.870222 (“no usable signal”). Per-label rates and Normal/DoS matrix are in docs/EVALUATION_REPORT.md and the CIC run report.
- Option B stopped at the API hard guard when free RAM fell below 2 GiB. No Option B predictions/metrics and no retry. First 162,276-row CIC run preserved.
- No training, fine-tuning, threshold changes, or source-test access.

### Section 4 — S0, documentation and permanent deviations

- S0 runtime audit completed on 64 raw NSL-KDD validation rows only; Stage 5–11 tensors finite. Inventory, trainability, G/C stats, tensor shapes and ranks saved to docs/pipeline_truth_check.json. No test partition loaded.
- Updated STATUS_VERIFIED.md, EVALUATION_REPORT.md, FINAL_REPORT_2026-11-26.md, INFERENCE_API.md, README_DEMO_SAMPLES.md and PROGRESS.md; created PHASEC_INTEGRITY_NOTE.md.
- Focused suite: 18 passed before numeric coercion regression; 21 passed in 9.73s after.
- Permanent deviations: missing sixth ablation; baseline per-row artifacts absent; no plain Transformer test numbers; evaluator original unrecovered; CIC Option B incomplete; Stage 6 qualification unchanged; CIC source-file identity unavailable; KDD Cup 99 not supplied; S4 remains skipped.
- Stage 6 qualification: original pre-registered criterion result NO_QUALIFYING_STAGE6_ARM. Candidate 09 / lambda_decorr=1.0 uses a revised criterion applied after results were seen on logical grounds. Residual contains measurable detection information beyond random features, but learned encoder does not exceed random projection of E_d; original rank gate fails; L_decorr effect is not distinguishable from noise on one seed.
- No Stage 5 file, snapshot, checkpoint, dataset, labels, splits, fitted preprocessing artifact, taxonomy, calibration or thresholds changed. No Git or installs.


## 2026-11-27 — Option B CIC continuation result

User instruction: “check ram and continue the pending work fast.” Fresh preflight: 3.81 GiB free RAM and 21.94 GiB free C: disk. The previous Option B RAM failure was addressed by a single resource-conscious retry with batch size 32 and chunk size 2,048; the same frozen model, 137,312-row seed-42 feature sample, mapping, metadata, and normalization mode were used. The API 2 GiB guard stayed active. Option B completed, ending at 2.681 GiB free RAM. No labels were read until predictions were saved. No training, fine-tuning, threshold change, test partition access, or protected-file change. Result: all rows Unfamiliar (0% accepted coverage); forced diagnostics are not accepted operational metrics. Combined metrics and hashes are in the CIC second-run report.


## 2026-11-27 ? Dashboard operational verification completed

- Streamlit 1.67.0 was already present in the project virtual environment; no package installation was performed.
- Streamlit AppTest loaded the approved frozen checkpoint on CPU and reported ?Frozen model loaded and verified?; checkpoint, Stage 5 and Stage 6 hashes matched the loaded model info.
- The dashboard uploaded docs/examples/inference_validation_input.csv (12 raw NSL-KDD validation feature rows; no labels), completed inference for 12 rows, and produced finite numeric outputs with no app exceptions.
- Exact source schema recognized as NSL-KDD: 41 available features, compatibility 1.0, trust High. Decisions: 7 Known attack, 4 Normal, 1 Unfamiliar. This is an engineering smoke test on validation data, not a new research metric.
- Prediction CSV and summary JSON serialized successfully (3,393 and 41,623 bytes); both corresponding download controls are wired in app.py.
- Headless Streamlit server returned HTTP 200 for /_stcore/health and /; the test server was stopped after verification.
- No training, fine-tuning, threshold change, package install, source-test access, or protected artifact modification.
