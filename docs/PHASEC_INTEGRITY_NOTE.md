# Phase C integrity note — 2026-11-27

## Ruling and hashes

The original one-shot manifest was superseded by the approved Phase C recovery manifest. The recovery manifest remains authoritative for the saved recovery results.

- Recovery manifest SHA-256: aab8674579ee39cbc54e9816ace0c76429413260102b4f4d33cfc6d3dd23c201; this matches the manifest hash recorded in results.json.
- Frozen recovery evaluator recorded SHA-256: a63b65debd48127e94825f5d51e205f877de33219dd6338f907be7ba231c8cb2.
- Current scripts/phaseC_recovery_execute.py SHA-256: 5b20ab90ee76f3872e65fc0fc9e76402f8abc55719da9d0eae6f2f363d2ecd3d.
- Read-only recovery-manifest audit: 25 of 26 entries match. The sole mismatch is phaseC_recovery_evaluator.
- All 24 recorded prediction CSV hashes match. The primary checkpoint, Stage 5 snapshot, Stage 6 checkpoint, taxonomy, evaluation label map, calibration and threshold-bearing calibration artifact hashes match their recorded values.
- Computed results.json SHA-256 for this audit: 640601c0395ce3fff0ca80be94a9b5a6cdf2d28f0284bfb0ab3615668e0a5601. The recovery manifest hash is recorded inside results.json; no separate expected results.json file hash was present in the manifest.

## Forensics

A read-only search of the recovery run directory and project scripts found no copy with the recorded evaluator hash, including same-size candidates and backup-like copies. The original evaluator version was not recovered. The current evaluator file modification time is 2026-11-27 02:23:49; results.json modification time is 2026-11-26 21:06:55. The saved results therefore predate the current evaluator file.

No original source copy was available for a unified diff. Consequently, the changed hunks cannot be classified as value-computing, reporting-only, or other. The original version is not recovered; no claim is made that the current evaluator is byte-for-byte or behaviorally equivalent.

If exact evaluator replay is ever required, the project owner must restore the original file from OneDrive version history or another owner-controlled backup. This is not required for the approved next steps below.

## Effect on saved results and execution

The saved results are unaffected because they were produced before the current evaluator modification and the saved result/prediction artifacts and protected model artifacts are hash-verified. Exact re-execution from the current evaluator script is not guaranteed.

Per the project-owner ruling dated 2026-11-27, the Phase C evaluator is complete and is no longer a protected file for execution. The S0 validation-only runtime audit and the CIC inference path do not execute scripts/phaseC_recovery_execute.py. This mismatch is therefore not a blocker for those tasks.

No source-test inference was rerun. No test feature partition was reopened. Neither freeze manifest was edited, and the evaluator was not rewritten to match the recorded hash.
