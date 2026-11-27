from __future__ import annotations

import gc
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from novel_ids.residual.stage6_training import (
    Stage6Trainer,
    Stage6TrainingConfig,
    screening_rejection_reasons,
)
from novel_ids.residual.stage6_probes import run_source_validation_probes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE5_SNAPSHOT = Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
STAGE5_SHA256 = "9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )


def make_config(raw: dict[str, Any], candidate: dict[str, Any], seed: int) -> Stage6TrainingConfig:
    return Stage6TrainingConfig(
        residual_dim=int(candidate["residual_dim"]),
        lambda_red=float(candidate["lambda_red"]),
        variance_floor=float(raw["training"]["variance_floor"]),
        projection_mode=str(candidate["projection_mode"]),
        sub_batch_size=int(raw["training"]["sub_batch_size_per_dataset"]),
        validation_batch_size=int(raw["training"]["validation_batch_size"]),
        max_steps=int(raw["training"]["max_steps"]),
        validation_interval=int(raw["training"]["validation_interval"]),
        early_stop_patience=int(raw["training"]["early_stop_patience"]),
        min_relative_improvement=float(raw["training"]["min_relative_improvement"]),
        warmup_fraction=float(raw["training"]["warmup_fraction"]),
        lambda_pred=float(raw["training"]["loss_coefficients"]["lambda_pred"]),
        lambda_info=float(raw["training"]["loss_coefficients"]["lambda_info"]),
        lambda_mag=float(raw["training"]["loss_coefficients"]["lambda_mag"]),
        lambda_proj=float(raw["training"]["loss_coefficients"]["lambda_proj"]),
        seed=int(seed),
        device="cuda",
    )


def candidate_key(candidate: dict[str, Any]) -> str:
    projection = str(candidate["projection_mode"])
    return (
        f"d{int(candidate['residual_dim'])}_{projection}"
        f"_lred{float(candidate['lambda_red']):g}"
    )


def gpu_memory() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def load_best_branches(trainer: Stage6Trainer, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("stage5_checkpoint_sha256", "").lower() != STAGE5_SHA256:
        raise RuntimeError(f"Stage5 checkpoint provenance mismatch in {checkpoint_path}")
    trainer.branches.load_state_dict(checkpoint["branches"], strict=True)
    trainer.branches.eval()
    return checkpoint


def train_one(
    raw: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
    run_dir: Path,
) -> tuple[dict[str, Any], Path]:
    config = make_config(raw, candidate, seed)
    print(
        "RUN_START",
        candidate_key(candidate),
        "seed", seed,
        "variance_floor", config.variance_floor,
        "gpu_before", gpu_memory(),
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    trainer = Stage6Trainer.from_project(
        stage5_snapshot=STAGE5_SNAPSHOT,
        expected_stage5_sha256=STAGE5_SHA256,
        run_dir=run_dir,
        config=config,
    )
    if trainer.device.type != "cuda":
        raise RuntimeError("Stage6 screening was required to run on CUDA but resolved to CPU.")
    if any(p.requires_grad for p in trainer.stage5.parameters()):
        raise RuntimeError("Stage5 unexpectedly unfrozen before Stage6 training.")
    fit_result = trainer.fit()
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Best checkpoint was not written: {checkpoint_path}")
    checkpoint = load_best_branches(trainer, checkpoint_path)
    validation = trainer.validate()
    rejection = screening_rejection_reasons(validation, config.residual_dim)
    memory = gpu_memory()
    result = {
        "candidate": candidate,
        "candidate_key": candidate_key(candidate),
        "seed": seed,
        "fit": fit_result,
        "best_checkpoint": str(checkpoint_path),
        "best_checkpoint_global_step": int(checkpoint["global_step"]),
        "best_checkpoint_stage5_sha256": checkpoint["stage5_checkpoint_sha256"],
        "validation": validation,
        "rejection_reasons": rejection,
        "passed_validation_gates": not rejection,
        "gpu_memory": memory,
    }
    write_json(run_dir / "screening_summary.json", result)
    print(
        "RUN_COMPLETE",
        candidate_key(candidate),
        "seed", seed,
        "best_step", checkpoint.get("best_step"),
        "validation_objective", validation["mean_validation_objective"],
        "rejections", rejection,
        "gpu_after", memory,
        flush=True,
    )
    del trainer, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result, checkpoint_path


def probe_checkpoint(
    raw: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
    checkpoint_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    config = make_config(raw, candidate, seed)
    trainer = Stage6Trainer.from_project(
        stage5_snapshot=STAGE5_SNAPSHOT,
        expected_stage5_sha256=STAGE5_SHA256,
        run_dir=run_dir,
        config=config,
    )
    checkpoint = load_best_branches(trainer, checkpoint_path)
    validation = trainer.validate()
    rejection = screening_rejection_reasons(validation, config.residual_dim)
    if rejection:
        raise RuntimeError(
            f"Finalist seed {seed} failed validation gates: {rejection}"
        )
    probe = run_source_validation_probes(trainer, seed=seed)
    result = {
        "candidate": candidate,
        "candidate_key": candidate_key(candidate),
        "seed": seed,
        "source_validation": validation,
        "probe": probe,
        "checkpoint": str(checkpoint_path),
        "stage5_sha256": checkpoint["stage5_checkpoint_sha256"],
        "gpu_memory": gpu_memory(),
    }
    write_json(run_dir / "probe_summary.json", result)
    del trainer, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/run_stage6_screening.py "
            r"C:\NOVEL_IDS_runs\stage6\approved_screening_run"
        )
    run_root = Path(sys.argv[1])
    if run_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Stage6 run directory: {run_root}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; approved Stage6 training requires GPU.")
    free, total = torch.cuda.mem_get_info()
    if free < int(5.0 * 1024**3):
        raise RuntimeError(
            f"Insufficient free GPU memory before training: {free} of {total} bytes."
        )

    config_path = PROJECT_ROOT / "config" / "stage6_projection.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidates = raw["screening_matrix"]["candidates"]
    if len(candidates) != 9:
        raise ValueError(f"Expected approved nine-run matrix, got {len(candidates)}.")
    if float(raw["training"]["variance_floor"]) != 1e-4:
        raise ValueError("Approved variance_floor must be exactly 1e-4.")
    if int(raw["screening_matrix"]["seed"]) != 42:
        raise ValueError("Screening seed must remain 42.")
    if int(raw["training"]["max_steps"]) != 10000:
        raise ValueError("Unexpected maximum-step configuration.")
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "experiment_config.json", raw)

    print(
        "STAGE6_SCREENING_BEGIN",
        "candidates", len(candidates),
        "variance_floor", raw["training"]["variance_floor"],
        "gpu", torch.cuda.get_device_name(0),
        "memory", gpu_memory(),
        flush=True,
    )

    screening_results = []
    for index, candidate in enumerate(candidates, start=1):
        run_dir = run_root / "screening" / f"{index:02d}_{candidate_key(candidate)}"
        result, _ = train_one(
            raw, candidate, int(raw["screening_matrix"]["seed"]), run_dir
        )
        screening_results.append(result)
        write_json(run_root / "screening_results.json", screening_results)

    eligible = [
        item for item in screening_results if item["passed_validation_gates"]
    ]
    eligible.sort(key=lambda item: item["validation"]["mean_validation_objective"])
    if len(eligible) < 2:
        final = {
            "status": "STOP_FEWER_THAN_TWO_ELIGIBLE_SCREENING_CANDIDATES",
            "screening_results": screening_results,
        }
        write_json(run_root / "final_report.json", final)
        print(final["status"], flush=True)
        return 2

    finalists = eligible[: int(raw["screening_matrix"]["finalist_selection"]["count"])]
    write_json(run_root / "selected_finalists.json", finalists)
    finalist_results = []
    for candidate_result in finalists:
        candidate = candidate_result["candidate"]
        key = candidate_key(candidate)
        seed_results = []
        screening_checkpoint = Path(candidate_result["best_checkpoint"])

        # Reuse the seed-42 screening checkpoint; do not retrain seed 42.
        seed42_probe = probe_checkpoint(
            raw,
            candidate,
            42,
            screening_checkpoint,
            run_root / "finalists" / key / "seed42_probe_loader",
        )
        seed_results.append(seed42_probe)

        candidate_failed = False
        for seed in (43, 44):
            seed_run_dir = run_root / "finalists" / key / f"seed{seed}_training"
            trained, checkpoint_path = train_one(
                raw, candidate, seed, seed_run_dir
            )
            if not trained["passed_validation_gates"]:
                seed_results.append({
                    "seed": seed,
                    "passed_validation_gates": False,
                    "rejection_reasons": trained["rejection_reasons"],
                    "training_summary": trained,
                })
                candidate_failed = True
                # Preserve the approved paired seed design: still run seed 44
                # even if seed 43 fails. The candidate will remain ineligible
                # because not all three seeds passed the validation gates.
                continue
            probe_result = probe_checkpoint(
                raw,
                candidate,
                seed,
                checkpoint_path,
                run_root / "finalists" / key / f"seed{seed}_probe_loader",
            )
            seed_results.append(probe_result)

        gains = [
            float(item["probe"]["mean_probe_gain"])
            for item in seed_results
            if "probe" in item
        ]
        candidate_summary = {
            "candidate": candidate,
            "candidate_key": key,
            "seed_results": seed_results,
            "all_finalist_seeds_passed": not candidate_failed and len(gains) == 3,
            "probe_gain_mean": float(np.mean(gains)) if gains else None,
            "probe_gain_sd": float(np.std(gains, ddof=1)) if len(gains) > 1 else None,
        }
        finalist_results.append(candidate_summary)
        write_json(run_root / "finalist_results.json", finalist_results)

    complete = [
        item for item in finalist_results
        if item["all_finalist_seeds_passed"] and item["probe_gain_mean"] is not None
    ]
    if not complete:
        final = {
            "status": "NO_FINALIST_COMPLETED_ALL_SEEDS_AND_GATES",
            "screening_results": screening_results,
            "finalist_results": finalist_results,
        }
    elif len(complete) == 1:
        final = {
            "status": "ONLY_ONE_FINALIST_COMPLETED_ALL_SEEDS",
            "eligible_finalist": complete[0],
            "screening_results": screening_results,
            "finalist_results": finalist_results,
        }
    else:
        ranked = sorted(complete, key=lambda item: item["probe_gain_mean"], reverse=True)
        first, second = ranked[:2]
        pooled_sd = math.sqrt(
            (first["probe_gain_sd"] ** 2 + second["probe_gain_sd"] ** 2) / 2.0
        )
        difference = first["probe_gain_mean"] - second["probe_gain_mean"]
        if first["probe_gain_mean"] <= first["probe_gain_sd"]:
            status = "R_ADDS_NO_MEASURABLE_DETECTION_INFORMATION"
            winner = None
        elif abs(difference) <= pooled_sd:
            if first["candidate"]["residual_dim"] != second["candidate"]["residual_dim"]:
                winner = min(
                    (first, second),
                    key=lambda item: item["candidate"]["residual_dim"],
                )
                status = "TIE_WITHIN_ONE_SEED_SD_SMALLER_WIDTH_PREFERRED"
            else:
                winner = None
                status = "INCONCLUSIVE_WITHIN_SEED_NOISE_NO_APPROVED_SAME_WIDTH_TIEBREAK"
        else:
            winner = first
            status = "WINNER_BY_MEAN_VALIDATION_PROBE_GAIN"
        final = {
            "status": status,
            "winner": winner,
            "pooled_seed_sd": pooled_sd,
            "probe_gain_difference": difference,
            "screening_results": screening_results,
            "finalist_results": finalist_results,
        }
    write_json(run_root / "final_report.json", final)
    print("STAGE6_EXPERIMENT_COMPLETE", final["status"], flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("STAGE6_STOPPED_ON_ERROR", traceback.format_exc(), flush=True)
        raise
