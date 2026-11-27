from __future__ import annotations

import ctypes
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from novel_ids.residual.stage6_data import load_stage6_label_artifact
from novel_ids.residual.stage6_training import Stage6Trainer, Stage6TrainingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE5_SNAPSHOT = Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
STAGE5_SHA256 = "9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b"
SCREENING_CONFIG = PROJECT_ROOT / "config" / "stage6_projection.yaml"
OUTPUT_ROOT = Path(r"C:\NOVEL_IDS_runs\stage6\phase2_decorr_20261125")
CANDIDATES = {
    "candidate09": {"residual_dim": 128, "projection_mode": "identity", "lambda_red": 1.0},
    "candidate07": {"residual_dim": 128, "projection_mode": "identity", "lambda_red": 0.01},
}
DECORR_WEIGHTS = (0.1, 1.0)
BOOTSTRAP_REPS = 2000


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temp.replace(path)


def all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, (int, np.integer, str, bool)) or value is None:
        return True
    return True


def paired_bootstrap_ci(y: np.ndarray, pred_base: np.ndarray, pred_aug: np.ndarray,
                         seed: int = 42) -> dict[str, float]:
    y = np.asarray(y).astype(str)
    pred_base = np.asarray(pred_base).astype(str)
    pred_aug = np.asarray(pred_aug).astype(str)
    labels = np.unique(y)
    observed = float(
        f1_score(y, pred_aug, labels=labels, average="macro", zero_division=0)
        - f1_score(y, pred_base, labels=labels, average="macro", zero_division=0)
    )
    rng = np.random.default_rng(seed)
    n = y.size
    gains = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, size=n)
        gains[i] = (
            f1_score(y[idx], pred_aug[idx], labels=labels, average="macro", zero_division=0)
            - f1_score(y[idx], pred_base[idx], labels=labels, average="macro", zero_division=0)
        )
    return {
        "observed_gain": observed,
        "ci95_lower": float(np.quantile(gains, 0.025)),
        "ci95_upper": float(np.quantile(gains, 0.975)),
        "bootstrap_replicates": int(BOOTSTRAP_REPS),
        "bootstrap_seed": int(seed),
    }


def make_config(raw: dict[str, Any], candidate: dict[str, Any], seed: int,
                lambda_decorr: float) -> Stage6TrainingConfig:
    t = raw["training"]
    loss = t["loss_coefficients"]
    return Stage6TrainingConfig(
        residual_dim=int(candidate["residual_dim"]),
        lambda_red=float(candidate["lambda_red"]),
        variance_floor=float(t["variance_floor"]),
        projection_mode=str(candidate["projection_mode"]),
        sub_batch_size=int(t["sub_batch_size_per_dataset"]),
        validation_batch_size=int(t["validation_batch_size"]),
        max_steps=int(t["max_steps"]),
        validation_interval=int(t["validation_interval"]),
        early_stop_patience=int(t["early_stop_patience"]),
        min_relative_improvement=float(t["min_relative_improvement"]),
        warmup_fraction=0.10,
        lambda_pred=float(loss["lambda_pred"]),
        lambda_info=float(loss["lambda_info"]),
        lambda_mag=float(loss["lambda_mag"]),
        lambda_proj=float(loss["lambda_proj"]),
        decorr_enabled=True,
        lambda_decorr=float(lambda_decorr),
        seed=int(seed),
        device="cuda",
    )


def fit_probe_pair(trainer: Stage6Trainer, dataset: str, seed: int) -> dict[str, Any]:
    label_path = trainer.provenance["source_label_artifact_paths"][dataset]
    label_artifact = load_stage6_label_artifact(label_path, dataset)
    y_train = np.asarray(label_artifact["train"]["labels"]).astype(str)
    y_val = np.asarray(label_artifact["validation"]["labels"]).astype(str)
    del label_artifact
    train_zs, train_r = trainer.extract_features(dataset, "train")
    val_zs, val_r = trainer.extract_features(dataset, "validation")

    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced",
                           random_state=seed),
    )
    augmented = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced",
                           random_state=seed),
    )
    base.fit(train_zs, y_train)
    augmented.fit(np.concatenate([train_zs, train_r], axis=1), y_train)
    pred_base = base.predict(val_zs)
    pred_aug = augmented.predict(np.concatenate([val_zs, val_r], axis=1))
    labels = np.unique(y_val)
    base_f1 = float(f1_score(y_val, pred_base, labels=labels, average="macro", zero_division=0))
    aug_f1 = float(f1_score(y_val, pred_aug, labels=labels, average="macro", zero_division=0))
    ci = paired_bootstrap_ci(y_val, pred_base, pred_aug, seed=seed)
    return {
        "baseline_Zs_macro_f1": base_f1,
        "augmented_Zs_R_macro_f1": aug_f1,
        "gain": aug_f1 - base_f1,
        "paired_bootstrap": ci,
        "train_rows": int(y_train.size),
        "validation_rows": int(y_val.size),
        "validation_support": {str(c): int(np.sum(y_val == c)) for c in labels},
    }


def gpu_memory() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def attach_gradient_monitor(trainer: Stage6Trainer) -> list[dict[str, Any]]:
    """Sample post-clipping Stage 6 gradient norms every 500 optimizer steps."""
    samples: list[dict[str, Any]] = []
    original_step = trainer.optimizer.step

    def monitored_step(*args, **kwargs):
        next_step = trainer.global_step + 1
        if next_step % 500 == 0:
            group_sq: dict[str, float] = {}
            total_sq = 0.0
            for name, parameter in trainer.branches.named_parameters():
                if parameter.grad is None:
                    continue
                norm_sq = float(parameter.grad.detach().float().square().sum().cpu())
                total_sq += norm_sq
                parts = name.split(".")
                group = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
                group_sq[group] = group_sq.get(group, 0.0) + norm_sq
            samples.append({
                "step": int(next_step),
                "global_post_clip_grad_norm": float(np.sqrt(total_sq)),
                "group_post_clip_grad_norms": {k: float(np.sqrt(v)) for k, v in group_sq.items()},
            })
        return original_step(*args, **kwargs)

    trainer.optimizer.step = monitored_step
    return samples


def run_arm(raw: dict[str, Any], candidate_name: str, candidate: dict[str, Any],
            lambda_decorr: float, seed: int, arm_dir: Path) -> dict[str, Any]:
    config = make_config(raw, candidate, seed, lambda_decorr)
    gpu_before = gpu_memory()
    torch.cuda.reset_peak_memory_stats()
    trainer = Stage6Trainer.from_project(
        stage5_snapshot=STAGE5_SNAPSHOT,
        expected_stage5_sha256=STAGE5_SHA256,
        run_dir=arm_dir,
        config=config,
    )
    if trainer.device.type != "cuda":
        raise RuntimeError("Phase 2 requires CUDA.")
    if any(p.requires_grad for p in trainer.stage5.parameters()):
        raise RuntimeError("Stage 5 must remain frozen.")
    gradient_norm_samples = attach_gradient_monitor(trainer)
    fit = trainer.fit()
    checkpoint_path = arm_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Missing best checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage5_checkpoint_sha256", "").lower() != STAGE5_SHA256:
        raise RuntimeError("Stage 5 snapshot provenance mismatch in Phase 2 checkpoint.")
    trainer.branches.load_state_dict(checkpoint["branches"], strict=True)
    trainer.branches.eval()
    validation = trainer.validate()
    probes = {
        name: fit_probe_pair(trainer, name, seed)
        for name in ("nsl_kdd", "unsw_nb15")
    }
    mean_gain = float(np.mean([probes[n]["gain"] for n in probes]))
    dimension = int(candidate["residual_dim"])
    floor_fraction = {
        name: float(validation[name]["near_zero_dimensions"]) / dimension
        for name in ("nsl_kdd", "unsw_nb15")
    }
    rank_threshold = 32 if dimension == 128 else 16
    rank_status = {
        name: bool(validation[name]["effective_rank_participation"] >= rank_threshold)
        for name in ("nsl_kdd", "unsw_nb15")
    }
    finite = all_finite(fit) and all_finite(validation) and all_finite(probes)
    reconstruction_positive = all(
        validation[n]["reconstruction_skill"] > 0 for n in ("nsl_kdd", "unsw_nb15")
    )
    floor_qualifies = max(floor_fraction.values()) <= 0.25
    qualifies = finite and reconstruction_positive and floor_qualifies
    loss_history_means = {
        key: float(np.mean([entry[key] for entry in trainer.loss_history]))
        for key in trainer.loss_history[0]
        if key != "step" and all(key in entry for entry in trainer.loss_history)
    } if trainer.loss_history else {}
    result = {
        "candidate": candidate_name,
        "candidate_parameters": candidate,
        "label": "failed the original rank gate",
        "lambda_decorr": float(lambda_decorr),
        "seed": int(seed),
        "fit": fit,
        "loss_history_means": loss_history_means,
        "gradient_norm_samples_post_clip": gradient_norm_samples,
        "validation": validation,
        "probe": probes,
        "mean_validation_probe_gain": mean_gain,
        "finite": finite,
        "positive_reconstruction_both_datasets": reconstruction_positive,
        "variance_floor_fraction": floor_fraction,
        "floor_qualifies": floor_qualifies,
        "rank_threshold": rank_threshold,
        "rank_gate_pass": rank_status,
        "qualifies_for_best_arm_selection": qualifies,
        "stage5_snapshot_sha256": checkpoint["stage5_checkpoint_sha256"],
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest(),
        "gpu_memory_before": gpu_before,
        "gpu_memory_after": gpu_memory(),
    }
    write_json(arm_dir / "phase2_arm_summary.json", result)
    del trainer, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result


def available_ram_gb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed.")
    return float(status.ullAvailPhys) / (1024 ** 3)


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("This fixed approved Phase 2 runner takes no arguments.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    if free_gpu < int(5.0 * 1024**3):
        raise RuntimeError(f"Insufficient free GPU memory: {free_gpu} of {total_gpu} bytes." )
    available_ram = available_ram_gb()
    if available_ram < 4.0:
        raise RuntimeError(f"Insufficient free RAM: {available_ram:.2f} GB; require >=4 GB.")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 2 output: {OUTPUT_ROOT}")
    torch.set_num_threads(4 if (os.cpu_count() or 1) >= 8 else 2)
    raw = yaml.safe_load(SCREENING_CONFIG.read_text(encoding="utf-8"))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT_ROOT / "phase2_config.json", {
        "candidates": CANDIDATES,
        "lambda_decorr": DECORR_WEIGHTS,
        "seeds": [42, 43, 44],
        "source_config": raw,
        "stage5_snapshot_sha256": STAGE5_SHA256,
    })
    results = []
    for candidate_name, candidate in CANDIDATES.items():
        for lam in DECORR_WEIGHTS:
            arm_name = f"{candidate_name}_lambda{lam:g}_seed42"
            print("PHASE2_ARM_START", arm_name, "gpu", gpu_memory(), flush=True)
            arm_dir = OUTPUT_ROOT / arm_name
            result = run_arm(raw, candidate_name, candidate, lam, 42, arm_dir)
            results.append(result)
            write_json(OUTPUT_ROOT / "phase2_partial_results.json", results)
            print("PHASE2_ARM_COMPLETE", arm_name,
                  "gain", result["mean_validation_probe_gain"],
                  "qualifies", result["qualifies_for_best_arm_selection"],
                  "gpu", result["gpu_memory_after"], flush=True)

    eligible = [r for r in results if r["qualifies_for_best_arm_selection"]]
    eligible.sort(key=lambda r: r["mean_validation_probe_gain"], reverse=True)
    final: dict[str, Any] = {"seed42_arms": results, "selected_best_arm": None, "three_seed_results": []}
    if eligible:
        best = eligible[0]
        final["selected_best_arm"] = {
            "candidate": best["candidate"],
            "lambda_decorr": best["lambda_decorr"],
            "mean_validation_probe_gain": best["mean_validation_probe_gain"],
        }
        final["three_seed_results"].append(best)
        candidate = CANDIDATES[best["candidate"]]
        for seed in (43, 44):
            arm_name = f"{best['candidate']}_lambda{best['lambda_decorr']:g}_seed{seed}"
            print("PHASE2_SEED_START", arm_name, "gpu", gpu_memory(), flush=True)
            result = run_arm(raw, best["candidate"], candidate, best["lambda_decorr"],
                             seed, OUTPUT_ROOT / arm_name)
            final["three_seed_results"].append(result)
            write_json(OUTPUT_ROOT / "phase2_partial_results.json", {
                "seed42_arms": results,
                "selected_best_arm": final["selected_best_arm"],
                "three_seed_results": final["three_seed_results"],
            })
            print("PHASE2_SEED_COMPLETE", arm_name,
                  "gain", result["mean_validation_probe_gain"],
                  "gpu", result["gpu_memory_after"], flush=True)
    else:
        final["status"] = "NO_PHASE2_ARM_QUALIFIED"
    write_json(OUTPUT_ROOT / "phase2_final_results.json", final)
    print("PHASE2_COMPLETE", final.get("status", "THREE_SEED_ARM_COMPLETE"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("PHASE2_STOPPED", traceback.format_exc(), flush=True)
        raise



