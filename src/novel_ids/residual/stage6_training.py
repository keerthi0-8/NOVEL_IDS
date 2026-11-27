from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from novel_ids.residual.stage6_data import load_stage6_input_artifact
from novel_ids.residual.stage6_decorr import residual_decorrelation_loss
from novel_ids.representation.dual_encoder import DualEncoder
from novel_ids.residual.stage6 import (
    Stage6LossCoefficients,
    Stage6ResidualBranch,
    combine_stage6_losses_with_projection,
    freeze_stage5,
    information_reconstruction_loss,
    initialize_fixed_semi_orthogonal_projection,
    magnitude_stability_loss,
    prediction_loss,
    projection_geometry_loss,
    redundancy_warmup,
    standardized_cross_covariance_loss,
)


STAGE5_PROTECTED_HASHES = {
    "src/novel_ids/representation/stage5_training.py": "EE3D8FE2DA5C76F9FBAC1D75E48EAA029E6A13AF68F96CF3BC81F13D22F98267",
    "src/novel_ids/representation/stage5_training_candidate.py": "0A1C75A046C9CE86BF122D4E4CECB65DF14365E2114760040B30D52787015130",
    "src/novel_ids/representation/dual_encoder.py": "1FF74D97E651C541B9AB6F84EC762F054D6CF03E24F054A8BDF9C8D723AC0FFC",
    "src/novel_ids/representation/classification_head.py": "D4D0C337AC8C9A234239461282DFD5A4753B7A660C9ADA91D9C404B77900EED8",
    "src/novel_ids/representation/reconstruction_decoder.py": "E043AEC1F54367A18A89DA762FE2BFD24A1898AA5FEA07AA2B81AB85F865243A",
    "src/novel_ids/representation/feature_identity.py": "D82C46AACCA3C4C53777A8AB7C2F1D0930FE8E44E2D554A912578482091942AC",
    "config/datasets.yaml": "ADE3CB9E5E12015542ECEB6BB2ACFA5F440551F72278ADE7C5BF4A955FE836BE",
    "config/preprocessing_policy.yaml": "123BDD2104D01C0839E93AAF7B2322072786AF7987AA7213A2247878CBCCD6EF",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_absolute_active_cross_correlation(
    z_s: torch.Tensor, residual: torch.Tensor, variance_floor: float
) -> float:
    """Mean absolute active cross-correlation using the approved per-batch stats."""
    zs = z_s.detach().float()
    r = residual.detach().float()
    zs = zs - zs.mean(dim=0, keepdim=True)
    r = r - r.mean(dim=0, keepdim=True)
    zs_var = zs.square().mean(dim=0)
    r_var = r.square().mean(dim=0)
    zs_active = zs_var >= variance_floor ** 2
    r_active = r_var >= variance_floor ** 2
    if not bool(zs_active.any()) or not bool(r_active.any()):
        return 0.0
    zs_std = zs_var[zs_active].sqrt()
    r_std = r_var[r_active].sqrt()
    corr = (
        zs[:, zs_active].T @ r[:, r_active]
    ) / (zs.shape[0] - 1)
    corr = corr / zs_std.unsqueeze(1) / r_std.unsqueeze(0)
    return float(corr.abs().mean().cpu())


INPUT_FIELDS = (
    "numeric_values",
    "categorical_values",
    "M_avail",
    "M_miss",
)


@dataclass(frozen=True)
class Stage6TrainingConfig:
    residual_dim: int
    lambda_red: float
    variance_floor: float
    projection_mode: str
    predictor_hidden_dim: int = 128
    residual_hidden_dim: int = 128
    decoder_hidden_dim: int = 128
    sub_batch_size: int = 256
    validation_batch_size: int = 512
    max_steps: int = 10000
    validation_interval: int = 500
    early_stop_patience: int = 5
    min_relative_improvement: float = 0.005
    warmup_fraction: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    lambda_pred: float = 1.0
    lambda_info: float = 1.0
    lambda_mag: float = 1.0
    lambda_proj: float = 1.0
    # Optional Stage6 ablation term; disabled preserves the original objective.
    decorr_enabled: bool = False
    lambda_decorr: float = 0.0
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        if self.residual_dim not in (64, 128):
            raise ValueError("residual_dim must be 64 or 128.")
        if self.lambda_red < 0 or self.variance_floor <= 0:
            raise ValueError("lambda_red must be non-negative and variance_floor positive.")
        if self.projection_mode not in ("trainable", "fixed_semi_orthogonal", "identity"):
            raise ValueError(
                "projection_mode must be trainable, fixed_semi_orthogonal, or identity."
            )
        if self.projection_mode == "identity" and self.residual_dim != 128:
            raise ValueError("The identity projection baseline requires residual_dim=128.")
        if self.projection_mode == "fixed_semi_orthogonal" and self.residual_dim != 64:
            raise ValueError("The fixed semi-orthogonal baseline is defined for 64->128 only.")
        if min(self.sub_batch_size, self.validation_batch_size, self.max_steps,
               self.validation_interval, self.early_stop_patience) <= 0:
            raise ValueError("Batch sizes, step counts, and patience must be positive.")
        if not 0 < self.warmup_fraction <= 1:
            raise ValueError("warmup_fraction must be in (0, 1].")
        if min(self.lambda_pred, self.lambda_info, self.lambda_mag, self.lambda_proj, self.lambda_decorr) < 0:
            raise ValueError("Loss coefficients must be non-negative.")


class Stage6Trainer:
    """Frozen-Stage5, two-source Stage6 trainer. Training path never reads labels."""

    def __init__(
        self,
        *,
        stage5: nn.Module,
        datasets: dict[str, dict[str, dict[str, Any]]],
        stage5_sha256: str,
        run_dir: str | Path,
        config: Stage6TrainingConfig,
        device: torch.device | str | None = None,
        provenance: dict[str, Any] | None = None,
    ):
        self.config = config
        self.device = torch.device(device or (
            "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        ))
        if set(datasets) != {"nsl_kdd", "unsw_nb15"}:
            raise ValueError("Stage6 requires exactly nsl_kdd and unsw_nb15.")
        self.datasets = datasets
        self.stage5 = freeze_stage5(stage5).to(self.device)
        self.stage5.eval()
        self.stage5_sha256 = stage5_sha256
        self.provenance = dict(provenance or {})
        if len(stage5_sha256) != 64:
            raise ValueError("A verified SHA256 for the Stage5 checkpoint is required.")

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.rng = np.random.default_rng(config.seed)

        self.branches = nn.ModuleDict({
            name: Stage6ResidualBranch(
                shared_dim=128,
                dataset_dim=128,
                predictor_hidden_dim=config.predictor_hidden_dim,
                residual_hidden_dim=config.residual_hidden_dim,
                residual_dim=config.residual_dim,
                decoder_hidden_dim=config.decoder_hidden_dim,
            )
            for name in ("nsl_kdd", "unsw_nb15")
        }).to(self.device)

        if config.projection_mode == "fixed_semi_orthogonal":
            for branch in self.branches.values():
                initialize_fixed_semi_orthogonal_projection(branch.projection.projection)

        trainable = [p for p in self.branches.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable Stage6 parameters.")
        self.optimizer = AdamW(
            trainable, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.global_step = 0
        self.loss_history: list[dict[str, float]] = []
        self.validation_history: list[dict[str, Any]] = []
        self.best_validation_objective = float("inf")
        self.best_step: int | None = None
        self.non_improving_evaluations = 0

        self._validate_data_contract()

    @classmethod
    def from_project(
        cls,
        *,
        stage5_snapshot: str | Path,
        expected_stage5_sha256: str,
        run_dir: str | Path,
        config: Stage6TrainingConfig,
    ) -> "Stage6Trainer":
        """Load only the accepted Stage5 snapshot and train/validation inputs."""
        snapshot = Path(stage5_snapshot).resolve()
        project_root = Path(__file__).resolve().parents[3]
        protected_hashes = {}
        for relative, expected_hash in STAGE5_PROTECTED_HASHES.items():
            protected_path = project_root / relative
            actual_hash = sha256_file(protected_path)
            if actual_hash.upper() != expected_hash:
                raise RuntimeError(
                    f"Protected Stage5 file hash mismatch: {relative}; "
                    f"expected {expected_hash}, got {actual_hash.upper()}."
                )
            protected_hashes[relative] = actual_hash
        digest = sha256_file(snapshot)
        if digest.lower() != expected_stage5_sha256.lower():
            raise RuntimeError(
                f"Stage5 snapshot SHA256 mismatch: expected {expected_stage5_sha256}, got {digest}."
            )
        checkpoint = torch.load(snapshot, map_location="cpu", weights_only=False)
        stage5_config = checkpoint["config"]
        identity_path = (
            Path("data") / "processed" / "unified_representation"
            / "feature_identity_vectors.npy"
        )
        identity = np.load(identity_path, allow_pickle=False)
        stage5 = DualEncoder(
            identity,
            dropout=float(stage5_config["dropout"]),
            value_masking_rate=float(stage5_config["value_masking_rate"]),
            feature_set_dropout_rate=float(stage5_config["feature_set_dropout_rate"]),
            feature_set_dropout_all_probability=float(
                stage5_config["feature_set_dropout_all_probability"]
            ),
        )
        stage5.load_state_dict(checkpoint["dual_encoder"], strict=True)
        freeze_stage5(stage5)

        data: dict[str, dict[str, dict[str, Any]]] = {}
        input_artifact_hashes = {}
        input_artifact_paths = {}
        label_artifact_paths = {}
        trainval_dir = project_root / "data" / "interim" / "stage6_trainval_only"
        for name in ("nsl_kdd", "unsw_nb15"):
            input_path = trainval_dir / f"{name}_inputs.joblib"
            label_path = trainval_dir / f"{name}_labels.joblib"
            if not input_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing split-isolated Stage6 artifacts for {name}; "
                    "run scripts/build_stage6_trainval_artifacts.py first."
                )
            input_artifact_hashes[name] = sha256_file(input_path)
            input_artifact_paths[name] = str(input_path.resolve())
            label_artifact_paths[name] = str(label_path.resolve())
            # This file physically contains only train/validation feature inputs;
            # labels are stored separately and are not opened by the trainer.
            data[name] = load_stage6_input_artifact(input_path, name)

        del checkpoint
        return cls(
            stage5=stage5,
            datasets=data,
            stage5_sha256=digest,
            run_dir=run_dir,
            config=config,
            provenance={
                "stage5_input_snapshot_path": str(snapshot),
                "stage5_protected_file_sha256": protected_hashes,
                "source_input_artifact_paths": input_artifact_paths,
                "source_input_artifact_sha256": input_artifact_hashes,
                "source_label_artifact_paths": label_artifact_paths,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
            },
        )

    def _validate_data_contract(self) -> None:
        for name, splits in self.datasets.items():
            if set(splits) != {"train", "validation"}:
                raise ValueError(f"{name}: only train and validation splits are permitted.")
            for split, arrays in splits.items():
                required = set(INPUT_FIELDS) | {
                    "numeric_token_mask", "categorical_token_mask"
                }
                if set(arrays) != required:
                    raise ValueError(f"{name}/{split}: unexpected or missing input fields.")
                n = len(arrays["numeric_values"])
                for key in INPUT_FIELDS:
                    if len(arrays[key]) != n:
                        raise ValueError(f"{name}/{split}: row count mismatch for {key}.")
                if n == 0:
                    raise ValueError(f"{name}/{split}: empty split.")
                if np.asarray(arrays["numeric_values"]).shape[1] != 78:
                    raise ValueError(f"{name}/{split}: expected 78 tokens.")
            if len(splits["train"]["numeric_values"]) < self.config.sub_batch_size:
                raise ValueError(f"{name}: training split smaller than approved sub-batch.")

    def _encode(self, dataset: str, split: str, indices: np.ndarray):
        arrays = self.datasets[dataset][split]
        numeric = torch.as_tensor(
            arrays["numeric_values"][indices], device=self.device, dtype=torch.float32
        )
        categorical = torch.as_tensor(
            arrays["categorical_values"][indices], device=self.device, dtype=torch.long
        )
        available = torch.as_tensor(
            arrays["M_avail"][indices], device=self.device, dtype=torch.float32
        )
        missing = torch.as_tensor(
            arrays["M_miss"][indices], device=self.device, dtype=torch.float32
        )
        numeric_mask = torch.as_tensor(
            arrays["numeric_token_mask"], device=self.device, dtype=torch.bool
        )
        categorical_mask = torch.as_tensor(
            arrays["categorical_token_mask"], device=self.device, dtype=torch.bool
        )
        self.stage5.eval()
        with torch.no_grad():
            output = self.stage5(
                numeric_values=numeric,
                categorical_values=categorical,
                M_avail=available,
                M_miss=missing,
                numeric_token_mask=numeric_mask,
                categorical_token_mask=categorical_mask,
                dataset=dataset,
                feature_descriptors=None,
            )
        return output["Z_s"].detach(), output["Z_d"].detach()

    def _forward_stage6(self, dataset: str, z_s: torch.Tensor, z_d: torch.Tensor):
        branch = self.branches[dataset]
        branch.train()
        return branch(z_s, z_d)

    def train_step(self) -> dict[str, float]:
        self.stage5.eval()
        self.optimizer.zero_grad(set_to_none=True)
        warmup_steps = max(1, int(round(self.config.max_steps * self.config.warmup_fraction)))
        lambda_red_now = redundancy_warmup(
            self.global_step, warmup_steps, self.config.lambda_red
        )
        lambda_decorr_now = (
            redundancy_warmup(self.global_step, warmup_steps, self.config.lambda_decorr)
            if self.config.decorr_enabled else 0.0
        )
        coefficients = Stage6LossCoefficients(
            prediction=self.config.lambda_pred,
            information=self.config.lambda_info,
            magnitude=self.config.lambda_mag,
            redundancy=lambda_red_now,
        )
        dataset_totals = []
        metrics: dict[str, float] = {}

        for name in ("nsl_kdd", "unsw_nb15"):
            train_size = len(self.datasets[name]["train"]["numeric_values"])
            indices = self.rng.choice(
                train_size, size=self.config.sub_batch_size, replace=False
            )
            z_s, z_d = self._encode(name, "train", indices)
            output = self._forward_stage6(name, z_s, z_d)
            l_pred = prediction_loss(output["Z_d_hat"], z_d)
            l_info = information_reconstruction_loss(output["E_d_hat"], output["E_d"])
            l_mag, mag_stats = magnitude_stability_loss(output["R"])
            l_red, red_stats = standardized_cross_covariance_loss(
                z_s, output["R"], variance_floor=self.config.variance_floor
            )
            if self.config.decorr_enabled:
                l_decorr, decorr_stats = residual_decorrelation_loss(
                    output["R"], variance_floor=self.config.variance_floor
                )
            else:
                l_decorr = None
                decorr_stats = None
            l_proj = projection_geometry_loss(
                output["R"], output["R_proj_objective"]
            )
            total = combine_stage6_losses_with_projection(
                l_pred=l_pred,
                l_info=l_info,
                l_mag=l_mag,
                l_red=l_red,
                l_proj=l_proj,
                coefficients=coefficients,
                projection_coefficient=self.config.lambda_proj,
            )
            if self.config.decorr_enabled:
                total = total + lambda_decorr_now * l_decorr
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite Stage6 total loss for {name}.")
            dataset_totals.append(0.5 * total)
            for key, value in {
                "L_pred": l_pred, "L_info": l_info, "L_mag": l_mag,
                "L_red": l_red, "L_proj": l_proj, "total": total,
            }.items():
                metrics[f"{name}_{key}"] = float(value.detach().cpu())
            metrics[f"{name}_masked_shared_dimensions"] = float(
                red_stats["masked_shared_dimensions"]
            )
            metrics[f"{name}_masked_residual_dimensions"] = float(
                red_stats["masked_residual_dimensions"]
            )
            metrics[f"{name}_L_decorr"] = (
                float(l_decorr.detach().cpu()) if self.config.decorr_enabled else 0.0
            )
            metrics[f"{name}_decorr_active_dimensions"] = (
                float(decorr_stats["active_dimensions"]) if self.config.decorr_enabled else 0.0
            )
            metrics[f"{name}_decorr_masked_dimensions"] = (
                float(decorr_stats["masked_dimensions"]) if self.config.decorr_enabled else 0.0
            )
            metrics[f"{name}_R_rms"] = float(mag_stats["rms"])
            metrics[f"{name}_E_rms"] = float(output["E_d"].float().square().mean().sqrt().detach().cpu())
            metrics[f"{name}_mean_abs_active_cross_correlation"] = (
                mean_absolute_active_cross_correlation(
                    z_s, output["R"], self.config.variance_floor
                )
            )
            metrics[f"{name}_R_dimension_std_min"] = float(
                mag_stats["dimension_std_min"]
            )

        combined = sum(dataset_totals)
        if not torch.isfinite(combined):
            raise FloatingPointError("Non-finite combined Stage6 loss.")
        combined.backward()
        trainable = [p for p in self.branches.parameters() if p.requires_grad]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite Stage6 gradient; optimizer step was skipped.")
        if any(p.grad is not None for p in self.stage5.parameters()):
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("Frozen Stage5 unexpectedly received gradients.")
        if self.config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(trainable, self.config.gradient_clip)
        self.optimizer.step()
        self.global_step += 1
        metrics["step"] = float(self.global_step)
        metrics["lambda_red_current"] = float(lambda_red_now)
        metrics["combined_total"] = float(combined.detach().cpu())
        self.loss_history.append(metrics.copy())
        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        """Label-free source validation for early stopping and collapse/skill checks."""
        self.stage5.eval()
        self.branches.eval()
        results: dict[str, Any] = {}
        for name in ("nsl_kdd", "unsw_nb15"):
            train_size = len(self.datasets[name]["train"]["numeric_values"])
            val_size = len(self.datasets[name]["validation"]["numeric_values"])
            # Baseline is the training-partition mean prediction error under the
            # current frozen predictor; no labels are loaded or consulted.
            error_sum = torch.zeros(128, device=self.device)
            for start in range(0, train_size, self.config.validation_batch_size):
                idx = np.arange(start, min(start + self.config.validation_batch_size, train_size))
                z_s, z_d = self._encode(name, "train", idx)
                z_hat = self.branches[name].predictor(z_s)
                error_sum += (z_d - z_hat).sum(dim=0)
            train_error_mean = error_sum / float(train_size)

            pred_sum = 0.0
            info_sum = 0.0
            recon_sum = 0.0
            baseline_sum = 0.0
            cross_corr_sum = 0.0
            error_rms_sum = 0.0
            residual_parts = []
            for start in range(0, val_size, self.config.validation_batch_size):
                idx = np.arange(start, min(start + self.config.validation_batch_size, val_size))
                z_s, z_d = self._encode(name, "validation", idx)
                output = self.branches[name](z_s, z_d)
                e = output["E_d"]
                batch = len(idx)
                pred = prediction_loss(output["Z_d_hat"], z_d)
                info = information_reconstruction_loss(output["E_d_hat"], e)
                pred_sum += float(pred) * batch
                info_sum += float(info) * batch
                recon_sum += float((output["E_d_hat"] - e).square().mean()) * batch
                baseline_sum += float((e - train_error_mean).square().mean()) * batch
                cross_corr_sum += mean_absolute_active_cross_correlation(
                    z_s, output["R"], self.config.variance_floor
                ) * batch
                error_rms_sum += float(e.square().mean().sqrt()) * batch
                residual_parts.append(output["R"].float().cpu())

            residual = torch.cat(residual_parts, dim=0)
            std = residual.std(dim=0, correction=0)
            centered = residual - residual.mean(dim=0, keepdim=True)
            covariance = centered.T @ centered / max(1, residual.shape[0] - 1)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
            eigen_sum = eigenvalues.sum()
            participation = (
                float((eigen_sum.square() / eigenvalues.square().sum().clamp_min(1e-12)))
                if eigen_sum > 0 else 0.0
            )
            reconstruction_mse = recon_sum / val_size
            baseline_mse = baseline_sum / val_size
            skill = 1.0 - reconstruction_mse / max(baseline_mse, 1e-12)
            pred_mse = pred_sum / val_size
            info_mse = info_sum / val_size
            results[name] = {
                "validation_objective": pred_mse + info_mse,
                "validation_L_pred": pred_mse,
                "validation_L_info": info_mse,
                "reconstruction_mse": reconstruction_mse,
                "mean_residual_baseline_mse": baseline_mse,
                "reconstruction_skill": skill,
                "residual_rms": float(residual.square().mean().sqrt()),
                "error_rms": error_rms_sum / val_size,
                "residual_to_error_rms_ratio": (
                    float(residual.square().mean().sqrt()) / max(error_rms_sum / val_size, 1e-12)
                ),
                "mean_abs_active_cross_correlation": cross_corr_sum / val_size,
                "residual_mean_row_norm": float(residual.norm(dim=1).mean()),
                "residual_p95_row_norm": float(torch.quantile(residual.norm(dim=1), 0.95)),
                "residual_std_min": float(std.min()),
                "residual_std_median": float(std.median()),
                "residual_std_max": float(std.max()),
                # Match the approved L_red standard-deviation mask exactly:
                # dimensions with std < variance_floor are counted as collapsed.
                "near_zero_dimensions": int((std < self.config.variance_floor).sum()),
                "near_zero_dimension_threshold": self.config.variance_floor,
                "effective_rank_participation": participation,
                "validation_rows": val_size,
            }
        results["mean_validation_objective"] = float(np.mean([
            results[n]["validation_objective"] for n in ("nsl_kdd", "unsw_nb15")
        ]))
        return results

    @torch.no_grad()
    def extract_features(
        self, dataset: str, split: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract frozen Z_s and Stage6 R without loading or using labels."""
        if dataset not in self.datasets or split not in ("train", "validation"):
            raise ValueError("Feature extraction permits source train/validation only.")
        self.stage5.eval()
        self.branches.eval()
        size = len(self.datasets[dataset][split]["numeric_values"])
        shared_parts = []
        residual_parts = []
        for start in range(0, size, self.config.validation_batch_size):
            indices = np.arange(start, min(start + self.config.validation_batch_size, size))
            z_s, z_d = self._encode(dataset, split, indices)
            output = self.branches[dataset](z_s, z_d)
            shared_parts.append(z_s.float().cpu().numpy())
            residual_parts.append(output["R"].float().cpu().numpy())
        return np.concatenate(shared_parts), np.concatenate(residual_parts)

    def save_checkpoint(self, *, best: bool = False) -> Path:
        target = self.run_dir / ("best.pt" if best else "last.pt")
        temporary = target.with_suffix(".tmp")
        payload = {
            "stage": 6,
            "global_step": self.global_step,
            "stage5_checkpoint_sha256": self.stage5_sha256,
            "provenance": self.provenance,
            "device": str(self.device),
            "config": asdict(self.config),
            "branches": self.branches.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
            "validation_history": self.validation_history,
            "best_validation_objective": self.best_validation_objective,
            "best_step": self.best_step,
        }
        torch.save(payload, temporary)
        temporary.replace(target)
        metadata = {
            "stage": 6,
            "global_step": self.global_step,
            "stage5_checkpoint_sha256": self.stage5_sha256,
            "provenance": self.provenance,
            "device": str(self.device),
            "config": asdict(self.config),
            "best_validation_objective": self.best_validation_objective,
            "best_step": self.best_step,
        }
        target.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return target

    def fit(self) -> dict[str, Any]:
        for _ in range(self.config.max_steps):
            self.train_step()
            if self.global_step % self.config.validation_interval != 0:
                continue
            warmup_steps = max(1, int(round(
                self.config.max_steps * self.config.warmup_fraction
            )))
            if self.global_step < warmup_steps:
                continue
            validation = self.validate()
            validation["step"] = self.global_step
            self.validation_history.append(validation)
            objective = validation["mean_validation_objective"]
            improved = (
                not np.isfinite(self.best_validation_objective)
                or objective <= self.best_validation_objective
                * (1.0 - self.config.min_relative_improvement)
            )
            if improved:
                self.best_validation_objective = objective
                self.best_step = self.global_step
                self.non_improving_evaluations = 0
                self.save_checkpoint(best=True)
            else:
                self.non_improving_evaluations += 1
            if self.non_improving_evaluations >= self.config.early_stop_patience:
                break
        self.save_checkpoint(best=False)
        return {
            "global_step": self.global_step,
            "best_step": self.best_step,
            "best_validation_objective": self.best_validation_objective,
            "non_improving_evaluations": self.non_improving_evaluations,
            "validation_history": self.validation_history,
            "checkpoint": str(self.run_dir / "best.pt"),
        }


def screening_rejection_reasons(
    validation: dict[str, Any], residual_dim: int
) -> list[str]:
    """Apply the approved validation-only nonfinite/collapse/skill rejection gates."""
    if residual_dim not in (64, 128):
        raise ValueError("residual_dim must be 64 or 128.")
    reasons: list[str] = []
    for name in ("nsl_kdd", "unsw_nb15"):
        if name not in validation:
            reasons.append(f"{name}: missing validation metrics")
            continue
        metrics = validation[name]
        numeric_values = [
            metrics.get("validation_objective"),
            metrics.get("reconstruction_skill"),
            metrics.get("residual_rms"),
            metrics.get("residual_std_min"),
            metrics.get("effective_rank_participation"),
        ]
        if any(value is None or not np.isfinite(float(value)) for value in numeric_values):
            reasons.append(f"{name}: non-finite or missing validation metric")
            continue
        if (
            float(metrics["effective_rank_participation"]) < 0.25 * residual_dim
            or int(metrics.get("near_zero_dimensions", residual_dim + 1))
            > 0.25 * residual_dim
        ):
            reasons.append(f"{name}: residual collapse")
        if float(metrics["reconstruction_skill"]) <= 0.0:
            reasons.append(f"{name}: reconstruction skill is not positive")
    return reasons
