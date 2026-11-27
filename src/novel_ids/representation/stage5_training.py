import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from novel_ids.representation.classification_head import (
    ClassificationHead,
    compute_class_weights,
)
from novel_ids.representation.reconstruction_decoder import ReconstructionDecoder
from novel_ids.representation.dual_encoder import (
    CATEGORICAL_CONFIG,
    DualEncoder,
    IDENTITY_DIM,
    OUTPUT_DIM,
    TOKEN_COUNT,
)
DEFAULT_RUNS_DIR = Path(r"C:\NOVEL_IDS_runs")
COMMON_POSITIONS = set(range(5))
@dataclass
class Stage5Config:
    """Stage 5 training and validation configuration."""
    dataset_names: tuple[str, str] = ("nsl_kdd", "unsw_nb15")
    batch_size_per_dataset: int = 256
    sub_batch_size: int = 256
    hidden_dim: int = 128
    decoder_hidden_dim: int = 128
    dropout: float = 0.1
    value_masking_rate: float = 0.05
    feature_set_dropout_rate: float = 0.10
    feature_set_dropout_all_probability: float = 0.05
    reconstruction_mask_rate: float = 0.20
    visible_reconstruction_weight: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    classification_weight: float = 1.0
    lambda_candidates: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0)
    lambda_reconstruction: float = 1.0
    lambda_ramp_steps: int | None = None
    warmup_steps: int = 500
    max_steps: int = 10000
    gradient_clip: float = 1.0
    validation_interval: int = 500
    min_validation_evaluations: int = 10
    max_consecutive_degrading_evaluations: int = 3
    degradation_threshold: float = 0.005
    checkpoint_tolerance: float = 0.05
    # Retained for compatibility with existing test/config callers; not used
    # by the approved three-consecutive-degrading-evaluations stop rule.
    patience: int = 5
    min_delta: float = 1e-4
    validation_mask_seed: int = 2026
    lambda_selection_phase: str = "A"
    lambda_selection_seed: int | None = None
    lambda_candidate_results: dict | None = None
    lambda_best_candidates: list | None = None
    lambda_eval_step: int = 0
    lambda_ramp_started: bool = False
    device: str = "cuda"
    mixed_precision: bool = True
    seed: int = 42
    runs_dir: Path = DEFAULT_RUNS_DIR
    experiment_name: str = "stage5"


def set_seed(seed: int) -> None:
    """Set fixed seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def get_device(preferred: str = "cuda") -> torch.device:
    """Return the best available device, with CPU fallback."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
class Stage5Trainer:
    """Training pipeline for Stage 5 representation learning."""
    def __init__(self, config: Stage5Config):
        self.config = config
        self.device = get_device(config.device)
        self.mixed_precision = config.mixed_precision and self.device.type == "cuda"
        # BF16 avoids FP16 overflow on valid high-magnitude UNSW-NB15 values.
        # BF16 has sufficient exponent range and does not require loss scaling.
        self.amp_dtype = torch.bfloat16
        self.scaler = None
        set_seed(config.seed)
        self.datasets: dict[str, dict[str, Any]] = {}
        self.models: dict[str, nn.Module] = {}
        self.label_encoders: dict[str, dict[str, int]] = {}
        self.class_weights: dict[str, torch.Tensor] = {}
        self.dual_encoder: DualEncoder | None = None
        self.classification_heads: dict[str, ClassificationHead] = {}
        self.reconstruction_decoders: dict[str, ReconstructionDecoder] = {}
        self.optimizer: AdamW | None = None
        self.scheduler: CosineAnnealingLR | None = None
        self.global_step = 0
        self.best_val_macro_f1 = -1.0
        self.patience_counter = 0
        self.best_dataset_f1 = {name: -1.0 for name in config.dataset_names}
        self.previous_dataset_f1: dict[str, float] = {}
        self.validation_evaluations = 0
        self.consecutive_degrading_evaluations = 0
        self.best_checkpoint_metrics: dict[str, Any] = {}
        self.best_checkpoint_step: int | None = None
        self.current_lambda = 0.0
        self.loss_history: list[dict[str, float]] = []
        self.runs_dir = config.runs_dir / config.experiment_name
        self.runs_dir.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_artifacts(self, config_path: str = "config/datasets.yaml") -> None:
        """Load unified representation artifacts for both datasets."""
        from novel_ids.onboarding.load_source_datasets import (
            load_config,
            unified_artifact_path,
        )
        cfg = load_config()
        for dataset_name in self.config.dataset_names:
            artifact_path = unified_artifact_path(cfg, dataset_name)
            artifact = joblib.load(artifact_path)
            train = artifact["train"]
            validation = artifact["validation"]
            train_labels = np.asarray(train["labels"]).astype(str)
            val_labels = np.asarray(validation["labels"]).astype(str)
            unique_labels = sorted(set(train_labels))
            label_to_index = {label: i for i, label in enumerate(unique_labels)}
            num_classes = len(unique_labels)
            train_label_ids = np.array(
                [label_to_index[l] for l in train_labels], dtype=np.int64
            )
            val_label_ids = np.array(
                [label_to_index[l] for l in val_labels], dtype=np.int64
            )
            weights = compute_class_weights(train_labels.tolist(), num_classes)
            numeric_mask = np.asarray(train["numeric_token_mask"], dtype=bool)
            categorical_mask = np.asarray(train["categorical_token_mask"], dtype=bool)
            numeric_positions = [
                i for i in range(TOKEN_COUNT) if numeric_mask[i]
            ]
            categorical_positions = [
                i for i in range(TOKEN_COUNT) if categorical_mask[i]
            ]
            cat_config = CATEGORICAL_CONFIG[dataset_name]
            cat_position_to_vocab: dict[int, int] = {}
            for pos, size in zip(cat_config["positions"], cat_config["vocab_sizes"]):
                # UNKNOWN is input id 0, not a decoder output class.
                cat_position_to_vocab[pos] = size
            available_positions = np.flatnonzero(
                np.any(np.asarray(train["M_avail"], dtype=np.float32) > 0.5, axis=0)
            ).tolist()
            dataset_specific_positions = [
                int(i) for i in available_positions if int(i) not in COMMON_POSITIONS
            ]
            self.datasets[dataset_name] = {
                "artifact": artifact,
                "train_numeric": np.asarray(train["numeric_values"], dtype=np.float32),
                "train_categorical": np.asarray(train["categorical_values"], dtype=np.int64),
                "train_M_avail": np.asarray(train["M_avail"], dtype=np.float32),
                "train_M_miss": np.asarray(train["M_miss"], dtype=np.float32),
                "train_labels": train_label_ids,
                "val_numeric": np.asarray(validation["numeric_values"], dtype=np.float32),
                "val_categorical": np.asarray(validation["categorical_values"], dtype=np.int64),
                "val_M_avail": np.asarray(validation["M_avail"], dtype=np.float32),
                "val_M_miss": np.asarray(validation["M_miss"], dtype=np.float32),
                "val_labels": val_label_ids,
                "numeric_mask": numeric_mask,
                "categorical_mask": categorical_mask,
                "numeric_positions": numeric_positions,
                "categorical_positions": categorical_positions,
                "cat_position_to_vocab": cat_position_to_vocab,
                "dataset_specific_positions": dataset_specific_positions,
                "label_to_index": label_to_index,
                "index_to_label": {i: l for l, i in label_to_index.items()},
                "num_classes": num_classes,
                "train_size": len(train_label_ids),
                "val_size": len(val_label_ids),
            }
            print(
                f"[{dataset_name}] train={len(train_label_ids):,}, "
                f"val={len(val_label_ids):,}, classes={num_classes}"
            )
    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------
    def build_models(self, identity_vectors_path: str | None = None) -> None:
        """Build DualEncoder, classification heads, and reconstruction decoders."""
        if identity_vectors_path is None:
            identity_vectors_path = (
                Path("data") / "processed" / "unified_representation" / "feature_identity_vectors.npy"
            )
        identity_vectors = np.load(identity_vectors_path, allow_pickle=False)
        self.dual_encoder = DualEncoder(
            identity_vectors,
            dropout=self.config.dropout,
            value_masking_rate=self.config.value_masking_rate,
            feature_set_dropout_rate=self.config.feature_set_dropout_rate,
            feature_set_dropout_all_probability=self.config.feature_set_dropout_all_probability,
        ).to(self.device)
        for dataset_name in self.config.dataset_names:
            ds = self.datasets[dataset_name]
            head = ClassificationHead(
                input_dim=OUTPUT_DIM,
                num_classes=ds["num_classes"],
            ).to(self.device)
            self.classification_heads[dataset_name] = head
            train_labels_str = [
                ds["index_to_label"][i] for i in ds["train_labels"].tolist()
            ]
            self.class_weights[dataset_name] = compute_class_weights(
                train_labels_str, ds["num_classes"]
            )
            head.class_weights = self.class_weights[dataset_name].to(self.device)
            decoder = ReconstructionDecoder(
                z_dim=OUTPUT_DIM,
                identity_dim=IDENTITY_DIM,
                hidden_dim=self.config.decoder_hidden_dim,
                numeric_features=ds["numeric_positions"],
                categorical_features=ds["cat_position_to_vocab"],
            ).to(self.device)
            self.reconstruction_decoders[dataset_name] = decoder
            print(
                f"[{dataset_name}] head: {ds['num_classes']} classes, "
                f"decoder: {len(ds['dataset_specific_positions'])} dataset-specific positions"
            )
        all_params = []
        all_params.extend(self.dual_encoder.parameters())
        for head in self.classification_heads.values():
            all_params.extend(head.parameters())
        for decoder in self.reconstruction_decoders.values():
            all_params.extend(decoder.parameters())
        self.optimizer = AdamW(
            all_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        effective_warmup = (
            self.config.warmup_steps
            if self.config.max_steps >= 5000
            else max(1, int(math.ceil(self.config.max_steps * 0.10)))
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.max_steps - effective_warmup),
        )
        self.warmup_steps = self.config.warmup_steps
        self.max_steps = self.config.max_steps
    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _make_reconstruction_mask(
        self,
        dataset_name: str,
        indices: np.ndarray,
        M_avail: torch.Tensor,
        M_miss: torch.Tensor,
        categorical: torch.Tensor,
        validation: bool = False,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Return hidden and visible masks from the actual selected batch.

        Validation masks are generated from one fixed full-validation RNG stream,
        so every candidate and seed sees the same held-out reconstruction targets.
        """
        ds = self.datasets[dataset_name]
        positions = ds["dataset_specific_positions"]
        if validation:
            dataset_offset = self.config.dataset_names.index(dataset_name) * 1009
            rng = np.random.default_rng(self.config.validation_mask_seed + dataset_offset)
            avail_all = ds["val_M_avail"] > 0.5
            observed_all = avail_all & (ds["val_M_miss"] < 0.5)
            for pos in ds["categorical_positions"]:
                observed_all[:, pos] &= ds["val_categorical"][:, pos] > 0
            observed_all[:, list(COMMON_POSITIONS)] = False
            random_all = rng.random(observed_all.shape)
            hidden_all = observed_all & (random_all < self.config.reconstruction_mask_rate)
            observed = observed_all[indices]
            hidden = hidden_all[indices]
        else:
            avail = M_avail.detach().cpu().numpy() > 0.5
            miss = M_miss.detach().cpu().numpy() > 0.5
            categorical_np = categorical.detach().cpu().numpy()
            observed = avail & ~miss
            for pos in ds["categorical_positions"]:
                observed[:, pos] &= categorical_np[:, pos] > 0
            observed[:, list(COMMON_POSITIONS)] = False
            hidden = observed & (
                np.random.random(observed.shape) < self.config.reconstruction_mask_rate
            )

        visible = observed & ~hidden
        hidden_masks = {
            pos: torch.as_tensor(hidden[:, pos], device=self.device, dtype=torch.bool)
            for pos in positions
        }
        visible_masks = {
            pos: torch.as_tensor(visible[:, pos], device=self.device, dtype=torch.bool)
            for pos in positions
        }
        return hidden_masks, visible_masks

    def _current_reconstruction_lambda(self) -> float:
        """Linear warm-up: zero on the first update, target at ramp end."""
        ramp_steps = self.config.lambda_ramp_steps
        if ramp_steps is None:
            ramp_steps = max(2, int(math.ceil(self.config.max_steps * 0.10)))
        ramp_steps = max(2, int(ramp_steps))
        fraction = min(1.0, self.global_step / float(ramp_steps - 1))
        return float(self.config.lambda_reconstruction) * fraction

    def _forward_batch(
        self,
        dataset_name: str,
        indices: np.ndarray,
        training: bool,
    ) -> dict[str, Any]:
        """Forward pass and masked-feature objective for one source dataset."""
        ds = self.datasets[dataset_name]
        device = self.device
        numeric = torch.as_tensor(ds["train_numeric"][indices], device=device, dtype=torch.float32)
        categorical = torch.as_tensor(ds["train_categorical"][indices], device=device, dtype=torch.long)
        M_avail = torch.as_tensor(ds["train_M_avail"][indices], device=device, dtype=torch.float32)
        M_miss = torch.as_tensor(ds["train_M_miss"][indices], device=device, dtype=torch.float32)
        labels = torch.as_tensor(ds["train_labels"][indices], device=device, dtype=torch.long)

        hidden_masks, visible_masks = self._make_reconstruction_mask(
            dataset_name, indices, M_avail, M_miss, categorical, validation=False
        )
        model_M_miss = M_miss.clone()
        for position, hidden in hidden_masks.items():
            model_M_miss[:, position] = torch.maximum(
                model_M_miss[:, position], hidden.to(model_M_miss.dtype)
            )

        numeric_mask = torch.as_tensor(ds["numeric_mask"], device=device, dtype=torch.bool)
        categorical_mask = torch.as_tensor(ds["categorical_mask"], device=device, dtype=torch.bool)
        output = self.dual_encoder(
            numeric_values=numeric,
            categorical_values=categorical,
            M_avail=M_avail,
            M_miss=model_M_miss,
            numeric_token_mask=numeric_mask,
            categorical_token_mask=categorical_mask,
            dataset=dataset_name,
            feature_descriptors=None,
        )
        z_s, z_d = output["Z_s"], output["Z_d"]
        logits = self.classification_heads[dataset_name](z_s)
        cls_loss = self.classification_heads[dataset_name].loss(logits, labels)

        positions = ds["dataset_specific_positions"]
        identity_vectors = self.dual_encoder.feature_identities
        predictions = self.reconstruction_decoders[dataset_name](
            z_d, identity_vectors, positions
        )
        recon_targets = {
            pos: numeric[:, pos] if pos in ds["numeric_positions"] else (categorical[:, pos] - 1)
            for pos in positions
        }
        masked_loss, masked_logged = self.reconstruction_decoders[
            dataset_name
        ].reconstruction_loss(
            predictions, recon_targets, hidden_masks,
            numeric_weight=1.0, categorical_weight=1.0,
        )
        visible_loss, visible_logged = self.reconstruction_decoders[
            dataset_name
        ].reconstruction_loss(
            predictions, recon_targets, visible_masks,
            numeric_weight=1.0, categorical_weight=1.0,
        )
        recon_loss = masked_loss + (
            self.config.visible_reconstruction_weight * visible_loss
        )
        current_lambda = self._current_reconstruction_lambda()
        total_loss = (
            self.config.classification_weight * cls_loss
            + current_lambda * recon_loss
        )
        return {
            "loss": total_loss,
            "classification_loss": cls_loss.detach(),
            "reconstruction_loss": recon_loss.detach(),
            "reconstruction_masked_loss": masked_loss.detach(),
            "reconstruction_visible_loss": visible_loss.detach(),
            "reconstruction_numeric": masked_logged["reconstruction_numeric"],
            "reconstruction_categorical": masked_logged["reconstruction_categorical"],
            "reconstruction_visible_numeric": visible_logged["reconstruction_numeric"],
            "reconstruction_visible_categorical": visible_logged["reconstruction_categorical"],
            "z_s": z_s.detach(),
            "z_d": z_d.detach(),
            "labels": labels.detach(),
            "logits": logits.detach(),
            "lambda_reconstruction_current": current_lambda,
        }

    def train_step(self) -> dict[str, float]:
        """One optimizer update with equally weighted, equal-size dataset batches."""
        if self.dual_encoder is None or self.optimizer is None:
            raise RuntimeError("Models and optimizer must be built before training.")
        self.dual_encoder.train()
        for head in self.classification_heads.values():
            head.train()
        for decoder in self.reconstruction_decoders.values():
            decoder.train()

        self.optimizer.zero_grad(set_to_none=True)
        lambda_used = self._current_reconstruction_lambda()
        batch_size = self.config.sub_batch_size
        metrics: dict[str, float] = {}
        dataset_total_losses = []
        for dataset_name in self.config.dataset_names:
            ds = self.datasets[dataset_name]
            replace_rows = ds["train_size"] < batch_size
            indices = np.random.choice(
                ds["train_size"], size=batch_size, replace=replace_rows
            )
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.mixed_precision,
            ):
                result = self._forward_batch(dataset_name, indices, training=True)
                weighted_loss = 0.5 * result["loss"]
            if not torch.isfinite(weighted_loss):
                raise FloatingPointError(
                    f"Non-finite Stage 5 loss for {dataset_name} at step {self.global_step}."
                )
            if self.scaler is not None:
                self.scaler.scale(weighted_loss).backward()
            else:
                weighted_loss.backward()

            dataset_total_losses.append(float(result["loss"].detach()))
            metrics[f"{dataset_name}_classification_loss"] = float(result["classification_loss"])
            metrics[f"{dataset_name}_reconstruction_loss"] = float(result["reconstruction_loss"])
            metrics[f"{dataset_name}_reconstruction_masked_loss"] = float(
                result["reconstruction_masked_loss"]
            )
            metrics[f"{dataset_name}_reconstruction_visible_loss"] = float(
                result["reconstruction_visible_loss"]
            )

        all_params = (
            list(self.dual_encoder.parameters())
            + [p for head in self.classification_heads.values() for p in head.parameters()]
            + [p for decoder in self.reconstruction_decoders.values() for p in decoder.parameters()]
        )
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if self.config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(all_params, self.config.gradient_clip)

        effective_warmup = (
            self.config.warmup_steps
            if self.config.max_steps >= 5000
            else max(1, int(math.ceil(self.config.max_steps * 0.10)))
        )
        if self.global_step < effective_warmup:
            lr = self.config.learning_rate * (self.global_step + 1) / effective_warmup
            for group in self.optimizer.param_groups:
                group["lr"] = lr

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.global_step += 1
        if self.global_step >= effective_warmup and self.scheduler is not None:
            self.scheduler.step()

        self.current_lambda = lambda_used
        metrics.update({
            "step": float(self.global_step),
            "classification_loss": float(np.mean([
                metrics[f"{name}_classification_loss"] for name in self.config.dataset_names
            ])),
            "reconstruction_loss": float(np.mean([
                metrics[f"{name}_reconstruction_loss"] for name in self.config.dataset_names
            ])),
            "total_loss": float(np.mean(dataset_total_losses)),
            "lambda_reconstruction_current": float(lambda_used),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        })
        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        """Validate both datasets using fixed seeded masked-feature targets."""
        self.dual_encoder.eval()
        for head in self.classification_heads.values():
            head.eval()
        for decoder in self.reconstruction_decoders.values():
            decoder.eval()

        results: dict[str, Any] = {}
        dataset_f1s: list[float] = []
        for dataset_name in self.config.dataset_names:
            ds = self.datasets[dataset_name]
            batch_size = 512
            all_preds: list[np.ndarray] = []
            all_labels: list[np.ndarray] = []
            cls_losses: list[float] = []
            recon_losses: list[float] = []
            for start in range(0, ds["val_size"], batch_size):
                end = min(start + batch_size, ds["val_size"])
                indices = np.arange(start, end)
                numeric = torch.as_tensor(ds["val_numeric"][indices], device=self.device, dtype=torch.float32)
                categorical = torch.as_tensor(ds["val_categorical"][indices], device=self.device, dtype=torch.long)
                M_avail = torch.as_tensor(ds["val_M_avail"][indices], device=self.device, dtype=torch.float32)
                M_miss = torch.as_tensor(ds["val_M_miss"][indices], device=self.device, dtype=torch.float32)
                labels = torch.as_tensor(ds["val_labels"][indices], device=self.device, dtype=torch.long)
                hidden_masks, _ = self._make_reconstruction_mask(
                    dataset_name, indices, M_avail, M_miss, categorical, validation=True
                )
                model_M_miss = M_miss.clone()
                for pos, hidden in hidden_masks.items():
                    model_M_miss[:, pos] = torch.maximum(
                        model_M_miss[:, pos], hidden.to(model_M_miss.dtype)
                    )
                numeric_mask = torch.as_tensor(ds["numeric_mask"], device=self.device, dtype=torch.bool)
                categorical_mask = torch.as_tensor(ds["categorical_mask"], device=self.device, dtype=torch.bool)
                output = self.dual_encoder(
                    numeric_values=numeric,
                    categorical_values=categorical,
                    M_avail=M_avail,
                    M_miss=model_M_miss,
                    numeric_token_mask=numeric_mask,
                    categorical_token_mask=categorical_mask,
                    dataset=dataset_name,
                    feature_descriptors=None,
                )
                z_s, z_d = output["Z_s"], output["Z_d"]
                logits = self.classification_heads[dataset_name](z_s)
                cls_loss = self.classification_heads[dataset_name].loss(logits, labels)
                cls_losses.append(float(cls_loss))
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                positions = ds["dataset_specific_positions"]
                predictions = self.reconstruction_decoders[dataset_name](
                    z_d, self.dual_encoder.feature_identities, positions
                )
                targets = {
                    pos: numeric[:, pos] if pos in ds["numeric_positions"] else (categorical[:, pos] - 1)
                    for pos in positions
                }
                recon_loss, _ = self.reconstruction_decoders[dataset_name].reconstruction_loss(
                    predictions, targets, hidden_masks,
                    numeric_weight=1.0, categorical_weight=1.0,
                )
                recon_losses.append(float(recon_loss))

            preds_all = np.concatenate(all_preds)
            labels_all = np.concatenate(all_labels)
            supported_labels = np.unique(labels_all)
            macro_f1 = f1_score(
                labels_all, preds_all, labels=supported_labels,
                average="macro", zero_division=0,
            )
            support_by_class = {
                str(ds["index_to_label"][int(label)]): int(np.sum(labels_all == label))
                for label in supported_labels
            }
            results[f"{dataset_name}_val_macro_f1"] = float(macro_f1)
            results[f"{dataset_name}_val_cls_loss"] = float(np.mean(cls_losses))
            results[f"{dataset_name}_val_recon_loss"] = float(np.mean(recon_losses))
            results[f"{dataset_name}_val_support_count"] = int(labels_all.size)
            results[f"{dataset_name}_val_supported_class_count"] = int(len(supported_labels))
            results[f"{dataset_name}_val_class_support"] = support_by_class
            results[f"{dataset_name}_val_supported_labels"] = [
                str(ds["index_to_label"][int(label)]) for label in supported_labels
            ]
            dataset_f1s.append(float(macro_f1))

        results["val_macro_f1_avg"] = float(np.mean(dataset_f1s))
        return results

    def save_checkpoint(self, is_best: bool = False, tag: str = "last") -> Path:
        """Atomically save best-eligible or last checkpoint and its metadata."""
        checkpoint = {
            "global_step": self.global_step,
            "best_val_macro_f1": self.best_val_macro_f1,
            "patience_counter": self.patience_counter,
            "best_dataset_f1": self.best_dataset_f1,
            "previous_dataset_f1": self.previous_dataset_f1,
            "validation_evaluations": self.validation_evaluations,
            "consecutive_degrading_evaluations": self.consecutive_degrading_evaluations,
            "best_checkpoint_metrics": self.best_checkpoint_metrics,
            "best_checkpoint_step": self.best_checkpoint_step,
            "loss_history": self.loss_history,
            "config": asdict(self.config),
            "dual_encoder": self.dual_encoder.state_dict(),
            "classification_heads": {
                name: head.state_dict() for name, head in self.classification_heads.items()
            },
            "reconstruction_decoders": {
                name: decoder.state_dict()
                for name, decoder in self.reconstruction_decoders.items()
            },
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "label_encoders": {
                name: ds["label_to_index"] for name, ds in self.datasets.items()
            },
            "class_weights": {
                name: weights.cpu() for name, weights in self.class_weights.items()
            },
        }
        filename = "best.pt" if is_best else f"{tag}.pt"
        target_path = self.runs_dir / filename
        tmp_path = target_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as handle:
            torch.save(checkpoint, handle)
        tmp_path.replace(target_path)
        metadata = {
            "global_step": self.global_step,
            "best_val_macro_f1": self.best_val_macro_f1,
            "best_dataset_f1": self.best_dataset_f1,
            "best_checkpoint_metrics": self.best_checkpoint_metrics,
            "best_checkpoint_step": self.best_checkpoint_step,
            "validation_evaluations": self.validation_evaluations,
            "consecutive_degrading_evaluations": self.consecutive_degrading_evaluations,
            "tag": tag,
            "is_best": is_best,
        }
        target_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Checkpoint saved: {target_path} (step={self.global_step})")
        return target_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model, optimizer, scheduler, scaler and validation-selection state."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.global_step = int(checkpoint["global_step"])
        self.best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", -1.0))
        self.patience_counter = int(checkpoint.get("patience_counter", 0))
        self.best_dataset_f1 = checkpoint.get(
            "best_dataset_f1", {name: -1.0 for name in self.config.dataset_names}
        )
        self.previous_dataset_f1 = checkpoint.get("previous_dataset_f1", {})
        self.validation_evaluations = int(checkpoint.get("validation_evaluations", 0))
        self.consecutive_degrading_evaluations = int(
            checkpoint.get("consecutive_degrading_evaluations", 0)
        )
        self.best_checkpoint_metrics = checkpoint.get("best_checkpoint_metrics", {})
        self.best_checkpoint_step = checkpoint.get("best_checkpoint_step")
        self.loss_history = checkpoint.get("loss_history", [])
        self.dual_encoder.load_state_dict(checkpoint["dual_encoder"])
        for name, state in checkpoint["classification_heads"].items():
            self.classification_heads[name].load_state_dict(state)
        for name, state in checkpoint["reconstruction_decoders"].items():
            self.reconstruction_decoders[name].load_state_dict(state)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        print(f"Checkpoint loaded: {path} (step={self.global_step})")

    def fit(self) -> dict[str, Any]:
        """Train to the step cap or approved validation-degradation stop rule."""
        effective_warmup = (
            self.config.warmup_steps if self.config.max_steps >= 5000
            else max(1, int(math.ceil(self.config.max_steps * 0.10)))
        )
        print(f"Starting Stage 5 training on {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(
            f"Max steps: {self.config.max_steps}; warmup: {effective_warmup}; "
            f"validation interval: {self.config.validation_interval}"
        )
        while self.global_step < self.config.max_steps:
            step_metrics = self.train_step()
            self.loss_history.append(step_metrics)
            if self.global_step % self.config.validation_interval != 0:
                continue

            val_metrics = self.validate()
            self.loss_history[-1].update(val_metrics)
            self.validation_evaluations += 1
            current = {
                name: float(val_metrics[f"{name}_val_macro_f1"])
                for name in self.config.dataset_names
            }
            degrading = bool(self.previous_dataset_f1) and any(
                self.previous_dataset_f1[name] - current[name]
                > self.config.degradation_threshold
                for name in self.config.dataset_names
            )
            if degrading:
                self.consecutive_degrading_evaluations += 1
            else:
                self.consecutive_degrading_evaluations = 0

            for name, value in current.items():
                self.best_dataset_f1[name] = max(self.best_dataset_f1[name], value)

            eligible = all(
                current[name] >= self.best_dataset_f1[name] - self.config.checkpoint_tolerance
                for name in self.config.dataset_names
            )
            average_f1 = float(val_metrics["val_macro_f1_avg"])
            if eligible and average_f1 > self.best_val_macro_f1:
                self.best_val_macro_f1 = average_f1
                self.best_checkpoint_metrics = {
                    "val_macro_f1_avg": average_f1,
                    **current,
                }
                self.best_checkpoint_step = self.global_step
                self.save_checkpoint(is_best=True, tag=f"step_{self.global_step}")

            self.previous_dataset_f1 = current
            print(
                f"Step {self.global_step}: "
                f"NSL-KDD F1={current.get('nsl_kdd', 0.0):.4f}, "
                f"UNSW-NB15 F1={current.get('unsw_nb15', 0.0):.4f}, "
                f"mean F1={average_f1:.4f}, eligible={eligible}, "
                f"degrading streak={self.consecutive_degrading_evaluations}"
            )
            if (
                self.validation_evaluations >= self.config.min_validation_evaluations
                and self.consecutive_degrading_evaluations
                >= self.config.max_consecutive_degrading_evaluations
            ):
                print(
                    "Early stopping: "
                    f"{self.consecutive_degrading_evaluations} consecutive degrading "
                    f"evaluations after {self.validation_evaluations} validations."
                )
                break

        self.save_checkpoint(is_best=False, tag="last")
        return {
            "best_val_macro_f1": self.best_val_macro_f1,
            "best_dataset_macro_f1": {
                name: self.best_checkpoint_metrics.get(name, -1.0)
                for name in self.config.dataset_names
            },
            "best_checkpoint_metrics": self.best_checkpoint_metrics,
            "best_checkpoint_step": self.best_checkpoint_step,
            "global_step": self.global_step,
            "validation_evaluations": self.validation_evaluations,
            "consecutive_degrading_evaluations": self.consecutive_degrading_evaluations,
            "loss_history": self.loss_history,
        }


def run_stage5_lambda_selection(
    config: Stage5Config | None = None,
    identity_vectors_path: str | None = None,
) -> dict[str, Any]:
    """Run the approved two-phase lambda screen and paired three-seed comparison.

    This function starts experiments only when explicitly called. The ordinary
    train_stage5() entry point continues to build a trainer without fitting it.
    """
    if config is None:
        config = Stage5Config()
    candidates = sorted({float(value) for value in config.lambda_candidates} | {0.0})
    phase_a: dict[str, Any] = {}
    for candidate in candidates:
        run_config = replace(
            config,
            lambda_reconstruction=candidate,
            seed=config.seed,
            experiment_name=f"{config.experiment_name}/phaseA_lambda_{candidate:g}",
        )
        trainer = train_stage5(run_config, identity_vectors_path)
        result = trainer.fit()
        phase_a[str(candidate)] = {
            "best_val_macro_f1": result["best_val_macro_f1"],
            "best_dataset_macro_f1": result["best_dataset_macro_f1"],
            "global_step": result["global_step"],
        }

    ranked = sorted(
        candidates,
        key=lambda value: phase_a[str(value)]["best_val_macro_f1"],
        reverse=True,
    )
    phase_b_candidates = [0.0] + [value for value in ranked if value != 0.0][:2]
    seeds = [config.seed, config.seed + 1, config.seed + 2]
    phase_b: dict[str, Any] = {}
    for candidate in phase_b_candidates:
        # Reuse Phase A's run as the first paired seed; only two additional
        # seed runs are needed for each Phase B candidate.
        per_seed = {
            str(config.seed): phase_a[str(candidate)]["best_dataset_macro_f1"]
        }
        for seed in seeds[1:]:
            run_config = replace(
                config,
                lambda_reconstruction=candidate,
                seed=seed,
                experiment_name=(
                    f"{config.experiment_name}/phaseB_lambda_{candidate:g}_seed_{seed}"
                ),
            )
            trainer = train_stage5(run_config, identity_vectors_path)
            result = trainer.fit()
            per_seed[str(seed)] = result["best_dataset_macro_f1"]
        phase_b[str(candidate)] = per_seed

    baseline = phase_b["0.0"]
    summaries: dict[str, Any] = {}
    accepted: list[float] = []
    for candidate in phase_b_candidates:
        values = phase_b[str(candidate)]
        dataset_summary = {}
        rejected = False
        for dataset_name in config.dataset_names:
            candidate_scores = np.asarray(
                [values[str(seed)][dataset_name] for seed in seeds], dtype=float
            )
            baseline_scores = np.asarray(
                [baseline[str(seed)][dataset_name] for seed in seeds], dtype=float
            )
            paired_differences = candidate_scores - baseline_scores
            paired_sd = float(np.std(paired_differences, ddof=1))
            threshold = max(0.05, 2.0 * paired_sd)
            mean_candidate = float(candidate_scores.mean())
            mean_baseline = float(baseline_scores.mean())
            drop = mean_baseline - mean_candidate
            dataset_rejected = drop > threshold if candidate != 0.0 else False
            rejected = rejected or dataset_rejected
            dataset_summary[dataset_name] = {
                "candidate_mean_macro_f1": mean_candidate,
                "baseline_mean_macro_f1": mean_baseline,
                "drop_from_baseline": float(drop),
                "paired_seed_difference_std": paired_sd,
                "rejection_threshold": threshold,
                "rejected": dataset_rejected,
            }
        if not rejected:
            accepted.append(candidate)
        summaries[str(candidate)] = {
            "datasets": dataset_summary,
            "mean_macro_f1": float(np.mean([
                dataset_summary[name]["candidate_mean_macro_f1"]
                for name in config.dataset_names
            ])),
            "rejected": rejected,
        }

    selected = max(accepted, key=lambda value: summaries[str(value)]["mean_macro_f1"])
    report = {
        "phase_a": phase_a,
        "phase_a_ranked_candidates": ranked,
        "phase_b_candidates": phase_b_candidates,
        "phase_b_seeds": seeds,
        "phase_b_reused_phase_a_seed": True,
        "phase_b_runs": phase_b,
        "candidate_comparison": summaries,
        "accepted_candidates": accepted,
        "selected_lambda": selected,
        "selection_rule": (
            "Reject if either dataset's mean macro-F1 drop from lambda=0 exceeds "
            "max(0.05, 2 * sample SD of paired per-seed differences); among passing "
            "candidates choose the highest mean of the two dataset mean macro-F1 scores."
        ),
    }
    output_dir = config.runs_dir / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lambda_selection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def train_stage5(
    config: Stage5Config | None = None,
    identity_vectors_path: str | None = None,
) -> Stage5Trainer:
    """Convenience entry point for Stage 5 training.
    Builds models, loads artifacts, and returns the trainer. Does not start
    training; call trainer.fit() to begin.
    """
    if config is None:
        config = Stage5Config()
    trainer = Stage5Trainer(config)
    trainer.load_artifacts()
    trainer.build_models(identity_vectors_path=identity_vectors_path)
    return trainer
if __name__ == "__main__":
    config = Stage5Config()
    trainer = train_stage5(config)
    print("Stage 5 trainer ready. Call trainer.fit() to start training.")
