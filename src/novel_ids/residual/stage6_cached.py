from __future__ import annotations
from pathlib import Path
from types import MethodType
from typing import Any
import numpy as np
import torch

def attach_stage6_cache(trainer: Any, cache_root: str | Path) -> None:
    """Bind the approved finite K=8 float32 Stage5 representation bank to Stage6Trainer.

    Training rows sample one precomputed draw per row using a deterministic,
    per-dataset RNG. Validation and explicit clean-evaluation contexts use draw 0.
    The Stage6 architecture and loss implementation remain untouched.
    """
    root=Path(cache_root)
    trainer._stage6_cache_train={}
    trainer._stage6_cache_validation={}
    trainer._stage6_cache_rng={
        "nsl_kdd":np.random.default_rng(int(trainer.config.seed)+20261125),
        "unsw_nb15":np.random.default_rng(int(trainer.config.seed)+20271125),
    }
    trainer._stage6_cache_force_clean=False
    for dataset in ("nsl_kdd","unsw_nb15"):
        train_path=root/"cache"/dataset/"train_zszd.npy"
        val_path=root/"cache"/dataset/"validation_zszd.npy"
        train=np.load(train_path,mmap_mode="r")
        val=np.load(val_path,mmap_mode="r")
        if train.dtype!=np.float32 or val.dtype!=np.float32:
            raise TypeError(f"{dataset}: Stage6 cache must be float32.")
        if train.ndim!=3 or train.shape[1:]!=(8,256):
            raise ValueError(f"{dataset}: unexpected train cache shape {train.shape}.")
        if val.ndim!=2 or val.shape[1]!=256:
            raise ValueError(f"{dataset}: unexpected validation cache shape {val.shape}.")
        expected_train=len(trainer.datasets[dataset]["train"]["numeric_values"])
        expected_val=len(trainer.datasets[dataset]["validation"]["numeric_values"])
        if train.shape[0]!=expected_train or val.shape[0]!=expected_val:
            raise ValueError(f"{dataset}: cache row counts differ from Stage6 train/validation artifacts.")
        trainer._stage6_cache_train[dataset]=train
        trainer._stage6_cache_validation[dataset]=val

    def cached_encode(self, dataset: str, split: str, indices: np.ndarray):
        idx=np.asarray(indices,dtype=np.int64)
        if split=="train":
            if self._stage6_cache_force_clean:
                draws=np.zeros(idx.size,dtype=np.int64)
            else:
                draws=self._stage6_cache_rng[dataset].integers(0,8,size=idx.size,dtype=np.int64)
            packed=np.asarray(self._stage6_cache_train[dataset][idx,draws,:],dtype=np.float32)
        elif split=="validation":
            packed=np.asarray(self._stage6_cache_validation[dataset][idx,:],dtype=np.float32)
        else:
            raise ValueError("Cached Stage6 path permits train and validation only.")
        if packed.shape!=(idx.size,256) or not np.isfinite(packed).all():
            raise FloatingPointError(f"{dataset}/{split}: invalid cached representation batch.")
        tensor=torch.from_numpy(np.ascontiguousarray(packed))
        zs=tensor[:,:128].to(self.device,non_blocking=False)
        zd=tensor[:,128:].to(self.device,non_blocking=False)
        return zs,zd

    trainer._encode=MethodType(cached_encode,trainer)
    trainer._stage6_cache_root=str(root)
    trainer._stage6_cache_draw_policy="training: deterministic per-row uniform draw 0..7; validation: clean cache"

def release_stage6_source_arrays(trainer: Any) -> dict[str, dict[str, int]]:
    """Release in-memory train/validation input arrays after cache row counts are verified.

    The on-disk dataset artifacts are never changed. Stage6Trainer's cached
    _encode path uses only the K=8/clean cache; its remaining methods need the
    original split row counts, which are retained in tiny placeholder arrays.
    """
    import gc
    counts: dict[str, dict[str, int]] = {}
    for dataset in ("nsl_kdd", "unsw_nb15"):
        counts[dataset] = {}
        for split in ("train", "validation"):
            n = len(trainer.datasets[dataset][split]["numeric_values"])
            counts[dataset][split] = int(n)
            trainer.datasets[dataset][split] = {
                "numeric_values": np.empty((n, 0), dtype=np.float32)
            }
    gc.collect()
    trainer._stage6_cache_source_row_counts = counts
    return counts
