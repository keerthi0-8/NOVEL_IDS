from __future__ import annotations

import copy
import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from novel_ids.representation.dual_encoder import DualEncoder
from novel_ids.residual.stage6 import (
    Stage6LossCoefficients,
    Stage6ResidualBranch,
    combine_stage6_losses_with_projection,
    freeze_stage5,
    information_reconstruction_loss,
    magnitude_stability_loss,
    prediction_loss,
    projection_geometry_loss,
    standardized_cross_covariance_loss,
)
from novel_ids.residual.stage6_data import load_stage6_input_artifact
from novel_ids.residual.stage6_decorr import residual_decorrelation_loss
from novel_ids.residual.stage6_fastpath import (
    make_augmentation_masks,
    sha256_file,
    stage5_encode_augmented,
)
from novel_ids.residual.stage6_training import STAGE5_PROTECTED_HASHES

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
OLD_ARM = Path(r"C:\NOVEL_IDS_runs\stage6\phase2_decorr_20261125\candidate09_lambda0.1_seed42")
INPUT_ROOT = ROOT / "data" / "interim" / "stage6_trainval_only"
SCHEMA = ROOT / "data" / "processed" / "unified_representation" / "unified_feature_schema.yaml"
OUTPUT_ROOT = Path(r"C:\NOVEL_IDS_runs\stage6\phase2_fastcache_20261125")
K = 8
BATCH_SIZE = 512
AUGMENTATION_SEED = 20261125
EXPECTED_SNAPSHOT = "9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b"
DATASETS = ("nsl_kdd", "unsw_nb15")


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def free_ram_gb() -> float:
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed.")
    return status.ullAvailPhys / (1024 ** 3)


def verify_protected() -> dict[str, str]:
    hashes = {}
    for relative, expected in STAGE5_PROTECTED_HASHES.items():
        actual = sha256_file(str(ROOT / relative))
        if actual.lower() != expected.lower():
            raise RuntimeError(f"Protected Stage5 hash mismatch: {relative}: {actual}")
        hashes[relative] = actual
    snapshot_hash = sha256_file(str(SNAPSHOT))
    if snapshot_hash.lower() != EXPECTED_SNAPSHOT:
        raise RuntimeError(f"Stage5 snapshot mismatch: {snapshot_hash}")
    hashes["stage5_snapshot"] = snapshot_hash
    return hashes


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temp.replace(path)


def make_stage5(device: torch.device):
    checkpoint = torch.load(SNAPSHOT, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    identity_path = ROOT / "data" / "processed" / "unified_representation" / "feature_identity_vectors.npy"
    identities = np.load(identity_path, allow_pickle=False)
    model = DualEncoder(
        identities,
        dropout=float(cfg["dropout"]),
        value_masking_rate=float(cfg["value_masking_rate"]),
        feature_set_dropout_rate=float(cfg["feature_set_dropout_rate"]),
        feature_set_dropout_all_probability=float(cfg["feature_set_dropout_all_probability"]),
    )
    model.load_state_dict(checkpoint["dual_encoder"], strict=True)
    del checkpoint
    return freeze_stage5(model).to(device).eval(), cfg


def encode_clean(stage5, arrays, dataset, start, stop):
    idx = np.arange(start, stop)
    zero_drop = np.zeros((len(idx), arrays["M_avail"].shape[1]), dtype=bool)
    return stage5_encode_augmented(
        stage5,
        arrays["numeric_values"][idx],
        arrays["categorical_values"][idx],
        arrays["M_avail"][idx],
        arrays["M_miss"][idx],
        arrays["numeric_token_mask"],
        arrays["categorical_token_mask"],
        zero_drop,
        dataset,
    )


def encode_augmented(stage5, arrays, dataset, start, stop, level3_positions, rng, cfg):
    idx = np.arange(start, stop)
    avail, miss, dropped, value_mask = make_augmentation_masks(
        arrays["M_avail"][idx],
        arrays["M_miss"][idx],
        arrays["numeric_token_mask"],
        arrays["categorical_token_mask"],
        level3_positions,
        rng,
        float(cfg["feature_set_dropout_rate"]),
        float(cfg["feature_set_dropout_all_probability"]),
        float(cfg["value_masking_rate"]),
    )
    z_s, z_d = stage5_encode_augmented(
        stage5,
        arrays["numeric_values"][idx],
        arrays["categorical_values"][idx],
        avail,
        miss,
        arrays["numeric_token_mask"],
        arrays["categorical_token_mask"],
        dropped,
        dataset,
    )
    return z_s, z_d, avail, miss, dropped, value_mask


def loss_bundle(branch, z_s, z_d, lambda_red=1.0, lambda_decorr=0.1):
    output = branch(z_s.float(), z_d.float())
    l_pred = prediction_loss(output["Z_d_hat"], z_d)
    l_info = information_reconstruction_loss(output["E_d_hat"], output["E_d"])
    l_mag, _ = magnitude_stability_loss(output["R"])
    l_red, _ = standardized_cross_covariance_loss(
        z_s, output["R"], variance_floor=1e-4
    )
    l_decorr, _ = residual_decorrelation_loss(
        output["R"], variance_floor=1e-4
    )
    l_proj = projection_geometry_loss(
        output["R"], output["R_proj_objective"]
    )
    coefficients = Stage6LossCoefficients(
        prediction=1.0, information=1.0, magnitude=1.0,
        redundancy=float(lambda_red),
    )
    total = combine_stage6_losses_with_projection(
        l_pred=l_pred, l_info=l_info, l_mag=l_mag, l_red=l_red,
        l_proj=l_proj, coefficients=coefficients, projection_coefficient=1.0,
    ) + float(lambda_decorr) * l_decorr
    return {
        "L_pred": float(l_pred.detach().cpu()),
        "L_info": float(l_info.detach().cpu()),
        "L_mag": float(l_mag.detach().cpu()),
        "L_red": float(l_red.detach().cpu()),
        "L_decorr": float(l_decorr.detach().cpu()),
        "L_proj": float(l_proj.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage5 cache generation.")
    if free_ram_gb() < 2.0:
        raise RuntimeError(f"Hard stop: free RAM below 2 GB ({free_ram_gb():.2f} GB).")
    if torch.cuda.mem_get_info()[0] < int(5.0 * 1024**3):
        raise RuntimeError("Cache build requires at least 5 GiB free GPU memory.")
    disk_free = __import__("shutil").disk_usage("C:\\").free / (1024**3)
    if disk_free < 10.0:
        raise RuntimeError(f"Hard stop: disk free below 10 GB ({disk_free:.2f} GB).")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

    protected_before = verify_protected()
    schema = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    if int(schema["source_identity_count"]) != 78 or len(schema["features"]) != 78:
        raise RuntimeError("Unified feature schema is not the approved 78-slot schema.")
    level3_positions = [
        int(item["index"]) for item in schema["features"] if int(item.get("level", 0)) == 3
    ]
    if len(level3_positions) != 8:
        raise RuntimeError(f"Expected 8 Level-3 slots across the 78-slot schema, got {level3_positions}.")

    stage5, stage5_cfg = make_stage5(torch.device("cuda"))
    if any(parameter.requires_grad for parameter in stage5.parameters()) or stage5.training:
        raise RuntimeError("Stage5 must remain frozen and eval-mode.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    (OUTPUT_ROOT / "cache").mkdir()
    build_log = OUTPUT_ROOT / "cache_build.log"
    start_time = time.perf_counter()
    log_time = start_time
    total_bytes = 0
    dataset_manifest = {}

    for ds_index, dataset in enumerate(DATASETS):
        input_path = INPUT_ROOT / f"{dataset}_inputs.joblib"
        input_hash = sha256_file(str(input_path))
        arrays_by_split = load_stage6_input_artifact(input_path, dataset)
        train_arrays = arrays_by_split["train"]
        val_arrays = arrays_by_split["validation"]
        train_n = len(train_arrays["numeric_values"])
        val_n = len(val_arrays["numeric_values"])
        ds_dir = OUTPUT_ROOT / "cache" / dataset
        ds_dir.mkdir()
        train_path = ds_dir / "train_zszd.npy"
        val_path = ds_dir / "validation_zszd.npy"
        train_cache = np.lib.format.open_memmap(
            train_path, mode="w+", dtype=np.float32, shape=(train_n, K, 256)
        )
        val_cache = np.lib.format.open_memmap(
            val_path, mode="w+", dtype=np.float32, shape=(val_n, 256)
        )
        dataset_level3 = [
            int(item["index"]) for item in schema["features"]
            if item.get("dataset") == dataset and int(item.get("level", 0)) == 3
        ]
        if len(dataset_level3) != 4:
            raise RuntimeError(f"{dataset}: expected four Level-3 slots, got {dataset_level3}.")

        # Training bank: draw 0 clean; draws 1-7 use fixed, independent row masks.
        for draw in range(K):
            rng = np.random.default_rng(AUGMENTATION_SEED + ds_index * 10000 + draw * 101)
            for begin in range(0, train_n, BATCH_SIZE):
                end = min(train_n, begin + BATCH_SIZE)
                if draw == 0:
                    zs, zd = encode_clean(stage5, train_arrays, dataset, begin, end)
                else:
                    zs, zd, _, _, _, _ = encode_augmented(
                        stage5, train_arrays, dataset, begin, end,
                        dataset_level3, rng, stage5_cfg,
                    )
                train_cache[begin:end, draw, :128] = zs.detach().cpu().numpy()
                train_cache[begin:end, draw, 128:] = zd.detach().cpu().numpy()
                now = time.perf_counter()
                if now - log_time >= 60:
                    elapsed = now - start_time
                    completed_fraction = (
                        ds_index + (draw + end / train_n) / K
                    ) / len(DATASETS)
                    eta = elapsed * (1.0 - completed_fraction) / max(completed_fraction, 1e-6)
                    line = (
                        f"{time.strftime('%H:%M:%S')} dataset={dataset} draw={draw}/{K-1} "
                        f"rows={end}/{train_n} elapsed_s={elapsed:.1f} eta_s={eta:.1f} "
                        f"free_ram_gb={free_ram_gb():.2f}"
                    )
                    print(line, flush=True)
                    with build_log.open("a", encoding="utf-8") as handle:
                        handle.write(line + "\n")
                    log_time = now
            train_cache.flush()
            line = f"{dataset} train draw {draw}/{K-1} complete"
            print(line, flush=True)
            with build_log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

        # Source validation is clean under the approved evaluation protocol.
        for begin in range(0, val_n, BATCH_SIZE):
            end = min(val_n, begin + BATCH_SIZE)
            zs, zd = encode_clean(stage5, val_arrays, dataset, begin, end)
            val_cache[begin:end, :128] = zs.detach().cpu().numpy()
            val_cache[begin:end, 128:] = zd.detach().cpu().numpy()
        train_cache.flush()
        val_cache.flush()
        del train_cache, val_cache

        # Reproduce selected seeded draws from the start of each split and compare.
        cached_train = np.load(train_path, mmap_mode="r")
        max_zs = 0.0
        max_zd = 0.0
        for draw in (0, 1, 3, 7):
            count = min(BATCH_SIZE, train_n)
            if draw == 0:
                zs_direct, zd_direct = encode_clean(stage5, train_arrays, dataset, 0, count)
            else:
                rng = np.random.default_rng(AUGMENTATION_SEED + ds_index * 10000 + draw * 101)
                zs_direct, zd_direct, _, _, _, _ = encode_augmented(
                    stage5, train_arrays, dataset, 0, count,
                    dataset_level3, rng, stage5_cfg,
                )
            diff_zs = np.max(np.abs(cached_train[:count, draw, :128] - zs_direct.cpu().numpy()))
            diff_zd = np.max(np.abs(cached_train[:count, draw, 128:] - zd_direct.cpu().numpy()))
            max_zs = max(max_zs, float(diff_zs))
            max_zd = max(max_zd, float(diff_zd))
        if max(max_zs, max_zd) > 1e-6:
            raise RuntimeError(
                f"{dataset}: cached/on-the-fly Stage5 mismatch Zs={max_zs} Zd={max_zd}."
            )
        cached_val = np.load(val_path, mmap_mode="r")
        val_count = min(BATCH_SIZE, val_n)
        val_zs, val_zd = encode_clean(stage5, val_arrays, dataset, 0, val_count)
        val_diff = max(
            float(np.max(np.abs(cached_val[:val_count, :128] - val_zs.cpu().numpy()))),
            float(np.max(np.abs(cached_val[:val_count, 128:] - val_zd.cpu().numpy()))),
        )
        if val_diff > 1e-6:
            raise RuntimeError(f"{dataset}: clean validation cache mismatch {val_diff}.")

        train_bytes = train_path.stat().st_size
        val_bytes = val_path.stat().st_size
        total_bytes += train_bytes + val_bytes
        dataset_manifest[dataset] = {
            "input_artifact_sha256": input_hash,
            "train_rows": train_n,
            "validation_rows": val_n,
            "train_cache_path": str(train_path),
            "validation_cache_path": str(val_path),
            "train_cache_shape": [train_n, K, 256],
            "validation_cache_shape": [val_n, 256],
            "train_cache_bytes": train_bytes,
            "validation_cache_bytes": val_bytes,
            "train_cache_sha256": sha256_file(str(train_path)),
            "validation_cache_sha256": sha256_file(str(val_path)),
            "level3_positions": dataset_level3,
            "max_abs_cached_vs_on_the_fly_Zs": max_zs,
            "max_abs_cached_vs_on_the_fly_Zd": max_zd,
            "max_abs_validation_clean_difference": val_diff,
        }
        del cached_train, cached_val, arrays_by_split, train_arrays, val_arrays
        torch.cuda.empty_cache()

    # Fixed-batch loss equivalence: cache vs direct augmentation, and CPU vs GPU.
    old_ckpt = torch.load(OLD_ARM / "best.pt", map_location="cpu", weights_only=False)
    branches_gpu = torch.nn.ModuleDict({
        ds: Stage6ResidualBranch(128, 128, 128, 128, 128, 128) for ds in DATASETS
    }).to("cuda")
    branches_gpu.load_state_dict(old_ckpt["branches"], strict=True)
    branches_gpu.eval()
    branches_cpu = copy.deepcopy(branches_gpu).to("cpu").eval()
    loss_checks = {}
    cpu_gpu_checks = {}

    for ds_index, dataset in enumerate(DATASETS):
        arrays_by_split = load_stage6_input_artifact(INPUT_ROOT / f"{dataset}_inputs.joblib", dataset)
        train_arrays = arrays_by_split["train"]
        ds_level3 = dataset_manifest[dataset]["level3_positions"]
        count = 256
        cache_forward_count = min(BATCH_SIZE, train_n)
        draw = 1
        rng = np.random.default_rng(AUGMENTATION_SEED + ds_index * 10000 + draw * 101)
        # Reproduce the cache builder's exact first 512-row RNG consumption and
        # Stage5 forward batch, then score the first 256 Stage6 rows.
        zs_full, zd_full, avail_full, miss_full, dropped_full, value_mask_full = encode_augmented(
            stage5, train_arrays, dataset, 0, cache_forward_count, ds_level3, rng, stage5_cfg
        )
        zs_direct, zd_direct = zs_full[:count].contiguous(), zd_full[:count].contiguous()
        cached = np.load(dataset_manifest[dataset]["train_cache_path"], mmap_mode="r")
        if cached.dtype != np.float32:
            raise RuntimeError(f"{dataset}: cache dtype must be float32, got {cached.dtype}.")
        zs_cached = torch.as_tensor(np.array(cached[:count, draw, :128], dtype=np.float32, copy=True), device="cuda")
        zd_cached = torch.as_tensor(np.array(cached[:count, draw, 128:], dtype=np.float32, copy=True), device="cuda")
        with torch.autocast(device_type="cuda", enabled=False):
            gpu_cached = loss_bundle(branches_gpu[dataset], zs_cached, zd_cached)
            gpu_direct = loss_bundle(branches_gpu[dataset], zs_direct, zd_direct)
        loss_diff = {key: abs(gpu_cached[key] - gpu_direct[key]) for key in gpu_cached}
        if max(loss_diff.values()) > 1e-5:
            raise RuntimeError(f"{dataset}: cached/on-the-fly Stage6 loss mismatch {loss_diff}.")
        loss_checks[dataset] = {
            "cached_losses": gpu_cached,
            "on_the_fly_losses": gpu_direct,
            "absolute_differences": loss_diff,
            "max_absolute_difference": max(loss_diff.values()),
            "tolerance": 1e-5,
            "passed": True,
        }

        # CPU/GPU comparison uses the exact same cached float32 inputs and
        # identical saved branch weights; only the compute device changes.
        cpu_losses = loss_bundle(
            branches_cpu[dataset], zs_cached.detach().cpu(), zd_cached.detach().cpu()
        )
        cpu_diff = {
            key: abs(cpu_losses[key] - gpu_cached[key]) / max(abs(cpu_losses[key]), abs(gpu_cached[key]), 1e-12)
            for key in gpu_cached
        }
        if max(cpu_diff.values()) > 1e-4:
            raise RuntimeError(f"{dataset}: CPU/GPU relative Stage6 loss mismatch {cpu_diff}.")
        cpu_gpu_checks[dataset] = {
            "cpu_losses": cpu_losses,
            "gpu_losses": gpu_cached,
            "relative_differences": cpu_diff,
            "max_relative_difference": max(cpu_diff.values()),
            "tolerance": 1e-4,
            "passed": True,
        }
        del cached, arrays_by_split, train_arrays
        torch.cuda.empty_cache()

    protected_after = verify_protected()
    if protected_before != protected_after:
        raise RuntimeError("Protected Stage5 hash manifest changed during cache build.")
    manifest = {
        "schema_version": 1,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage5_snapshot_sha256": EXPECTED_SNAPSHOT,
        "protected_stage5_hashes": protected_after,
        "stage5_config": stage5_cfg,
        "augmentation": {
            "K": K,
            "draw_0": "clean",
            "draws_1_to_7": "seeded feature-set dropout (p=0.1, all-specific p=0.05) and value masking (p=0.05)",
            "seed_base": AUGMENTATION_SEED,
            "validation": "clean only; validation scoring and Stage6 validation are clean by protocol",
            "finite_bank_deviation": "finite K=8 augmentation bank replaces fresh random augmentation each optimizer step",
            "stage5_module_mode": "eval, frozen, no_grad; ordinary neural Dropout disabled",
            "feature_dropout_pooling": "Level-3 token positions from the approved 78-slot unified schema are excluded from pooling when dropped",
        },
        "dtype": "float32",
        "feature_layout": "concatenate Z_s[128] and Z_d[128]",
        "total_cache_bytes": total_bytes,
        "total_cache_gib": total_bytes / (1024**3),
        "datasets": dataset_manifest,
        "stage6_loss_equivalence": loss_checks,
        "cpu_gpu_loss_equivalence": cpu_gpu_checks,
        "max_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "free_ram_after_build_gb": free_ram_gb(),
        "free_disk_after_build_gb": __import__("shutil").disk_usage("C:\\").free / (1024**3),
    }
    write_json(OUTPUT_ROOT / "cache_manifest.json", manifest)
    print(json.dumps({
        "status": "CACHE_READY",
        "cache_root": str(OUTPUT_ROOT),
        "total_cache_gib": manifest["total_cache_gib"],
        "loss_equivalence": loss_checks,
        "cpu_gpu_equivalence": cpu_gpu_checks,
        "free_ram_gb": manifest["free_ram_after_build_gb"],
        "free_disk_gb": manifest["free_disk_after_build_gb"],
        "max_gpu_allocated_bytes": manifest["max_gpu_allocated_bytes"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
