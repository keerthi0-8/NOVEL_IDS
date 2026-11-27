from pathlib import Path
p=Path(r"C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project\scripts\build_stage6_embedding_cache.py")
s=p.read_text(encoding="utf-8")
old='''        count = 256
        draw = 1
        rng = np.random.default_rng(AUGMENTATION_SEED + ds_index * 10000 + draw * 101)
        zs_direct, zd_direct, avail, miss, dropped, value_mask = encode_augmented(
            stage5, train_arrays, dataset, 0, count, ds_level3, rng, stage5_cfg
        )
        cached = np.load(dataset_manifest[dataset]["train_cache_path"], mmap_mode="r")
        zs_cached = torch.as_tensor(cached[:count, draw, :128], device="cuda")
        zd_cached = torch.as_tensor(cached[:count, draw, 128:], device="cuda")
        gpu_cached = loss_bundle(branches_gpu[dataset], zs_cached, zd_cached)
        gpu_direct = loss_bundle(branches_gpu[dataset], zs_direct, zd_direct)
        loss_diff = {
            key: abs(gpu_cached[key] - gpu_direct[key]) for key in gpu_cached
        }
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

        cpu_zs, cpu_zd = stage5_encode_augmented(
            stage5_cpu,
            train_arrays["numeric_values"][:count],
            train_arrays["categorical_values"][:count],
            avail, miss,
            train_arrays["numeric_token_mask"],
            train_arrays["categorical_token_mask"],
            dropped,
            dataset,
        )
        cpu_losses = loss_bundle(branches_cpu[dataset], cpu_zs, cpu_zd)
        cpu_diff = {key: abs(cpu_losses[key] - gpu_direct[key]) for key in gpu_direct}
        if max(cpu_diff.values()) > 1e-5:
            raise RuntimeError(f"{dataset}: CPU/GPU Stage6 loss mismatch {cpu_diff}.")
        cpu_gpu_checks[dataset] = {
            "cpu_losses": cpu_losses,
            "gpu_losses": gpu_direct,
            "absolute_differences": cpu_diff,
            "max_absolute_difference": max(cpu_diff.values()),
            "tolerance": 1e-5,
            "passed": True,
        }
'''
new='''        count = 256
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
'''
if s.count(old)!=1: raise SystemExit(f"Expected exactly one loss-equivalence block, found {s.count(old)}")
s=s.replace(old,new)
s=s.replace('''    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
''','''    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
''')
s=s.replace('''    stage5_cpu = copy.deepcopy(stage5).to("cpu").eval()
''','')
p.write_text(s,encoding="utf-8")
print("PATCHED",p)
