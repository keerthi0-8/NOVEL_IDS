from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import yaml

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "interim" / "stage6_trainval_only"
TAX_PATH = ROOT / "config" / "taxonomy_unified.yaml"
STAGE5 = Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
STAGE6 = Path(r"C:\NOVEL_IDS_runs\stage6\accepted_candidate09_lambda1_seed42\best.pt")
STAGE5_SHA = "9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b"
STAGE6_SHA = "1c66f6f8d1a029266e5a637b5db4d6f49881988bf16462d2c3fbc09cd704ecbd"
PROTECTED = {
    "src/novel_ids/representation/stage5_training.py": "EE3D8FE2DA5C76F9FBAC1D75E48EAA029E6A13AF68F96CF3BC81F13D22F98267",
    "src/novel_ids/representation/stage5_training_candidate.py": "0A1C75A046C9CE86BF122D4E4CECB65DF14365E2114760040B30D52787015130",
    "src/novel_ids/representation/dual_encoder.py": "1FF74D97E651C541B9AB6F84EC762F054D6CF03E24F054A8BDF9C8D723AC0FFC",
    "src/novel_ids/representation/classification_head.py": "D4D0C337AC8C9A234239461282DFD5A4753B7A660C9ADA91D9C404B77900EED8",
    "src/novel_ids/representation/reconstruction_decoder.py": "E043AEC1F54367A18A89DA762FE2BFD24A1898AA5FEA07AA2B81AB85F865243A",
    "src/novel_ids/representation/feature_identity.py": "D82C46AACCA3C4C53777A8AB7C2F1D0930FE8E44E2D554A912578482091942AC",
    "config/datasets.yaml": "ADE3CB9E5E12015542ECEB6BB2ACFA5F440551F72278ADE7C5BF4A955FE836BE",
    "config/preprocessing_policy.yaml": "123BDD2104D01C0839E93AAF7B2322072786AF7987AA7213A2247878CBCCD6EF",
}
DATASETS = ("nsl_kdd", "unsw_nb15")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log(path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def verify_provenance():
    for rel, expected in PROTECTED.items():
        actual = sha(ROOT / rel)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"HARD STOP protected Stage5 hash mismatch: {rel}: {actual}")
    if sha(STAGE5).lower() != STAGE5_SHA.lower():
        raise RuntimeError("HARD STOP Stage5 snapshot hash mismatch")
    if sha(STAGE6).lower() != STAGE6_SHA.lower():
        raise RuntimeError("HARD STOP frozen Stage6 checkpoint hash mismatch")
    manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test_partitions_loaded") is not False or manifest.get("allowed_splits") != ["train", "validation"]:
        raise RuntimeError("HARD STOP train/validation-only manifest contract mismatch")
    for ds in DATASETS:
        for kind in ("input", "label"):
            file = ART / f"{ds}_{'inputs' if kind == 'input' else 'labels'}.joblib"
            expected = manifest["datasets"][ds][f"{kind}_sha256"]
            if sha(file).lower() != expected.lower():
                raise RuntimeError(f"HARD STOP {ds} {kind} artifact hash mismatch")
    return manifest


def load_taxonomy():
    doc = yaml.safe_load(TAX_PATH.read_text(encoding="utf-8"))
    if doc.get("status") != "approved":
        raise RuntimeError("Approved unified taxonomy is required.")
    return doc


def build_features(ds: str):
    x = joblib.load(ART / f"{ds}_inputs.joblib")
    y = joblib.load(ART / f"{ds}_labels.joblib")
    out = {}
    for split in ("train", "validation"):
        arr = x[split]
        num_mask = np.asarray(arr["numeric_token_mask"], dtype=bool)
        cat_mask = np.asarray(arr["categorical_token_mask"], dtype=bool)
        num_idx = np.flatnonzero(num_mask)
        cat_idx = np.flatnonzero(cat_mask)
        numeric = np.asarray(arr["numeric_values"][:, num_idx], dtype=np.float32)
        categorical = np.asarray(arr["categorical_values"][:, cat_idx], dtype=np.int64)
        missing = np.asarray(arr["M_miss"][:, num_idx], dtype=np.float32)
        available = np.asarray(arr["M_avail"], dtype=np.float32)
        missing_all = np.asarray(arr["M_miss"], dtype=np.float32)
        labels = np.asarray(y[split]["labels"]).astype(str)
        # Preserve measurement-missingness as explicit input columns; structural
        # availability is retained separately and is never imputed as a value.
        out[split] = {
            "numeric": numeric,
            "categorical": categorical,
            "numeric_missing": missing,
            "available": available,
            "missing_all": missing_all,
            "labels": labels,
        }
    # Fit numeric medians and categorical vocabulary on training rows only.
    medians = np.nanmedian(out["train"]["numeric"], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    train_cat = out["train"]["categorical"].astype(str)
    encoder.fit(train_cat)
    for split in ("train", "validation"):
        row = out[split]
        numeric = np.where(np.isfinite(row["numeric"]), row["numeric"], medians).astype(np.float32)
        categorical = encoder.transform(row["categorical"].astype(str)).astype(np.float32, copy=False)
        row["X"] = np.concatenate(
            [numeric, categorical, row["numeric_missing"], row["available"], row["missing_all"]],
            axis=1,
        ).astype(np.float32, copy=False)
        if not np.isfinite(row["X"]).all():
            raise FloatingPointError(f"{ds}/{split}: non-finite baseline features after train-fit imputation")
    return out, {"numeric_medians": medians.tolist(), "category_sizes": [len(x) for x in encoder.categories_],
                 "category_values": [list(map(str, values)) for values in encoder.categories_],
                 "feature_dim": int(out["train"]["X"].shape[1]), "numeric_positions": num_idx.tolist(),
                 "categorical_positions": cat_idx.tolist()}


def encode_labels(labels: np.ndarray, taxonomy: dict, ds: str, mode: str, classes_override=None):
    if mode == "native":
        classes = list(classes_override) if classes_override is not None else sorted(np.unique(labels).tolist())
        mapping = {name: i for i, name in enumerate(classes)}
        try:
            return np.asarray([mapping[x] for x in labels], dtype=np.int64), classes
        except KeyError as exc:
            raise RuntimeError(f"Validation label absent from native training classes for {ds}: {exc}") from exc
    mp = {r["original_label"]: r["unified_class"] for r in taxonomy["datasets"][ds]["mappings"]}
    mapped = [mp[x] for x in labels]
    classes = (list(classes_override) if classes_override is not None else
               [name for name in taxonomy["classes"] if name in set(mapped)])
    index = {name: i for i, name in enumerate(classes)}
    try:
        encoded = np.asarray([index[name] for name in mapped], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"Validation unified label absent from training classes for {ds}: {exc}") from exc
    return encoded, classes


def metrics(y, pred, classes):
    present = np.unique(y)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=present, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=present, average="weighted", zero_division=0)),
        "per_class": {classes[int(c)]: {"precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(present)},
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(len(classes))).tolist(),
    }


def fit_hgb(Xtr, ytr, Xva, yva, class_names, seed, max_leaf_nodes):
    model = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=120, max_leaf_nodes=max_leaf_nodes,
        l2_regularization=1.0, early_stopping=False, random_state=seed, verbose=1,
    )
    weights = compute_sample_weight(class_weight="balanced", y=ytr)
    model.fit(Xtr, ytr, sample_weight=weights)
    pred = model.predict(Xva)
    return model, metrics(yva, pred, class_names)


class PlainMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def class_weights(y: np.ndarray, n_classes: int, device):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.power(np.maximum(counts, 1.0), -0.5)
    w = w / w.mean()
    w = np.maximum(w, w.max() / 15.0)
    w = w / w.mean()
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def fit_mlp(Xtr, ytr, Xva, yva, class_names, seed, device, log_path, tag, max_epochs=40):
    n_classes = len(class_names)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = PlainMLP(Xtr.shape[1], n_classes).to(device)
    weights = class_weights(ytr, n_classes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=device)
    xv = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    yv = torch.as_tensor(yva, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    best = -1.0; best_state = None; best_epoch = 0; stale = 0
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(ytr))
        losses = []
        for begin in range(0, len(order), 1024):
            idx = torch.as_tensor(order[begin:begin+1024], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xt[idx])
            loss = F.cross_entropy(logits, yt[idx], weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{tag}: non-finite MLP loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred = model(xv).argmax(1).cpu().numpy()
        score = float(f1_score(yva, pred, labels=np.unique(yva), average="macro", zero_division=0))
        elapsed = time.perf_counter() - start
        log(log_path, f"MLP_EPOCH tag={tag} seed={seed} epoch={epoch} loss={np.mean(losses):.6f} val_macro_f1={score:.6f} best={best:.6f} elapsed_s={elapsed:.1f}")
        if score > best + 1e-6:
            best = score; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 6:
            break
    if best_state is None:
        raise RuntimeError(f"{tag}: MLP produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xv).argmax(1).cpu().numpy()
    return model, metrics(yva, pred, list(class_names)), best_epoch, time.perf_counter()-start


def main(args):
    out_dir = Path(args.run_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    log_path = out_dir / "progress.log"
    manifest = verify_provenance()
    taxonomy = load_taxonomy()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.set_num_threads(2)
    log(log_path, f"BASELINES_START device={device} stage5_sha={STAGE5_SHA} stage6_sha={STAGE6_SHA} trainval_only=True")
    all_results = {"device": str(device), "stage5_sha256": STAGE5_SHA, "stage6_sha256": STAGE6_SHA,
                   "taxonomy_sha256": sha(TAX_PATH), "datasets": {}}
    for ds in DATASETS:
        data, feature_meta = build_features(ds)
        all_results["datasets"][ds] = {"feature_meta": feature_meta, "label_modes": {}}
        for mode in ("native", "unified"):
            ytr, classes = encode_labels(data["train"]["labels"], taxonomy, ds, mode)
            yva, _ = encode_labels(data["validation"]["labels"], taxonomy, ds, mode, classes_override=classes)
            if len(np.unique(ytr)) != len(classes):
                raise RuntimeError(f"{ds}/{mode}: missing training class in encoded labels")
            Xtr, Xva = data["train"]["X"], data["validation"]["X"]
            entry = {"classes": classes, "train_rows": int(len(ytr)), "validation_rows": int(len(yva))}
            # Validation-only hyperparameter selection for HGB; the selected
            # setting is then repeated across the three pre-registered seeds.
            tune = []
            for leaves in (15, 31):
                model, score = fit_hgb(Xtr, ytr, Xva, yva, classes, 42, leaves)
                tune.append({"max_leaf_nodes": leaves, "validation": score})
                log(log_path, f"HGB_TUNE dataset={ds} mode={mode} leaves={leaves} macro_f1={score['macro_f1']:.6f}")
                del model
            chosen = max(tune, key=lambda x: x["validation"]["macro_f1"])["max_leaf_nodes"]
            hgb_runs = []
            for seed in (42, 43, 44):
                model, score = fit_hgb(Xtr, ytr, Xva, yva, classes, seed, chosen)
                hgb_runs.append({"seed": seed, "validation": score})
                joblib.dump(model, out_dir / f"{ds}_{mode}_hgb_seed{seed}.joblib")
                log(log_path, f"HGB_DONE dataset={ds} mode={mode} seed={seed} macro_f1={score['macro_f1']:.6f}")
                del model
            mlp_runs = []
            for seed in (42, 43, 44):
                model, score, epoch, elapsed = fit_mlp(
                    Xtr, ytr, Xva, yva, classes, seed, device, log_path,
                    f"{ds}_{mode}", max_epochs=40,
                )
                mlp_runs.append({"seed": seed, "best_epoch": epoch, "elapsed_seconds": elapsed, "validation": score})
                torch.save({"state_dict": model.state_dict(), "seed": seed, "classes": classes,
                            "input_dim": int(Xtr.shape[1]), "best_epoch": epoch},
                           out_dir / f"{ds}_{mode}_mlp_seed{seed}.pt")
                del model
            entry["hgb"] = {"tuning": tune, "chosen_max_leaf_nodes": chosen, "runs": hgb_runs}
            entry["plain_mlp"] = {"runs": mlp_runs}
            all_results["datasets"][ds]["label_modes"][mode] = entry
            (out_dir / "results.partial.json").write_text(json.dumps(all_results, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2, allow_nan=False), encoding="utf-8")
    log(log_path, "BASELINES_COMPLETE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    raise SystemExit(main(args))
