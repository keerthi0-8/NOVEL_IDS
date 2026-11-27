import json
import sys
from pathlib import Path

DEFAULT = Path(r"C:\NOVEL_IDS_runs\phaseC_recovery_20261126\results.json")

def find_reports(obj, path="root"):
    if isinstance(obj, dict):
        if isinstance(obj.get("per_class"), dict) and "confusion_matrix" in obj:
            yield path, obj
        for key, value in obj.items():
            yield from find_reports(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from find_reports(value, f"{path}[{i}]")

def pct(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"

def print_report(path, m):
    classes = m["per_class"]
    total = sum(int(v.get("support", 0)) for v in classes.values())

    weighted_precision = (
        sum(v["precision"] * v["support"] for v in classes.values()) / total
        if total else None
    )
    weighted_recall = (
        sum(v["recall"] * v["support"] for v in classes.values()) / total
        if total else None
    )

    print("\n" + "=" * 100)
    print("MULTICLASS EVALUATION REPORT")
    print("Result:", path)
    print("=" * 100)

    print(f"{'Accuracy':25s}: {pct(m.get('accuracy'))}")
    print(f"{'Macro precision':25s}: {pct(m.get('precision'))}")
    print(f"{'Macro recall':25s}: {pct(m.get('recall'))}")
    print(f"{'Macro F1':25s}: {pct(m.get('macro_f1', m.get('f1')))}")
    print(f"{'Weighted precision':25s}: {pct(weighted_precision)}")
    print(f"{'Weighted recall':25s}: {pct(weighted_recall)}")
    print(f"{'Weighted F1':25s}: {pct(m.get('weighted_f1'))}")
    print(f"{'Total support':25s}: {total:,}")

    print("\nPER-CLASS METRICS")
    print("-" * 100)
    print(f"{'Class':22s} {'Precision':>12s} {'Recall':>12s} {'F1-score':>12s} {'Support':>12s}")
    print("-" * 100)

    for name, values in classes.items():
        print(
            f"{name:22.22s} "
            f"{pct(values.get('precision')):>12s} "
            f"{pct(values.get('recall')):>12s} "
            f"{pct(values.get('f1')):>12s} "
            f"{values.get('support', 0):>12,}"
        )

    print("-" * 100)
    print(f"{'TOTAL':22s} {'':12s} {'':12s} {'':12s} {total:>12,}")

    print("\nCONFUSION MATRIX")
    for row in m["confusion_matrix"]:
        print(" ".join(f"{int(value):8d}" for value in row))
    print("=" * 100)

def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not source.is_file():
        raise FileNotFoundError(f"Results file not found: {source}")

    data = json.loads(source.read_text(encoding="utf-8"))
    reports = list(find_reports(data))

    if not reports:
        raise RuntimeError("No saved multiclass metric objects found in this JSON.")

    print(f"Reading saved evaluation results: {source}")
    print(f"Found {len(reports)} multiclass result(s).")

    for path, metrics in reports:
        print_report(path, metrics)

if __name__ == "__main__":
    main()
