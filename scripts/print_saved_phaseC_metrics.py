
import json
from pathlib import Path

RESULTS_FILE = Path(r"C:\NOVEL_IDS_runs\phaseC_recovery_20261126\results.json")


def find_model_results(obj, path="root"):
    """Find saved model metric dictionaries recursively."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}"

            if (
                isinstance(value, dict)
                and "accuracy" in value
                and "per_class" in value
                and "confusion_matrix" in value
            ):
                yield key, value, current_path
            else:
                yield from find_model_results(value, current_path)

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from find_model_results(value, f"{path}[{index}]")


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        results = json.load(file)

    models = list(find_model_results(results))

    if not models:
        print("No saved model metric records found.")
        return

    for model_name, metrics, location in models:
        print("\n" + "=" * 85)
        print(f"MODEL: {model_name}")
        print(f"RESULT LOCATION: {location}")
        print("=" * 85)

        print(f"Accuracy       : {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision      : {metrics['precision'] * 100:.2f}%")
        print(f"Recall         : {metrics['recall'] * 100:.2f}%")
        print(f"Macro F1       : {metrics['macro_f1'] * 100:.2f}%")
        print(f"Weighted F1    : {metrics['weighted_f1'] * 100:.2f}%")
        print(f"FPR            : {metrics['fpr'] * 100:.2f}%")
        print(f"FNR            : {metrics['fnr'] * 100:.2f}%")

        print("\nPER-CLASS METRICS")
        print("-" * 85)
        print(
            f"{'Class':<22}{'Precision':>12}"
            f"{'Recall':>12}{'F1':>12}{'Support':>12}"
        )
        print("-" * 85)

        for class_name, values in metrics["per_class"].items():
            print(
                f"{class_name:<22}"
                f"{values['precision'] * 100:>11.2f}%"
                f"{values['recall'] * 100:>11.2f}%"
                f"{values['f1'] * 100:>11.2f}%"
                f"{values['support']:>12}"
            )

        print("\nCONFUSION MATRIX")
        print("Class order:", list(metrics["per_class"].keys()))
        for row in metrics["confusion_matrix"]:
            print(row)

    print("\nFinished reading saved evaluation results.")
    print("No model training or inference was performed.")


if __name__ == "__main__":
    main()
