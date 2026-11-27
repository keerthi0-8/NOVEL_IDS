from __future__ import annotations

import argparse
import json
from pathlib import Path

from novel_ids.inference.api import load_model, onboard_csv, predict_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVEL-IDS zero-shot streaming CSV inference")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--labels-column", default=None)
    parser.add_argument("--normalization-input", default=None,
                        help="Separate unlabeled target CSV used only for Option A calibration.")
    parser.add_argument("--feature-metadata", default=None,
                        help="Optional external YAML feature metadata for semantic onboarding.")
    parser.add_argument("--normalization-mode", choices=("option_a_target", "option_b_strict_source_level1"),
                        default="option_a_target",
                        help="Option A uses unlabeled target statistics; Option B uses source-training statistics for Level-1 numeric features.")
    parser.add_argument("--checkpoint-dir", default=r"C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed43")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()

    try:
        model = load_model(args.checkpoint_dir)
        onboarding = onboard_csv(
            args.input, model=model, label_column=args.labels_column,
            normalization_path=args.normalization_input, chunksize=args.chunk_size,
            feature_metadata_path=args.feature_metadata,
            normalization_mode=args.normalization_mode,
        )
        predictions, summary = predict_csv(
            args.input, model, onboarding=onboarding,
            label_column=args.labels_column, batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        output_path = Path(args.output).expanduser()
        summary_path = Path(args.summary).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({
            "rows": len(predictions),
            "output": str(output_path.resolve()),
            "summary": str(summary_path.resolve()),
            "decision_counts": summary["decision_counts"],
        }, indent=2))
    except Exception as exc:
        parser.exit(2, f"NOVEL-IDS inference failed ({type(exc).__name__}): {exc}\n")


if __name__ == "__main__":
    main()
