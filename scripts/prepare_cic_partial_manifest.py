#!/usr/bin/env python
"""Create a hash-verified inventory for the available CIC-DDoS2019 CSV files.

This is an inventory-only step. It does not load CSV contents, sample rows,
modify source files, or claim that the available files form the full dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPORTED_MISSING = ("MSSQL.csv", "NetBIOS.csv", "Syn.csv", "TFTP.csv", "UDP.csv")
DEFAULT_SOURCE = Path(r"G:\My Drive\CiC-DDos")
DEFAULT_OUTPUT = Path(r"C:\NOVEL_IDS_runs\phaseD_cic_partial_20261126")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Record file metadata without streaming multi-gigabyte Drive files for SHA-256.",
    )
    args = parser.parse_args()

    source = args.source.expanduser()
    output = args.output.expanduser()
    if not source.is_dir():
        raise SystemExit(
            f"Source folder is unavailable: {source}. Reconnect Google Drive; "
            "no inventory was generated."
        )

    output.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(
        (p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"),
        key=lambda p: str(p.relative_to(source)).casefold(),
    )

    rows: list[dict[str, object]] = []
    found_names: set[str] = set()
    for path in csv_paths:
        rel = path.relative_to(source).as_posix()
        name = path.name
        found_names.add(name.casefold())
        try:
            stat = path.stat()
            digest = "" if args.metadata_only else sha256_file(path)
            rows.append({
                "relative_path": rel,
                "filename": name,
                "bytes": stat.st_size,
                "sha256": digest,
                "readable_for_hash": None if args.metadata_only else True,
                "status": "present_hash_deferred" if args.metadata_only else "present",
                "error": "",
            })
        except OSError as exc:
            rows.append({
                "relative_path": rel,
                "filename": name,
                "bytes": "",
                "sha256": "",
                "readable_for_hash": False,
                "status": "hash_error",
                "error": f"{type(exc).__name__}: {exc}",
            })

    reported_missing = [
        name for name in REPORTED_MISSING if name.casefold() not in found_names
    ]
    inventory_path = output / "cic_file_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "relative_path", "filename", "bytes", "sha256",
                "readable_for_hash", "status", "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source),
        "output_directory": str(output),
        "inventory_only": True,
        "hash_mode": "deferred_metadata_only" if args.metadata_only else "sha256_full_read",
        "csv_file_count": len(rows),
        "total_bytes": sum(int(r["bytes"]) for r in rows if r["bytes"] != ""),
        "hash_errors": sum(r["status"] == "hash_error" for r in rows),
        "user_reported_missing_filenames": list(REPORTED_MISSING),
        "reported_missing_still_absent": reported_missing,
        "reported_missing_now_present": [
            name for name in REPORTED_MISSING if name.casefold() in found_names
        ],
        "expected_total_file_count": None,
        "expected_total_note": (
            "Not inferred. Reconcile with an authoritative original dataset manifest."
        ),
        "source_files_modified": False,
        "csv_contents_loaded": False,
    }
    summary_path = output / "cic_inventory_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Inventory: {inventory_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
