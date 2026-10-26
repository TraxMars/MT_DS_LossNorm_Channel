"""Read-only SHA256, historical configuration, split and warm-up verification."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT / "analysis/stage_features"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def inventory(root):
    return {str(path) for path in root.rglob("*") if path.is_file()
            and not any(part in {".git", "__pycache__"} for part in path.relative_to(root).parts)}


def verify():
    started = time.monotonic()
    baseline = json.loads((REPORT_DIR / "BASELINE_MANIFEST.json").read_text())
    mutable = set(baseline["mutable_original_files"])
    mismatches, checked = [], 0
    for filename, expected in baseline["files"].items():
        if filename in mutable:
            continue
        path = Path(filename)
        checked += 1
        if not path.is_file():
            mismatches.append(dict(path=filename, error="missing"))
        elif path.stat().st_size != expected["size_bytes"] or digest(path) != expected["sha256"]:
            mismatches.append(dict(path=filename, error="content changed"))
    # New files in protected trees must also be detected.
    roots = [PROJECT.parent / "MT_DS_LossNorm", PROJECT.parent / "Dataset"]
    roots += [PROJECT / name for name in ("Results", "summary", "verification_runs", "launch_logs")]
    for root in roots:
        previous = {name for name in baseline["files"] if Path(name).is_relative_to(root)}
        for added in sorted(inventory(root) - previous):
            mismatches.append(dict(path=added, error="new file in protected tree"))
    current_main = subprocess.check_output(["git", "rev-parse", "main"], cwd=PROJECT, text=True).strip()
    if current_main != baseline["main_commit"]:
        mismatches.append(dict(error="main commit changed", actual=current_main))
    runs = {}
    code_names = ("train.py", "data.py", "objectives.py", "evaluate.py",
                  "networks/vnet_ds.py", "networks/__init__.py")
    configs = json.loads((REPORT_DIR / "EXPERIMENT_CONFIGS.json").read_text())
    for name, saved in configs.items():
        directory = PROJECT / "Results/seed_42" / name
        split_record = saved["data_split_manifest.json"]
        sets, counts = {}, {}
        for split, record in split_record["splits"].items():
            path = Path(record["path"])
            ids = [x.strip() for x in path.read_text().splitlines() if x.strip()]
            sets[split], counts[split] = set(ids), len(ids)
            if digest(path) != record["sha256"] or ids != record["ids"]:
                mismatches.append(dict(run=name, error=f"live {split} list differs from experiment"))
        if len(set.union(*sets.values())) != sum(map(len, sets.values())):
            mismatches.append(dict(run=name, error="data split overlap"))
        train_ids = split_record["splits"]["train"]["ids"]
        labelnum = saved["config.json"]["labelnum"]
        if (split_record["labeled_ids"] != train_ids[:labelnum]
                or split_record["unlabeled_ids"] != train_ids[labelnum:]):
            mismatches.append(dict(run=name, error="labeled ordering mismatch"))
        snapshot_mismatches = []
        for relative in code_names:
            reference = baseline["files"][str(PROJECT / "code" / relative)]["sha256"]
            if digest(directory / "code" / relative) != reference:
                snapshot_mismatches.append(relative)
        if snapshot_mismatches:
            mismatches.append(dict(run=name, error="saved code differs from initial active code",
                                   files=snapshot_mismatches))
        with (directory / "training_metrics.csv").open() as handle:
            rows = {int(row["iteration"]): row for row in csv.DictReader(handle)}
        boundary = {step: float(rows[step]["consistency_weight"]) for step in (1, 2000, 2010)}
        if boundary != {1: 0.0, 2000: 0.0, 2010: 0.3}:
            mismatches.append(dict(run=name, error="historical warm-up log differs", values=boundary))
        if saved["run_status.json"]["state"] != "completed":
            mismatches.append(dict(run=name, error="historical experiment incomplete"))
        runs[name] = dict(split_counts=counts, splits_disjoint=True,
                          labeled_prefix_matches=True, saved_source_matches_initial_main=True,
                          warmup_weights=boundary, state=saved["run_status.json"]["state"])
    return dict(status="PASS" if not mismatches else "FAIL", checked_files=checked,
                original_file_count=len(baseline["files"]), changed_allowed_files=sorted(mutable),
                main_commit=current_main, mismatches=mismatches, runs=runs,
                elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_DIR / "INTEGRITY_RESULTS.json")
    args = parser.parse_args()
    result = verify()
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
