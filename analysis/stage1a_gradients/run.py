"""Extract full-patch frozen eval-mode parameter gradients without unlabeled GT."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import subprocess
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

from core import (PROJECT, SCALES, STAGES, deterministic_crop, gradient_validity,
                  independent_losses, parameter_gradients, read_image_only,
                  read_labeled_case, sha256, stage_parameters)
from networks import VNetDS


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protect_before(output):
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "HEAD"],
                                      cwd=PROJECT, text=True).splitlines()
    paths = {PROJECT / name for name in tracked if not name.startswith("analysis/stage1a_gradients/")}
    paths.update((PROJECT / "Results").rglob("*.pth"))
    source = PROJECT.parent / "MT_DS_LossNorm"
    paths.update(p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    # Never hash/read HDF5 datasets, including unlabeled GT. Lists are safe.
    paths.update((PROJECT.parent / "Dataset").rglob("*.list"))
    paths = sorted(p for p in paths if p.is_file() and p.suffix.lower() not in {".h5", ".hdf5"})
    hashes = {str(p): dict(sha256=sha256(p), size=p.stat().st_size) for p in paths}
    record = dict(base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                                       text=True).strip(), files=hashes,
                  data_policy="HDF5 contents not hashed: no unlabeled GT reads. Read-only data file stats and access logs checked separately.")
    write_json(output / "protection_before.json", record)
    return record


def protect_after(before):
    failures = []
    for name, expected in before["files"].items():
        path = Path(name)
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            failures.append(name)
    return dict(status="PASS" if not failures else "FAIL", checked_files=len(before["files"]),
                changed_or_missing=failures)


def state_snapshot(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def state_unchanged(model, before):
    return all(torch.equal(tensor.detach().cpu(), before[name]) for name, tensor in model.state_dict().items())


def analyze(args, output):
    started = time.monotonic()
    before = protect_before(output)
    checkpoint_path = PROJECT / "Results/seed_42/LA_8l/last_checkpoint.pth"
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["dataset"] != "LA" or config["labelnum"] != 8 or checkpoint["iteration"] != 15000:
        raise ValueError("Expected paired LA 10% iteration-15000 checkpoint")
    manifest = json.loads((checkpoint_path.parent / "data_split_manifest.json").read_text())
    if sha256(Path(manifest["splits"]["train"]["path"])) != manifest["splits"]["train"]["sha256"]:
        raise AssertionError("Train list has changed")
    labeled = manifest["labeled_ids"]
    selection_rng = np.random.default_rng(args.seed)
    if not 1 <= args.unlabeled_cases <= len(manifest["unlabeled_ids"]):
        raise ValueError("Invalid unlabeled case count")
    chosen = sorted(selection_rng.choice(len(manifest["unlabeled_ids"]), args.unlabeled_cases,
                                          replace=False).tolist())
    unlabeled = [manifest["unlabeled_ids"][i] for i in chosen]
    if set(labeled) & set(unlabeled):
        raise AssertionError("Labeled/unlabeled overlap")
    root = Path(config["dataset_root"])
    patch = tuple(config["patch_size"])
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Diagnostic precision policy: prevent TF32 rounding from obscuring the
    # linear decomposition of four independent head gradients. No train code changes.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(False)  # Original CUDA CE has no strict-deterministic implementation.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this execution context")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    student = VNetDS(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True).to(device).eval()
    teacher = VNetDS(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True).to(device).eval()
    student.load_state_dict(checkpoint["student"], strict=True)
    teacher.load_state_dict(checkpoint["teacher"], strict=True)
    teacher.requires_grad_(False)
    del checkpoint
    mapping = stage_parameters(student)
    parameter_map = {stage: spec.describe() for stage, spec in mapping.items()}
    write_json(output / "parameter_mapping.json", parameter_map)
    design = dict(dataset="LA", label_percent=10, iteration=15000, checkpoint=str(checkpoint_path),
                  checkpoint_sha256=checkpoint_hash, checkpoint_pairing="student and teacher in same serialized checkpoint",
                  model_mode="eval; Student autograd enabled; no optimizer instantiated",
                  patch=list(patch), repeats=args.repeats, seed=args.seed, labeled_ids=labeled,
                  unlabeled_ids=unlabeled, unlabeled_population=len(manifest["unlabeled_ids"]),
                  normalization_eps=config["normalization_eps"], consistency_weight=config["consistency"],
                  batch_size=1, interpretation="Per-case patch diagnostics; global Dice is evaluated at batch_size=1, not a reconstruction of a batch_size=2 training update",
                  augmentation="none; independent uniform image-shape-only crops, predeclared seeds",
                  absolute_gradient_floor=args.absolute_floor, relative_gradient_floor=args.relative_floor,
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                  diagnostic_precision="float32; cuDNN/matmul TF32 disabled; no mixed precision",
                  python_torch=torch.__version__, created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write_json(output / "design.json", design)
    initial_student, initial_teacher = state_snapshot(student), state_snapshot(teacher)
    torch_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    access_log, observations, file_stats = [], {"L": [], "U": []}, {}
    raw_dir = output / "raw"
    raw_dir.mkdir()
    arrays = {}
    for domain, cases in (("L", labeled), ("U", unlabeled)):
        arrays[domain] = {}
        for stage, spec in mapping.items():
            shape = (len(cases) * args.repeats, 4, spec.channels, spec.parameter.numel() // spec.channels)
            arrays[domain][stage] = np.lib.format.open_memmap(raw_dir / f"{domain}_{stage}.npy", mode="w+",
                                                            dtype=np.float32, shape=shape)
            arrays[domain][stage][:] = np.nan
    channel_path = output / "per_observation_channels.csv"
    loss_rows, reconstruction_checks = [], []
    with channel_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=("domain", "case", "repeat", "stage", "scale", "channel",
                                                    "status", "raw_norm", "validity_floor", "raw_mean_scale_contribution_norm",
                                                    "lossnorm_scale_contribution_norm", "normalization_factor"))
        writer.writeheader()
        for domain, cases in (("L", labeled), ("U", unlabeled)):
            for case_index, case in enumerate(cases):
                path = root / "2018LA_Seg_TrainingSet" / case / "mri_norm2.h5"
                stat = path.stat()
                file_stats[str(path)] = (stat.st_size, stat.st_mtime_ns)
                if domain == "L":
                    image, label = read_labeled_case(path, access_log, case)
                else:
                    image, label = read_image_only(path, access_log, "unlabeled", case), None
                for repeat in range(args.repeats):
                    obs_index = case_index * args.repeats + repeat
                    x, y, crop = deterministic_crop(image, label, patch, args.seed, case, repeat)
                    x = x.to(device)
                    outputs, features = student(x, deepsup=True, return_features=True)
                    for stage, spec in mapping.items():
                        tensor = features["encoder" if stage.startswith("E") else "decoder"][stage]
                        if tensor.shape[1] != spec.channels or not tensor.requires_grad:
                            raise AssertionError("Feature-to-parameter channel mapping failed")
                    targets = None
                    if domain == "U":
                        with torch.no_grad():
                            targets = teacher(x, deepsup=True)
                        if any(t.requires_grad for t in targets):
                            raise AssertionError("Teacher target attached to autograd")
                    losses, total, details = independent_losses(outputs, labels=y.to(device) if y is not None else None,
                                                               teacher_outputs=targets)
                    values, total_value = [float(t.detach()) for t in losses], float(total.detach())
                    gradients, check = parameter_gradients(losses, total, mapping,
                                                          eps=config["normalization_eps"],
                                                          weight=1.0 if domain == "L" else config["consistency"])
                    observation = dict(case=case, repeat=repeat, index=obs_index, crop=crop, aggregate_loss=total_value,
                                       losses=values, normalized_scale_factor=check["normalized_scale_factor"])
                    if y is not None:
                        observation["labeled_foreground_fraction"] = float(y.float().mean())
                    observations[domain].append(observation)
                    reconstruction_checks.append(dict(domain=domain, case=case, repeat=repeat, **check))
                    for scale_index, (scale, loss, detail) in enumerate(zip(SCALES, values, details)):
                        loss_rows.append(dict(domain=domain, case=case, repeat=repeat, scale=scale,
                                              loss=loss, ce=detail.get("ce"), dice=detail.get("dice"),
                                              aggregate_loss=total_value,
                                              lossnorm_denominator=check["lossnorm_denominator"],
                                              normalized_scale_factor=check["normalized_scale_factor"]))
                        for stage, spec in mapping.items():
                            gradient = gradients[stage][scale_index]
                            if gradient is None:
                                norms = [None] * spec.channels
                                statuses = ["unreachable"] * spec.channels
                                floors = [None] * spec.channels
                            else:
                                arrays[domain][stage][obs_index, scale_index] = gradient
                                norms, statuses, floors = gradient_validity(gradient, args.absolute_floor,
                                                                          args.relative_floor)
                            for channel, (norm, status, floor) in enumerate(zip(norms, statuses, floors)):
                                writer.writerow(dict(domain=domain, case=case, repeat=repeat, stage=stage, scale=scale,
                                                     channel=channel, status=status, raw_norm=norm, validity_floor=floor,
                                                     raw_mean_scale_contribution_norm=None if norm is None else norm / 4,
                                                     lossnorm_scale_contribution_norm=None if norm is None else norm * check["normalized_scale_factor"],
                                                     normalization_factor=check["normalized_scale_factor"]))
                    del outputs, features, losses, total, gradients, targets, x, y
                for array in arrays[domain].values():
                    array.flush()
                print(f"DONE {domain} case {case_index+1}/{len(cases)} {case}; {args.repeats} crops; "
                      f"elapsed {time.monotonic()-started:.1f}s", flush=True)
    write_json(output / "observations.json", observations)
    write_json(output / "hdf5_access_log.json", access_log)
    write_json(output / "gradient_reconstruction.json", reconstruction_checks)
    with (output / "losses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(loss_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(loss_rows)
    invariants = dict(student_state_unchanged=state_unchanged(student, initial_student),
                      teacher_state_unchanged=state_unchanged(teacher, initial_teacher),
                      student_grad_slots_unchanged=all(p.grad is None for p in student.parameters()),
                      teacher_no_gradients=all(p.grad is None and not p.requires_grad for p in teacher.parameters()),
                      student_and_teacher_eval=not student.training and not teacher.training,
                      torch_rng_unchanged=torch.equal(torch.get_rng_state(), torch_rng),
                      cuda_rng_unchanged=cuda_rng is None or torch.equal(torch.cuda.get_rng_state(device), cuda_rng),
                      unlabeled_gt_never_accessed=all(row["key"] == "image" for row in access_log if row["role"] == "unlabeled"),
                      data_file_stats_unchanged=all((Path(name).stat().st_size, Path(name).stat().st_mtime_ns) == values
                                                    for name, values in file_stats.items()),
                      checkpoint_unchanged=sha256(checkpoint_path) == checkpoint_hash)
    protection = protect_after(before)
    write_json(output / "protection_after.json", protection)
    if not all(invariants.values()) or protection["status"] != "PASS":
        raise AssertionError(f"Diagnostic invariants failed: {invariants}")
    result = dict(status="PASS", invariants=invariants, observations=sum(map(len, observations.values())),
                  optimizer_instances=0, optimizer_steps=0, checkpoint_writes=0,
                  protected_files=protection["checked_files"], elapsed_seconds=time.monotonic()-started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None)
    write_json(output / "extraction_results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--unlabeled-cases", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--absolute-floor", type=float, default=1e-12)
    parser.add_argument("--relative-floor", type=float, default=1e-6)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parent / "runs"):
        parser.error("All analysis outputs must be under analysis/stage1a_gradients/runs")
    if args.repeats < 2 or min(args.absolute_floor, args.relative_floor) < 0:
        parser.error("Require at least two crops and nonnegative validity floors")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = analyze(args, output)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        write_json(output / "extraction_results.json", dict(status="FAIL", traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
