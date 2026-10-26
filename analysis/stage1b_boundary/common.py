"""Stage 1B utilities; Stage 0/1A and production code are read-only imports."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
A_ROOT = PROJECT / "analysis/stage1a_gradients"
A_RUN = A_ROOT / "runs/la10_iter15000_seed142_fp32"
sys.path.insert(0, str(A_ROOT))
from core import (SCALES, STAGES, deterministic_crop, gradient_validity, independent_losses,
                  read_image_only, read_labeled_case, sha256, signed_cosine, stage_parameters)
from stats_tools import (bootstrap_pair_mean, case_means, describe, masked_mean,
                         observation_patterns, pairwise_cosine)
from networks import VNetDS


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(clean(row) for row in rows)


def fixed_rng(*parts):
    seed = int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def require_stage1b_run(path):
    path = Path(path).resolve()
    if path == ROOT / 'runs' or not path.is_relative_to(ROOT / 'runs'):
        raise ValueError('Results must stay inside the independent Stage1B/runs directory')
    return path


def protected_snapshot():
    # All pre-existing project/source artifacts, including untracked Stage1A raw
    # gradients. HDF5 content is never hashed/read for protection.
    paths = set()
    for base in (PROJECT, PROJECT.parent / "MT_DS_LossNorm"):
        for path in base.rglob("*"):
            if not path.is_file() or any(part in {".git", "__pycache__"} for part in path.relative_to(base).parts):
                continue
            if base == PROJECT and (path.is_relative_to(ROOT) or path.name == "test_stage1b_boundary.py"):
                continue
            if path.suffix.lower() not in {".h5", ".hdf5"}:
                paths.add(path)
    paths.update((PROJECT.parent / "Dataset").rglob("*.list"))
    return dict(base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                files={str(p): dict(size=p.stat().st_size, sha256=sha256(p)) for p in sorted(paths)},
                data_policy="Read-only HDF5; no container hashing to avoid unlabeled GT bytes. Lists hashed but test IDs never used for selection.")


def verify_protection(snapshot):
    changed = []
    for name, expected in snapshot["files"].items():
        path = Path(name)
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            changed.append(name)
    return dict(status="PASS" if not changed else "FAIL", checked_files=len(snapshot["files"]), changed=changed)


def load_models(device):
    design = json.loads((A_RUN / "design.json").read_text())
    path = Path(design["checkpoint"])
    assert sha256(path) == design["checkpoint_sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["iteration"] == 15000 and saved["config"]["dataset"] == "LA"
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    torch.manual_seed(design["seed"])
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(design["seed"])
    models = []
    for key in ("student", "teacher"):
        model = VNetDS(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True).to(device).eval()
        model.load_state_dict(saved[key], strict=True)
        models.append(model)
    models[1].requires_grad_(False)
    return *models, design


def state_snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def unchanged(model, snapshot):
    return all(torch.equal(v.detach().cpu(), snapshot[k]) for k, v in model.state_dict().items())


def load_case_and_crops(domain, case, design, observations, access):
    root = PROJECT.parent / "Dataset/LA/2018LA_Seg_TrainingSet"
    path = root / case / "mri_norm2.h5"
    if domain == "L":
        image, label = read_labeled_case(path, access, case)
    else:
        image, label = read_image_only(path, access, "unlabeled", case), None
    crops = []
    for obs in observations:
        x, y, meta = deterministic_crop(image, label, design["patch"], design["seed"], case, obs["repeat"])
        meta = json.loads(json.dumps(meta))  # A JSON stores padding tuples as lists.
        assert meta == obs["crop"], "A/B crop mismatch"
        crops.append((obs, x, y, meta))
    return image, label, crops


def crop_native(array, meta):
    padding = meta["padding"]
    padded = np.pad(array, padding)
    return padded[tuple(slice(s, s+p) for s, p in zip(meta["starts"], meta["patch"]))]


def extract_terms(terms, mapping):
    """No .backward/.zero_grad/optimizer; objective-invalid differs from unused."""
    parameters = tuple(spec.parameter for spec in mapping.values())
    prior = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    result = {stage: [] for stage in mapping}
    for scale, term in zip(SCALES, terms):
        gradients = (torch.autograd.grad(term.loss, parameters, retain_graph=True, allow_unused=True)
                     if term.valid else [None] * len(parameters))
        for (stage, spec), gradient in zip(mapping.items(), gradients):
            if term.valid and (gradient is not None) != (scale in spec.reachable):
                raise AssertionError(f"Unexpected reachability: {stage}/{scale}")
            value = spec.channel_vectors(gradient)
            result[stage].append(None if value is None else value.detach().cpu().numpy().copy())
    for parameter, original in zip(parameters, prior):
        assert (original is None) == (parameter.grad is None)
        if original is not None:
            assert torch.equal(original, parameter.grad)
    return result
