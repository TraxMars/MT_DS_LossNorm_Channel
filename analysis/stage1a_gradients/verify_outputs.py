"""Independently validate saved diagnostic masks, shapes, losses and case cosines."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from core import SCALES, sha256


def verify(output):
    started = time.monotonic()
    design = json.loads((output / "design.json").read_text())
    mapping = json.loads((output / "parameter_mapping.json").read_text())
    observations = json.loads((output / "observations.json").read_text())
    extraction = json.loads((output / "extraction_results.json").read_text())
    assert extraction["status"] == "PASS" and all(extraction["invariants"].values())
    repeats = design["repeats"]
    channels = sum(spec["channels"] for spec in mapping.values())
    sample_cosines = {}
    for stage, spec in mapping.items():
        arrays = {domain: np.load(output / "raw" / f"{domain}_{stage}.npy", mmap_mode="r") for domain in ("L", "U")}
        for domain, array in arrays.items():
            assert array.shape == (len(observations[domain]), 4, spec["channels"], spec["values_per_channel"])
            for si, scale in enumerate(SCALES):
                if scale not in spec["reachable_scales"]:
                    assert np.isnan(array[:,si]).all(), (stage, scale)
                else:
                    assert np.isfinite(array[:,si]).all(), (stage, scale)
        if stage in ("E1", "D4", "D1"):
            si = 3
            for channel in (0, spec["channels"] - 1):
                for li in (0, len(design["labeled_ids"]) - 1):
                    for ui in (0, len(design["unlabeled_ids"]) - 1):
                        left = arrays["L"][li*repeats:(li+1)*repeats,si,channel].astype(np.float64).mean(axis=0)
                        right = arrays["U"][ui*repeats:(ui+1)*repeats,si,channel].astype(np.float64).mean(axis=0)
                        denominator = np.linalg.norm(left) * np.linalg.norm(right)
                        sample_cosines[(stage, "D1", str(channel), design["labeled_ids"][li], design["unlabeled_ids"][ui])] = float(left @ right / denominator)
    matched, pair_rows = 0, 0
    with (output / "lu_case_pairs.csv").open() as handle:
        for row in csv.DictReader(handle):
            pair_rows += 1
            key = tuple(row[k] for k in ("stage", "scale", "channel", "labeled_case", "unlabeled_case"))
            if key in sample_cosines:
                np.testing.assert_allclose(float(row["raw_cosine"]), sample_cosines[key], atol=1e-12, rtol=1e-12)
                matched += 1
            if row["valid"] == "True":
                assert row["L_status"] == row["U_status"] == "valid"
                assert -1 <= float(row["raw_cosine"]) <= 1
            else:
                assert row["raw_cosine"] == ""
    assert matched == len(sample_cosines) == 24
    statuses, norm_rows, invalid_observations = Counter(), 0, set()
    with (output / "per_observation_channels.csv").open() as handle:
        for row in csv.DictReader(handle):
            norm_rows += 1
            statuses[row["status"]] += 1
            if row["status"] == "unreachable":
                assert row["raw_norm"] == row["lossnorm_scale_contribution_norm"] == ""
            elif row["status"] == "valid":
                assert float(row["raw_norm"]) > float(row["validity_floor"])
            elif row["status"] == "below_threshold":
                assert 0 < float(row["raw_norm"]) <= float(row["validity_floor"])
                invalid_observations.add(tuple(row[k] for k in ("domain","case","repeat","stage","scale","channel")))
    assert norm_rows == sum(map(len, observations.values())) * 4 * channels
    suppressed = 0
    with (output / "per_observation_cross_scale.csv").open() as handle:
        for row in csv.DictReader(handle):
            bad = any(tuple(row[k] for k in ("domain","case","repeat","stage")) + (row[scale],row["channel"])
                      in invalid_observations for scale in ("scale_a","scale_b"))
            if bad:
                assert row["cosine"] == "" and row["status"] != "valid"
                suppressed += 1
    assert not invalid_observations or suppressed > 0
    maximum_loss_error = 0
    for domain, values in observations.items():
        for row in values:
            maximum_loss_error = max(maximum_loss_error, abs(np.mean(row["losses"]) - row["aggregate_loss"]))
            np.testing.assert_allclose(np.mean(row["losses"]), row["aggregate_loss"], rtol=1e-6, atol=1e-7)
            expected_factor = (1 if domain == "L" else design["consistency_weight"]) / (
                4 * max(row["aggregate_loss"], design["normalization_eps"]))
            assert row["normalized_scale_factor"] == expected_factor
    audit = json.loads((output / "hdf5_access_log.json").read_text())
    assert len(audit) == len(design["labeled_ids"]) * 2 + len(design["unlabeled_ids"])
    assert all(row["key"] == "image" for row in audit if row["role"] == "unlabeled")
    protection = json.loads((output / "protection_before.json").read_text())
    for filename, record in protection["files"].items():
        assert Path(filename).suffix.lower() not in {".h5", ".hdf5"}
        assert sha256(filename) == record["sha256"], filename
    pca = json.loads((output / "pca_clusters.json").read_text())
    for stage, result in pca.items():
        assert result.get("rank", 0) <= result["algebraic_rank_upper_bound"]
    assert pca["D1"]["status"] == "CONSTANT_PROFILE"
    result = dict(status="PASS", saved_norm_rows=norm_rows, norm_statuses=dict(statuses),
                  case_pair_rows=pair_rows, independently_recomputed_signed_cosines=matched,
                  correctly_suppressed_tiny_observation_cosines=suppressed,
                  maximum_loss_formula_absolute_error=maximum_loss_error,
                  protected_file_hashes_rechecked=len(protection["files"]),
                  elapsed_seconds=time.monotonic()-started)
    (output / "output_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run.resolve()), indent=2))
