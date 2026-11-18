import gzip
import hashlib
import json

import numpy as np
import pandas as pd

from shared.criteo.study import load_fresh


def _fixture(root):
    frame = pd.DataFrame({f"f{i}": np.arange(120, dtype=float) for i in range(12)})
    frame["treatment"] = [0] * 24 + [1] * 96
    frame["visit"] = (np.arange(120) % 3 == 0).astype("int8")
    frame["conversion"] = (np.arange(120) % 11 == 0).astype("int8")
    frame["exposure"] = (np.arange(120) % 2).astype("int8")
    path = root / "criteo-research-uplift-v2.1.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "criteo_manifest.json").write_text(json.dumps({"revision": "fixture", "files": [{"path": path.name, "sha256": digest}]}))


def test_fresh_sample_is_reproducible_ratio_preserving_and_disjoint(tmp_path):
    _fixture(tmp_path)
    first, meta = load_fresh(tmp_path, n=40, seed=19, exclude_n=10, exclude_seed=7, chunksize=13)
    second, meta2 = load_fresh(tmp_path, n=40, seed=19, exclude_n=10, exclude_seed=7, chunksize=13)
    assert first.equals(second)
    assert meta["excluded_overlap"] == 0
    assert meta["sample_counts"] == [8, 32]
    assert first.treatment.mean() == .8
    assert set(first.columns) == {*(f"f{i}" for i in range(12)), "treatment", "visit", "conversion", "exposure", "row_id"}
    old, _, _ = __import__("shared.causal_data", fromlist=["criteo"]).criteo(tmp_path, n=10, seed=7, chunksize=13)
    assert not set(first.row_id).intersection(old.row_id)


def test_sampling_does_not_change_when_outcomes_are_changed(tmp_path):
    _fixture(tmp_path)
    first, _ = load_fresh(tmp_path, n=40, seed=19, exclude_n=10, exclude_seed=7, chunksize=17)
    path = tmp_path / "criteo-research-uplift-v2.1.csv.gz"
    with gzip.open(path, "rt") as stream:
        frame = pd.read_csv(stream)
    frame[["visit", "conversion", "exposure"]] = 1 - frame[["visit", "conversion", "exposure"]]
    frame.to_csv(path, index=False, compression="gzip")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "criteo_manifest.json").write_text(json.dumps({"revision": "fixture", "files": [{"path": path.name, "sha256": digest}]}))
    second, _ = load_fresh(tmp_path, n=40, seed=19, exclude_n=10, exclude_seed=7, chunksize=17)
    assert first.row_id.tolist() == second.row_id.tolist()


def test_npz_cache_is_validated_and_reused(tmp_path):
    _fixture(tmp_path)
    cache = tmp_path / "cache"
    first, first_meta = load_fresh(tmp_path, n=40, seed=19, exclude_n=10, exclude_seed=7, cache_dir=cache)
    second, second_meta = load_fresh(tmp_path, n=40, seed=19, exclude_n=10, exclude_seed=7, cache_dir=cache)
    assert first.equals(second)
    assert first_meta["cache_hit"] is False
    assert second_meta["cache_hit"] is True
    assert len(first_meta["exclusion_sampler_sha256"]) == 64
    assert "sampler" in first_meta["cache_key"]


def test_zero_exclusion_and_manifest_corruption_are_explicit(tmp_path):
    _fixture(tmp_path)
    sampled, metadata = load_fresh(tmp_path, n=10, seed=3, exclude_n=0, chunksize=19)
    assert len(sampled) == 10
    assert metadata["excluded_overlap"] == 0
    path = tmp_path / "criteo-research-uplift-v2.1.csv.gz"
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    try:
        load_fresh(tmp_path, n=10, seed=3, exclude_n=0, chunksize=19)
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("manifest corruption was not rejected")
