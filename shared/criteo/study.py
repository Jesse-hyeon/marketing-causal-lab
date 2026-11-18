"""Reproducible, leakage-safe Criteo sampling for the second study.

The public file is scanned end to end.  Sampling priorities are generated without
looking at outcomes, and the prior 120k-row teaching sample can be excluded by
its original row IDs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared.causal_data import criteo as _old_criteo


FEATURES = [f"f{i}" for i in range(12)]
REQUIRED = FEATURES + ["treatment", "visit", "conversion", "exposure"]


def _source_sha(root: Path, filename: str) -> tuple[str, str]:
    digest = hashlib.sha256()
    with (root / filename).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    actual_sha = digest.hexdigest()
    manifest = root / "criteo_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for item in payload.get("files", []):
            if item.get("path") == filename:
                expected_sha = item["sha256"]
                if expected_sha != actual_sha:
                    raise ValueError("Criteo source SHA-256 differs from its manifest")
                return actual_sha, str(payload.get("revision", "unknown"))
    return actual_sha, "unknown"


def _ids_hash(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def _sampler_sha() -> str:
    """Bind exclusion IDs to the exact legacy sampler implementation."""
    return hashlib.sha256(inspect.getsource(_old_criteo).encode()).hexdigest()


def _cache_key(source_sha, sampler_sha, n, seed, exclude_n, exclude_seed):
    return f"criteo-fresh-v2-{source_sha[:16]}-sampler{sampler_sha[:16]}-n{n}-s{seed}-ex{exclude_n}-{exclude_seed}"


def _load_cache(cache_path: Path, expected_key: str, expected_n: int, expected_sampler_sha: str):
    meta_path = cache_path.with_suffix(".json")
    if not cache_path.exists() or not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text())
    if (metadata.get("cache_key") != expected_key or metadata.get("n") != expected_n
            or metadata.get("exclusion_sampler_sha256") != expected_sampler_sha):
        return None
    with np.load(cache_path, allow_pickle=False) as values:
        arrays = {name: values[name] for name in values.files}
    missing = set(REQUIRED + ["row_id"]) - set(arrays)
    if missing or len(arrays["row_id"]) != expected_n:
        raise ValueError(f"Criteo cache is missing keys or has wrong length: {sorted(missing)}")
    lengths = {len(array) for array in arrays.values()}
    if lengths != {expected_n} or any(array.dtype == object for array in arrays.values()):
        raise ValueError("Criteo cache arrays have inconsistent lengths or object dtype")
    if len(np.unique(arrays["row_id"])) != expected_n:
        raise ValueError("Criteo cache contains duplicate row_id values")
    if not np.isin(arrays["treatment"], [0, 1]).all() or not np.isin(arrays["visit"], [0, 1]).all() or not np.isin(arrays["conversion"], [0, 1]).all() or not np.isin(arrays["exposure"], [0, 1]).all():
        raise ValueError("Criteo cache contains non-binary labels")
    if metadata.get("sample_row_id_sha256") != _ids_hash(arrays["row_id"]):
        raise ValueError("Criteo cache row ID hash does not match metadata")
    if metadata.get("sample_counts") != [int((arrays["treatment"] == 0).sum()), int((arrays["treatment"] == 1).sum())]:
        raise ValueError("Criteo cache treatment counts do not match metadata")
    return arrays, metadata


def _save_cache(cache_path: Path, arrays: dict[str, np.ndarray], metadata: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    cache_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_fresh(
    root,
    n: int = 1_000_000,
    seed: int = 20261111,
    exclude_n: int = 120_000,
    exclude_seed: int = 42,
    cache_dir=None,
    chunksize: int = 400_000,
):
    """Return a new treatment-ratio-preserving sample disjoint from the old one.

    ``exposure`` is retained for data-audit diagnostics only; it is never used
    for sampling, splitting, feature construction, or model selection.
    """
    if int(n) != n or n <= 0 or int(exclude_n) != exclude_n or exclude_n < 0:
        raise ValueError("n must be positive and exclude_n must be non-negative integers")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    root = Path(root)
    filename = "criteo-research-uplift-v2.1.csv.gz"
    source_sha, revision = _source_sha(root, filename)
    sampler_sha = _sampler_sha()
    cache_key = _cache_key(source_sha, sampler_sha, n, seed, exclude_n, exclude_seed)
    cache_path = Path(cache_dir) / f"{cache_key}.npz" if cache_dir else None

    if cache_path:
        cached = _load_cache(cache_path, cache_key, int(n), sampler_sha)
        if cached is not None:
            arrays, metadata = cached
            # The cache key binds the source revision and exclusion parameters;
            # the producer records the exclusion hash and zero overlap. A cache
            # hit therefore avoids a second full-file scan.
            if metadata.get("excluded_overlap") != 0:
                raise ValueError("Criteo cache records an excluded-row overlap")
            return _frame(arrays), {**metadata, "cache_hit": True, "excluded_overlap": 0}

    if exclude_n:
        old, _, _ = _old_criteo(root, n=exclude_n, seed=exclude_seed, chunksize=chunksize)
        old_ids = np.asarray(old["row_id"], dtype=np.int64)
    else:
        old_ids = np.empty(0, dtype=np.int64)
    old_set = set(old_ids.tolist())
    old_hash = _ids_hash(old_ids)

    rng = np.random.default_rng(seed)
    pools = {0: None, 1: None}
    population = np.zeros(2, dtype=np.int64)
    eligible = np.zeros(2, dtype=np.int64)
    offset = 0
    dtypes = {feature: "float64" for feature in FEATURES}
    dtypes.update({name: "int8" for name in ["treatment", "visit", "conversion", "exposure"]})
    usecols = REQUIRED
    with pd.read_csv(root / filename, usecols=usecols, dtype=dtypes, chunksize=chunksize) as chunks:
        for chunk in chunks:
            row_ids = np.arange(offset, offset + len(chunk), dtype=np.int64)
            offset += len(chunk)
            chunk["row_id"] = row_ids
            priorities = rng.random(len(chunk))
            chunk["_priority"] = priorities
            for arm in (0, 1):
                arm_mask = chunk.treatment.eq(arm).to_numpy()
                population[arm] += int(arm_mask.sum())
                keep_mask = arm_mask & ~np.isin(row_ids, list(old_set))
                eligible[arm] += int(keep_mask.sum())
                part = chunk.loc[keep_mask].copy()
                if pools[arm] is None:
                    pools[arm] = part
                else:
                    pools[arm] = pd.concat([pools[arm], part], ignore_index=True)
                pools[arm] = pools[arm].nsmallest(n, "_priority")

    total_eligible = int(eligible.sum())
    if n > total_eligible:
        raise ValueError(f"requested n={n} exceeds eligible rows={total_eligible}")
    target_treated = int(round(n * eligible[1] / total_eligible))
    target_control = n - target_treated
    if target_control > len(pools[0]) or target_treated > len(pools[1]):
        raise ValueError("sampling target exceeds one treatment arm")
    selected = pd.concat([
        pools[0].nsmallest(target_control, "_priority"),
        pools[1].nsmallest(target_treated, "_priority"),
    ], ignore_index=True)
    selected = selected.sample(frac=1, random_state=seed).drop(columns="_priority").reset_index(drop=True)
    arrays = {column: selected[column].to_numpy() for column in REQUIRED + ["row_id"]}
    metadata = {
        "cache_key": cache_key,
        "dataset": "criteo/criteo-uplift",
        "source_file": filename,
        "source_sha256": source_sha,
        "revision": revision,
        "exclusion_sampler_sha256": sampler_sha,
        "seed": int(seed),
        "n": int(n),
        "exclude_n": int(exclude_n),
        "exclude_seed": int(exclude_seed),
        "excluded_row_id_sha256": old_hash,
        "sample_row_id_sha256": _ids_hash(arrays["row_id"]),
        "population_counts": population.tolist(),
        "eligible_counts": eligible.tolist(),
        "sample_counts": [target_control, target_treated],
        "excluded_overlap": 0,
        "cache_hit": False,
        "exposure_used_for_sampling": False,
        "outcomes_used_for_sampling": False,
    }
    if cache_path:
        _save_cache(cache_path, arrays, metadata)
    return _frame(arrays), metadata


def _frame(arrays):
    frame = pd.DataFrame({name: arrays[name] for name in REQUIRED + ["row_id"]})
    frame.attrs["feature_columns"] = FEATURES
    frame.attrs["audit_only_columns"] = ["exposure"]
    return frame
