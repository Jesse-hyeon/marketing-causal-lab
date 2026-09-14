import __future__
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


def _load_search_analysis():
    """Execute definitions only; never run Notebook fit or display cells."""
    notebook_path = Path(__file__).resolve().parents[1] / "02. prototype.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    core_ids = [
        "core-setup",
        "core-data",
        "core-transform",
        "core-priors",
        "core-alignment",
        "core-signatures",
        "core-discovery",
        "core-suite",
    ]
    search_ids = ["search-protocol", "search-engine", "search-evaluate"]
    cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and (
            "causal-definitions" in cell.get("metadata", {}).get("tags", [])
            or "causal-experiment-definitions"
            in cell.get("metadata", {}).get("tags", [])
        )
    ]
    if [cell["id"] for cell in cells] != [*core_ids, *search_ids]:
        raise AssertionError("Notebook search definition cells are missing or out of order")

    module = types.ModuleType("notebook_stability_search")
    module.__file__ = str(notebook_path)
    sys.modules[module.__name__] = module
    for cell in cells:
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        code = compile(
            source,
            f"{notebook_path}#{cell['id']}",
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )
        exec(code, module.__dict__)
    return module


SEARCH = _load_search_analysis()


def _graph_record(
    config_id,
    *,
    phase="dev",
    window="full96",
    links=(),
    target="Target",
    variables=("Target", "X"),
    status="complete",
):
    max_lag = max([lag for _, lag, _, _ in links], default=1)
    graph = np.full((len(variables), len(variables), max_lag + 1), "", dtype="<U3")
    values = np.full(graph.shape, np.nan)
    for source, lag, mark, value in links:
        source_index = list(variables).index(source)
        target_index = list(variables).index(target)
        graph[source_index, target_index, lag] = mark
        values[source_index, target_index, lag] = value
    record_key = f"{phase}|{window}|{config_id}|standard|"
    record = {
        "record_key": record_key,
        "config_id": config_id,
        "phase": phase,
        "window": window,
        "variant": "standard",
        "target": target,
        "variables": list(variables),
        "status": status,
        "errors": [],
    }
    if status == "complete":
        record.update(
            graph=graph.tolist(),
            val_matrix=SEARCH.search_json_safe(values),
            p_matrix=SEARCH.search_json_safe(np.full(graph.shape, np.nan)),
        )
    else:
        record["errors"] = [{"type": "RuntimeError", "message": "fixture failure"}]
    return record


def _selected_document(*, shock_2020=True, null_detections=1, failed_null=False):
    config = {
        "id": "base__parcorr__t1__a0p01",
        "input_set": "base",
        "inputs": ["X"],
        "ci_test": "ParCorr",
        "tau_max": 1,
        "pc_alpha": 0.01,
    }
    edge = [("X", 1, "-->", 0.4)]
    attempts = [
        _graph_record(config["id"], window=window, links=edge)
        for window in SEARCH.SEARCH_PROTOCOL["dev_windows"]
    ]
    for window in SEARCH.SEARCH_PROTOCOL["validation_windows"]:
        attempts.append(
            _graph_record(config["id"], phase="validation", window=window, links=edge)
        )
    attempts.extend(
        [
            _graph_record(
                config["id"] + "__cash",
                phase="validation",
                window="cash133",
                links=edge,
            ),
            _graph_record(
                config["id"] + "__lag3",
                phase="validation",
                window="lag3_133",
                links=edge,
            ),
        ]
    )
    blocks = [(start, start + 12) for start in range(0, 120, 12)] + [(120, 133)]
    for start, stop in blocks:
        keep = shock_2020 or (start, stop) != (60, 72)
        attempts.append(
            _graph_record(
                config["id"],
                phase="jackknife",
                window=f"raw_{start}_{stop}",
                links=edge if keep else (),
            )
        )
    for index in range(19):
        status = "failed" if failed_null and index == 18 else "complete"
        attempts.append(
            _graph_record(
                config["id"],
                phase="null",
                window=f"roll_{12 + index}",
                links=edge if index < null_detections else (),
                status=status,
            )
        )
    return {
        "protocol": copy.deepcopy(SEARCH.SEARCH_PROTOCOL),
        "configs": [config],
        "attempts": attempts,
        "frozen_selection": [config["id"]],
        "null_offsets": list(range(12, 31)),
    }


class SearchProtocolTests(unittest.TestCase):
    def test_grid_and_development_windows_are_frozen_before_future_validation(self):
        configs = SEARCH.search_configs()

        self.assertEqual(len(configs), 72)
        self.assertEqual(len({row["id"] for row in configs}), 72)
        self.assertEqual(
            SEARCH.SEARCH_PROTOCOL["dev_windows"],
            {
                "full96": [0, 96],
                "first72": [0, 72],
                "last72": [24, 96],
                "first84": [0, 84],
                "last84": [12, 96],
            },
        )
        self.assertTrue(
            all(stop <= 96 for _, stop in SEARCH.SEARCH_PROTOCOL["dev_windows"].values())
        )
        self.assertEqual(SEARCH.SEARCH_PROTOCOL["validation_windows"]["late37"], [96, 133])

    def test_raw_block_exclusion_accounts_for_log_difference_and_six_lag_history(self):
        baseline = set(SEARCH.search_reference_points(132, burn=6))
        retained = set(
            SEARCH.search_reference_points(132, burn=6, excluded_raw_blocks=[(60, 72)])
        )

        # Frame reference r is raw month r+1. Its raw dependency is [r-6, r+1].
        self.assertEqual(baseline - retained, set(range(59, 78)))
        self.assertIn(58, retained)
        self.assertIn(78, retained)

    def test_circular_shift_excludes_cross_seam_histories_and_rolls_predictors_jointly(self):
        dates = pd.date_range("2020-01-01", periods=30, freq="MS")
        raw = pd.DataFrame(
            {
                "Target": np.exp(np.linspace(1.0, 2.0, len(dates))),
                "X": np.exp(np.linspace(0.2, 1.7, len(dates))),
                "Y": np.exp(2 * np.linspace(0.2, 1.7, len(dates))),
            },
            index=dates,
        )
        config = {
            "id": "joint",
            "input_set": "fixture",
            "inputs": ["X", "Y"],
            "ci_test": "ParCorr",
            "tau_max": 1,
            "pc_alpha": 0.05,
        }
        captured = {}

        class FakeDataFrame:
            def __init__(self, data, **kwargs):
                captured["data"] = np.asarray(data)
                captured["refs"] = np.asarray(kwargs["reference_points"])

        class FakeLPCMCI:
            def __init__(self, dataframe, cond_ind_test, verbosity):
                self.dataframe = dataframe
                self.cond_ind_test = cond_ind_test

            def run_lpcmci(self, **kwargs):
                shape = (3, 3, kwargs["tau_max"] + 1)
                return {
                    "graph": np.full(shape, "", dtype="<U3"),
                    "val_matrix": np.full(shape, np.nan),
                    "p_matrix": np.full(shape, np.nan),
                }

        with mock.patch("tigramite.data_processing.DataFrame", FakeDataFrame), mock.patch(
            "tigramite.lpcmci.LPCMCI", FakeLPCMCI
        ), mock.patch.object(SEARCH, "search_ci_test", return_value=object()), mock.patch.object(
            SEARCH, "search_instrument_ci", return_value=object()
        ):
            result = SEARCH.search_fit(raw, config, shift=12, target="Target")

        self.assertEqual(result["status"], "complete", result.get("errors"))
        np.testing.assert_allclose(captured["data"][:, 2], 2 * captured["data"][:, 1])
        self.assertEqual(
            set(range(6, 29)) - set(captured["refs"]),
            set(range(12, 18)),
        )

    def test_robust_parcorr_tiny_fit_records_safe_lags_and_positive_degrees_of_freedom(self):
        dates = pd.date_range("2018-01-01", periods=48, freq="MS")
        rng = np.random.default_rng(7)
        raw = pd.DataFrame(
            {
                "Target": np.exp(np.cumsum(rng.normal(0.01, 0.04, len(dates))) + 4),
                "X": np.exp(np.cumsum(rng.normal(0.00, 0.05, len(dates))) + 3),
            },
            index=dates,
        )
        config = {
            "id": "robust-real-fit",
            "input_set": "fixture",
            "inputs": ["X"],
            "ci_test": "RobustParCorr",
            "tau_max": 1,
            "pc_alpha": 0.05,
            "seed": 11,
        }

        result = SEARCH.search_fit(raw, config, target="Target")

        self.assertEqual(result["status"], "complete", result.get("errors"))
        self.assertLessEqual(result["max_requested_lag"], 6)
        self.assertGreater(result["min_df"], 0)
        self.assertGreater(result["effective_n"], 0)

    def test_ci_guard_checks_actual_xyz_node_lags_not_only_tau_max_argument(self):
        diagnostics = {"run_test_events": [], "dependence_events": []}
        ci = SEARCH.search_instrument_ci(
            SEARCH.search_ci_test("ParCorr", seed=3), diagnostics
        )

        with self.assertRaisesRegex(RuntimeError, "six-month guard"):
            ci.run_test(
                X=[(0, 0)],
                Y=[(1, 0)],
                Z=[(0, -7)],
                tau_max=2,
            )


class SearchIdentityAndRankingTests(unittest.TestCase):
    def test_definition_hash_binds_both_definition_families_but_not_outputs(self):
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "id": "core",
                    "metadata": {"tags": ["causal-definitions"]},
                    "source": ["CORE = 1\n"],
                    "outputs": [{"text": ["old"]}],
                },
                {
                    "cell_type": "code",
                    "id": "search",
                    "metadata": {"tags": ["causal-experiment-definitions"]},
                    "source": ["SEARCH = 1\n"],
                },
                {"cell_type": "markdown", "id": "note", "metadata": {}, "source": ["old"]},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            baseline = SEARCH.search_definition_sha256(path)

            notebook["cells"][0]["outputs"] = [{"text": ["new"]}]
            notebook["cells"][2]["source"] = ["new prose"]
            path.write_text(json.dumps(notebook), encoding="utf-8")
            self.assertEqual(baseline, SEARCH.search_definition_sha256(path))

            notebook["cells"][0]["source"] = ["CORE = 2\n"]
            path.write_text(json.dumps(notebook), encoding="utf-8")
            self.assertNotEqual(baseline, SEARCH.search_definition_sha256(path))

    def test_cache_identity_changes_with_input_protocol_config_and_source(self):
        raw = pd.DataFrame({"x": [1.0]})
        raw.attrs["input_sha256"] = "input-a"
        configs = SEARCH.search_configs()[:1]
        with tempfile.TemporaryDirectory() as directory:
            notebook = Path(directory) / "fixture.ipynb"
            notebook.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "code",
                                "id": "core",
                                "metadata": {"tags": ["causal-definitions"]},
                                "source": ["A = 1\n"],
                            },
                            {
                                "cell_type": "code",
                                "id": "search",
                                "metadata": {"tags": ["causal-experiment-definitions"]},
                                "source": ["B = 1\n"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            baseline = SEARCH.search_manifest(raw, SEARCH.SEARCH_PROTOCOL, notebook, configs)
            hashes = {SEARCH.search_manifest_hash(baseline)}

            changed_input = raw.copy()
            changed_input.attrs["input_sha256"] = "input-b"
            hashes.add(
                SEARCH.search_manifest_hash(
                    SEARCH.search_manifest(
                        changed_input, SEARCH.SEARCH_PROTOCOL, notebook, configs
                    )
                )
            )
            protocol = SEARCH.search_protocol_copy({"seed": 9})
            hashes.add(
                SEARCH.search_manifest_hash(
                    SEARCH.search_manifest(raw, protocol, notebook, configs)
                )
            )
            changed_config = [dict(configs[0], pc_alpha=0.02, inputs=["different"])]
            hashes.add(
                SEARCH.search_manifest_hash(
                    SEARCH.search_manifest(
                        raw, SEARCH.SEARCH_PROTOCOL, notebook, changed_config
                    )
                )
            )
            contents = json.loads(notebook.read_text(encoding="utf-8"))
            contents["cells"][1]["source"] = ["B = 2\n"]
            notebook.write_text(json.dumps(contents), encoding="utf-8")
            hashes.add(
                SEARCH.search_manifest_hash(
                    SEARCH.search_manifest(raw, SEARCH.SEARCH_PROTOCOL, notebook, configs)
                )
            )

        self.assertEqual(len(hashes), 5)

    def test_ranking_uses_rates_over_all_variable_lag_opportunities(self):
        small = {
            "id": "small",
            "input_set": "small-set",
            "inputs": ["X"],
            "ci_test": "ParCorr",
            "tau_max": 1,
            "pc_alpha": 0.01,
        }
        large = {
            "id": "large",
            "input_set": "large-set",
            "inputs": ["A", "B"],
            "ci_test": "ParCorr",
            "tau_max": 2,
            "pc_alpha": 0.01,
        }
        records = []
        for window in SEARCH.SEARCH_PROTOCOL["dev_windows"]:
            records.append(
                _graph_record(small["id"], window=window, links=[("X", 1, "-->", 0.2)])
            )
            records.append(
                _graph_record(
                    large["id"],
                    window=window,
                    variables=("Target", "A", "B"),
                    links=[
                        ("A", 0, "-->", 0.2),
                        ("A", 1, "-->", 0.2),
                        ("B", 2, "-->", 0.2),
                    ],
                )
            )

        ranked = SEARCH.search_rank(records, [small, large]).set_index("config_id")

        self.assertEqual(ranked.loc["small", "eligible_opportunities"], 2)
        self.assertEqual(ranked.loc["large", "eligible_opportunities"], 6)
        self.assertAlmostEqual(ranked.loc["small", "stable_adj_rate"], 0.5)
        self.assertAlmostEqual(ranked.loc["large", "stable_adj_rate"], 0.5)


class SearchEvaluationTests(unittest.TestCase):
    def test_selection_is_persisted_before_validation_and_never_retuned_on_late_failure(self):
        low = {
            "id": "base__parcorr__t1__a0p01",
            "input_set": "base",
            "inputs": ["X"],
            "ci_test": "ParCorr",
            "tau_max": 1,
            "pc_alpha": 0.01,
        }
        high = dict(low, id="base__parcorr__t1__a0p05", pc_alpha=0.05)
        dates = pd.date_range("2015-01-01", periods=133, freq="MS")
        raw = pd.DataFrame(
            {
                SEARCH.SEARCH_PROTOCOL["target"]: np.exp(np.linspace(1, 2, 133)),
                SEARCH.SEARCH_PROTOCOL["cash_target"]: np.exp(np.linspace(1, 2, 133)),
                "X": np.exp(np.linspace(2, 3, 133)),
            },
            index=dates,
        )
        raw.attrs["input_sha256"] = "orchestration-fixture"

        with tempfile.TemporaryDirectory() as cache_root:
            call_count = 0
            saw_frozen_before_validation = False

            def fake_fit(raw_slice, config, target=SEARCH.SEARCH_PROTOCOL["target"], **kwargs):
                nonlocal call_count, saw_frozen_before_validation
                call_count += 1
                is_dev = call_count <= 10
                if not is_dev and not saw_frozen_before_validation:
                    result_files = list(Path(cache_root).glob("*/results.json"))
                    self.assertEqual(len(result_files), 1)
                    checkpoint = json.loads(result_files[0].read_text(encoding="utf-8"))
                    self.assertEqual(checkpoint["frozen_selection"], [low["id"]])
                    self.assertEqual(checkpoint["status"], "selection_frozen")
                    self.assertTrue(checkpoint["validation_plan"][low["id"]])
                    saw_frozen_before_validation = True

                if is_dev:
                    links = [("X", 1, "-->", 0.4)] if config["id"] == low["id"] else []
                elif kwargs.get("shift") is not None or len(raw_slice) == 37:
                    links = []
                else:
                    links = [("X", 1, "-->", 0.4)]
                return _graph_record(
                    config["id"],
                    target=target,
                    variables=(target, "X"),
                    links=links,
                )

            with mock.patch.object(SEARCH, "load_data", return_value=raw), mock.patch.object(
                SEARCH, "search_fit", side_effect=fake_fit
            ):
                document = SEARCH.search_run(
                    "ignored.csv",
                    cache_root,
                    progress=False,
                    notebook_path=SEARCH.__file__,
                    configs=[low, high],
                )

        self.assertTrue(saw_frozen_before_validation)
        self.assertEqual(document["frozen_selection"], [low["id"]])
        self.assertEqual(len(document["attempts"]), 48)
        self.assertEqual(document["status"], "complete")
        selected = document["evaluation"]["selected_links"][0]
        self.assertEqual(selected["late_status"], "temporally-unconfirmed")
        self.assertFalse(selected["strong_directed_tier"])

    def test_failures_stay_in_denominators_and_cannot_promote_an_incomplete_config(self):
        document = _selected_document(failed_null=True)
        config_id = document["frozen_selection"][0]
        null = SEARCH.search_presence(
            document["attempts"],
            config_id,
            "null",
            {f"roll_{12 + i}" for i in range(19)},
            ("X", 1),
        )

        self.assertEqual(null["denominator"], 19)
        self.assertEqual(null["complete"], 18)
        self.assertEqual(null["failed"], 1)
        evaluation = SEARCH.search_evaluate(document)
        selected = evaluation["selected_links"][0]
        self.assertFalse(selected["stable_association_operational"])
        self.assertEqual(len(evaluation["failures"]), 1)

        missing = _selected_document()
        missing["attempts"] = [
            row
            for row in missing["attempts"]
            if not (row["phase"] == "null" and row["window"] == "roll_30")
        ]
        missing_selected = SEARCH.search_evaluate(missing)["selected_links"][0]
        self.assertEqual(missing_selected["null"]["denominator"], 19)
        self.assertEqual(missing_selected["null"]["complete"], 18)
        self.assertEqual(missing_selected["null"]["failed"], 1)
        self.assertFalse(missing_selected["stable_association_operational"])

    def test_shock_deletions_are_mandatory_and_not_averaged_into_eighty_percent(self):
        selected = SEARCH.search_evaluate(
            _selected_document(shock_2020=False)
        )["selected_links"][0]

        self.assertEqual(selected["jackknife"]["denominator"], 11)
        self.assertGreaterEqual(selected["jackknife"]["adjacency_frequency"], 0.8)
        self.assertTrue(selected["pre_shock_association"])
        self.assertEqual(selected["critical_shocks"]["status"], "shock-sensitive")
        self.assertFalse(selected["stable_association_operational"])
        self.assertFalse(selected["strong_directed_tier"])

    def test_null_threshold_is_the_frozen_integer_one_of_nineteen(self):
        one = SEARCH.search_evaluate(
            _selected_document(null_detections=1)
        )["selected_links"][0]
        two = SEARCH.search_evaluate(
            _selected_document(null_detections=2)
        )["selected_links"][0]

        self.assertTrue(one["stable_association_operational"])
        self.assertFalse(two["stable_association_operational"])

    def test_adjacency_endpoint_direction_and_sign_are_separate_claims(self):
        document = _selected_document()
        for record in document["attempts"]:
            if record.get("status") != "complete":
                continue
            graph = np.asarray(record["graph"], dtype="<U3")
            nonempty = np.argwhere(graph != "")
            for source, target, lag in nonempty:
                graph[source, target, lag] = "<->"
            record["graph"] = graph.tolist()

        selected = SEARCH.search_evaluate(document)["selected_links"][0]

        self.assertTrue(selected["stable_association_operational"])
        self.assertFalse(selected["stable_direction"])
        self.assertEqual(selected["dev"]["modal_mark"], "<->")
        self.assertEqual(selected["dev"]["modal_sign"], 1)

        unknown = _graph_record(
            "unknown", links=[("X", 1, "x-x", 0.3)]
        )
        unknown_link = SEARCH.search_target_links(unknown)[("X", 1)]
        self.assertFalse(unknown_link["supported"])

    def test_sign_must_match_across_development_full_and_extended_checks(self):
        document = _selected_document()
        for record in document["attempts"]:
            if record["phase"] != "validation" or record["window"] != "first108":
                continue
            values = np.asarray(record["val_matrix"], dtype=object)
            values[1, 0, 1] = -0.4
            record["val_matrix"] = SEARCH.search_json_safe(values)

        selected = SEARCH.search_evaluate(document)["selected_links"][0]

        self.assertFalse(selected["stable_association_operational"])

    def test_any_ancestry_conflicting_endpoint_blocks_direction_promotion(self):
        document = _selected_document()
        record = next(
            row
            for row in document["attempts"]
            if row["phase"] == "jackknife" and row["window"] == "raw_0_12"
        )
        graph = np.asarray(record["graph"], dtype="<U3")
        graph[1, 0, 1] = "<->"
        record["graph"] = graph.tolist()

        selected = SEARCH.search_evaluate(document)["selected_links"][0]

        self.assertTrue(selected["stable_association_operational"])
        self.assertFalse(selected["stable_direction"])


if __name__ == "__main__":
    unittest.main()
