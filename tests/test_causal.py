import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import __future__
import json
import sys
import types


def _load_notebook_analysis():
    """Load only the Notebook's tagged definitions, not its analysis/display cells."""
    notebook_path = Path(__file__).resolve().parents[1] / "02. prototype.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    expected = ["core-setup","core-data","core-transform","core-priors","core-alignment","core-signatures","core-discovery","core-suite"]
    cells = [
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "causal-definitions" in cell.get("metadata", {}).get("tags", [])
    ]
    if [cell["id"] for cell in cells] != expected:
        raise AssertionError("Notebook analysis definition cells are missing or out of order")
    module = types.ModuleType("notebook_analysis")
    module.__file__ = str(notebook_path)
    # dataclasses resolves annotations through sys.modules during class creation.
    sys.modules[module.__name__] = module
    for cell in cells:
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        code = compile(
            source, f"{notebook_path}#{cell['id']}", "exec",
            flags=__future__.annotations.compiler_flag, dont_inherit=True,
        )
        exec(code, module.__dict__)
    required = ["EXPLANATORY","TARGET","RunConfig","_result_signature","build_link_assumptions","load_data","prepare_data","preprocess_logdiff","reference_lag"]
    if any(not hasattr(module, name) for name in required):
        raise AssertionError("Required Notebook analysis definitions are missing")
    return module


NOTEBOOK_ANALYSIS = _load_notebook_analysis()
EXPLANATORY = NOTEBOOK_ANALYSIS.EXPLANATORY
TARGET = NOTEBOOK_ANALYSIS.TARGET
RunConfig = NOTEBOOK_ANALYSIS.RunConfig
_result_signature = NOTEBOOK_ANALYSIS._result_signature
build_link_assumptions = NOTEBOOK_ANALYSIS.build_link_assumptions
load_data = NOTEBOOK_ANALYSIS.load_data
prepare_data = NOTEBOOK_ANALYSIS.prepare_data
preprocess_logdiff = NOTEBOOK_ANALYSIS.preprocess_logdiff
reference_lag = NOTEBOOK_ANALYSIS.reference_lag
from causal_graph import edge_table, plot_pag, review_graph, upstream_paths


def empty_graph(size, tau_max=1):
    return np.full((size, size, tau_max + 1), "", dtype="<U3")


def set_contemporaneous(graph, left, right, mark):
    mirrors = {
        "-->": "<--",
        "<--": "-->",
        "o->": "<-o",
        "<-o": "o->",
        "o-o": "o-o",
        "<->": "<->",
    }
    graph[left, right, 0] = mark
    graph[right, left, 0] = mirrors[mark]


def review(**overrides):
    values = {
        "signature": "run-1",
        "source": "A",
        "target": "B",
        "lag": 0,
        "original_mark": "o->",
        "proposed_mark": "-->",
        "rationale": "A can precede and transmit to B.",
        "counterargument": "An omitted common driver may explain the relation.",
    }
    values.update(overrides)
    return values


class PriorAndPreparationTests(unittest.TestCase):
    def test_repository_data_has_complete_positive_monthly_baseline_inputs(self):
        source = Path(__file__).resolve().parents[1] / "data" / "com_copper_raw.csv"

        raw = load_data(source)
        selected = raw.loc[:, [TARGET, *EXPLANATORY]]

        self.assertEqual(len(raw), 133)
        pd.testing.assert_index_equal(
            raw.index,
            pd.date_range("2015-01-01", "2026-01-01", freq="MS", name="dt"),
        )
        self.assertFalse(selected.isna().any().any())
        self.assertTrue((selected > 0).all().all())

    def test_empty_prior_enumerates_all_contemporaneous_and_lagged_candidates(self):
        assumptions = build_link_assumptions(["A", "B"], tau_max=2)

        self.assertEqual(sum(map(len, assumptions.values())), 10)
        self.assertEqual(assumptions[0][(1, 0)], "o?o")
        self.assertEqual(assumptions[1][(0, 0)], "o?o")
        self.assertEqual(assumptions[0][(0, -1)], "o?>")
        self.assertEqual(assumptions[0][(0, -2)], "o?>")
        self.assertNotIn("", {mark for links in assumptions.values() for mark in links.values()})

    def test_evidence_block_removes_only_the_named_lagged_candidate(self):
        variables = ["A", "B"]
        baseline = build_link_assumptions(variables, tau_max=2)
        constrained = build_link_assumptions(
            variables,
            tau_max=2,
            blocks=[
                {
                    "source": "A",
                    "target": "B",
                    "lag": 1,
                    "rationale": "Fixture: this candidate is definitionally impossible.",
                }
            ],
        )

        changed = [
            (target, key)
            for target in baseline
            for key in baseline[target]
            if baseline[target][key] != constrained[target][key]
        ]
        self.assertEqual(changed, [(1, (0, -1))])
        self.assertEqual(constrained[1][(0, -1)], "")

    def test_log_differences_require_positive_selected_values_and_keep_order(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="MS")
        columns = [TARGET, *EXPLANATORY]
        raw = pd.DataFrame(
            {
                name: [base, base * 2, base * 4]
                for name, base in zip(columns, [100.0, 10.0, 20.0, 30.0])
            },
            index=dates,
        )

        transformed = preprocess_logdiff(raw, columns)

        self.assertEqual(list(transformed.columns), columns)
        self.assertTrue(transformed.iloc[0].isna().all())
        np.testing.assert_allclose(transformed.iloc[1:].to_numpy(), np.log(2.0))

        invalid = raw.copy()
        invalid.loc[dates[1], EXPLANATORY[0]] = 0.0
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            preprocess_logdiff(invalid, columns)

    def test_availability_delay_preserves_calendar_and_uses_earlier_reference_value(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="MS")
        columns = [TARGET, *EXPLANATORY]
        raw = pd.DataFrame(index=dates)
        for offset, name in enumerate(columns, start=1):
            raw[name] = np.array([1.0, 2.0, 8.0, 16.0]) * offset
        delayed_variable = EXPLANATORY[2]
        config = RunConfig(
            name="inventory-delay",
            delays={delayed_variable: 1},
        )

        prepared, metadata = prepare_data(raw, config)

        self.assertEqual(prepared.index.tolist(), dates[2:].tolist())
        self.assertAlmostEqual(prepared.loc[dates[2], TARGET], np.log(4.0))
        self.assertAlmostEqual(prepared.loc[dates[2], delayed_variable], np.log(2.0))
        self.assertEqual(
            metadata["reference_periods_by_variable"][delayed_variable][0],
            "2024-02",
        )
        self.assertEqual(metadata["analysis_availability_dates"][0], "2024-03")

    def test_internal_missing_month_is_rejected_instead_of_compressing_time(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="MS")
        columns = [TARGET, *EXPLANATORY]
        raw = pd.DataFrame(
            {name: np.arange(1.0, 6.0) for name in columns},
            index=dates,
        )
        raw.loc[dates[2], EXPLANATORY[0]] = np.nan

        with self.assertRaisesRegex(ValueError, "missing values"):
            prepare_data(raw, RunConfig(name="gap"))

    def test_missing_calendar_row_is_rejected_instead_of_compressing_lags(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="MS").delete(2)
        columns = [TARGET, *EXPLANATORY]
        raw = pd.DataFrame(
            {name: np.arange(1.0, 5.0) for name in columns},
            index=dates,
        )

        with self.assertRaisesRegex(ValueError, "continuous month-start grid"):
            prepare_data(raw, RunConfig(name="missing-month"))

    def test_reference_lag_translates_between_analysis_and_observation_periods(self):
        self.assertEqual(reference_lag(1, source_delay=2, target_delay=0), 3)
        self.assertEqual(reference_lag(2, source_delay=0, target_delay=1), 1)

    def test_signature_binds_domain_algorithm_and_discovery_output(self):
        frame = pd.DataFrame(
            {name: [0.1, 0.2] for name in [TARGET, *EXPLANATORY]},
            index=pd.date_range("2024-01-01", periods=2, freq="MS"),
        )
        config = RunConfig(name="signature")
        metadata = {
            "input_sha256": "raw-input",
            "transform": "logdiff",
            "delays_months": {},
            "raw_date_bounds": ["2024-01", "2024-02"],
            "effective_date_bounds": ["2024-01", "2024-02"],
            "actual_target_dates": ["2024-01", "2024-02"],
            "algorithm_parameters": {"method": "LPCMCI", "pc_alpha": 0.05},
            "reference_burn": 6,
            "reference_points": [],
            "domain_blocks": [],
            "graph_sha256": "graph-a",
        }
        versions = {"python": "test"}
        baseline = _result_signature(frame, config, metadata, versions)

        for key, changed_value in (
            (
                "domain_blocks",
                [{"source": "A", "target": "B", "lag": 1, "rationale": "blocked"}],
            ),
            ("algorithm_parameters", {"method": "LPCMCI", "pc_alpha": 0.01}),
            ("graph_sha256", "graph-b"),
        ):
            with self.subTest(changed=key):
                changed = dict(metadata)
                changed[key] = changed_value
                self.assertNotEqual(
                    baseline,
                    _result_signature(frame, config, changed, versions),
                )


class GraphPresentationTests(unittest.TestCase):
    def test_symmetric_unknown_tau_zero_mark_is_visible_but_not_a_causal_path(self):
        graph = empty_graph(2)
        graph[0, 1, 0] = "x-x"
        graph[1, 0, 0] = "x-x"

        table = edge_table(graph, ["A", "Target"])
        paths = upstream_paths(graph, ["A", "Target"], "Target", horizon=1)

        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["edge_class"], "unclassified")
        self.assertTrue(paths.empty)

    def test_edge_table_canonicalizes_tau_zero_and_preserves_unknown_marks(self):
        graph = empty_graph(3)
        set_contemporaneous(graph, 0, 1, "-->")
        graph[1, 2, 1] = "x-x"
        values = np.full(graph.shape, np.nan)
        pvalues = np.full(graph.shape, np.nan)
        values[0, 1, 0] = -0.4
        pvalues[0, 1, 0] = 0.02

        result = edge_table(graph, ["A", "B", "C"], values, pvalues)

        self.assertEqual(len(result), 2)
        directed = result.loc[result["mark"] == "-->"].iloc[0]
        self.assertEqual(directed["edge_class"], "directed")
        self.assertAlmostEqual(directed["statistic"], -0.4)
        self.assertAlmostEqual(directed["p_value"], 0.02)
        unknown = result.loc[result["mark"] == "x-x"].iloc[0]
        self.assertEqual(unknown["edge_class"], "unclassified")

    def test_paths_distinguish_definite_possible_and_exclude_noncausal_marks(self):
        variables = ["A", "B", "C", "D", "Target"]
        graph = empty_graph(len(variables))
        graph[0, 4, 1] = "-->"
        graph[1, 4, 1] = "o->"
        graph[2, 4, 1] = "<->"
        graph[3, 4, 1] = "x-x"

        result = upstream_paths(graph, variables, "Target", horizon=2, max_edges=1)

        self.assertEqual(set(result["source"]), {"A", "B"})
        self.assertEqual(
            dict(zip(result["source"], result["path_type"])),
            {"A": "definite", "B": "possible"},
        )
        self.assertEqual(set(result["max_edge_lag"]), {1})

    def test_assumed_edge_is_explicitly_labelled_in_path(self):
        graph = empty_graph(2)
        graph[0, 1, 1] = "-->"

        result = upstream_paths(
            graph,
            ["A", "Target"],
            "Target",
            horizon=1,
            max_edges=1,
            assumed_edges={"lag1:A->Target"},
        )

        self.assertEqual(result.iloc[0]["path_type"], "reviewed_assumption")

    def test_path_key_distinguishes_endpoint_evidence_for_same_node_route(self):
        definite = empty_graph(2)
        definite[0, 1, 1] = "-->"
        possible = empty_graph(2)
        possible[0, 1, 1] = "o->"

        definite_path = upstream_paths(definite, ["A", "Target"], "Target", horizon=1).iloc[0]
        possible_path = upstream_paths(possible, ["A", "Target"], "Target", horizon=1).iloc[0]

        self.assertEqual(definite_path["path"], possible_path["path"])
        self.assertNotEqual(definite_path["path_type"], possible_path["path_type"])
        self.assertNotEqual(definite_path["path_key"], possible_path["path_key"])

    def test_plot_draws_every_supported_endpoint_mark(self):
        marks = ("-->", "<--", "o->", "<-o", "o-o", "<->")
        for lag in (0, 1):
            for mark in marks:
                with self.subTest(lag=lag, mark=mark):
                    graph = empty_graph(2)
                    if lag == 0:
                        set_contemporaneous(graph, 0, 1, mark)
                    else:
                        graph[0, 1, 1] = mark

                    figure = plot_pag(graph, ["A", "B"])
                    try:
                        self.assertIsInstance(figure, plt.Figure)
                        figure.canvas.draw()
                    finally:
                        plt.close(figure)


class GraphReviewTests(unittest.TestCase):
    def test_fractional_and_boolean_review_lags_are_rejected(self):
        graph = empty_graph(2)
        graph[0, 1, 1] = "o-o"

        for invalid_lag in (1.5, True):
            with self.subTest(lag=invalid_lag):
                with self.assertRaisesRegex(ValueError, "integer"):
                    review_graph(
                        graph,
                        ["A", "B"],
                        "run-1",
                        [review(lag=invalid_lag, original_mark="o-o")],
                    )

    def test_valid_circle_completion_is_mirrored_and_audited(self):
        graph = empty_graph(2)
        set_contemporaneous(graph, 0, 1, "o->")

        augmented, decisions = review_graph(
            graph,
            ["A", "B"],
            "run-1",
            [review(sources=["https://example.test/evidence"])],
        )

        self.assertEqual(graph[0, 1, 0], "o->")
        self.assertEqual(augmented[0, 1, 0], "-->")
        self.assertEqual(augmented[1, 0, 0], "<--")
        self.assertEqual(
            decisions.iloc[0]["evidence_label"],
            "literature_informed_hypothesis",
        )

    def test_stale_signature_fails_without_mutating_original(self):
        graph = empty_graph(2)
        set_contemporaneous(graph, 0, 1, "o->")
        before = graph.copy()

        with self.assertRaisesRegex(ValueError, "stale graph signature"):
            review_graph(
                graph,
                ["A", "B"],
                "run-1",
                [review(signature="old-run")],
            )

        np.testing.assert_array_equal(graph, before)

    def test_review_is_atomic_when_any_proposal_is_invalid(self):
        graph = empty_graph(3)
        set_contemporaneous(graph, 0, 1, "o->")
        set_contemporaneous(graph, 1, 2, "o->")
        before = graph.copy()

        with self.assertRaises(ValueError):
            review_graph(
                graph,
                ["A", "B", "C"],
                "run-1",
                [
                    review(),
                    review(
                        source="B",
                        target="C",
                        original_mark="o->",
                        proposed_mark="<--",
                    ),
                ],
            )

        np.testing.assert_array_equal(graph, before)

    def test_nonaccepted_disposition_cannot_silently_change_graph(self):
        graph = empty_graph(2)
        set_contemporaneous(graph, 0, 1, "o->")
        before = graph.copy()

        with self.assertRaisesRegex(ValueError, "disposition must be 'accepted'"):
            review_graph(
                graph,
                ["A", "B"],
                "run-1",
                [review(disposition="rejected")],
            )

        np.testing.assert_array_equal(graph, before)

    def test_established_arrowheads_and_bidirected_edges_cannot_be_overwritten(self):
        for original in ("-->", "<->"):
            with self.subTest(original=original):
                graph = empty_graph(2)
                set_contemporaneous(graph, 0, 1, original)
                with self.assertRaisesRegex(ValueError, "cannot change"):
                    review_graph(
                        graph,
                        ["A", "B"],
                        "run-1",
                        [review(original_mark=original)],
                    )

    def test_new_contemporaneous_directed_cycle_is_rejected(self):
        graph = empty_graph(3)
        set_contemporaneous(graph, 0, 1, "-->")
        set_contemporaneous(graph, 1, 2, "-->")
        set_contemporaneous(graph, 0, 2, "o-o")

        with self.assertRaisesRegex(ValueError, "directed cycle"):
            review_graph(
                graph,
                ["A", "B", "C"],
                "run-1",
                [
                    review(
                        source="A",
                        target="C",
                        original_mark="o-o",
                        proposed_mark="<--",
                    )
                ],
            )

    def test_temporal_feedback_is_not_mistaken_for_same_time_cycle(self):
        graph = empty_graph(2)
        graph[0, 1, 1] = "o-o"
        graph[1, 0, 1] = "-->"

        augmented, _ = review_graph(
            graph,
            ["A", "B"],
            "run-1",
            [review(lag=1, original_mark="o-o")],
        )

        self.assertEqual(augmented[0, 1, 1], "-->")
        self.assertEqual(augmented[1, 0, 1], "-->")


if __name__ == "__main__":
    unittest.main()
