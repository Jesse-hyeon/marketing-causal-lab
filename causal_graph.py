"""Utilities for inspecting and reviewing Tigramite-style PAG/DPAG graphs.

The functions in this module preserve uncertainty in endpoint marks.  In
particular, ``<->`` is treated as a latent-compatible association rather than
as a directed causal route, and paths containing circle endpoints are only
reported as endpoint-compatible candidates.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


KNOWN_MARKS = {"", "-->", "<--", "o->", "<-o", "o-o", "<->"}
MIRROR_MARK = {
    "": "",
    "-->": "<--",
    "<--": "-->",
    "o->": "<-o",
    "<-o": "o->",
    "o-o": "o-o",
    "<->": "<->",
}

EDGE_COLUMNS = [
    "source",
    "target",
    "lag",
    "mark",
    "edge_class",
    "statistic",
    "p_value",
    "edge_key",
]
PATH_COLUMNS = [
    "source",
    "target",
    "edge_count",
    "total_lag",
    "max_edge_lag",
    "path_type",
    "path",
    "timestamps",
    "edge_keys",
    "path_key",
]
DECISION_COLUMNS = [
    "signature",
    "source",
    "target",
    "lag",
    "original_mark",
    "proposed_mark",
    "rationale",
    "counterargument",
    "sources",
    "disposition",
    "evidence_label",
    "edge_key",
]


def _validate_graph(graph: Any, variables: Sequence[str]) -> np.ndarray:
    array = np.asarray(graph)
    if array.ndim != 3:
        raise ValueError("graph must have shape (variables, variables, lags)")
    n_rows, n_cols, n_lags = array.shape
    if n_rows != n_cols or n_rows != len(variables) or n_lags < 1:
        raise ValueError(
            "graph shape must be (len(variables), len(variables), tau_max + 1)"
        )
    if len(set(variables)) != len(variables):
        raise ValueError("variables must be unique")
    return array


def _validate_optional_matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have the same shape as graph")
    return array


def _token(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _edge_key(source: str, target: str, lag: int) -> str:
    return f"{source}[t-{lag}]->{target}[t]"


def _stored_edge_key(i: int, j: int, lag: int, variables: Sequence[str]) -> str:
    if lag == 0:
        left, right = sorted((i, j))
        return f"tau0:{variables[left]}|{variables[right]}"
    return f"lag{lag}:{variables[i]}->{variables[j]}"


def _finite_or_nan(array: np.ndarray | None, index: tuple[int, int, int]) -> float:
    if array is None:
        return np.nan
    try:
        value = float(array[index])
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _classify(mark: str, lag: int) -> str:
    if mark not in KNOWN_MARKS:
        return "unclassified"
    if lag > 0 and mark in {"<--", "<-o"}:
        return "temporal_conflict"
    return {
        "-->": "directed",
        "<--": "directed",
        "o->": "partially_directed",
        "<-o": "partially_directed",
        "o-o": "unoriented",
        "<->": "latent_compatible",
        "": "unclassified",
    }[mark]


def _validate_contemporaneous_mirror(array: np.ndarray) -> None:
    n_vars = array.shape[0]
    for i in range(n_vars):
        diagonal = _token(array[i, i, 0])
        if diagonal:
            raise ValueError("contemporaneous self-edges are not supported")
        for j in range(i + 1, n_vars):
            mark = _token(array[i, j, 0])
            mirror = _token(array[j, i, 0])
            if not mark and not mirror:
                continue
            if mark not in MIRROR_MARK:
                if mirror and mirror != mark:
                    raise ValueError(
                        f"unknown contemporaneous mark {mark!r} cannot be mirrored safely"
                    )
                # Unknown endpoint syntax remains visible to edge_table but is
                # excluded from orientation, review, and path traversal.
                continue
            expected = MIRROR_MARK[mark]
            if mirror != expected:
                raise ValueError(
                    f"inconsistent contemporaneous marks at ({i}, {j}): "
                    f"{mark!r} vs {mirror!r}; expected {expected!r}"
                )


def edge_table(
    graph: Any,
    variables: Sequence[str],
    values: Any = None,
    pvalues: Any = None,
) -> pd.DataFrame:
    """Return one stable row per stored PAG/DPAG adjacency.

    Contemporaneous edges are canonicalized to the upper triangle.  Unknown
    lagged tokens are retained as ``unclassified``; they are never silently
    oriented.  ``statistic`` and ``p_value`` are copied only for retained rows
    and non-finite entries become ``NaN``.
    """

    array = _validate_graph(graph, variables)
    value_array = _validate_optional_matrix(values, array.shape, "values")
    pvalue_array = _validate_optional_matrix(pvalues, array.shape, "pvalues")
    _validate_contemporaneous_mirror(array)

    rows: list[dict[str, Any]] = []
    n_vars, _, n_lags = array.shape
    for lag in range(n_lags):
        pairs = (
            ((i, j) for i in range(n_vars) for j in range(i + 1, n_vars))
            if lag == 0
            else ((i, j) for i in range(n_vars) for j in range(n_vars))
        )
        for i, j in pairs:
            mark = _token(array[i, j, lag])
            if not mark:
                continue
            rows.append(
                {
                    "source": variables[i],
                    "target": variables[j],
                    "lag": lag,
                    "mark": mark,
                    "edge_class": _classify(mark, lag),
                    "statistic": _finite_or_nan(value_array, (i, j, lag)),
                    "p_value": _finite_or_nan(pvalue_array, (i, j, lag)),
                    "edge_key": _stored_edge_key(i, j, lag, variables),
                }
            )
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


@dataclass(frozen=True)
class _Route:
    source: tuple[int, int]
    target: tuple[int, int]
    edge_key: str
    mark: str
    definite: bool


def _base_orientations(i: int, j: int, lag: int, mark: str) -> list[tuple[int, int, bool]]:
    """Return (source variable, target variable, definite) orientations."""
    if lag > 0:
        if mark == "-->":
            return [(i, j, True)]
        if mark in {"o->", "o-o"}:
            return [(i, j, False)]
        return []
    if mark == "-->":
        return [(i, j, True)]
    if mark == "<--":
        return [(j, i, True)]
    if mark == "o->":
        return [(i, j, False)]
    if mark == "<-o":
        return [(j, i, False)]
    if mark == "o-o":
        return [(i, j, False), (j, i, False)]
    return []


def _time_routes(array: np.ndarray, variables: Sequence[str], horizon: int) -> list[_Route]:
    n_vars, _, n_lags = array.shape
    routes: list[_Route] = []
    for lag in range(n_lags):
        pairs = (
            ((i, j) for i in range(n_vars) for j in range(i + 1, n_vars))
            if lag == 0
            else ((i, j) for i in range(n_vars) for j in range(n_vars))
        )
        for i, j in pairs:
            mark = _token(array[i, j, lag])
            for source_var, target_var, definite in _base_orientations(i, j, lag, mark):
                for target_time in range(-horizon, 1):
                    source_time = target_time - lag
                    if source_time < -horizon:
                        continue
                    routes.append(
                        _Route(
                            (source_var, source_time),
                            (target_var, target_time),
                            _stored_edge_key(i, j, lag, variables),
                            mark,
                            definite,
                        )
                    )
    return routes


def upstream_paths(
    graph: Any,
    variables: Sequence[str],
    target: str,
    horizon: int = 6,
    max_edges: int = 3,
    assumed_edges: Iterable[str] = (),
) -> pd.DataFrame:
    """Enumerate short endpoint-compatible paths ending at ``target(t)``.

    Definite paths contain only tail-to-arrowhead orientations.  Possible
    paths contain at least one circle endpoint.  Neither category certifies a
    realizable MAG/DAG or an intervention effect.  Bidirected and unknown
    marks are excluded.  If a path uses an edge key supplied in
    ``assumed_edges``, its class is ``reviewed_assumption``.
    """

    array = _validate_graph(graph, variables)
    _validate_contemporaneous_mirror(array)
    if target not in variables:
        raise ValueError(f"target {target!r} is not in variables")
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    if not isinstance(max_edges, int) or max_edges < 1 or max_edges > 3:
        raise ValueError("max_edges must be an integer from 1 to 3")

    routes = _time_routes(array, variables, horizon)
    incoming: dict[tuple[int, int], list[_Route]] = defaultdict(list)
    for route in routes:
        incoming[route.target].append(route)
    for candidates in incoming.values():
        candidates.sort(key=lambda route: (route.source[1], route.source[0], route.edge_key))

    assumed = set(assumed_edges)
    target_node = (variables.index(target), 0)
    rows: list[dict[str, Any]] = []

    def visit(
        current: tuple[int, int],
        reverse_nodes: list[tuple[int, int]],
        reverse_routes: list[_Route],
    ) -> None:
        if reverse_routes:
            nodes = list(reversed(reverse_nodes))
            used_routes = list(reversed(reverse_routes))
            keys = tuple(route.edge_key for route in used_routes)
            marked_keys = tuple(
                f"{route.edge_key}:{route.mark}" for route in used_routes
            )
            labels = tuple(f"{variables[var]}(t{time:+d})" for var, time in nodes)
            path_type = (
                "reviewed_assumption"
                if any(key in assumed for key in keys)
                else "definite"
                if all(route.definite for route in used_routes)
                else "possible"
            )
            rows.append(
                {
                    "source": variables[nodes[0][0]],
                    "target": target,
                    "edge_count": len(used_routes),
                    "total_lag": nodes[-1][1] - nodes[0][1],
                    "max_edge_lag": max(
                        route.target[1] - route.source[1] for route in used_routes
                    ),
                    "path_type": path_type,
                    "path": " -> ".join(labels),
                    "timestamps": tuple(time for _, time in nodes),
                    "edge_keys": keys,
                    "path_key": "|".join(
                        [
                            *(f"{label}@{time}" for label, time in zip(labels, (node[1] for node in nodes))),
                            *(f"edge:{key}" for key in marked_keys),
                            f"class:{path_type}",
                        ]
                    ),
                }
            )
        if len(reverse_routes) >= max_edges:
            return
        visited = set(reverse_nodes)
        for route in incoming.get(current, []):
            if route.source in visited:
                continue
            visit(route.source, reverse_nodes + [route.source], reverse_routes + [route])

    visit(target_node, [target_node], [])
    if not rows:
        return pd.DataFrame(columns=PATH_COLUMNS)
    result = pd.DataFrame(rows, columns=PATH_COLUMNS)
    return result.drop_duplicates("path_key").sort_values(
        ["edge_count", "path_key"], kind="stable", ignore_index=True
    )


def _endpoint_arrowheads(mark: str) -> tuple[bool, bool]:
    """Return arrowhead presence at the stored left and right endpoints."""
    return mark.startswith("<"), mark.endswith(">")


def _expanded_adjacencies(
    array: np.ndarray, variables: Sequence[str]
) -> tuple[dict[tuple[int, int], set[tuple[int, int]]], list[tuple[tuple[int, int], tuple[int, int], bool, bool]]]:
    horizon = max(1, array.shape[2] - 1)
    directed: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    endpoints: list[tuple[tuple[int, int], tuple[int, int], bool, bool]] = []
    n_vars = array.shape[0]
    for lag in range(array.shape[2]):
        pairs = (
            ((i, j) for i in range(n_vars) for j in range(i + 1, n_vars))
            if lag == 0
            else ((i, j) for i in range(n_vars) for j in range(n_vars))
        )
        for i, j in pairs:
            mark = _token(array[i, j, lag])
            if not mark or mark not in KNOWN_MARKS:
                continue
            for target_time in range(-horizon, 1):
                source_time = target_time - lag
                if source_time < -horizon:
                    continue
                left = (i, source_time)
                right = (j, target_time)
                arrow_left, arrow_right = _endpoint_arrowheads(mark)
                endpoints.append((left, right, arrow_left, arrow_right))
                if mark == "-->":
                    directed[left].add(right)
                elif lag == 0 and mark == "<--":
                    directed[right].add(left)
    return directed, endpoints


def _has_path(
    adjacency: Mapping[tuple[int, int], set[tuple[int, int]]],
    start: tuple[int, int],
    target: tuple[int, int],
    *,
    require_nonempty: bool = True,
) -> bool:
    stack = list(adjacency.get(start, ()))
    seen: set[tuple[int, int]] = set()
    if not require_nonempty and start == target:
        return True
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, ()))
    return False


def _validate_ancestral_consistency(array: np.ndarray, variables: Sequence[str]) -> None:
    directed, endpoints = _expanded_adjacencies(array, variables)
    for node in directed:
        if _has_path(directed, node, node):
            raise ValueError("review would create a directed cycle in the time-expanded graph")
    for left, right, arrow_left, arrow_right in endpoints:
        if arrow_left and _has_path(directed, left, right):
            raise ValueError("review would create an arrowhead-into-ancestor conflict")
        if arrow_right and _has_path(directed, right, left):
            raise ValueError("review would create an arrowhead-into-ancestor conflict")


def review_graph(
    graph: Any,
    variables: Sequence[str],
    signature: str,
    reviews: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply explicit circle-endpoint reviews atomically.

    Reviews can only replace unresolved circle endpoints with a compatible
    direction.  Established tails/arrowheads and adjacencies are preserved.
    Any stale, invalid, duplicated, or ancestrally inconsistent proposal raises
    ``ValueError``; the input array is never mutated.
    """

    array = _validate_graph(graph, variables)
    _validate_contemporaneous_mirror(array)
    # Object dtype prevents a caller's short fixed-width string dtype from
    # truncating a completed three-character endpoint mark.
    augmented = np.array(array, dtype=object, copy=True)
    index = {name: i for i, name in enumerate(variables)}
    required = {
        "signature",
        "source",
        "target",
        "lag",
        "original_mark",
        "proposed_mark",
        "rationale",
        "counterargument",
    }
    seen_keys: set[str] = set()
    prepared: list[tuple[int, int, int, str, str, Mapping[str, Any], str]] = []

    for position, review in enumerate(reviews):
        missing = required.difference(review)
        if missing:
            raise ValueError(f"review {position} is missing fields: {sorted(missing)}")
        if review["signature"] != signature:
            raise ValueError(f"review {position} has a stale graph signature")
        source = str(review["source"])
        target = str(review["target"])
        if source not in index or target not in index:
            raise ValueError(f"review {position} names an unknown variable")
        raw_lag = review["lag"]
        if isinstance(raw_lag, (bool, np.bool_)) or not isinstance(
            raw_lag, (int, np.integer)
        ):
            raise ValueError(f"review {position} lag must be an integer")
        lag = int(raw_lag)
        if source == target and lag == 0:
            raise ValueError("contemporaneous self-edge reviews are invalid")
        if lag < 0 or lag >= array.shape[2]:
            raise ValueError(f"review {position} lag is outside the graph")
        i, j = index[source], index[target]
        original = _token(review["original_mark"])
        proposed = _token(review["proposed_mark"])
        if _token(array[i, j, lag]) != original:
            raise ValueError(f"review {position} original_mark is stale")
        allowed = {"o->": {"-->"}, "<-o": {"<--"}, "o-o": {"-->", "<--"}}
        if proposed not in allowed.get(original, set()):
            raise ValueError(
                f"review {position} cannot change {original!r} to {proposed!r}"
            )
        if lag > 0 and proposed != "-->":
            raise ValueError("lagged reviews cannot orient an edge backward in time")
        rationale = str(review["rationale"]).strip()
        counterargument = str(review["counterargument"]).strip()
        if not rationale or not counterargument:
            raise ValueError(f"review {position} needs rationale and counterargument")
        disposition = str(review.get("disposition", "accepted")).strip().lower()
        if disposition != "accepted":
            raise ValueError(
                f"review {position} disposition must be 'accepted' to change the graph"
            )
        key = _stored_edge_key(i, j, lag, variables)
        if key in seen_keys:
            raise ValueError(f"duplicate or conflicting proposal for {key}")
        seen_keys.add(key)
        prepared.append((i, j, lag, original, proposed, review, key))

    for i, j, lag, _original, proposed, _review, _key in prepared:
        augmented[i, j, lag] = proposed
        if lag == 0:
            augmented[j, i, lag] = MIRROR_MARK[proposed]

    _validate_contemporaneous_mirror(augmented)
    _validate_ancestral_consistency(augmented, variables)

    rows: list[dict[str, Any]] = []
    for i, j, lag, original, proposed, review, key in prepared:
        raw_sources = review.get("sources", [])
        if raw_sources is None:
            sources: list[str] = []
        elif isinstance(raw_sources, str):
            sources = [raw_sources]
        else:
            sources = [str(source) for source in raw_sources]
        rows.append(
            {
                "signature": signature,
                "source": variables[i],
                "target": variables[j],
                "lag": lag,
                "original_mark": original,
                "proposed_mark": proposed,
                "rationale": str(review["rationale"]).strip(),
                "counterargument": str(review["counterargument"]).strip(),
                "sources": sources,
                "disposition": "accepted",
                "evidence_label": (
                    "literature_informed_hypothesis" if sources else "model_hypothesis"
                ),
                "edge_key": key,
            }
        )
    return augmented, pd.DataFrame(rows, columns=DECISION_COLUMNS)


_LABELS = {
    "Com_LME_Cu_3M": "Copper 3M",
    "Com_LME_Cu_Cash": "Copper cash",
    "Com_LME_Cu_cash": "Copper cash",
    "Com_LME_Cu_Stock": "Copper stock",
    "Com_Brent": "Brent",
    "Eco_US_DXY": "US dollar index",
    "Idx_DxyUSD": "US dollar index",
    "Com_BrentCrudeOil": "Brent crude oil",
    "Com_LME_Cu_Inv": "Copper inventory",
}


def _draw_marked_edge(
    axis: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    mark: str,
    curvature: float,
    identifying_assumption: bool = False,
) -> None:
    color = (
        "#d97706"
        if identifying_assumption
        else "#4c566a"
        if mark in KNOWN_MARKS
        else "#b23a48"
    )
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={curvature}",
        mutation_scale=12,
        linewidth=1.4,
        color=color,
        # Node radius is roughly nine points at the configured scatter size;
        # sixteen points leaves endpoint glyphs visibly beyond its boundary.
        shrinkA=16,
        shrinkB=16,
        zorder=2,
    )
    axis.add_patch(patch)
    # FancyArrowPatch exposes the actual curved, node-shrunk path in data
    # coordinates.  Placing unresolved endpoint markers on those vertices
    # keeps them outside the node glyph instead of hiding them at its center.
    path_vertices = patch.get_path().vertices
    visible_start = path_vertices[0]
    visible_end = path_vertices[-1]
    if len(path_vertices) >= 3:
        control = path_vertices[1]

        def curve_point(fraction: float) -> np.ndarray:
            return (
                (1 - fraction) ** 2 * visible_start
                + 2 * (1 - fraction) * fraction * control
                + fraction**2 * visible_end
            )

        if mark.startswith("<"):
            axis.add_patch(
                FancyArrowPatch(
                    curve_point(0.18),
                    visible_start,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.4,
                    color=color,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=4,
                )
            )
        if mark.endswith(">"):
            axis.add_patch(
                FancyArrowPatch(
                    curve_point(0.82),
                    visible_end,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.4,
                    color=color,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=4,
                )
            )
    if mark.startswith("o"):
        axis.plot(
            *visible_start,
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=color,
            zorder=7,
        )
    if mark.endswith("o"):
        axis.plot(
            *visible_end,
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=color,
            zorder=7,
        )
    if mark not in KNOWN_MARKS:
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        axis.plot(*midpoint, marker="x", color=color, markersize=7, zorder=4)


def plot_pag(
    graph: Any,
    variables: Sequence[str],
    title: str = "PAG / DPAG",
    horizon: int | None = None,
    values: Any = None,
    assumed_edges: Iterable[str] = (),
) -> Figure:
    """Plot a time-unrolled PAG/DPAG while retaining exact endpoint marks."""

    array = _validate_graph(graph, variables)
    _validate_optional_matrix(values, array.shape, "values")
    _validate_contemporaneous_mirror(array)
    max_lag = array.shape[2] - 1
    assumed = set(assumed_edges)
    if horizon is None:
        horizon = max_lag
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")

    width = max(7.0, 2.0 + 1.7 * (horizon + 1))
    height = max(4.0, 1.2 + 0.9 * len(variables))
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    n_vars = len(variables)
    times = list(range(-horizon, 1))
    # Fix the transform before FancyArrowPatch calculates point-based shrink.
    axis.set_xlim(-horizon - 0.35, 0.35)
    axis.set_ylim(-0.55, max(n_vars - 0.45, 0.55))

    for time in times:
        for var in range(n_vars):
            axis.scatter(time, n_vars - 1 - var, s=230, facecolor="#eef3f8", edgecolor="#2f4858", zorder=5)

    for lag in range(array.shape[2]):
        pairs = (
            ((i, j) for i in range(n_vars) for j in range(i + 1, n_vars))
            if lag == 0
            else ((i, j) for i in range(n_vars) for j in range(n_vars))
        )
        for i, j in pairs:
            mark = _token(array[i, j, lag])
            if not mark:
                continue
            for target_time in times:
                source_time = target_time - lag
                if source_time < -horizon:
                    continue
                start = (source_time, n_vars - 1 - i)
                end = (target_time, n_vars - 1 - j)
                curvature = 0.12 if lag else 0.18 * (1 if (i + j) % 2 else -1)
                _draw_marked_edge(
                    axis,
                    start,
                    end,
                    mark,
                    curvature,
                    identifying_assumption=(
                        _stored_edge_key(i, j, lag, variables) in assumed
                    ),
                )

    axis.set_xticks(times, ["t" if time == 0 else f"t{time}" for time in times])
    axis.set_yticks(
        range(n_vars - 1, -1, -1),
        [_LABELS.get(variable, variable) for variable in variables],
    )
    axis.grid(axis="x", color="#d8dee9", linewidth=0.8, zorder=0)
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    legend = [
        Line2D([0], [0], color="#4c566a", marker=">", markevery=[1], label="--> tail to arrowhead"),
        Line2D([0], [0], color="#4c566a", marker="o", markerfacecolor="white", label="o  unresolved endpoint"),
        Line2D([0], [0], color="#4c566a", marker="<", label="<-> latent-compatible"),
        Line2D([0], [0], color="#b23a48", marker="x", label="unknown / unclassified"),
    ]
    if assumed:
        legend.append(
            Line2D(
                [0],
                [0],
                color="#d97706",
                linewidth=2,
                label="Codex identifying assumption",
            )
        )
    axis.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    return figure


__all__ = ["edge_table", "upstream_paths", "review_graph", "plot_pag"]
