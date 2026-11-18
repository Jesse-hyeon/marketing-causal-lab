"""Small, dependency-light utilities for uplift and causal ML notebooks.

The functions here intentionally keep the estimators transparent: all learners
use scikit-learn models, and evaluation uses only held-out observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPS = 0.03


def _arrays(X, T, Y):
    X = np.asarray(X, dtype=float)
    T = np.asarray(T, dtype=int).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(T) or len(T) != len(Y):
        raise ValueError("X must be 2-D and have the same number of rows as T and Y")
    if not np.isin(T, [0, 1]).all():
        raise ValueError("T must contain only 0 and 1")
    if np.unique(T).size != 2:
        raise ValueError("Both treatment groups are required")
    return X, T, Y


def _regressor(seed: int):
    return HistGradientBoostingRegressor(
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        learning_rate=0.08,
        l2_regularization=2.0,
        random_state=seed,
    )


def _propensity(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=seed),
    )


@dataclass
class _SLearner:
    model: object

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.model.predict(np.column_stack([X, np.ones(len(X))])) - self.model.predict(
            np.column_stack([X, np.zeros(len(X))])
        )


@dataclass
class _TLearner:
    control: object
    treated: object

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.treated.predict(X) - self.control.predict(X)


@dataclass
class _XLearner:
    control: object
    treated: object
    tau_control: object
    tau_treated: object
    propensity: object

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        mu0 = self.control.predict(X)
        mu1 = self.treated.predict(X)
        tau0 = self.tau_control.predict(X)
        tau1 = self.tau_treated.predict(X)
        e = np.clip(self.propensity.predict_proba(X)[:, 1], EPS, 1 - EPS)
        return e * tau0 + (1.0 - e) * tau1


def fit_meta(X, T, Y, seed: int = 42):
    """Fit S-, T-, and X-learners; each returned value has ``predict(X)``."""
    X, T, Y = _arrays(X, T, Y)
    models = {}

    s_model = _regressor(seed)
    s_model.fit(np.column_stack([X, T]), Y)
    models["S"] = _SLearner(s_model)

    c0, c1 = _regressor(seed), _regressor(seed + 1)
    c0.fit(X[T == 0], Y[T == 0])
    c1.fit(X[T == 1], Y[T == 1])
    models["T"] = _TLearner(c0, c1)

    mu0, mu1 = _regressor(seed + 2), _regressor(seed + 3)
    mu0.fit(X[T == 0], Y[T == 0])
    mu1.fit(X[T == 1], Y[T == 1])
    d0 = mu1.predict(X[T == 0]) - Y[T == 0]
    d1 = Y[T == 1] - mu0.predict(X[T == 1])
    tau0, tau1 = _regressor(seed + 4), _regressor(seed + 5)
    tau0.fit(X[T == 0], d0)
    tau1.fit(X[T == 1], d1)
    prop = _propensity(seed + 6)
    prop.fit(X, T)
    models["X"] = _XLearner(mu0, mu1, tau0, tau1, prop)
    return models


def fit_nuisance(X, T, Y, seed: int = 42):
    """Fit outcome regressions and a propensity model on a training split."""
    X, T, Y = _arrays(X, T, Y)
    mu0, mu1 = _regressor(seed), _regressor(seed + 1)
    mu0.fit(X[T == 0], Y[T == 0])
    mu1.fit(X[T == 1], Y[T == 1])
    e = _propensity(seed + 2)
    e.fit(X, T)
    return {"mu0": mu0, "mu1": mu1, "e": e}


def nuisance_scores(nuisance: Mapping[str, object], X, T, Y):
    """Return nuisance predictions and the AIPW pseudo-outcome."""
    T = np.asarray(T, dtype=int).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    mu0 = np.asarray(nuisance["mu0"].predict(X), dtype=float)
    mu1 = np.asarray(nuisance["mu1"].predict(X), dtype=float)
    e = np.clip(np.asarray(nuisance["e"].predict_proba(X)[:, 1], dtype=float), EPS, 1 - EPS)
    dr = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
    return {"e": e, "mu0": mu0, "mu1": mu1, "dr": dr}


def _rank(scores, k, seed):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = len(scores)
    count = min(n, max(1, int(np.ceil(n * k))))
    rng = np.random.default_rng(seed)
    # A random secondary key makes equal-score handling independent of outcomes.
    order = np.lexsort((rng.random(n), -scores))
    policy = np.zeros(n, dtype=int)
    policy[order[:count]] = 1
    return order, policy


def evaluate(scores, T, Y, e, mu0, mu1, k: float = 0.3, seed: int = 42):
    """Evaluate ranking and top-k policy on an independent evaluation split."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    T = np.asarray(T, dtype=int).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    e = np.clip(np.asarray(e, dtype=float).reshape(-1), EPS, 1 - EPS)
    mu0, mu1 = np.asarray(mu0, dtype=float), np.asarray(mu1, dtype=float)
    n = len(scores)
    if not all(len(a) == n for a in (T, Y, e, mu0, mu1)) or not 0 < k <= 1:
        raise ValueError("all inputs must have equal length and k must be in (0, 1]")
    order, policy = _rank(scores, k, seed)
    z = T * Y / e - (1 - T) * Y / (1 - e)
    gain = np.r_[0.0, np.cumsum(z[order]) / n]
    x = np.arange(n + 1, dtype=float) / n
    auuc = float(np.trapezoid(gain, x))
    random_area = float(gain[-1] / 2.0)
    qini = auuc - random_area
    top = order[: max(1, int(np.ceil(n * k)))]
    treated, control = top[T[top] == 1], top[T[top] == 0]
    top_uplift = float(np.mean(Y[treated]) - np.mean(Y[control])) if len(treated) and len(control) else np.nan
    incremental_rate = float(np.sum(z[top]) / n)

    dr0 = mu0 + (1 - T) * (Y - mu0) / (1 - e)
    dr1 = mu1 + T * (Y - mu1) / e
    policy_value = float(np.mean(np.where(policy, dr1, dr0)))
    random_value = float(np.mean(dr0) + policy.mean() * np.mean(dr1 - dr0))
    return {
        "AUUC": auuc,
        "Qini": float(qini),
        "Uplift@30%": top_uplift,
        "IncrementalRate": incremental_rate,
        "PolicyValue": policy_value,
        "RandomValue": random_value,
        "PolicyGain": policy_value - random_value,
    }


def uplift_curve(scores, T, Y, e):
    """Return ranked fractions, IPW cumulative gain, and its random baseline."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    T = np.asarray(T, dtype=int).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    e = np.clip(np.asarray(e, dtype=float).reshape(-1), EPS, 1 - EPS)
    if not (len(scores) == len(T) == len(Y) == len(e)):
        raise ValueError("all inputs must have equal length")
    order, _ = _rank(scores, 1.0, 42)
    z = T * Y / e - (1 - T) * Y / (1 - e)
    fractions = np.arange(len(scores) + 1, dtype=float) / len(scores)
    gain = np.r_[0.0, np.cumsum(z[order]) / len(scores)]
    random_gain = fractions * gain[-1]
    return fractions, gain, random_gain


def paired_bootstrap(scores_a, scores_b, T, Y, e, mu0, mu1, metric="Qini", B=200, seed=42, k=0.3):
    """Paired bootstrap CI for the metric difference, preserving row pairing."""
    arrays = [np.asarray(a) for a in (scores_a, scores_b, T, Y, e, mu0, mu1)]
    n = len(arrays[0])
    if not all(len(a) == n for a in arrays):
        raise ValueError("all inputs must have equal length")
    observed = evaluate(*arrays[0:1], *arrays[2:], k=k, seed=seed)[metric] - evaluate(
        arrays[1], *arrays[2:], k=k, seed=seed
    )[metric]
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        diffs[i] = evaluate(arrays[0][idx], arrays[2][idx], arrays[3][idx], arrays[4][idx], arrays[5][idx], arrays[6][idx], k=k, seed=seed + i + 1)[metric] - evaluate(
            arrays[1][idx], arrays[2][idx], arrays[3][idx], arrays[4][idx], arrays[5][idx], arrays[6][idx], k=k, seed=seed + i + 1
        )[metric]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"estimate": float(observed), "lo": float(lo), "hi": float(hi)}


def fit_advanced(X, T, Y, seed: int = 42):
    """Cross-fitted AIPW DR learner and constant-effect DML estimate."""
    X, T, Y = _arrays(X, T, Y)
    kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
    pseudo = np.empty(len(Y))
    m_hat = np.empty(len(Y))
    e_hat = np.empty(len(Y))
    for train_idx, test_idx in kfold.split(X):
        nuisance = fit_nuisance(X[train_idx], T[train_idx], Y[train_idx], seed=seed)
        fold_scores = nuisance_scores(nuisance, X[test_idx], T[test_idx], Y[test_idx])
        pseudo[test_idx] = fold_scores["dr"]
        m_hat[test_idx] = (1 - fold_scores["e"]) * fold_scores["mu0"] + fold_scores["e"] * fold_scores["mu1"]
        e_hat[test_idx] = fold_scores["e"]
    dr_model = _regressor(seed + 10)
    dr_model.fit(X, pseudo)
    y_resid = Y - m_hat
    t_resid = T - e_hat
    denominator = float(np.mean(t_resid**2))
    if denominator <= 0:
        raise ValueError("propensity residual variance is zero")
    estimate = float(np.mean(t_resid * y_resid) / denominator)
    influence = t_resid * (y_resid - estimate * t_resid) / denominator
    se = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
    return {
        "dr_model": dr_model,
        "DR": dr_model,
        "pseudo_outcome": pseudo,
        "dml_ate": estimate,
        "dml_se": se,
        "DML": {"estimate": estimate, "se": se},
    }
