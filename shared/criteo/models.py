"""Criteo-specific encoders and meta-learners.

The corrected Criteo benchmark documents f0, f2, f7 and f10 as continuous;
the other eight anonymized fields are hashed categorical values.  This module
keeps that distinction explicit so it can be compared with an all-numeric
baseline without leaking validation/test category information into training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [f"f{i}" for i in range(12)]
CONTINUOUS_FEATURES = {"f0", "f2", "f7", "f10"}
CATEGORICAL_FEATURES = [name for name in FEATURES if name not in CONTINUOUS_FEATURES]
DEFAULT_CATEGORICAL_INDICES = np.array(
    [FEATURES.index(name) for name in CATEGORICAL_FEATURES], dtype=int
)
_CLIP = 0.03


class CriteoFeatureEncoder:
    """Fit train-only numeric imputation and categorical value mappings.

    Categorical values are encoded as integer levels.  At most 63 frequent
    training values are retained; code 63 is reserved for rare and unseen
    values.  This gives a 64-level categorical feature, within sklearn's
    ``max_bins=255`` limit.
    """

    def __init__(self, categorical_indices: Iterable[int] = DEFAULT_CATEGORICAL_INDICES):
        indices = np.asarray(list(categorical_indices), dtype=int)
        if np.any(indices < 0) or np.any(indices >= len(FEATURES)):
            raise ValueError("categorical indices must be between 0 and 11")
        self.categorical_indices = np.unique(indices)
        self.categorical_mask = np.zeros(len(FEATURES), dtype=bool)
        self.categorical_mask[self.categorical_indices] = True
        self._maps: dict[int, dict[float, int]] = {}
        self._medians: dict[int, float] = {}
        self._fitted = False

    @staticmethod
    def _array(values):
        if isinstance(values, pd.DataFrame):
            if set(FEATURES).issubset(values.columns):
                values = values[FEATURES]
            values = values.to_numpy()
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(FEATURES):
            raise ValueError("expected a 2-D array/DataFrame with 12 Criteo features")
        return array

    def fit(self, values):
        array = self._array(values)
        self._maps = {}
        self._medians = {}
        for column in range(array.shape[1]):
            values_column = array[:, column]
            if self.categorical_mask[column]:
                finite = values_column[np.isfinite(values_column)]
                counts = pd.Series(finite).value_counts()
                # Stable tie-break by value makes the mapping reproducible.
                frequent = sorted(counts.items(), key=lambda item: (-int(item[1]), float(item[0])))[:63]
                self._maps[column] = {float(value): code for code, (value, _) in enumerate(frequent)}
            else:
                finite = values_column[np.isfinite(values_column)]
                if len(finite) == 0:
                    raise ValueError(f"continuous feature {column} has no finite training values")
                self._medians[column] = float(np.median(finite))
        self._fitted = True
        return self

    def transform(self, values):
        if not self._fitted:
            raise RuntimeError("fit must be called before transform")
        array = self._array(values)
        result = np.empty_like(array, dtype=np.float64)
        for column in range(array.shape[1]):
            if self.categorical_mask[column]:
                mapping = self._maps[column]
                result[:, column] = [
                    mapping.get(float(value), 63) if np.isfinite(value) else 63
                    for value in array[:, column]
                ]
            else:
                result[:, column] = np.where(
                    np.isfinite(array[:, column]), array[:, column], self._medians[column]
                )
        return result

    def fit_transform(self, values):
        return self.fit(values).transform(values)


def _regressor(seed: int, categorical_mask=None):
    mask = None if categorical_mask is None else np.asarray(categorical_mask, dtype=bool)
    return HistGradientBoostingRegressor(
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        learning_rate=0.06,
        l2_regularization=2.0,
        early_stopping=False,
        categorical_features=mask,
        random_state=seed,
    )


def _validate(X, T, Y):
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.int8).reshape(-1)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[1] != 12 or len(X) != len(T) or len(T) != len(Y):
        raise ValueError("X must have shape (n, 12) and match T/Y")
    if not np.isin(T, [0, 1]).all() or np.unique(T).size != 2:
        raise ValueError("T must contain both 0 and 1")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("X and Y must be finite after encoding")
    return X, T, Y


class _ConstantPropensity:
    def __init__(self, value):
        self.value = float(np.clip(value, _CLIP, 1 - _CLIP))

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.value), np.full(len(X), self.value)])


@dataclass
class _S:
    model: object

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict(np.column_stack([X, np.ones(len(X))])) - self.model.predict(
            np.column_stack([X, np.zeros(len(X))])
        )


@dataclass
class _T:
    control: object
    treated: object

    def predict(self, X):
        return self.treated.predict(X) - self.control.predict(X)


@dataclass
class _X:
    control: object
    treated: object
    tau_control: object
    tau_treated: object
    propensity: object

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        e = np.clip(self.propensity.predict_proba(X)[:, 1], _CLIP, 1 - _CLIP)
        return e * self.tau_control.predict(X) + (1 - e) * self.tau_treated.predict(X)


def fit_outcomes(X, T, Y, categorical_mask=None, seed=42):
    """Fit the two outcome regressions used by the T/X learners."""
    X, T, Y = _validate(X, T, Y)
    control = _regressor(seed, categorical_mask)
    treated = _regressor(seed + 1, categorical_mask)
    control.fit(X[T == 0], Y[T == 0])
    treated.fit(X[T == 1], Y[T == 1])
    return control, treated


def fit_candidates(X, T, Y, categorical_mask=None, seed=42):
    """Fit S/T/X learners with identical outcome-model hyperparameters."""
    X, T, Y = _validate(X, T, Y)
    if categorical_mask is not None and len(categorical_mask) != X.shape[1]:
        raise ValueError("categorical_mask must have 12 entries")
    s = _regressor(seed, None if categorical_mask is None else [*categorical_mask, False])
    s.fit(np.column_stack([X, T]), Y)
    control, treated = fit_outcomes(X, T, Y, categorical_mask, seed + 1)

    d_control = treated.predict(X[T == 0]) - Y[T == 0]
    d_treated = Y[T == 1] - control.predict(X[T == 1])
    tau_control = _regressor(seed + 3, categorical_mask)
    tau_treated = _regressor(seed + 4, categorical_mask)
    tau_control.fit(X[T == 0], d_control)
    tau_treated.fit(X[T == 1], d_treated)
    propensity = _ConstantPropensity(np.mean(T))
    return {
        "S": _S(s),
        "T": _T(control, treated),
        "X": _X(control, treated, tau_control, tau_treated, propensity),
    }


def fit_crossfit_dr(X, T, Y, groups, categorical_mask=None, seed=42):
    """Fit grouped cross-fitted DR and constant-effect PLR/DML estimates.

    ``groups`` should identify identical raw feature profiles.  GroupKFold
    keeps each profile in a single fold, preventing duplicated anonymized
    profiles from crossing a nuisance-model validation boundary.  The DML
    number is a constant-effect partially-linear estimate; it is not a CATE
    estimate under heterogeneous effects.
    """
    X, T, Y = _validate(X, T, Y)
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(Y):
        raise ValueError("groups must be a one-dimensional array matching X")
    if np.unique(groups).size < 3:
        raise ValueError("at least three groups are required for 3-fold cross-fitting")

    pseudo = np.empty(len(Y), dtype=float)
    m_hat = np.empty(len(Y), dtype=float)
    e_hat = np.empty(len(Y), dtype=float)
    splitter = GroupKFold(n_splits=3)
    for train_idx, holdout_idx in splitter.split(X, T, groups):
        if np.unique(T[train_idx]).size < 2:
            raise ValueError("each nuisance training fold must contain both treatment arms")
        control, treated = fit_outcomes(
            X[train_idx], T[train_idx], Y[train_idx], categorical_mask, seed=seed
        )
        e = float(np.clip(np.mean(T[train_idx]), _CLIP, 1 - _CLIP))
        # ``control`` predicts Y(0), ``treated`` predicts Y(1).
        mu0 = control.predict(X[holdout_idx])
        mu1 = treated.predict(X[holdout_idx])
        m_hat[holdout_idx] = (1 - e) * mu0 + e * mu1
        e_hat[holdout_idx] = e
        th = T[holdout_idx]
        yh = Y[holdout_idx]
        pseudo[holdout_idx] = mu1 - mu0 + th * (yh - mu1) / e - (1 - th) * (yh - mu0) / (1 - e)

    t_resid = T - e_hat
    y_resid = Y - m_hat
    denominator = float(np.mean(t_resid**2))
    if denominator <= 0:
        raise ValueError("cross-fitted treatment residual variance is zero")
    estimate = float(np.mean(t_resid * y_resid) / denominator)
    influence = t_resid * (y_resid - estimate * t_resid) / denominator
    se = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
    final_model = _regressor(seed + 100, categorical_mask)
    final_model.fit(X, pseudo)
    return {
        "DR": final_model,
        "DML": {
            "estimate": estimate,
            "se": se,
            "lo": estimate - 1.96 * se,
            "hi": estimate + 1.96 * se,
        },
        "pseudo": pseudo,
    }
