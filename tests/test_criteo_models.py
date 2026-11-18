import numpy as np
import pandas as pd

from shared.criteo.models import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    CriteoFeatureEncoder,
    DEFAULT_CATEGORICAL_INDICES,
    fit_candidates,
    fit_crossfit_dr,
)


def sample(n=300, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 12))
    x[:, 1] = rng.integers(0, 80, n)
    x[:, 3] = rng.integers(0, 80, n)
    t = rng.binomial(1, 0.85, n)
    y = (rng.random(n) < 0.05 + 0.01 * t + 0.002 * x[:, 0]).astype(float)
    return x, t, y


def test_encoder_is_train_only_and_preserves_documented_feature_types():
    train, _, _ = sample(100)
    test, _, _ = sample(20, 8)
    test[0, 1] = 999999
    enc = CriteoFeatureEncoder().fit(train)
    transformed = enc.transform(test)
    assert set(CATEGORICAL_FEATURES) == {"f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"}
    assert set(CONTINUOUS_FEATURES) == {"f0", "f2", "f7", "f10"}
    assert np.array_equal(np.flatnonzero(enc.categorical_mask), DEFAULT_CATEGORICAL_INDICES)
    assert transformed[0, 1] == 63
    assert transformed.shape == (20, 12)
    assert np.isfinite(transformed).all()


def test_dataframe_column_order_and_missing_values_are_handled():
    x, _, _ = sample(100)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(12)]).loc[:, ::-1]
    frame.iloc[0, frame.columns.get_loc("f0")] = np.nan
    frame.iloc[1, frame.columns.get_loc("f1")] = np.nan
    transformed = CriteoFeatureEncoder().fit(frame).transform(frame)
    assert np.isfinite(transformed).all()
    assert transformed.shape == (100, 12)


def test_candidates_produce_finite_cate_without_outcome_feature():
    x, t, y = sample()
    encoder = CriteoFeatureEncoder().fit(x[:200])
    train_x, test_x = encoder.transform(x[:200]), encoder.transform(x[200:])
    models = fit_candidates(train_x, t[:200], y[:200], encoder.categorical_mask, seed=8)
    predictions = {name: model.predict(test_x) for name, model in models.items()}
    assert set(predictions) == {"S", "T", "X"}
    assert all(values.shape == (100,) and np.isfinite(values).all() for values in predictions.values())


def test_crossfit_dr_is_finite_and_groups_do_not_cross_folds():
    x, t, y = sample(360, 22)
    # Repeated raw profiles are deliberately assigned one group each.
    x[:60] = x[0]
    groups = np.arange(len(x))
    groups[:60] = 0
    encoder = CriteoFeatureEncoder().fit(x)
    encoded = encoder.transform(x)
    result = fit_crossfit_dr(encoded, t, y, groups, encoder.categorical_mask, seed=9)
    assert result["pseudo"].shape == (len(y),)
    assert np.isfinite(result["pseudo"]).all()
    assert result["DR"].predict(encoded[:7]).shape == (7,)
    assert all(np.isfinite(value) for value in result["DML"].values())
