import numpy as np

from shared.causal_utils import _XLearner, evaluate, fit_advanced, fit_meta, fit_nuisance, nuisance_scores, paired_bootstrap


def synthetic(seed=7, n=600):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-0.5 * x[:, 0]))
    t = rng.binomial(1, p)
    tau = 0.4 + 0.5 * x[:, 0]
    y = 0.2 * x[:, 1] + t * tau + rng.normal(scale=0.25, size=n)
    return x, t, y


def test_meta_learners_and_nuisance_shapes():
    x, t, y = synthetic()
    models = fit_meta(x, t, y)
    assert set(models) == {"S", "T", "X"}
    assert all(models[name].predict(x[:17]).shape == (17,) for name in models)
    nuisance = fit_nuisance(x, t, y)
    scores = nuisance_scores(nuisance, x[:31], t[:31], y[:31])
    assert all(scores[name].shape == (31,) for name in scores)
    assert np.all((scores["e"] > 0) & (scores["e"] < 1))


def test_evaluation_is_deterministic():
    x, t, y = synthetic()
    nuisance = fit_nuisance(x, t, y)
    ns = nuisance_scores(nuisance, x, t, y)
    metrics_a = evaluate(ns["dr"], t, y, ns["e"], ns["mu0"], ns["mu1"], seed=11)
    metrics_b = evaluate(ns["dr"], t, y, ns["e"], ns["mu0"], ns["mu1"], seed=11)
    assert metrics_a == metrics_b
    assert set(metrics_a) == {"AUUC", "Qini", "Uplift@30%", "IncrementalRate", "PolicyValue", "RandomValue", "PolicyGain"}
    assert np.isfinite(metrics_a["Qini"])


def test_paired_bootstrap_and_advanced_outputs():
    x, t, y = synthetic(n=300)
    nuisance = fit_nuisance(x, t, y)
    ns = nuisance_scores(nuisance, x, t, y)
    result = paired_bootstrap(ns["dr"], np.zeros(len(y)), t, y, ns["e"], ns["mu0"], ns["mu1"], B=12, seed=3)
    assert set(result) == {"estimate", "lo", "hi"}
    assert all(np.isfinite(list(result.values())))
    advanced = fit_advanced(x, t, y)
    assert advanced["pseudo_outcome"].shape == (len(y),)
    assert np.isfinite(advanced["dml_ate"]) and advanced["dml_se"] >= 0
    assert advanced["dr_model"].predict(x[:5]).shape == (5,)


def test_incremental_rate_uses_population_denominator():
    scores = np.array([4.0, 3.0, 2.0, 1.0])
    treatment = np.array([1, 0, 1, 0])
    outcome = np.array([1, 0, 0, 0])
    e = np.full(4, 0.5)
    metrics = evaluate(scores, treatment, outcome, e, np.zeros(4), np.zeros(4), k=0.5)
    # Top two rows have IPW effects 2 and 0; divide by all four rows.
    assert metrics["IncrementalRate"] == 0.5
    assert metrics["AUUC"] == 0.4375
    assert metrics["Qini"] == 0.1875
    assert metrics["PolicyValue"] == 0.5
    assert metrics["RandomValue"] == 0.25
    assert metrics["PolicyGain"] == 0.25
    full = evaluate(scores, treatment, outcome, e, np.zeros(4), np.zeros(4), k=1)
    assert full["PolicyGain"] == 0


def test_x_learner_combines_control_and_treated_imputations_with_propensity_weights():
    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)

    class Propensity:
        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.25), np.full(len(X), 0.75)])

    model = _XLearner(Constant(0), Constant(0), Constant(2), Constant(6), Propensity())
    # e*tau_control + (1-e)*tau_treated = .75*2 + .25*6.
    assert np.allclose(model.predict(np.zeros((3, 1))), 3.0)


def test_dml_recovers_constant_effect_with_confounding():
    rng = np.random.default_rng(19)
    n = 1200
    x = rng.normal(size=(n, 2))
    p = 1 / (1 + np.exp(-(1.3 * x[:, 0] - 0.7 * x[:, 1])))
    treatment = rng.binomial(1, p)
    true_effect = 1.25
    outcome = 2.0 * x[:, 0] - 1.5 * x[:, 1] + true_effect * treatment + rng.normal(0, 0.15, n)
    result = fit_advanced(x, treatment, outcome, seed=23)
    assert abs(result["DML"]["estimate"] - true_effect) < 0.15
