"""Tests for the robust_exact tree method (ported from RobustTrees)."""
import json

import numpy as np
import pytest

import testing as tm
import xgboost as xgb

pytestmark = pytest.mark.skipif(**tm.no_sklearn())


def _moons():
    from sklearn.datasets import make_moons
    return make_moons(n_samples=200, noise=0.3, random_state=0)


def _train(tree_method, **extra):
    X, y = _moons()
    params = {"objective": "binary:logistic", "tree_method": tree_method,
              "max_depth": 3, "eta": 0.1, "nthread": 1, **extra}
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=20), X, y


def test_robust_params_in_config():
    bst, _, _ = _train("robust_exact", robust_eps=0.05)
    cfg = json.loads(bst.save_config())
    gbm = cfg["learner"]["gradient_booster"]
    assert gbm["gbtree_train_param"]["updater"] == "robust_grow_colmaker,prune"
    rparam = gbm["updater"]["robust_grow_colmaker"]["robust_colmaker_train_param"]
    assert float(rparam["robust_eps"]) == pytest.approx(0.05)


def test_robust_eps_zero_matches_exact():
    exact, X, _ = _train("exact")
    robust, _, _ = _train("robust_exact", robust_eps=0.0)
    dm = xgb.DMatrix(X)
    np.testing.assert_allclose(exact.predict(dm), robust.predict(dm), rtol=1e-6)


def test_robust_eps_changes_model():
    exact, X, _ = _train("exact")
    robust, _, _ = _train("robust_exact", robust_eps=0.3)
    dm = xgb.DMatrix(X)
    assert not np.allclose(exact.predict(dm), robust.predict(dm))


def test_config_round_trip():
    bst, X, _ = _train("robust_exact", robust_eps=0.3)
    cfg = bst.save_config()
    other = xgb.Booster()
    other.load_model(bst.save_raw())
    other.load_config(cfg)
    assert json.loads(other.save_config()) == json.loads(cfg)
    dm = xgb.DMatrix(X)
    np.testing.assert_allclose(bst.predict(dm), other.predict(dm))


def test_sklearn_robust_exact():
    X, y = _moons()
    clf = xgb.XGBClassifier(tree_method="robust_exact", robust_eps=0.05,
                            max_depth=3, learning_rate=0.1, n_estimators=20)
    clf.fit(X, y)
    assert np.mean(clf.predict(X) == y) > 0.8
