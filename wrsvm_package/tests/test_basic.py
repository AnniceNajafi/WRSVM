"""Tests covering all four decomposition strategies."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from wrsvm import WRSVMClassifier, inject_outliers_minority


@pytest.fixture
def toy_data():
    X, y = make_classification(
        n_samples=90, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, weights=[0.5, 0.3, 0.2],
        random_state=0,
    )
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.mark.parametrize("strategy", ["cs", "simmsvm", "ovo", "ovr"])
def test_fit_predict(toy_data, strategy):
    X, y = toy_data
    clf = WRSVMClassifier(strategy=strategy,
                          C=10.0, gamma=0.5, upsilon=0.2).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset(set(clf.classes_))
    assert (preds == y).mean() > 0.5


def test_decision_function_cs(toy_data):
    X, y = toy_data
    clf = WRSVMClassifier(strategy="cs",
                          C=10.0, gamma=0.5, upsilon=0.2).fit(X, y)
    scores = clf.decision_function(X)
    assert scores.shape == (X.shape[0], len(clf.classes_))


def test_decision_function_unsupported_on_ovo(toy_data):
    X, y = toy_data
    clf = WRSVMClassifier(strategy="ovo").fit(X, y)
    with pytest.raises(NotImplementedError):
        clf.decision_function(X)


def test_minority_noise_preserves_majority(toy_data):
    X, y = toy_data
    y_noisy = inject_outliers_minority(X, y, outlier_rate=0.3, seed=0)
    classes, sizes = np.unique(y, return_counts=True)
    maj = classes[sizes.argmax()]
    assert (y_noisy == maj).sum() >= (y == maj).sum()
