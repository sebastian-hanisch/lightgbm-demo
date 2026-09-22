"""LightGBM (Ke et al. 2017): dieselbe additive Boosting-Idee und dasselbe regularisierte Ziel wie xgboost-demo (`F_m = F_{m-1} + Lernrate * Baum_m`, Blattwert `-G/(H+λ)`, Split-Gain
`0.5*[GL²/(HL+λ)+GR²/(HR+λ)-G²/(H+λ)]-γ`) - der Unterschied liegt allein im Baumkern (`lgbm_tree.py`): Histogramm-Split-Suche statt exakter Suche, blattweises statt ebenenweises Wachsen."""

from dataclasses import dataclass

import numpy as np

import lgbm_tree as T

EPS = 1e-12


@dataclass(frozen=True)
class Ensemble:
    trees: tuple
    f0: float
    learning_rate: float
    task: str
    lam: float
    gamma: float
    n_train: int
    edges: tuple


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def grad_hess(y, F, task):
    if task == "reg":
        return F - y, np.ones_like(F)
    p = _sigmoid(F)
    return p - y, p * (1.0 - p)


def loss_value(y, F, task):
    if task == "reg":
        return 0.5 * (y - F) ** 2
    p = np.clip(_sigmoid(F), EPS, 1.0 - EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def init_value(y, task):
    if task == "reg":
        return float(np.mean(y))
    p = np.clip(float(np.mean(y)), 1e-6, 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


def fit(X, y, task, num_leaves=31, max_depth=None, max_bin=63, lam=1.0, gamma=0.0, min_child_weight=1.0, min_child_samples=1,
        n_rounds=60, learning_rate=0.1, subsample=1.0, seed=0, policy="leaf"):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    edges = T.build_bin_edges(X, max_bin)
    f0 = init_value(y, task)
    F = np.full(n, f0)
    rng = np.random.default_rng(seed)
    trees = []
    for _ in range(n_rounds):
        if subsample < 1.0:
            k = max(2, int(round(subsample * n)))
            idx = np.sort(rng.choice(n, k, replace=False))
        else:
            idx = np.arange(n)
        g, h = grad_hess(y[idx], F[idx], task)
        tree = T.grow(X[idx], g, h, edges, num_leaves=num_leaves, max_depth=max_depth, lam=lam, gamma=gamma,
                       min_child_weight=min_child_weight, min_child_samples=min_child_samples, policy=policy)
        F = F + learning_rate * T.predict_value(tree, X)
        trees.append(tree)
    return Ensemble(tuple(trees), f0, learning_rate, task, lam, gamma, n, tuple(edges))


def predict_raw(ensemble, X, upto=None):
    trees = ensemble.trees[:upto] if upto else ensemble.trees
    F = np.full(len(X), ensemble.f0)
    for t in trees:
        F = F + ensemble.learning_rate * T.predict_value(t, X)
    return F


def predict_value(ensemble, X, upto=None):
    F = predict_raw(ensemble, X, upto)
    return F if ensemble.task == "reg" else _sigmoid(F)


def predict(ensemble, X, upto=None):
    v = predict_value(ensemble, X, upto)
    return (v > 0.5).astype(int) if ensemble.task == "class" else v


def n_leaves_total(ensemble):
    return int(sum(t.n_leaves for t in ensemble.trees))
